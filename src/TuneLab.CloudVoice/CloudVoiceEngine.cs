using System.Net.Http;
using System.Text.Json;
using TuneLab.Foundation;
using TuneLab.SDK;
using TuneLab.CloudVoice.Providers;
using TuneLab.CloudVoice.Voice;

namespace TuneLab.CloudVoice;

// TuneLab V1 voice 插件：云端歌声合成引擎。
// 引擎 id "CloudVoice" 声明在 manifest.json（写进工程文件，不可变）；本类负责生命周期、声库目录、
// 后端能力 schema（属性面板 / 自动化轨 / 回显轨 / 自动音高）与声明面。
// 所有合成状态归 CloudVoiceSession；声明面全部是纯函数（宿主会在参数 commit 时重算并 diff 到 UI）。
public sealed class CloudVoiceEngine : IVoiceSynthesisEngine, IExtensionSettings
{
    public CloudVoiceEngine()
    {
        RebuildCatalog();
    }

    // 插件包目录：DLL 经 Assembly.Location 自定位（勿用工作目录 / AppContext.BaseDirectory，那是宿主目录）。
    public static string PackageDir
    {
        get
        {
            try
            {
                return System.IO.Path.GetDirectoryName(typeof(CloudVoiceEngine).Assembly.Location) ?? "";
            }
            catch (Exception)
            {
                return "";
            }
        }
    }

    // —— 引擎生命周期 ——

    // 声库目录：Init 期构建缓存，get 立即返回。宿主/UI 会同步读，绝不可阻塞。
    public IReadOnlyOrderedMap<string, VoiceSourceInfo> VoiceSourceInfos => mInfos;
    public IReadOnlyList<VoiceSourceLayoutItem> VoiceSourceLayout => mLayout;

    // 懒调用（首次用到前）；失败抛异常由宿主在调用边界捕获并归责插件。
    public void Init()
    {
        mInitialized = true;
        RebuildCatalog();
    }

    public void Destroy() { }

    // 每个 part 一个会话；voiceId 烘在 context.VoiceId，按路由决定用哪个 provider。
    public IVoiceSynthesisSession CreateSession(IVoiceSynthesisContext context)
    {
        if (!mInitialized)
            Init();

        if (!mRoutes.TryGetValue(context.VoiceId, out var route))
        {
            // 目录里没有这个 id（服务商改了声库 / 旧工程）：有服务地址按 http 试一次，
            // 否则落到内置演示。
            route = mSettings.HasCloudEndpoint
                ? new CloudVoiceRoute(CloudVoiceSettings.ProviderHttp, context.VoiceId)
                : new CloudVoiceRoute(CloudVoiceSettings.ProviderMock, context.VoiceId);
        }

        ICloudVoiceProvider provider = route.ProviderKind == CloudVoiceSettings.ProviderHttp
            ? new HttpCloudVoiceProvider(mSettings)
            : MockCloudVoiceProvider.Instance;

        return new CloudVoiceSession(context, mSettings, route, provider, mSchema, this);
    }

    // —— 声明面：全部来自服务商 schema（老服务端回退到内置默认声明） ——

    public IReadOnlyOrderedMap<PropertyKey, AutomationConfig> GetAutomationConfigs(IVoiceSynthesisPartPropertyContext context)
        => mAutomationConfigs;

    public IReadOnlyOrderedMap<PropertyKey, AutomationConfig> GetSynthesizedParameterConfigs(IVoiceSynthesisPartPropertyContext context)
        => mSynthesizedParameterConfigs;

    public ObjectConfig GetPartPropertyConfig(IVoiceSynthesisPartPropertyContext context)
        => mPartConfig;

    public ObjectConfig GetNotePropertyConfig(IVoiceSynthesisNotePropertyContext context)
        => mNoteConfig;

    // per-phoneme 属性：本引擎暂不声明（空 map = 所有音素无属性）。
    public IReadOnlyMap<int, ObjectConfig> GetPhonemePropertyConfigs(IVoiceSynthesisNotePropertyContext context)
        => mPhonemeConfigs;

    // —— IExtensionSettings：扩展级持久设置（端点 / API Key） ——

    public ObjectConfig GetSettingsConfig(IExtensionSettingsContext context)
        => CloudVoiceSettings.GetSettingsConfig(context.Settings, L10n.CurrentLanguage);

    // 宿主在加载完成后灌一次（早于 Init / 会话），用户在设置窗保存后再灌一次。
    public void ApplySettings(PropertyObject settings)
    {
        mSettings = CloudVoiceSettings.FromPropertyObject(settings);
        TuneLabContext.Global.GetLogger().Info(
            "[CloudVoice] settings applied: endpoint="
            + (mSettings.HasCloudEndpoint ? mSettings.Endpoint : "(none, offline demo)"));
        RebuildCatalog();
    }

    // —— 内部：声库目录 + 后端 schema 拉取 ——

    // 没填服务地址 -> 内置离线演示；填了地址 -> 自动 GET /voices（可选带 schema），
    // 再按需 GET /schema。网络失败时退回演示声库 + 默认声明，插件仍然可用。
    void RebuildCatalog()
    {
        var snapshot = FetchCatalogAndSchema();
        ApplyCatalog(snapshot.Catalog, snapshot.Schema);
    }

    sealed record CatalogSnapshot(CloudVoiceCatalog Catalog, CloudVoiceSchema Schema);

    CatalogSnapshot FetchCatalogAndSchema()
    {
        var language = L10n.CurrentLanguage;
        var logger = TuneLabContext.Global.GetLogger();
        var offline = new CatalogSnapshot(CloudVoiceCatalogBuilder.BuildOfflineDemo(PackageDir, language), CloudVoiceSchema.Defaults);

        if (!mSettings.HasCloudEndpoint)
            return offline;

        var voicesUrl = mSettings.ResolveVoicesUrl();
        if (string.IsNullOrWhiteSpace(voicesUrl))
        {
            logger.Warning("[CloudVoice] invalid service URL: " + mSettings.Endpoint + " (falling back to offline demo)");
            return offline;
        }

        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            if (!string.IsNullOrWhiteSpace(mSettings.ApiKey))
                http.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", "Bearer " + mSettings.ApiKey);

            var json = http.GetStringAsync(voicesUrl).GetAwaiter().GetResult();
            var catalogResult = CloudVoiceCatalogBuilder.ParseProviderCatalog(json, PackageDir, language);
            if (!catalogResult.Success || catalogResult.Catalog == null)
            {
                logger.Warning("[CloudVoice] provider voice catalog rejected (" + voicesUrl + "): " + catalogResult.Error);
                return offline;
            }

            var schema = FetchSchema(http, json, voicesUrl, logger, language);
            logger.Info("[CloudVoice] pulled " + catalogResult.Catalog.Routes.Count + " voicebank(s) from " + voicesUrl
                + "; schema: " + Describe(schema));
            return new CatalogSnapshot(catalogResult.Catalog, schema);
        }
        catch (Exception exception)
        {
            logger.Warning("[CloudVoice] could not pull the voice catalog from " + voicesUrl + ": " + exception.Message);
            return offline;
        }
    }

    // schema 可以随 /voices 一起返回（顶层 "schema" 字段），也可以单独 GET <voices 同级的 /schema>。
    // 老服务端两者都没有 -> 回退 Defaults（= 改版前的固定声明），保证向后兼容。
    CloudVoiceSchema FetchSchema(HttpClient http, string voicesJson, string voicesUrl, ILogger logger, string? language)
    {
        var schemaJson = ExtractJsonProperty(voicesJson, "schema");
        if (schemaJson == null)
        {
            var schemaUrl = ResolveSiblingUrl(voicesUrl, "schema");
            if (!string.IsNullOrWhiteSpace(schemaUrl))
            {
                try
                {
                    schemaJson = http.GetStringAsync(schemaUrl).GetAwaiter().GetResult();
                }
                catch (Exception exception)
                {
                    logger.Info("[CloudVoice] no schema at " + schemaUrl + " (" + exception.Message + "); using the default declarations.");
                }
            }
        }

        if (string.IsNullOrWhiteSpace(schemaJson))
            return CloudVoiceSchema.Defaults;

        var schema = CloudVoiceSchema.Parse(schemaJson);
        if (schema == null)
        {
            logger.Warning("[CloudVoice] provider schema is invalid; using the default declarations.");
            return CloudVoiceSchema.Defaults;
        }
        return schema;
    }

    static string Describe(CloudVoiceSchema schema)
        => schema.PartProperties.Count + " part prop(s), "
           + schema.NoteProperties.Count + " note prop(s), "
           + schema.Automations.Count + " automation(s), "
           + schema.SynthesizedParameters.Count + " echo track(s), pitch."
           + (schema.PitchAuto ? "auto" : "manual");

    static string? ExtractJsonProperty(string json, string name)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty(name, out var value))
                return value.GetRawText();
        }
        catch (JsonException)
        {
            // ignore: the caller validates the catalog separately
        }
        return null;
    }

    static string ResolveSiblingUrl(string url, string sibling)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
            return "";
        string path = uri.AbsolutePath;
        int slash = path.LastIndexOf('/');
        string target = slash >= 0 ? path[..(slash + 1)] + sibling : "/" + sibling;
        return new UriBuilder(uri) { Path = target, Query = "", Fragment = "" }.Uri.ToString();
    }

    // 服务端的 schema 变了（例如先启动了旧服务端、后来换成带 /schema 的新服务端，用户再保存设置）：
    // 通知所有活跃会话重新询问声明，属性面板/画线轨不用重启 TuneLab 就能刷新。
    internal event Action<CloudVoiceSchema>? SchemaChanged;

    void ApplyCatalog(CloudVoiceCatalog catalog, CloudVoiceSchema schema)
    {
        bool schemaChanged = mSchema.Signature != schema.Signature;
        var routes = new Dictionary<string, CloudVoiceRoute>(StringComparer.Ordinal);
        foreach (var pair in catalog.Routes)
            routes[pair.Key] = pair.Value;

        mInfos = catalog.Infos;
        mLayout = catalog.Layout;
        mRoutes = routes;
        mSchema = schema;
        mPartConfig = schema.BuildPartPropertyConfig();
        mNoteConfig = schema.BuildNotePropertyConfig();
        mAutomationConfigs = schema.BuildAutomationConfigs();
        mSynthesizedParameterConfigs = schema.BuildSynthesizedParameterConfigs();

        if (schemaChanged)
            SchemaChanged?.Invoke(schema);
    }

    CloudVoiceSettings mSettings = CloudVoiceSettings.Defaults;
    IReadOnlyOrderedMap<string, VoiceSourceInfo> mInfos = new OrderedMap<string, VoiceSourceInfo>();
    IReadOnlyList<VoiceSourceLayoutItem> mLayout = [];
    IReadOnlyDictionary<string, CloudVoiceRoute> mRoutes = new Dictionary<string, CloudVoiceRoute>();
    CloudVoiceSchema mSchema = CloudVoiceSchema.Defaults;
    bool mInitialized;

    ObjectConfig mPartConfig = CloudVoiceSchema.Defaults.BuildPartPropertyConfig();
    ObjectConfig mNoteConfig = CloudVoiceSchema.Defaults.BuildNotePropertyConfig();
    OrderedMap<PropertyKey, AutomationConfig> mAutomationConfigs = CloudVoiceSchema.Defaults.BuildAutomationConfigs();
    OrderedMap<PropertyKey, AutomationConfig> mSynthesizedParameterConfigs = CloudVoiceSchema.Defaults.BuildSynthesizedParameterConfigs();
    readonly IReadOnlyMap<int, ObjectConfig> mPhonemeConfigs = new Map<int, ObjectConfig>();
}
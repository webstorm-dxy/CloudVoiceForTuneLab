using TuneLab.Foundation;
using TuneLab.SDK;

namespace TuneLab.CloudVoice;

// 扩展设置（IExtensionSettings）：用户只需要填服务地址与 API Key。
// 声库目录不在这里配置——插件在 Init 时自动向服务商 GET <endpoint 同级的 /voices> 拉取。
public sealed record CloudVoiceSettings
{
    public const string ProviderMock = "mock";
    public const string ProviderHttp = "http";

    public string Endpoint { get; init; } = "";
    public string ApiKey { get; init; } = "";
    public int SampleRate { get; init; } = 48000;
    public int TimeoutSeconds { get; init; } = 120;
    public int Retries { get; init; } = 1;

    public static CloudVoiceSettings Defaults { get; } = new();

    // 没填地址时插件退回内置离线演示声库；填了地址就只列服务商拉回来的声库。
    public bool HasCloudEndpoint => !string.IsNullOrWhiteSpace(Endpoint);

    // 服务商声库目录端点：把 synth endpoint 的最后一段换成 voices
    // （http://host/v1/synthesize -> http://host/v1/voices，http://host/synthesize -> http://host/voices）。
    public string ResolveVoicesUrl()
    {
        if (!Uri.TryCreate(Endpoint.Trim(), UriKind.Absolute, out var endpoint))
            return "";
        string path = endpoint.AbsolutePath;
        int slash = path.LastIndexOf('/');
        string voicesPath = slash >= 0 ? path[..(slash + 1)] + "voices" : "/voices";
        return new UriBuilder(endpoint) { Path = voicesPath, Query = "", Fragment = "" }.Uri.ToString();
    }

    public static CloudVoiceSettings FromPropertyObject(PropertyObject settings)
    {
        var defaults = Defaults;
        return new CloudVoiceSettings
        {
            Endpoint = settings.GetString("endpoint", defaults.Endpoint).Trim(),
            ApiKey = settings.GetString("api_key", defaults.ApiKey),
        };
    }

    // 设置面板 schema：只有两个字段，且必须是 context.Settings 的纯函数、Init 之前可调。
    // 注意：不能使用 ButtonConfig / IButtonClickContext —— 它们不在 TuneLab 2.1.x 已发布宿主 SDK 里，
    // 用了会导致扩展整个加载失败（built against an incompatible SDK）。
    public static ObjectConfig GetSettingsConfig(PropertyObject current, string? language)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>();
        properties.Add(("endpoint", L10n.T(language, "Service URL", "服务地址")),
            TextBoxConfig.Create(current.GetString("endpoint", "")));
        properties.Add(("api_key", L10n.T(language, "API key", "API 密钥")),
            TextBoxConfig.Create(current.GetString("api_key", "")).WithPassword());
        return ObjectConfig.Create(properties);
    }
}
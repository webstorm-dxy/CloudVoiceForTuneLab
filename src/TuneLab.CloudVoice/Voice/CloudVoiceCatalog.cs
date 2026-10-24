using System.Text.Json;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace TuneLab.CloudVoice;

// 一个声库 id 到「用哪个 provider、provider 侧传什么 voice id」的映射。
public sealed record CloudVoiceRoute(string ProviderKind, string ProviderVoiceId);

// 引擎向宿主暴露的声库目录（一次性构建、get 立即返回；Init 期拉取/缓存的口径）。
public sealed class CloudVoiceCatalog
{
    public OrderedMap<string, VoiceSourceInfo> Infos { get; } = new();
    public List<VoiceSourceLayoutItem> Layout { get; } = new();
    public Dictionary<string, CloudVoiceRoute> Routes { get; } = new(StringComparer.Ordinal);
}

// 声库目录构建：
//   · BuildOfflineDemo：没填服务地址时用的内置离线演示声库；
//   · ParseProviderCatalog：解析服务商 GET /voices 返回的声库目录（插件自动拉取，不需要用户手填）。
public static class CloudVoiceCatalogBuilder
{
    public sealed record ProviderCatalogResult(bool Success, CloudVoiceCatalog? Catalog, string? Error);

    public static CloudVoiceCatalog BuildOfflineDemo(string packageDir, string? language)
    {
        var catalog = new CloudVoiceCatalog();
        var ids = new List<string>();
        foreach (var voice in MockVoices)
        {
            var id = voice.Id;
            catalog.Infos.Add(id, new VoiceSourceInfo
            {
                Name = L10n.T(language, voice.NameEn, voice.NameZh),
                Description = L10n.T(language, voice.DescriptionEn, voice.DescriptionZh),
            });
            catalog.Routes[id] = new CloudVoiceRoute(CloudVoiceSettings.ProviderMock, id);
            ids.Add(id);
        }
        if (ids.Count > 0)
            catalog.Layout.Add(VoiceSourceLayoutItem.Group(
                L10n.T(language, "Built-in demo (offline)", "内置演示（离线）"),
                ids.Select(VoiceSourceLayoutItem.Voice).ToList()));
        return catalog;
    }

    // 兼容两种响应形态：{ "voices": [...] } 或顶层数组。每个条目至少要有 id；
    // 可选 voice（服务商侧 voice 参数，缺省 = id）、description、portrait（包内/绝对路径）。
    public static ProviderCatalogResult ParseProviderCatalog(string json, string packageDir, string? language)
    {
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException exception)
        {
            return new ProviderCatalogResult(false, null, "invalid JSON: " + exception.Message);
        }

        using (document)
        {
            JsonElement root = document.RootElement;
            JsonElement array;
            if (root.ValueKind == JsonValueKind.Array)
                array = root;
            else if (root.ValueKind == JsonValueKind.Object
                     && root.TryGetProperty("voices", out var voices)
                     && voices.ValueKind == JsonValueKind.Array)
                array = voices;
            else
                return new ProviderCatalogResult(false, null, "response has no 'voices' array");

            var catalog = new CloudVoiceCatalog();
            var ids = new List<string>();
            foreach (var entry in array.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Object)
                    continue;
                string? id = ReadString(entry, "id");
                if (string.IsNullOrWhiteSpace(id))
                    continue;
                id = id.Trim();
                if (catalog.Infos.ContainsKey(id))
                    continue;
                var name = ReadString(entry, "name");
                var description = ReadString(entry, "description") ?? "";
                var providerVoice = ReadString(entry, "voice");
                catalog.Infos.Add(id, new VoiceSourceInfo
                {
                    Name = string.IsNullOrWhiteSpace(name) ? id : name!,
                    Description = description,
                    Portrait = ResolvePortrait(packageDir, ReadString(entry, "portrait")),
                });
                catalog.Routes[id] = new CloudVoiceRoute(
                    CloudVoiceSettings.ProviderHttp,
                    string.IsNullOrWhiteSpace(providerVoice) ? id : providerVoice!.Trim());
                ids.Add(id);
            }

            if (ids.Count == 0)
                return new ProviderCatalogResult(false, null, "the provider returned an empty voice list");

            catalog.Layout.Add(VoiceSourceLayoutItem.Group(
                L10n.T(language, "Cloud (provider)", "云端（服务商）"),
                ids.Select(VoiceSourceLayoutItem.Voice).ToList()));
            return new ProviderCatalogResult(true, catalog, null);
        }
    }

    static string? ReadString(JsonElement element, string property)
    {
        if (!element.TryGetProperty(property, out var value))
            return null;
        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Number => value.GetRawText(),
            _ => null,
        };
    }

    static ImageResource? ResolvePortrait(string packageDir, string? portrait)
    {
        if (string.IsNullOrWhiteSpace(portrait) || string.IsNullOrEmpty(packageDir))
            return null;
        try
        {
            var path = Path.IsPathRooted(portrait) ? portrait : Path.Combine(packageDir, portrait);
            return File.Exists(path) ? new FileImageResource(path) : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    static readonly (string Id, string NameEn, string NameZh, string DescriptionEn, string DescriptionZh)[] MockVoices =
    [
        ("mock-soprano", "Demo · Soprano", "演示 · 女高音",
            "Offline mock engine: additive sine voice with vibrato, soprano timbre. Fill in a service URL to use the provider's voicebanks.",
            "离线 mock 引擎：带颤音的正弦叠加女高音音色；填写服务地址后改用服务商声库。"),
        ("mock-alto", "Demo · Alto", "演示 · 中音",
            "Offline mock engine: warmer alto timbre. Needs no network.",
            "离线 mock 引擎：更温暖的中音音色，无需网络。"),
        ("mock-whisper", "Demo · Whisper", "演示 · 气声",
            "Offline mock engine: soft breathy timbre. Needs no network.",
            "离线 mock 引擎：轻柔气声音色，无需网络。"),
    ];
}
using System.Globalization;
using System.Text.Json;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace TuneLab.CloudVoice;

// —— 后端能力 schema：属性面板 / 可画线自动化轨 / 合成回显轨 / 自动音高 都由服务商返回 ——

public enum CloudPropertyKind { Slider, Integer, Number, Combo, Checkbox, Text }

public sealed record CloudPropertyOption(string Value, string? Label);

// 一个可在属性面板里编辑的标量参数（part 级或 note 级）。
// NumberDefault / TextDefault / BooleanDefault 按 Kind 取对应默认值；Combo 用 TextDefault。
public sealed record CloudPropertySpec(
    string Id,
    string? Label,
    CloudPropertyKind Kind,
    double NumberDefault,
    double Min,
    double Max,
    double Step,
    string? Unit,
    string? TextDefault,
    bool BooleanDefault,
    IReadOnlyList<CloudPropertyOption> Options);

// 一条由用户画线的自动化轨。Default == null 表示分段型（无数据 = 不绘制，插值不越过空段）。
public sealed record CloudAutomationSpec(
    string Id,
    string? Label,
    double Min,
    double Max,
    double? Default,
    string? Color,
    string? Unit,
    string? MinLabel,
    string? MaxLabel);

// 一条只读回显轨（合成产物）：参数值来自服务商响应。
public sealed record CloudSynthesizedParameterSpec(
    string Id,
    string? Label,
    double Min,
    double Max,
    string? Color,
    string? Unit);

public sealed class CloudVoiceSchema
{
    public bool PitchAuto { get; init; }
    public IReadOnlyList<CloudPropertySpec> PartProperties { get; init; } = [];
    public IReadOnlyList<CloudPropertySpec> NoteProperties { get; init; } = [];
    public IReadOnlyList<CloudAutomationSpec> Automations { get; init; } = [];
    public IReadOnlyList<CloudSynthesizedParameterSpec> SynthesizedParameters { get; init; } = [];

    public IEnumerable<string> PartPropertyKeys => PartProperties.Select(p => p.Id);
    public IEnumerable<string> NotePropertyKeys => NoteProperties.Select(p => p.Id);
    public IEnumerable<string> AutomationKeys => Automations.Select(a => a.Id);
    public IEnumerable<string> SynthesizedParameterKeys => SynthesizedParameters.Select(p => p.Id);

    // 声明面指纹：服务端 schema 变化时用于判断是否需要让宿主重新询问声明。
    public string Signature => string.Join("|",
        "auto:" + (PitchAuto ? "1" : "0"),
        "part:" + string.Join(",", PartPropertyKeys),
        "note:" + string.Join(",", NotePropertyKeys),
        "auto-track:" + string.Join(",", AutomationKeys),
        "synth:" + string.Join(",", SynthesizedParameterKeys));

    // 老服务端没有 schema 时的默认行为，等价于改版前插件的声明面；同时是解析失败的回退。
    public static CloudVoiceSchema Defaults { get; } = BuildDefaults();

    public ObjectConfig BuildPartPropertyConfig()
        => ObjectConfig.Create(BuildPropertyMap(PartProperties));

    public ObjectConfig BuildNotePropertyConfig()
        => ObjectConfig.Create(BuildPropertyMap(NoteProperties));

    public OrderedMap<PropertyKey, AutomationConfig> BuildAutomationConfigs()
    {
        var language = L10n.CurrentLanguage;
        var map = new OrderedMap<PropertyKey, AutomationConfig>();
        foreach (var spec in Automations)
        {
            var config = AutomationConfig.Create(spec.Min, spec.Max);
            if (spec.Default is { } defaultValue)
                config = config.WithDefault(defaultValue);
            if (!string.IsNullOrWhiteSpace(spec.Color))
                config = config.WithColor(spec.Color!);
            if (FormatFor(spec.Unit) is { } format)
                config = config.WithFormat(format);
            if (!string.IsNullOrWhiteSpace(spec.MinLabel))
                config = config.WithMinLabel(spec.MinLabel!);
            if (!string.IsNullOrWhiteSpace(spec.MaxLabel))
                config = config.WithMaxLabel(spec.MaxLabel!);
            map.Add((spec.Id, LabelOf(spec.Label, spec.Id, language)), config);
        }
        return map;
    }

    public OrderedMap<PropertyKey, AutomationConfig> BuildSynthesizedParameterConfigs()
    {
        var language = L10n.CurrentLanguage;
        var map = new OrderedMap<PropertyKey, AutomationConfig>();
        foreach (var spec in SynthesizedParameters)
        {
            var config = AutomationConfig.Create(spec.Min, spec.Max);
            if (!string.IsNullOrWhiteSpace(spec.Color))
                config = config.WithColor(spec.Color!);
            if (FormatFor(spec.Unit) is { } format)
                config = config.WithFormat(format);
            map.Add((spec.Id, LabelOf(spec.Label, spec.Id, language)), config);
        }
        return map;
    }

    // 服务商返回的 schema 对象；解析失败返回 null（调用方回退 Defaults）。
    public static CloudVoiceSchema? Parse(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                return null;

            bool pitchAuto = false;
            if (root.TryGetProperty("pitch", out var pitch) && pitch.ValueKind == JsonValueKind.Object
                && pitch.TryGetProperty("auto", out var auto) && auto.ValueKind is JsonValueKind.True)
                pitchAuto = true;

            return new CloudVoiceSchema
            {
                PitchAuto = pitchAuto,
                PartProperties = ParseProperties(root, "partProperties"),
                NoteProperties = ParseProperties(root, "noteProperties"),
                Automations = ParseAutomations(root),
                SynthesizedParameters = ParseSynthesizedParameters(root),
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    // —— 内部：解析 ——

    static IReadOnlyList<CloudPropertySpec> ParseProperties(JsonElement root, string property)
    {
        if (!root.TryGetProperty(property, out var array) || array.ValueKind != JsonValueKind.Array)
            return [];
        var list = new List<CloudPropertySpec>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in array.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                continue;
            var id = ReadString(item, "id");
            if (string.IsNullOrWhiteSpace(id) || !seen.Add(id))
                continue;

            var kindText = (ReadString(item, "type") ?? "").Trim().ToLowerInvariant();
            var min = ReadDouble(item, "min", 0);
            var max = ReadDouble(item, "max", 0);
            var step = ReadDouble(item, "step", 0);
            bool hasRange = max > min;
            var kind = kindText switch
            {
                "slider" => CloudPropertyKind.Slider,
                "integer" or "int" => CloudPropertyKind.Integer,
                "combo" or "enum" or "select" => CloudPropertyKind.Combo,
                "checkbox" or "bool" or "boolean" => CloudPropertyKind.Checkbox,
                "text" or "string" => CloudPropertyKind.Text,
                "number" or "draggable" => CloudPropertyKind.Number,
                _ => hasRange ? CloudPropertyKind.Slider : CloudPropertyKind.Number,
            };

            var options = ParseOptions(item);
            var defaultText = ReadString(item, "default");
            if (kind == CloudPropertyKind.Combo && defaultText == null && options.Count > 0)
                defaultText = options[0].Value;

            bool booleanDefault = false;
            if (item.TryGetProperty("default", out var defaultElement))
            {
                switch (defaultElement.ValueKind)
                {
                    case JsonValueKind.True: booleanDefault = true; break;
                    case JsonValueKind.False: booleanDefault = false; break;
                    case JsonValueKind.String: defaultText ??= defaultElement.GetString(); break;
                }
            }
            if (kind == CloudPropertyKind.Checkbox && defaultElement.NotNullBoolean())
                booleanDefault = defaultElement.GetBoolean();

            list.Add(new CloudPropertySpec(
                id!, ReadString(item, "label"), kind,
                ReadDouble(item, "default", 0), min, max, step,
                ReadString(item, "unit"), defaultText, booleanDefault, options));
        }
        return list;
    }

    static IReadOnlyList<CloudPropertyOption> ParseOptions(JsonElement item)
    {
        if (!item.TryGetProperty("options", out var options) || options.ValueKind != JsonValueKind.Array)
            return [];
        var list = new List<CloudPropertyOption>();
        foreach (var option in options.EnumerateArray())
        {
            if (option.ValueKind == JsonValueKind.String)
            {
                var text = option.GetString() ?? "";
                list.Add(new CloudPropertyOption(text, text));
            }
            else if (option.ValueKind == JsonValueKind.Object)
            {
                var value = ReadString(option, "value");
                if (value == null)
                    continue;
                list.Add(new CloudPropertyOption(value, ReadString(option, "label")));
            }
        }
        return list;
    }

    static IReadOnlyList<CloudAutomationSpec> ParseAutomations(JsonElement root)
    {
        if (!root.TryGetProperty("automations", out var array) || array.ValueKind != JsonValueKind.Array)
            return [];
        var list = new List<CloudAutomationSpec>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in array.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                continue;
            var id = ReadString(item, "id");
            if (string.IsNullOrWhiteSpace(id) || !seen.Add(id))
                continue;
            double? defaultValue = item.TryGetProperty("default", out var defaultElement) && defaultElement.NotNullNumber()
                ? defaultElement.GetDouble()
                : null;
            list.Add(new CloudAutomationSpec(
                id!, ReadString(item, "label"),
                ReadDouble(item, "min", 0), ReadDouble(item, "max", 1), defaultValue,
                ReadString(item, "color"), ReadString(item, "unit"),
                ReadString(item, "minLabel"), ReadString(item, "maxLabel")));
        }
        return list;
    }

    static IReadOnlyList<CloudSynthesizedParameterSpec> ParseSynthesizedParameters(JsonElement root)
    {
        if (!root.TryGetProperty("synthesizedParameters", out var array) || array.ValueKind != JsonValueKind.Array)
            return [];
        var list = new List<CloudSynthesizedParameterSpec>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in array.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.Object)
                continue;
            var id = ReadString(item, "id");
            if (string.IsNullOrWhiteSpace(id) || !seen.Add(id))
                continue;
            list.Add(new CloudSynthesizedParameterSpec(
                id!, ReadString(item, "label"),
                ReadDouble(item, "min", 0), ReadDouble(item, "max", 1),
                ReadString(item, "color"), ReadString(item, "unit")));
        }
        return list;
    }

    static OrderedMap<PropertyKey, IControllerConfig> BuildPropertyMap(IReadOnlyList<CloudPropertySpec> specs)
    {
        var language = L10n.CurrentLanguage;
        var map = new OrderedMap<PropertyKey, IControllerConfig>();
        foreach (var spec in specs)
            map.Add((spec.Id, LabelOf(spec.Label, spec.Id, language)), BuildPropertyConfig(spec));
        return map;
    }

    static IControllerConfig BuildPropertyConfig(CloudPropertySpec spec)
    {
        switch (spec.Kind)
        {
            case CloudPropertyKind.Integer:
            {
                var slider = SliderConfig.Integer(spec.NumberDefault, spec.Min, spec.Max);
                return FormatFor(spec.Unit) is { } format ? slider.WithFormat(format) : slider;
            }
            case CloudPropertyKind.Slider:
            {
                var slider = Math.Abs(spec.Step - 1) < 1e-9
                    ? SliderConfig.Integer(spec.NumberDefault, spec.Min, spec.Max)
                    : SliderConfig.Linear(spec.NumberDefault, spec.Min, spec.Max);
                return FormatFor(spec.Unit) is { } format ? slider.WithFormat(format) : slider;
            }
            case CloudPropertyKind.Combo:
            {
                var items = spec.Options.Count == 0
                    ? new List<ComboBoxItem> { new("") }
                    : spec.Options.Select(option => new ComboBoxItem(option.Value, option.Label ?? option.Value)).ToList();
                var defaultText = spec.TextDefault ?? spec.Options.FirstOrDefault()?.Value ?? "";
                var defaultOption = spec.Options.FirstOrDefault(option => string.Equals(option.Value, defaultText, StringComparison.Ordinal))
                                    ?? spec.Options.FirstOrDefault();
                var defaultItem = defaultOption == null
                    ? items[0]
                    : new ComboBoxItem(defaultOption.Value, defaultOption.Label ?? defaultOption.Value);
                return ComboBoxConfig.Create(items).WithDefault(defaultItem);
            }
            case CloudPropertyKind.Checkbox:
                return CheckBoxConfig.Create(spec.BooleanDefault);
            case CloudPropertyKind.Text:
                return TextBoxConfig.Create(spec.TextDefault ?? "");
            default:
            {
                var box = DraggableNumberBoxConfig.Create(spec.NumberDefault);
                bool hasRange = spec.Max > spec.Min;
                if (hasRange)
                    box = box.WithRange(spec.Min, spec.Max);
                if (spec.Step > 0)
                    box = box.WithStep(spec.Step);
                if (FormatFor(spec.Unit) is { } format)
                    box = box.WithFormat(format);
                return box;
            }
        }
    }

    // 单位只影响显示；解析时去掉单位后缀，仍按不变文化读数字。
    static INumberFormat? FormatFor(string? unit)
    {
        if (string.IsNullOrWhiteSpace(unit))
            return null;
        var suffix = unit!;
        return NumberFormat.Custom(
            value => value.ToString("0.###", CultureInfo.InvariantCulture) + suffix,
            text =>
            {
                var trimmed = text.Trim();
                if (trimmed.EndsWith(suffix, StringComparison.Ordinal))
                    trimmed = trimmed[..^suffix.Length].Trim();
                return double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ? value : null;
            });
    }

    static string LabelOf(string? label, string id, string? language)
        => string.IsNullOrWhiteSpace(label) ? id : label!;

    static string? ReadString(JsonElement element, string property)
        => element.TryGetProperty(property, out var value)
            ? value.ValueKind switch
            {
                JsonValueKind.String => value.GetString(),
                JsonValueKind.Number => value.GetRawText(),
                _ => null,
            }
            : null;

    static double ReadDouble(JsonElement element, string property, double defaultValue)
        => element.TryGetProperty(property, out var value)
           && value.ValueKind == JsonValueKind.Number
           && value.TryGetDouble(out var result)
            ? result
            : defaultValue;

    // —— 默认 schema：老服务端兼容（与改版前声明面一致） ——

    static CloudVoiceSchema BuildDefaults()
    {
        var language = L10n.CurrentLanguage;
        return new CloudVoiceSchema
        {
            PitchAuto = false,
            NoteProperties =
            [
                new CloudPropertySpec("dynamics", L10n.T(language, "Dynamics", "力度"),
                    CloudPropertyKind.Slider, 0, -1, 1, 0, null, null, false, []),
                new CloudPropertySpec("breathiness", L10n.T(language, "Breathiness", "气声"),
                    CloudPropertyKind.Slider, 0.2, 0, 1, 0, null, null, false, []),
            ],
            Automations =
            [
                new CloudAutomationSpec("dynamics", L10n.T(language, "Dynamics", "力度"), -1, 1, 0,
                    "#E5A573", null, L10n.T(language, "Soft", "弱"), L10n.T(language, "Hard", "强")),
                new CloudAutomationSpec("breathiness", L10n.T(language, "Breathiness", "气声"), 0, 1, 0.2,
                    "#73C2E5", null, L10n.T(language, "Dry", "干"), L10n.T(language, "Airy", "气")),
            ],
            SynthesizedParameters =
            [
                new CloudSynthesizedParameterSpec("energy", L10n.T(language, "Energy (synthesized)", "能量（合成回显）"),
                    0, 100, "#8ADF73", null),
            ],
        };
    }
}

file static class JsonElementExtensions
{
    public static bool NotNullBoolean(this JsonElement element)
        => element.ValueKind is JsonValueKind.True or JsonValueKind.False;

    public static bool NotNullNumber(this JsonElement element)
        => element.ValueKind == JsonValueKind.Number;
}
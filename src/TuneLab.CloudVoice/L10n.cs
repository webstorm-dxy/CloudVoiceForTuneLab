namespace TuneLab.CloudVoice;

// 插件自译：宿主不查表，DisplayText / 声库名都由插件按 TuneLabContext.Global.Language 产出。
public static class L10n
{
    public static bool IsChinese(string? language)
        => !string.IsNullOrEmpty(language) && language.StartsWith("zh", StringComparison.OrdinalIgnoreCase);

    public static string T(string? language, string english, string chinese)
        => IsChinese(language) ? chinese : english;

    public static string CurrentLanguage => TuneLab.SDK.TuneLabContext.Global.Language;
}

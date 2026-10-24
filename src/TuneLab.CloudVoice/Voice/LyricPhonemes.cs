namespace TuneLab.CloudVoice.Voice;

public readonly record struct PredictedPhoneme(string Symbol, double Duration, double StretchWeight);

public sealed record PredictedSyllable(
    IReadOnlyList<PredictedPhoneme> Leading,
    IReadOnlyList<PredictedPhoneme> Body,
    double BodyOffset,
    bool Rest);

// 轻量 G2P：把「一个音符一个音节」的歌词解析成 引导辅音 + 主体（元音 + 尾辅音）。
// 这是给 mock 引擎和「云端没返回音素」的兜底用的，不追求音韵学正确——真实 G2P 归云端服务。
// 规则：
//   · 空串 / 纯空白 / "R" / 孤立 "-" → 休止（Rest）。
//   · 去掉结尾的声调数字（ma3 → ma）。
//   · 拉丁字母：前置辅音簇（w=0） + 元音簇（w=1） + 尾辅音簇（w=0）。
//   · 汉字 / 其它文字：整字当作一个主体音素（w=1）显示，音频用 "a" 的共振峰发音。
public static class LyricPhonemes
{
    const double ConsonantDuration = 0.07;
    const double VowelDuration = 0.28;

    public static PredictedSyllable Predict(string? lyric)
    {
        var text = (lyric ?? "").Trim();
        if (text.Length == 0 || text == "-" || text.Equals("R", StringComparison.OrdinalIgnoreCase))
            return new PredictedSyllable([], [], 0, true);

        // 去掉结尾声调数字（ma3 / ma5）。
        while (text.Length > 0 && text[^1] is >= '0' and <= '9')
            text = text[..^1];
        if (text.Length == 0)
            return new PredictedSyllable([], [], 0, true);

        var leading = new List<PredictedPhoneme>();
        var body = new List<PredictedPhoneme>();

        if (!IsLatin(text))
        {
            body.Add(new PredictedPhoneme(text, VowelDuration, 1));
            return new PredictedSyllable(leading, body, 0, false);
        }

        var lower = text.ToLowerInvariant();
        int firstVowel = 0;
        while (firstVowel < lower.Length && !IsVowel(lower[firstVowel]))
            firstVowel++;
        int lastVowel = lower.Length - 1;
        while (lastVowel >= firstVowel && !IsVowel(lower[lastVowel]))
            lastVowel--;

        if (firstVowel >= lower.Length || lastVowel < firstVowel)
        {
            // 没有元音：全部当主体音素（w=0），也能显示与发音。
            foreach (char c in lower)
                body.Add(new PredictedPhoneme(c.ToString(), ConsonantDuration, 0));
            return new PredictedSyllable(leading, body, 0, false);
        }

        for (int i = 0; i < firstVowel; i++)
            leading.Add(new PredictedPhoneme(lower[i].ToString(), ConsonantDuration, 0));
        for (int i = firstVowel; i <= lastVowel; i++)
            body.Add(new PredictedPhoneme(lower[i].ToString(), VowelDuration, 1));
        for (int i = lastVowel + 1; i < lower.Length; i++)
            body.Add(new PredictedPhoneme(lower[i].ToString(), ConsonantDuration, 0));

        double bodyOffset = leading.Count * ConsonantDuration;
        return new PredictedSyllable(leading, body, bodyOffset, false);
    }

    static bool IsLatin(string text) => text.All(c => c < 0x2E80);

    static bool IsVowel(char c) => c is 'a' or 'e' or 'i' or 'o' or 'u';
}

using TuneLab.Foundation;

namespace TuneLab.CloudVoice.Audio;

public static class AudioAnalysis
{
    // 从音频算 RMS 包络（20Hz 采样，窗长 50ms），归一到 0..100 作为 energy 回显。
    // 云端没有返回 energy 曲线时用这个兜底，让只读回显轨始终有内容。
    public static IReadOnlyList<Point> RmsEnvelope(float[] samples, int sampleRate, double startTime)
    {
        if (samples.Length == 0 || sampleRate <= 0)
            return [];

        const double rate = 20.0;
        int window = Math.Max(1, (int)(sampleRate / rate));
        int count = Math.Max(1, samples.Length / window);
        var raw = new double[count];
        double peak = 0;
        for (int i = 0; i < count; i++)
        {
            int from = i * window;
            int to = Math.Min(samples.Length, from + window);
            double sum = 0;
            for (int index = from; index < to; index++)
                sum += samples[index] * samples[index];
            double rms = Math.Sqrt(sum / Math.Max(1, to - from));
            raw[i] = rms;
            if (rms > peak) peak = rms;
        }

        var points = new List<Point>(count);
        for (int i = 0; i < count; i++)
        {
            double value = peak > 1e-6 ? Math.Clamp(raw[i] / peak * 100.0, 0, 100) : 0;
            points.Add(new Point(startTime + i / rate, value));
        }
        return points;
    }
}

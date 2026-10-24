using TuneLab.Foundation;
using TuneLab.SDK;
using TuneLab.CloudVoice.Audio;
using TuneLab.CloudVoice.Protocol;
using TuneLab.CloudVoice.Voice;

namespace TuneLab.CloudVoice.Providers;

// 离线 mock 合成：把请求里的音高曲线、音符与歌词渲染成一段带颤音的加法合成音频，
// 同时回填音素 / 音高 / energy 回显。目的是让整条插件链路（分块、调度、产物、界面）在无网络、
// 无云服务时也能被完整验证，不代表真实歌声品质。
public sealed class MockCloudVoiceProvider : ICloudVoiceProvider
{
    public static readonly MockCloudVoiceProvider Instance = new(0);

    readonly int mLatencyMs;

    public MockCloudVoiceProvider(int latencyMs = 320)
    {
        mLatencyMs = Math.Max(0, latencyMs);
    }

    public void Dispose() { }

    public async Task<CloudSynthesisResult?> SynthesizeAsync(CloudSynthesisRequest request, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        try
        {
            progress?.Report(0.2);
            if (mLatencyMs > 0)
            {
                await Task.Delay(mLatencyMs / 2, cancellationToken).ConfigureAwait(false);
                progress?.Report(0.5);
                await Task.Delay(mLatencyMs / 2, cancellationToken).ConfigureAwait(false);
            }
            progress?.Report(0.85);
            return Render(request);
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    CloudSynthesisResult Render(CloudSynthesisRequest request)
    {
        int sampleRate = request.SampleRate > 0 ? request.SampleRate : 48000;
        double duration = Math.Max(0.05, request.EndTime - request.StartTime);
        int count = Math.Max(1, (int)Math.Round(duration * sampleRate));
        var audio = new float[count];
        var timbre = TimbreOf(request.Voice);
        var pitchSegments = new List<IReadOnlyList<Point>>();
        var phonemes = new Map<string, CloudSyllable>();

        foreach (var note in request.Notes)
        {
            var syllable = SyllableOf(note, out var rest);
            if (!rest)
                phonemes.Add(note.Key, syllable);

            if (rest || note.Duration <= 0)
                continue;

            int from = Math.Clamp((int)Math.Round((note.Start - request.StartTime) * sampleRate), 0, count);
            int to = Math.Clamp((int)Math.Round((note.Start + note.Duration - request.StartTime) * sampleRate), 0, count);
            if (to <= from)
                continue;

            var pitchCurve = new List<Point>(Math.Max(8, (int)(note.Duration * 20)));
            double phase = 0;
            int attack = Math.Max(1, (int)(0.02 * sampleRate));
            int release = Math.Max(1, (int)(0.05 * sampleRate));
            int length = to - from;
            for (int i = from; i < to; i++)
            {
                double time = request.StartTime + (double)i / sampleRate;
                double semitone = SamplePitch(request.Pitch, time) + VibratoDepth * Math.Sin(2 * Math.PI * VibratoRate * (time - note.Start));
                double frequency = 440.0 * Math.Pow(2, (semitone - 69) / 12.0);
                phase += 2 * Math.PI * frequency / sampleRate;

                double sample = 0;
                for (int harmonic = 0; harmonic < timbre.Length; harmonic++)
                    sample += timbre[harmonic] * Math.Sin(phase * (harmonic + 1));
                sample *= 0.22;

                int position = i - from;
                double envelope = position < attack ? (double)position / attack
                    : position >= length - release ? (double)(length - position) / Math.Max(1, release)
                    : 1.0;
                audio[i] += (float)(sample * envelope);

                if (position % Math.Max(1, sampleRate / 20) == 0)
                    pitchCurve.Add(new Point(time, semitone));
            }

            if (pitchCurve.Count >= 2)
                pitchSegments.Add(pitchCurve);
        }

        Normalize(audio);

        var parameters = new Map<string, IReadOnlyList<IReadOnlyList<Point>>>();
        var energy = AudioAnalysis.RmsEnvelope(audio, sampleRate, request.StartTime);
        if (energy.Count >= 2)
            parameters.Add("energy", [energy]);

        return new CloudSynthesisResult
        {
            Audio = audio,
            SampleRate = sampleRate,
            AudioStartTime = request.StartTime,
            Phonemes = phonemes,
            PitchSegments = pitchSegments,
            Parameters = parameters,
            Meta = "mock:" + request.Voice,
        };
    }

    static CloudSyllable SyllableOf(CloudRequestNote note, out bool rest)
    {
        rest = note.Rest;
        if (rest)
            return new CloudSyllable { Leading = [], Body = [] };

        if (note.Phonemes is { } pinned && (pinned.Leading.Count > 0 || pinned.Body.Count > 0))
        {
            return new CloudSyllable
            {
                Leading = pinned.Leading.Select(ToPhoneme).ToList(),
                Body = pinned.Body.Select(ToPhoneme).ToList(),
                BodyOffset = pinned.BodyOffset,
            };
        }

        var predicted = LyricPhonemes.Predict(note.Lyric);
        rest = predicted.Rest;
        if (rest)
            return new CloudSyllable { Leading = [], Body = [] };
        return new CloudSyllable
        {
            Leading = predicted.Leading.Select(p => ToPhoneme(p.Symbol, p.Duration, p.StretchWeight)).ToList(),
            Body = predicted.Body.Select(p => ToPhoneme(p.Symbol, p.Duration, p.StretchWeight)).ToList(),
            BodyOffset = predicted.BodyOffset,
        };
    }

    static SynthesizedPhoneme ToPhoneme(CloudPhoneme phoneme)
        => ToPhoneme(phoneme.Symbol, phoneme.Duration, phoneme.StretchWeight);

    static SynthesizedPhoneme ToPhoneme(string symbol, double duration, double weight)
        => new() { Symbol = symbol, Duration = duration, StretchWeight = weight };

    // 请求里的 pitch 是 100Hz 均匀折线（全局秒）；线性插值，越界钳到端点。
    static double SamplePitch(List<CloudPoint> points, double time)
    {
        if (points.Count == 0)
            return 60;
        if (points.Count == 1)
            return points[0].V;
        if (time <= points[0].T)
            return points[0].V;
        if (time >= points[^1].T)
            return points[^1].V;

        int low = 0, high = points.Count - 1;
        while (high - low > 1)
        {
            int middle = (low + high) / 2;
            if (points[middle].T <= time) low = middle;
            else high = middle;
        }
        double span = points[high].T - points[low].T;
        if (span <= 1e-9)
            return points[low].V;
        double ratio = (time - points[low].T) / span;
        return points[low].V + (points[high].V - points[low].V) * ratio;
    }

    static void Normalize(float[] audio)
    {
        float peak = 0;
        foreach (var sample in audio)
            peak = Math.Max(peak, Math.Abs(sample));
        if (peak <= 1e-6)
            return;
        float scale = 0.72f / peak;
        for (int i = 0; i < audio.Length; i++)
            audio[i] *= scale;
    }

    static double[] TimbreOf(string? voiceId)
    {
        if (voiceId == null || voiceId.Contains("alto", StringComparison.OrdinalIgnoreCase))
            return [1.0, 0.55, 0.3, 0.14, 0.06];
        if (voiceId.Contains("whisper", StringComparison.OrdinalIgnoreCase))
            return [1.0, 0.25, 0.12, 0.08, 0.04];
        return [1.0, 0.42, 0.2, 0.09, 0.04];
    }

    const double VibratoRate = 5.4;      // Hz
    const double VibratoDepth = 0.22;    // 半音
}
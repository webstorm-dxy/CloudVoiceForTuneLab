using TuneLab.Foundation;
using TuneLab.SDK;
using TuneLab.CloudVoice.Protocol;

namespace TuneLab.CloudVoice.Providers;

// 云端一次返回的音节（音素几何，与宿主 SynthesizedPhoneme 同形）。
public sealed class CloudSyllable
{
    public required IReadOnlyList<SynthesizedPhoneme> Leading { get; init; }
    public required IReadOnlyList<SynthesizedPhoneme> Body { get; init; }
    public double BodyOffset { get; init; }
}

// provider 交付给会话的合成结果：音频（单声道 float，native 采样率）+ 可选回显产物。
public sealed class CloudSynthesisResult
{
    public required float[] Audio { get; init; }
    public required int SampleRate { get; init; }
    public required double AudioStartTime { get; init; }
    public IReadOnlyMap<string, CloudSyllable> Phonemes { get; init; } = new Map<string, CloudSyllable>();
    public IReadOnlyList<IReadOnlyList<Point>> PitchSegments { get; init; } = [];
    public IReadOnlyMap<string, IReadOnlyList<IReadOnlyList<Point>>> Parameters { get; init; } = new Map<string, IReadOnlyList<IReadOnlyList<Point>>>();
    public string? Meta { get; init; }
}

// 合成 provider：mock（离线演示）与 http（云端）两个实现。
// 契约：取消时返回 null（不抛 OperationCanceledException）；真错误抛 CloudProviderException（带是否可重试）。
public interface ICloudVoiceProvider : IDisposable
{
    Task<CloudSynthesisResult?> SynthesizeAsync(CloudSynthesisRequest request, IProgress<double>? progress, CancellationToken cancellationToken);
}

public sealed class CloudProviderException : Exception
{
    public bool IsTransient { get; }

    public CloudProviderException(string message, bool isTransient, Exception? inner = null)
        : base(message, inner)
    {
        IsTransient = isTransient;
    }
}

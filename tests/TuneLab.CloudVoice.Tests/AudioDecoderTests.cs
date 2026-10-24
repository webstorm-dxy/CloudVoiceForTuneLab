using System.Buffers.Binary;
using TuneLab.CloudVoice.Audio;
using Xunit;

namespace TuneLab.CloudVoice.Tests;

public class AudioDecoderTests
{
    [Fact]
    public void PcmFloat32_Stereo_DownmixesToMono()
    {
        var bytes = new byte[8];
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0), 1.0f);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4), 0.0f);

        var decoded = AudioDecoder.Decode(bytes, "pcm_f32le", 48000, 2);

        Assert.Equal(48000, decoded.SampleRate);
        Assert.Single(decoded.Samples);
        Assert.Equal(0.5f, decoded.Samples[0], 6);
    }

    [Fact]
    public void PcmInt16_Mono_DecodesScale()
    {
        var bytes = new byte[4];
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(0), 16384);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(2), -32768);

        var decoded = AudioDecoder.Decode(bytes, "pcm_s16le", 44100, 1);

        Assert.Equal(2, decoded.Samples.Length);
        Assert.Equal(0.5f, decoded.Samples[0], 6);
        Assert.Equal(-1.0f, decoded.Samples[1], 6);
    }

    [Fact]
    public void Wav_ReadsSampleRateAndFrames()
    {
        var wav = TestAudio.Wav(440, 0.1, 32000, channels: 2);

        var decoded = AudioDecoder.Decode(wav, "wav", 48000, 2);

        Assert.Equal(32000, decoded.SampleRate);
        Assert.Equal(3200, decoded.Samples.Length);
        Assert.Contains(decoded.Samples, sample => Math.Abs(sample) > 0.1f);
    }

    [Fact]
    public void Wav_WithoutFormatHint_IsSniffed()
    {
        var wav = TestAudio.Wav(440, 0.02, 48000);

        var decoded = AudioDecoder.Decode(wav, null, 44100, null);

        Assert.Equal(48000, decoded.SampleRate);
        Assert.NotEmpty(decoded.Samples);
    }

    [Fact]
    public void UnsupportedFormat_Throws()
    {
        Assert.Throws<FormatException>(() => AudioDecoder.Decode([1, 2, 3], "opus", 48000, 1));
    }

    [Fact]
    public void RmsEnvelope_IsBoundedAndPositioned()
    {
        var samples = new float[48000];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)(0.5 * Math.Sin(2 * Math.PI * 440 * i / 48000.0));

        var envelope = AudioAnalysis.RmsEnvelope(samples, 48000, 1.5);

        Assert.NotEmpty(envelope);
        Assert.All(envelope, point => Assert.InRange(point.Y, 0, 100));
        Assert.Equal(1.5, envelope[0].X, 6);
    }
}

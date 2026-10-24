using System.Buffers.Binary;

namespace TuneLab.CloudVoice.Audio;

// 云端音频解码：把协议里的内联 base64 / 下载字节解成宿主需要的单声道 float 采样。
// 支持 PCM f32le / s16le / WAV（PCM 16/24/32 与 IEEE float32），多声道下混为单声道。
// 采样率随解码结果返回，宿主对与工程率不一致的段自动重采样。
public static class AudioDecoder
{
    public sealed record DecodedAudio(float[] Samples, int SampleRate);

    public static DecodedAudio Decode(byte[] bytes, string? format, int fallbackSampleRate, int? declaredChannels)
    {
        var normalized = (format ?? "").Trim().ToLowerInvariant();
        switch (normalized)
        {
            case "":
            case "wav":
            case "audio/wav":
            case "audio/x-wav":
                return DecodeWav(bytes, fallbackSampleRate);
            case "pcm_f32le":
            case "f32le":
            case "pcm-float32":
            case "float32":
                return new DecodedAudio(DecodePcm(bytes, 4, declaredChannels ?? 1), fallbackSampleRate);
            case "pcm_s16le":
            case "s16le":
            case "pcm-int16":
            case "int16":
                return new DecodedAudio(DecodePcm(bytes, 2, declaredChannels ?? 1), fallbackSampleRate);
            default:
                if (LooksLikeWav(bytes))
                    return DecodeWav(bytes, fallbackSampleRate);
                throw new FormatException($"Unsupported cloud audio format '{format}'.");
        }
    }

    static bool LooksLikeWav(byte[] bytes)
        => bytes.Length >= 12 && bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F'
           && bytes[8] == 'W' && bytes[9] == 'A' && bytes[10] == 'V' && bytes[11] == 'E';

    static float[] DecodePcm(byte[] bytes, int bytesPerSample, int channels)
    {
        if (channels < 1) channels = 1;
        int frameCount = bytes.Length / (bytesPerSample * channels);
        var mono = new float[frameCount];
        int offset = 0;
        for (int frame = 0; frame < frameCount; frame++)
        {
            double sum = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                sum += bytesPerSample == 4
                    ? BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(offset))
                    : BinaryPrimitives.ReadInt16LittleEndian(bytes.AsSpan(offset)) / 32768.0;
                offset += bytesPerSample;
            }
            mono[frame] = (float)(sum / channels);
        }
        return mono;
    }

    // 最小 WAV 解析：RIFF/WAVE + fmt + data。支持 PCM(1) / IEEE float(3)，位深 8/16/24/32。
    static DecodedAudio DecodeWav(byte[] bytes, int fallbackSampleRate)
    {
        if (!LooksLikeWav(bytes))
            throw new FormatException("The cloud audio is not a WAV file.");

        int sampleRate = fallbackSampleRate;
        int channels = 1;
        int bitsPerSample = 16;
        int audioFormat = 1;
        int dataOffset = -1;
        int dataLength = 0;

        int position = 12;
        while (position + 8 <= bytes.Length)
        {
            var chunkId = System.Text.Encoding.ASCII.GetString(bytes, position, 4);
            int chunkSize = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(position + 4));
            int body = position + 8;
            if (chunkSize < 0 || body > bytes.Length) break;

            if (chunkId == "fmt ")
            {
                if (chunkSize >= 16)
                {
                    audioFormat = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body));
                    channels = Math.Max(1, (int)BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body + 2)));
                    sampleRate = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(body + 4));
                    bitsPerSample = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(body + 14));
                }
            }
            else if (chunkId == "data")
            {
                dataOffset = body;
                dataLength = Math.Min(chunkSize, bytes.Length - body);
                if (audioFormat is 1 or 3 && bitsPerSample > 0)
                    break; // fmt 已读到时可以提前结束
            }

            position = body + chunkSize + (chunkSize & 1);
        }

        if (dataOffset < 0) throw new FormatException("The cloud WAV has no data chunk.");
        if (sampleRate <= 0) sampleRate = fallbackSampleRate;
        if (audioFormat != 1 && audioFormat != 3)
            throw new FormatException($"Unsupported WAV encoding (format {audioFormat}, {bitsPerSample} bit).");

        int bytesPerSample = bitsPerSample / 8;
        if (bytesPerSample <= 0) throw new FormatException("Unsupported WAV bit depth.");
        int frameCount = dataLength / (bytesPerSample * channels);
        var mono = new float[frameCount];
        var span = bytes.AsSpan(dataOffset, dataLength);

        for (int frame = 0; frame < frameCount; frame++)
        {
            double sum = 0;
            int offset = frame * bytesPerSample * channels;
            for (int channel = 0; channel < channels; channel++)
            {
                double value = audioFormat == 3
                    ? ReadFloat(span, offset, bitsPerSample)
                    : ReadInt(span, offset, bitsPerSample);
                sum += value;
                offset += bytesPerSample;
            }
            mono[frame] = (float)(sum / channels);
        }

        return new DecodedAudio(mono, sampleRate);
    }

    static double ReadFloat(ReadOnlySpan<byte> span, int offset, int bitsPerSample)
    {
        return bitsPerSample switch
        {
            32 => BinaryPrimitives.ReadSingleLittleEndian(span.Slice(offset)),
            64 => BinaryPrimitives.ReadDoubleLittleEndian(span.Slice(offset)),
            _ => throw new FormatException($"Unsupported float WAV bit depth {bitsPerSample}."),
        };
    }

    static double ReadInt(ReadOnlySpan<byte> span, int offset, int bitsPerSample)
    {
        return bitsPerSample switch
        {
            8 => (span[offset] - 128) / 128.0,
            16 => BinaryPrimitives.ReadInt16LittleEndian(span.Slice(offset)) / 32768.0,
            24 => ReadInt24(span, offset) / 8388608.0,
            32 => BinaryPrimitives.ReadInt32LittleEndian(span.Slice(offset)) / 2147483648.0,
            _ => throw new FormatException($"Unsupported PCM WAV bit depth {bitsPerSample}."),
        };
    }

    static int ReadInt24(ReadOnlySpan<byte> span, int offset)
    {
        int value = span[offset] | (span[offset + 1] << 8) | (span[offset + 2] << 16);
        if ((value & 0x800000) != 0) value |= unchecked((int)0xFF000000);
        return value;
    }
}
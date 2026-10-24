using System.Text.Json;
using TuneLab.CloudVoice.Protocol;
using Xunit;

namespace TuneLab.CloudVoice.Tests;

public class ProtocolTests
{
    [Fact]
    public void SerializeRequest_UsesCamelCaseProtocolFields()
    {
        var json = CloudProtocol.SerializeRequest(TestAudio.SampleRequest());

        Assert.Contains("\"protocol\":\"tunelab-cloudvoice\"", json);
        Assert.Contains("\"version\":1", json);
        Assert.Contains("\"voice\":\"test-voice\"", json);
        Assert.Contains("\"sampleRate\":48000", json);
        Assert.Contains("\"notes\"", json);
        Assert.Contains("\"key\":\"note-1\"", json);
        Assert.Contains("\"lyric\":\"la\"", json);
        Assert.Contains("\"pitch\"", json);
        Assert.Contains("\"parameters\"", json);
        // null Model 不应写进 JSON（协议允许缺省）。
        Assert.DoesNotContain("\"model\"", json);
    }

    [Fact]
    public void DeserializeResponse_ReadsAudioPhonemesPitchAndParameters()
    {
        var json = TestAudio.Json(new CloudSynthesisResponse
        {
            Protocol = "tunelab-cloudvoice",
            Version = 1,
            SampleRate = 48000,
            AudioStart = 2.5,
            Audio = new CloudAudio { Format = "wav", Channels = 1, Data = "AQID" },
            Phonemes = new Dictionary<string, CloudPhonemes>
            {
                ["note-1"] = new()
                {
                    Leading = [new CloudPhoneme { Symbol = "l", Duration = 0.06, StretchWeight = 0 }],
                    Body = [new CloudPhoneme { Symbol = "a", Duration = 0.3, StretchWeight = 1 }],
                    BodyOffset = 0.06,
                },
            },
            Pitch = [new CloudPoint { T = 2.5, V = 60 }, new CloudPoint { T = 3.0, V = 62 }],
            Parameters = new Dictionary<string, List<CloudPoint>>
            {
                ["energy"] = [new CloudPoint { T = 2.5, V = 55 }, new CloudPoint { T = 3.0, V = 60 }],
            },
        });

        var response = CloudProtocol.DeserializeResponse(json);

        Assert.Equal(48000, response.SampleRate);
        Assert.Equal(2.5, response.AudioStart);
        Assert.Equal("wav", response.Audio!.Format);
        Assert.Equal("l", response.Phonemes!["note-1"].Leading[0].Symbol);
        Assert.Equal(2, response.Pitch!.Count);
        Assert.Equal(55, response.Parameters!["energy"][0].V);
    }

    [Fact]
    public void DeserializeResponse_AcceptsPascalCaseAndUnknownFields()
    {
        var response = CloudProtocol.DeserializeResponse(
            "{\"SampleRate\":44100,\"audio\":{\"format\":\"wav\",\"data\":\"AA==\"},\"unknown\":true}");

        Assert.Equal(44100, response.SampleRate);
        Assert.Equal("wav", response.Audio!.Format);
    }

    [Fact]
    public void DeserializeResponse_InvalidJson_ThrowsFormatException()
    {
        Assert.Throws<FormatException>(() => CloudProtocol.DeserializeResponse("not json"));
    }
}

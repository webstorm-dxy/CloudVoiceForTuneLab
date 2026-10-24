using System.Text.Json;
using TuneLab.CloudVoice;
using TuneLab.CloudVoice.Providers;
using TuneLab.CloudVoice.Protocol;
using Xunit;

namespace TuneLab.CloudVoice.Tests;

public class HttpCloudVoiceProviderTests
{
    static CloudVoiceSettings SettingsFor(FakeCloudServer server, int retries = 0, int timeoutSeconds = 5) => new()
    {
        Endpoint = server.Endpoint,
        ApiKey = "secret-token",
        SampleRate = 48000,
        TimeoutSeconds = timeoutSeconds,
        Retries = retries,
    };

    static string SuccessBody(CloudSynthesisRequest request)
    {
        var wav = TestAudio.Wav(440, Math.Max(0.05, request.EndTime - request.StartTime), request.SampleRate);
        return TestAudio.Json(new CloudSynthesisResponse
        {
            Protocol = CloudProtocol.Name,
            Version = CloudProtocol.Version,
            SampleRate = request.SampleRate,
            AudioStart = request.StartTime,
            Audio = new CloudAudio
            {
                Format = "wav",
                Channels = 1,
                Data = Convert.ToBase64String(wav),
            },
            Phonemes = new Dictionary<string, CloudPhonemes>
            {
                ["note-1"] = new()
                {
                    Leading = [],
                    Body = [new CloudPhoneme { Symbol = "srv-a", Duration = 0.2, StretchWeight = 1 }],
                    BodyOffset = 0.0,
                },
            },
            Pitch = request.Pitch,
            Parameters = new Dictionary<string, List<CloudPoint>>
            {
                ["energy"] = [new CloudPoint { T = request.StartTime, V = 70 }, new CloudPoint { T = request.EndTime, V = 80 }],
            },
            Meta = new Dictionary<string, object> { ["noteCount"] = request.Notes.Count },
        });
    }

    [Fact]
    public async Task Synthesize_SendsProtocolRequestAndDecodesResponse()
    {
        using var server = new FakeCloudServer((request, _) =>
        {
            var parsed = JsonSerializer.Deserialize<CloudSynthesisRequest>(request.Body, CloudProtocol.JsonOptions)!;
            return (200, SuccessBody(parsed));
        });
        using var provider = new HttpCloudVoiceProvider(SettingsFor(server));

        var result = await provider.SynthesizeAsync(TestAudio.SampleRequest(), null, CancellationToken.None);

        Assert.NotNull(result);
        Assert.Equal(48000, result!.SampleRate);
        Assert.Equal(1.0, result.AudioStartTime);
        Assert.NotEmpty(result.Audio);
        Assert.Equal("srv-a", result.Phonemes["note-1"].Body[0].Symbol);
        Assert.Single(result.PitchSegments);
        Assert.True(result.Parameters.ContainsKey("energy"));

        var captured = Assert.Single(server.Requests);
        Assert.Equal("POST", captured.Method);
        Assert.Equal("/synthesize", captured.Path);
        Assert.Equal("Bearer secret-token", captured.Headers["Authorization"]);
        using var body = JsonDocument.Parse(captured.Body);
        Assert.Equal("tunelab-cloudvoice", body.RootElement.GetProperty("protocol").GetString());
        Assert.False(body.RootElement.TryGetProperty("model", out _));
        Assert.Equal("test-voice", body.RootElement.GetProperty("voice").GetString());
        Assert.Equal("la", body.RootElement.GetProperty("notes")[0].GetProperty("lyric").GetString());
    }

    [Fact]
    public async Task Synthesize_RetriesTransientFailures()
    {
        using var server = new FakeCloudServer((request, count) =>
        {
            if (count == 1)
                return (503, "{\"error\":\"temporarily down\"}");
            var parsed = JsonSerializer.Deserialize<CloudSynthesisRequest>(request.Body, CloudProtocol.JsonOptions)!;
            return (200, SuccessBody(parsed));
        });
        using var provider = new HttpCloudVoiceProvider(SettingsFor(server, retries: 1));

        var result = await provider.SynthesizeAsync(TestAudio.SampleRequest(), null, CancellationToken.None);

        Assert.NotNull(result);
        Assert.Equal(2, server.Requests.Count);
    }

    [Fact]
    public async Task Synthesize_DoesNotRetryClientErrors()
    {
        using var server = new FakeCloudServer((_, _) => (400, "{\"error\":\"bad request\"}"));
        using var provider = new HttpCloudVoiceProvider(SettingsFor(server, retries: 3));

        var exception = await Assert.ThrowsAsync<CloudProviderException>(
            () => provider.SynthesizeAsync(TestAudio.SampleRequest(), null, CancellationToken.None));

        Assert.False(exception.IsTransient);
        Assert.Single(server.Requests);
    }

    [Fact]
    public async Task Synthesize_ErrorFieldInSuccessfulResponse_Fails()
    {
        using var server = new FakeCloudServer((_, _) => (200, "{\"error\":\"voice not found\"}"));
        using var provider = new HttpCloudVoiceProvider(SettingsFor(server));

        var exception = await Assert.ThrowsAsync<CloudProviderException>(
            () => provider.SynthesizeAsync(TestAudio.SampleRequest(), null, CancellationToken.None));

        Assert.Contains("voice not found", exception.Message);
        Assert.False(exception.IsTransient);
    }

    [Fact]
    public async Task Synthesize_PreCanceled_ReturnsNullWithoutRequest()
    {
        using var server = new FakeCloudServer((_, _) => (200, "{}"));
        using var provider = new HttpCloudVoiceProvider(SettingsFor(server));
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var result = await provider.SynthesizeAsync(TestAudio.SampleRequest(), null, cts.Token);

        Assert.Null(result);
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task Synthesize_Timeout_IsTransientFailure()
    {
        using var server = new FakeCloudServer((_, _) =>
        {
            Thread.Sleep(1500);
            return (200, "{}");
        });
        // 直接构造 1 秒超时的设置（设置面板不暴露这些高级项；provider 尊重实际值）。
        using var shortTimeout = new HttpCloudVoiceProvider(SettingsFor(server, retries: 0) with { TimeoutSeconds = 1 });

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        Exception? caught = null;
        try
        {
            await shortTimeout.SynthesizeAsync(TestAudio.SampleRequest(), null, CancellationToken.None);
        }
        catch (Exception ex)
        {
            caught = ex;
        }
        stopwatch.Stop();

        var exception = Assert.IsType<CloudProviderException>(caught);
        Assert.True(exception.IsTransient, $"elapsed={stopwatch.ElapsedMilliseconds}ms message={exception.Message} inner={exception.InnerException?.GetType().Name}");
    }

    [Fact]
    public async Task Synthesize_MissingEndpoint_FailsFast()
    {
        using var provider = new HttpCloudVoiceProvider(new CloudVoiceSettings());

        var exception = await Assert.ThrowsAsync<CloudProviderException>(
            () => provider.SynthesizeAsync(TestAudio.SampleRequest(), null, CancellationToken.None));

        Assert.Contains("not configured", exception.Message);
        Assert.False(exception.IsTransient);
    }
}

public class MockCloudVoiceProviderTests
{
    [Fact]
    public async Task Synthesize_ProducesAudioForNotesAndSkipsRests()
    {
        var request = TestAudio.SampleRequest("la", 0.0, 1.0);
        request.Notes.Add(new CloudRequestNote { Key = "note-2", Start = 1.0, Duration = 0.5, Pitch = 62, Lyric = "ma" });
        request.Notes.Add(new CloudRequestNote { Key = "note-3", Start = 1.5, Duration = 0.5, Pitch = 64, Lyric = "", Rest = true });
        request.EndTime = 2.0;

        var result = await MockCloudVoiceProvider.Instance.SynthesizeAsync(request, null, CancellationToken.None);

        Assert.NotNull(result);
        Assert.Equal(48000, result!.SampleRate);
        Assert.Equal(96000, result.Audio.Length);
        Assert.Contains(result.Audio, sample => Math.Abs(sample) > 0.01f);
        Assert.True(result.Phonemes.ContainsKey("note-1"));
        Assert.True(result.Phonemes.ContainsKey("note-2"));
        Assert.False(result.Phonemes.ContainsKey("note-3"));
        Assert.Equal(2, result.PitchSegments.Count);
        Assert.True(result.Parameters.ContainsKey("energy"));
    }

    [Fact]
    public async Task Synthesize_Canceled_ReturnsNull()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var provider = new MockCloudVoiceProvider(latencyMs: 200);

        var result = await provider.SynthesizeAsync(TestAudio.SampleRequest(), null, cts.Token);

        Assert.Null(result);
    }
}
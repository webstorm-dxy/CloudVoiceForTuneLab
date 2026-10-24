using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using TuneLab.Foundation;
using TuneLab.SDK;
using TuneLab.CloudVoice.Audio;
using TuneLab.CloudVoice.Protocol;

namespace TuneLab.CloudVoice.Providers;

// 云端 HTTP provider：把协议请求 POST 到设置里的端点，解码响应里的音频 / 音素 / 音高 / 参数。
// 重试策略：网络错误、超时、429、5xx 视为可重试；4xx 与协议错误直接失败（重试也不会好）。
// 取消语义：调用方取消时返回 null（正常调度结局），不向上抛 OperationCanceledException。
public sealed class HttpCloudVoiceProvider : ICloudVoiceProvider
{
    readonly CloudVoiceSettings mSettings;
    readonly HttpClient mHttp;

    public HttpCloudVoiceProvider(CloudVoiceSettings settings)
    {
        mSettings = settings;
        // 超时按「每次尝试」用 linked CTS 控制，HttpClient 自身不设超时（否则与重试叠加）。
        mHttp = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = true })
        {
            Timeout = Timeout.InfiniteTimeSpan,
        };
    }

    public void Dispose() => mHttp.Dispose();

    public async Task<CloudSynthesisResult?> SynthesizeAsync(CloudSynthesisRequest request, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(mSettings.Endpoint))
            throw new CloudProviderException("The cloud endpoint is not configured. Fill it in Settings → Extensions → Cloud Voice.", false);
        if (!Uri.TryCreate(mSettings.Endpoint.Trim(), UriKind.Absolute, out var endpointUri)
            || (endpointUri.Scheme != Uri.UriSchemeHttp && endpointUri.Scheme != Uri.UriSchemeHttps))
            throw new CloudProviderException($"The cloud endpoint '{mSettings.Endpoint}' is not a valid http(s) URL.", false);

        var payload = CloudProtocol.SerializeRequest(request);
        int attempts = 1 + Math.Max(0, mSettings.Retries);

        for (int attempt = 1; attempt <= attempts; attempt++)
        {
            if (cancellationToken.IsCancellationRequested)
                return null;

            try
            {
                return await SendOnceAsync(endpointUri, payload, request, progress, cancellationToken).ConfigureAwait(false);
            }
            catch (CloudProviderException ex) when (ex.IsTransient && attempt < attempts)
            {
                progress?.Report(0.05 * attempt);
                try
                {
                    await Task.Delay(TimeSpan.FromMilliseconds(400 * attempt), cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return null;
                }
            }
        }

        return null; // 不可达（循环内要么 return 要么抛）
    }

    async Task<CloudSynthesisResult?> SendOnceAsync(Uri endpointUri, string payload, CloudSynthesisRequest request, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(Math.Max(1, mSettings.TimeoutSeconds)));

        using var message = new HttpRequestMessage(HttpMethod.Post, endpointUri)
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json"),
        };
        ApplyHeaders(message);

        progress?.Report(0.15);
        HttpResponseMessage response;
        string body;
        try
        {
            response = await mHttp.SendAsync(message, HttpCompletionOption.ResponseContentRead, timeout.Token).ConfigureAwait(false);
            body = await response.Content.ReadAsStringAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return null;
        }
        catch (OperationCanceledException ex)
        {
            throw new CloudProviderException($"The cloud request timed out after {mSettings.TimeoutSeconds}s.", true, ex);
        }
        catch (HttpRequestException ex)
        {
            throw new CloudProviderException("The cloud request failed: " + ex.Message, true, ex);
        }

        using (response)
        {
            if (!response.IsSuccessStatusCode)
            {
                bool transient = response.StatusCode is HttpStatusCode.RequestTimeout or HttpStatusCode.TooManyRequests
                    || (int)response.StatusCode >= 500;
                throw new CloudProviderException(
                    $"The cloud service returned HTTP {(int)response.StatusCode} ({response.ReasonPhrase}). {Truncate(body)}",
                    transient);
            }
        }

        progress?.Report(0.55);

        var parsed = CloudProtocol.DeserializeResponse(body);
        if (!string.IsNullOrWhiteSpace(parsed.Error))
            throw new CloudProviderException("The cloud service reported an error: " + parsed.Error!.Trim(), false);
        if (parsed.Audio == null)
            throw new CloudProviderException("The cloud response contains no audio.", false);

        int fallbackRate = parsed.SampleRate ?? parsed.Audio.SampleRate ?? request.SampleRate;
        if (fallbackRate <= 0) fallbackRate = 48000;

        byte[] bytes;
        if (!string.IsNullOrWhiteSpace(parsed.Audio.Data))
        {
            try
            {
                bytes = Convert.FromBase64String(parsed.Audio.Data!.Trim());
            }
            catch (FormatException ex)
            {
                throw new CloudProviderException("The cloud audio payload is not valid base64.", false, ex);
            }
        }
        else if (!string.IsNullOrWhiteSpace(parsed.Audio.Url))
        {
            bytes = await DownloadAsync(endpointUri, parsed.Audio.Url!.Trim(), progress, cancellationToken).ConfigureAwait(false);
        }
        else
        {
            throw new CloudProviderException("The cloud response has neither 'audio.data' nor 'audio.url'.", false);
        }

        progress?.Report(0.8);
        var decoded = AudioDecoder.Decode(bytes, parsed.Audio.Format, fallbackRate, parsed.Audio.Channels);
        if (decoded.Samples.Length == 0)
            throw new CloudProviderException("The cloud returned empty audio.", false);
        if (parsed.SampleRate is > 0 && parsed.SampleRate != decoded.SampleRate)
            decoded = decoded with { SampleRate = decoded.SampleRate }; // 解码器已用 WAV 头/采样率优先，保持一致

        progress?.Report(0.9);
        return new CloudSynthesisResult
        {
            Audio = decoded.Samples,
            SampleRate = decoded.SampleRate,
            AudioStartTime = parsed.AudioStart ?? request.StartTime,
            Phonemes = ToPhonemes(parsed.Phonemes),
            PitchSegments = ToSegments(parsed.Pitch),
            Parameters = ToParameters(parsed.Parameters),
            Meta = parsed.Meta == null ? null : JsonSerializer.Serialize(parsed.Meta),
        };
    }

    // 插件设置只有服务地址 + API Key：鉴权固定用 Authorization: Bearer。
    void ApplyHeaders(HttpRequestMessage message)
    {
        if (!string.IsNullOrWhiteSpace(mSettings.ApiKey))
            message.Headers.TryAddWithoutValidation("Authorization", "Bearer " + mSettings.ApiKey);
    }

    async Task<byte[]> DownloadAsync(Uri endpointUri, string url, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        Uri? audioUri = Uri.TryCreate(url, UriKind.Absolute, out var absolute) ? absolute : new Uri(endpointUri, url);
        if (audioUri.Scheme != Uri.UriSchemeHttp && audioUri.Scheme != Uri.UriSchemeHttps)
            throw new CloudProviderException("The cloud audio URL must be http(s).", false);

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(Math.Max(1, mSettings.TimeoutSeconds)));
        try
        {
            using var message = new HttpRequestMessage(HttpMethod.Get, audioUri);
            // 不给跨源音频地址发 API Key（避免把凭据泄露给第三方存储）。
            if (!string.IsNullOrWhiteSpace(mSettings.ApiKey) && Uri.Compare(audioUri, endpointUri, UriComponents.Host | UriComponents.Port, UriFormat.Unescaped, StringComparison.OrdinalIgnoreCase) == 0)
                message.Headers.TryAddWithoutValidation("Authorization", "Bearer " + mSettings.ApiKey);
            progress?.Report(0.65);
            using var response = await mHttp.SendAsync(message, HttpCompletionOption.ResponseContentRead, timeout.Token).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
                throw new CloudProviderException($"Downloading the cloud audio failed: HTTP {(int)response.StatusCode} ({response.ReasonPhrase}).", true);
            return await response.Content.ReadAsByteArrayAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (OperationCanceledException ex)
        {
            throw new CloudProviderException("Downloading the cloud audio timed out.", true, ex);
        }
        catch (HttpRequestException ex)
        {
            throw new CloudProviderException("Downloading the cloud audio failed: " + ex.Message, true, ex);
        }
    }

    static IReadOnlyMap<string, CloudSyllable> ToPhonemes(Dictionary<string, CloudPhonemes>? source)
    {
        var map = new Map<string, CloudSyllable>();
        if (source == null)
            return map;
        foreach (var pair in source)
        {
            if (string.IsNullOrEmpty(pair.Key) || pair.Value == null)
                continue;
            map[pair.Key] = new CloudSyllable
            {
                Leading = pair.Value.Leading.Select(ToPhoneme).ToList(),
                Body = pair.Value.Body.Select(ToPhoneme).ToList(),
                BodyOffset = pair.Value.BodyOffset,
            };
        }
        return map;
    }

    static SynthesizedPhoneme ToPhoneme(CloudPhoneme phoneme)
        => new() { Symbol = phoneme.Symbol ?? "", Duration = Math.Max(0, phoneme.Duration), StretchWeight = Math.Max(0, phoneme.StretchWeight) };

    static IReadOnlyList<IReadOnlyList<Point>> ToSegments(List<CloudPoint>? points)
    {
        if (points == null || points.Count < 2)
            return [];
        var segment = new List<Point>(points.Count);
        foreach (var point in points)
        {
            if (!double.IsFinite(point.T) || !double.IsFinite(point.V))
                continue;
            segment.Add(new Point(point.T, point.V));
        }
        return segment.Count >= 2 ? [segment] : [];
    }

    static IReadOnlyMap<string, IReadOnlyList<IReadOnlyList<Point>>> ToParameters(Dictionary<string, List<CloudPoint>>? source)
    {
        var map = new Map<string, IReadOnlyList<IReadOnlyList<Point>>>();
        if (source == null)
            return map;
        foreach (var pair in source)
        {
            var segments = ToSegments(pair.Value);
            if (segments.Count > 0)
                map[pair.Key] = segments;
        }
        return map;
    }

    static string Truncate(string text)
    {
        text = text.Trim();
        if (text.Length == 0)
            return "";
        return text.Length <= 300 ? text : text[..300] + "…";
    }
}
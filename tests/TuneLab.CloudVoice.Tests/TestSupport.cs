using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using TuneLab.CloudVoice.Protocol;

namespace TuneLab.CloudVoice.Tests;

// 最小 HTTP/1.1 测试服务器：TcpListener 手写，跨平台且不占 URL ACL。
// 每个请求交给 handler 决定状态码与响应体；请求被完整捕获供断言。
public sealed class FakeCloudServer : IDisposable
{
    public sealed record CapturedRequest(string Method, string Path, IReadOnlyDictionary<string, string> Headers, string Body);

    readonly TcpListener mListener;
    readonly Thread mThread;
    readonly Func<CapturedRequest, int, (int Status, string Body)> mHandler;
    readonly List<CapturedRequest> mRequests = [];
    int mCount;
    volatile bool mDisposed;

    // handler 参数：(请求, 从 1 开始的第几次请求) → (状态码, 响应体)。
    public FakeCloudServer(Func<CapturedRequest, int, (int Status, string Body)> handler)
    {
        mHandler = handler;
        mListener = new TcpListener(IPAddress.Loopback, 0);
        mListener.Start();
        Port = ((IPEndPoint)mListener.LocalEndpoint).Port;
        mThread = new Thread(AcceptLoop) { IsBackground = true };
        mThread.Start();
    }

    public int Port { get; }
    public string Endpoint => $"http://127.0.0.1:{Port}/synthesize";

    public IReadOnlyList<CapturedRequest> Requests
    {
        get
        {
            lock (mRequests)
                return mRequests.ToList();
        }
    }

    void AcceptLoop()
    {
        while (!mDisposed)
        {
            TcpClient client;
            try
            {
                client = mListener.AcceptTcpClient();
            }
            catch (Exception)
            {
                return;
            }

            try
            {
                using (client)
                {
                    var request = ReadRequest(client);
                    int count;
                    lock (mRequests)
                    {
                        mCount++;
                        count = mCount;
                        mRequests.Add(request);
                    }
                    var (status, body) = mHandler(request, count);
                    WriteResponse(client, status, body);
                }
            }
            catch (Exception)
            {
                // 测试服务器：单个连接失败不影响后续请求。
            }
        }
    }

    static CapturedRequest ReadRequest(TcpClient client)
    {
        var stream = client.GetStream();
        var buffer = new MemoryStream();
        var chunk = new byte[8192];
        int headerEnd = -1;
        while (headerEnd < 0)
        {
            int read = stream.Read(chunk, 0, chunk.Length);
            if (read <= 0)
                throw new IOException("connection closed before headers");
            buffer.Write(chunk, 0, read);
            headerEnd = FindHeaderEnd(buffer.GetBuffer(), (int)buffer.Length);
        }

        var all = buffer.GetBuffer();
        var headerText = Encoding.ASCII.GetString(all, 0, headerEnd);
        var lines = headerText.Split("\r\n", StringSplitOptions.None);
        var requestLine = lines[0].Split(' ');
        var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in lines.Skip(1))
        {
            int colon = line.IndexOf(':');
            if (colon <= 0) continue;
            headers[line[..colon].Trim()] = line[(colon + 1)..].Trim();
        }

        int contentLength = headers.TryGetValue("Content-Length", out var value) && int.TryParse(value, out var parsed) ? parsed : 0;
        var body = new MemoryStream();
        int alreadyRead = (int)buffer.Length - (headerEnd + 4);
        if (alreadyRead > 0)
            body.Write(all, headerEnd + 4, Math.Min(alreadyRead, contentLength));
        while (body.Length < contentLength)
        {
            int read = stream.Read(chunk, 0, Math.Min(chunk.Length, (int)(contentLength - body.Length)));
            if (read <= 0) break;
            body.Write(chunk, 0, read);
        }

        return new CapturedRequest(requestLine[0], requestLine[1], headers, Encoding.UTF8.GetString(body.ToArray()));
    }

    static int FindHeaderEnd(byte[] buffer, int length)
    {
        for (int i = 0; i + 3 < length; i++)
        {
            if (buffer[i] == '\r' && buffer[i + 1] == '\n' && buffer[i + 2] == '\r' && buffer[i + 3] == '\n')
                return i;
        }
        return -1;
    }

    static void WriteResponse(TcpClient client, int status, string body)
    {
        var bytes = Encoding.UTF8.GetBytes(body);
        var reason = status switch
        {
            200 => "OK",
            400 => "Bad Request",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Status",
        };
        var head = Encoding.ASCII.GetBytes(
            $"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {bytes.Length}\r\nConnection: close\r\n\r\n");
        var stream = client.GetStream();
        stream.Write(head);
        stream.Write(bytes);
        stream.Flush();
    }

    public void Dispose()
    {
        mDisposed = true;
        try { mListener.Stop(); } catch (Exception) { }
    }
}

public static class TestAudio
{
    // 生成 16-bit PCM 单声道 WAV 字节：frequency 的短正弦，duration 秒。
    public static byte[] Wav(float frequency, double duration, int sampleRate, int channels = 1)
    {
        int frames = Math.Max(1, (int)Math.Round(duration * sampleRate));
        var pcm = new MemoryStream();
        Span<byte> two = stackalloc byte[2];
        for (int i = 0; i < frames; i++)
        {
            short sample = (short)(Math.Sin(2 * Math.PI * frequency * i / sampleRate) * 12000);
            BinaryPrimitives.WriteInt16LittleEndian(two, sample);
            for (int c = 0; c < channels; c++)
                pcm.Write(two);
        }

        var bytes = pcm.ToArray();
        var wav = new MemoryStream();
        var writer = new BinaryWriter(wav);
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + bytes.Length);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * 2);
        writer.Write((short)(channels * 2));
        writer.Write((short)16);
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(bytes.Length);
        writer.Write(bytes);
        writer.Flush();
        return wav.ToArray();
    }

    public static CloudSynthesisRequest SampleRequest(string lyric = "la", double start = 1.0, double duration = 0.5)
    {
        var request = new CloudSynthesisRequest
        {
            Voice = "test-voice",
            SampleRate = 48000,
            StartTime = start,
            EndTime = start + duration,
        };
        request.Notes.Add(new CloudRequestNote
        {
            Key = "note-1",
            Start = start,
            Duration = duration,
            Pitch = 60,
            Lyric = lyric,
            Properties = { ["dynamics"] = 0.0, ["breathiness"] = 0.2 },
        });
        request.Pitch.Add(new CloudPoint { T = start, V = 60 });
        request.Pitch.Add(new CloudPoint { T = start + duration, V = 62 });
        request.Parameters["dynamics"] = [new CloudPoint { T = start, V = 0 }];
        return request;
    }

    public static string Json(object value) => JsonSerializer.Serialize(value, CloudProtocol.JsonOptions);
}
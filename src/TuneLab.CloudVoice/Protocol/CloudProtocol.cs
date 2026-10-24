using System.Text.Json;
using System.Text.Json.Serialization;

namespace TuneLab.CloudVoice.Protocol;

// 云端歌声合成协议 v1（tunelab-cloudvoice/1）的线上模型。
//
// 请求：POST <endpoint>，Content-Type: application/json，Authorization 头按设置拼。
//   {
//     "protocol": "tunelab-cloudvoice", "version": 1,
//     "voice": "<provider voice id>", "model": "...", "sampleRate": 48000,
//     "startTime": 1.5, "endTime": 4.0,
//     "notes": [ { "key": "<note id>", "start": 1.5, "duration": 0.5, "pitch": 60, "lyric": "la",
//                  "rest": false, "phonemes": { "leading": [...], "body": [...], "bodyOffset": 0.0 },
//                  "properties": { "dynamics": 0.0, "breathiness": 0.2 } } ],
//     "pitch": [ { "t": 1.5, "v": 60.0 }, ... ],                  // 已解析的最终音高（含 PitchDeviation），100Hz
//     "parameters": { "dynamics": [ { "t": 1.5, "v": 0.1 }, ... ], ... }  // 50Hz 采样，分段轨的 NaN 段不发送
//   }
//
// 响应：200 + JSON
//   {
//     "protocol": "tunelab-cloudvoice", "version": 1,
//     "sampleRate": 48000, "audioStart": 1.5,
//     "audio": { "format": "pcm_f32le" | "pcm_s16le" | "wav", "channels": 1, "data": "<base64>" }
//            | { "format": "wav", "url": "https://..." },
//     "phonemes": { "<note key>": { "leading": [...], "body": [...], "bodyOffset": 0.0 } },  // 可选
//     "pitch": [ { "t": 1.5, "v": 60.0 }, ... ],                  // 可选：合成音高回显
//     "parameters": { "energy": [ { "t": 1.5, "v": 55.0 }, ... ] },// 可选：参数回显（当前只消费 energy）
//     "meta": { ... }                                              // 可选：任意调试信息
//   }
// 错误响应：非 2xx，或 200 但带 "error" 字段（非空字符串）。
public static class CloudProtocol
{
    public const string Name = "tunelab-cloudvoice";
    public const int Version = 1;

    public static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true,
        WriteIndented = false,
    };

    public static string SerializeRequest(CloudSynthesisRequest request)
        => JsonSerializer.Serialize(request, JsonOptions);

    public static CloudSynthesisResponse DeserializeResponse(string json)
    {
        CloudSynthesisResponse? response;
        try
        {
            response = JsonSerializer.Deserialize<CloudSynthesisResponse>(json, JsonOptions);
        }
        catch (JsonException ex)
        {
            throw new FormatException("The cloud response is not valid protocol JSON: " + ex.Message, ex);
        }
        return response ?? throw new FormatException("The cloud response is empty JSON.");
    }
}

public sealed class CloudSynthesisRequest
{
    public string Protocol { get; set; } = CloudProtocol.Name;
    public int Version { get; set; } = CloudProtocol.Version;
    public string Voice { get; set; } = "";
    public string? Model { get; set; }
    public int SampleRate { get; set; }
    public double StartTime { get; set; }
    public double EndTime { get; set; }
    // "auto" = 后端可自行预测音高（若后端声明的 pitch.auto 且用户没要求手动）；
    // "manual" = 请按 pitch 曲线唱。具体由服务端能力决定，后端可忽略。
    public string? PitchMode { get; set; }
    // part 级后端参数（来自 schema.partProperties），值类型为 number / bool / string。
    public Dictionary<string, object> PartProperties { get; set; } = [];
    public List<CloudRequestNote> Notes { get; set; } = [];
    public List<CloudPoint> Pitch { get; set; } = [];
    public Dictionary<string, List<CloudPoint>> Parameters { get; set; } = [];
}

public sealed class CloudRequestNote
{
    public string Key { get; set; } = "";
    public double Start { get; set; }
    public double Duration { get; set; }
    public int Pitch { get; set; }
    public string Lyric { get; set; } = "";
    public bool Rest { get; set; }
    public CloudPhonemes? Phonemes { get; set; }
    // note 级后端参数（来自 schema.noteProperties），值类型为 number / bool / string。
    public Dictionary<string, object> Properties { get; set; } = [];
}

public sealed class CloudPhonemes
{
    public List<CloudPhoneme> Leading { get; set; } = [];
    public List<CloudPhoneme> Body { get; set; } = [];
    public double BodyOffset { get; set; }
}

public sealed class CloudPhoneme
{
    public string Symbol { get; set; } = "";
    public double Duration { get; set; }
    public double StretchWeight { get; set; }
}

// 曲线点：t = 全局秒，v = 该曲线坐标值（音高为半音、参数为参数值）。
public sealed class CloudPoint
{
    public double T { get; set; }
    public double V { get; set; }
}

public sealed class CloudSynthesisResponse
{
    public string? Protocol { get; set; }
    public int? Version { get; set; }
    public int? SampleRate { get; set; }
    public double? AudioStart { get; set; }
    public CloudAudio? Audio { get; set; }
    public string? Error { get; set; }
    public Dictionary<string, CloudPhonemes>? Phonemes { get; set; }
    public List<CloudPoint>? Pitch { get; set; }
    public Dictionary<string, List<CloudPoint>>? Parameters { get; set; }
    public Dictionary<string, object>? Meta { get; set; }
}

public sealed class CloudAudio
{
    public string Format { get; set; } = "pcm_f32le";
    public int? SampleRate { get; set; }
    public int? Channels { get; set; }
    // base64 内联音频；与 Url 二选一。
    public string? Data { get; set; }
    // 音频下载地址（绝对 URL，或相对 endpoint 的路径）。默认不携带 Authorization 头，跨源时更安全。
    public string? Url { get; set; }
}
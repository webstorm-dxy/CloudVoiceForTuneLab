# tunelab-cloudvoice/1 协议规范

Cloud Voice 插件把一块（按音符间隙切出的）音乐片段描述成一份 JSON，POST 给用户配置的端点，
再把响应里的音频与回显产物发布给 TuneLab。本文是**服务端实现者**需要的字段字典。

> 要从零搭建一个服务端（最小实现、接真实模型、部署、验收与排障），先读
> [server-guide.md](server-guide.md)；本文负责"每个字段是什么"。

- 传输：HTTP/1.1 POST，`Content-Type: application/json`；响应 200 + JSON。
- 鉴权：固定 `Authorization: Bearer <API key>`（插件设置里只有服务地址与 API key 两个字段）。
- 超时 / 重试：每次尝试独立超时 120s，网络错误 / 超时 / 429 / 5xx 自动重试 1 次（固定退避）。
- 取消：宿主取消时插件会中止在途请求；服务端应把客户端断连当作取消处理。
- 声库目录：插件加载时自动请求 **`GET <endpoint 同级 /voices>`**，响应为
  `{ "voices": [ { "id", "name", "description", "voice"? } ] }` 或顶层数组；每个声库至少要有 `id`。
  这个端点不是合成协议的一部分，但服务商必须提供它（否则插件退回内置离线演示声库）。

## 请求

```json
{
  "protocol": "tunelab-cloudvoice",
  "version": 1,
  "voice": "provider-voice-id",
  "sampleRate": 48000,
  "startTime": 1.5,
  "endTime": 4.0,
  "pitchMode": "auto",
  "partProperties": { "PitchMode": "predict", "Language": "mandarin" },
  "notes": [
    {
      "key": "note-runtime-id",
      "start": 1.5,
      "duration": 1.2,
      "pitch": 60,
      "lyric": "la",
      "rest": false,
      "phonemes": {
        "leading": [ { "symbol": "l", "duration": 0.06, "stretchWeight": 0 } ],
        "body":    [ { "symbol": "a", "duration": 0.3,  "stretchWeight": 1 } ],
        "bodyOffset": 0.06
      },
      "properties": { "language": "", "musical_type": "singing" }
    }
  ],
  "pitch": [
    { "t": 1.50, "v": 60.0 },
    { "t": 1.51, "v": 60.0 }
  ],
  "parameters": {
    "DYN": [ { "t": 1.50, "v": 0.0 }, { "t": 1.52, "v": 0.1 } ],
    "PIT": [ { "t": 1.50, "v": 0.0 }, { "t": 1.52, "v": 0.0 } ]
  }
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `voice` | 设置里声库条目的 `voice`（省略时是 `id`）。 |
| `model` | 可选；当前插件不发送，服务端可忽略。 |
| `sampleRate` | 期望输出采样率（插件固定请求 48000）。响应可返回别的采样率，宿主会重采样。 |
| `pitchMode` | `auto` = 后端声明了自动音高（schema `pitch.auto`）；`manual` = 请按 `pitch` 曲线渲染。具体是否预测仍由后端引擎参数（如 part 属性 `PitchMode`）决定，后端可忽略。 |
| `partProperties` | part 级后端参数（来自 schema `partProperties` 的 key），值是 number / boolean / string。 |
| `notes[].properties` | note 级后端参数（来自 schema `noteProperties`），值是 number / boolean / string。 |
| `startTime` / `endTime` | 本块在工程时间轴上的全局秒范围。**响应音频默认按 `startTime` 对齐**，可用 `audioStart` 覆盖。 |
| `notes[].key` | note 的运行期 id；响应里的 `phonemes` 用它作 key。 |
| `notes[].start` / `duration` | 全局秒。**延音（melisma）已经被合并进链头 note 的 duration**，不会作为独立 note 出现。 |
| `notes[].pitch` | MIDI 半音整数（自由区的兜底音高）。 |
| `notes[].lyric` | 歌词原文（汉字可能是原样汉字，G2P 归服务端）。空串/纯空白/R = 休止。 |
| `notes[].rest` | 休止标记；为 true 时该 note 不应发声。 |
| `notes[].phonemes` | 仅当用户在 TuneLab 里**钉死音素**时出现（此时必须遵守）；否则缺省、由服务端 G2P + 定时。 |
| `pitch` | 已解析的最终音高曲线：100Hz 采样、全局秒 → 半音。已包含用户钉死的音高轨与颤音等 PitchDeviation。自动音高模式下它仍会发送，可作为「用户期望」的提示；是否当作绝对 F0 由后端决定。 |
| `parameters` | 可画线自动化轨的 50Hz 采样，key = schema `automations` 的 id（后端可自行定义）；整块为默认值的轨会被省略。分段轨的 NaN 段不会出现。 |

## 后端能力 schema（GET /schema）

插件加载时先拉声库目录（`GET /voices`），再从中取 `schema` 字段；没有就单独
`GET <endpoint 同级 /schema>`。schema 决定**属性面板里有哪些参数、能画哪些自动化轨、
有哪些只读回显轨、以及后端是否支持自动音高**；服务端没有 schema 时插件回退到固定默认声明
（`dynamics` / `breathiness` 两条可画线轨 + `energy` 回显 + 两个 note 参数，自动音高关闭）。

```json
{
  "pitch": { "auto": true },
  "partProperties": [
    { "id": "PitchMode", "label": "Pitch mode 音高模式", "type": "combo", "default": "predict",
      "options": [ { "value": "tunelab", "label": "TuneLab" }, { "value": "predict", "label": "Auto" } ] },
    { "id": "Language", "label": "Language 语言", "type": "combo", "default": "mandarin",
      "options": [ { "value": "mandarin", "label": "mandarin" }, { "value": "english", "label": "english" } ] }
  ],
  "noteProperties": [
    { "id": "language", "label": "Language 语言", "type": "combo", "default": "",
      "options": [ { "value": "", "label": "Follow part" }, { "value": "mandarin", "label": "mandarin" } ] },
    { "id": "musical_type", "label": "Musical type 音乐类型", "type": "combo", "default": "singing",
      "options": [ { "value": "singing", "label": "singing" }, { "value": "rap", "label": "rap" } ] }
  ],
  "automations": [
    { "id": "DYN", "label": "Dynamics 力度", "min": -1, "max": 1, "default": 0, "color": "#73ACE5" },
    { "id": "PIT", "label": "Pitch bend 弯音", "min": -1, "max": 1, "default": 0 }
  ],
  "synthesizedParameters": [
    { "id": "energy", "label": "Energy 能量", "min": 0, "max": 100, "color": "#8ADF73" }
  ]
}
```

- `pitch.auto`：后端是否有自动音高能力。为 true 时插件会在请求里带 `pitchMode: "auto"`；
  后端返回的 `pitch` 回显会作为编辑器的**合成音高**显示（自动预测的 F0），而不是用户画线。
- `partProperties` / `noteProperties`：属性面板字段。`type` 支持
  `slider` / `integer` / `number` / `combo` / `checkbox` / `text`；数字字段用
  `min` / `max` / `step` / `default` / `unit`，combo 用 `options` + 默认值。
  用户改的值按 **part / note** 两级原样回传到请求的 `partProperties` / `notes[].properties`。
- `automations`：可画线轨。`default` 存在 = 连续轨（无数据时取默认值）；缺省 = 分段轨（没画就不绘制）。
  `color` / `unit` 只影响显示。
- `synthesizedParameters`：只读回显轨。服务端在响应 `parameters` 里按同样的 id 返回曲线，
  插件画成不可编辑轨（例如能量）。

## 响应

```json
{
  "protocol": "tunelab-cloudvoice",
  "version": 1,
  "sampleRate": 48000,
  "audioStart": 1.5,
  "audio": {
    "format": "pcm_f32le",
    "channels": 1,
    "data": "<base64>"
  },
  "phonemes": {
    "note-runtime-id": {
      "leading": [],
      "body": [ { "symbol": "a", "duration": 0.3, "stretchWeight": 1 } ],
      "bodyOffset": 0.0
    }
  },
  "pitch": [ { "t": 1.5, "v": 60.0 }, { "t": 4.0, "v": 67.0 } ],
  "parameters": { "energy": [ { "t": 1.5, "v": 55.0 }, { "t": 4.0, "v": 80.0 } ] },
  "meta": { "anything": "for debugging" }
}
```

- `audio`：二选一 ——
  - `data`：base64 内联音频；`format` 支持 `pcm_f32le` / `pcm_s16le` / `wav`；
  - `url`：音频下载地址（绝对 URL 或相对 endpoint 的路径；**跨源 URL 不会带 API key**）。
  - 多声道会下混成单声道；WAV 可带自己的采样率。
- `phonemes`：可选。key = 请求里的 `notes[].key`。**键存在 + 空列表 = 该 note 已算完但不发音**
  （例如休止）；键缺失 = 没答案，宿主会让相邻 note 一起留白。延音 note 不会出现在请求里，插件会
  自己回填空音节。
- `phonemes[].leading` / `body`：引导辅音 + 主体（元音/尾辅音）。每项只报 `symbol` / `duration`（秒）/
  `stretchWeight`（0 = 刚性辅音，>0 = 可伸元音）；定位、跨 note 去重叠、melisma 铺设由宿主统一做。
- `phonemes[].bodyOffset`：主体起点相对 note 头的有符号偏移（秒）。
- `pitch`：可选，合成音高回显折线（全局秒 → 半音）。缺省时插件用请求里已解析的音高曲线回显。
  **自动音高模式下这里应返回后端实际渲染出的 F0**，编辑器会把它作为只读合成音高显示。
  建议采样间隔 ≤10ms（OpenCyan 服务端直接回模型帧 5ms），否则编辑器里的音高线会呈折线。
- `parameters`：可选，按 schema `synthesizedParameters` 的 id 返回只读回显曲线（如 `energy`）。
  缺省 `energy` 时插件用返回音频的 RMS 包络兜底。
- `error`：非空字符串 → 本次合成失败（HTTP 200 也可以报错），失败文案会显示在状态带。

## 错误与重试约定

| 情况 | 插件行为 |
| --- | --- |
| 连接失败 / DNS / 超时 | 可重试，按设置重试后仍失败 → 该块 Failed |
| HTTP 429 / 5xx | 可重试 |
| HTTP 4xx（除 429） | 不重试，直接 Failed（带状态码与服务端响应片段） |
| 200 但 `error` 非空 | 不重试，直接 Failed |
| `audio` 缺失 / base64 无效 / 解码失败 | 不重试，直接 Failed |

失败只影响该块：状态带显示红色与错误文案；用户编辑该块或修好设置后重新合成。

## 最小服务端示例

`tools/mock_cloud_server.py` 是一个无依赖的 Python 参考实现（POST /synthesize + GET /voices），
它把请求渲染成正弦波形并故意返回 `srv-` 前缀的音素，方便验证"音素确实来自云端响应"。

```bash
python3 tools/mock_cloud_server.py --port 18765 --log /tmp/cloud-requests.jsonl
curl -s http://127.0.0.1:18765/synthesize -H 'Content-Type: application/json' -d '{
  "protocol": "tunelab-cloudvoice", "version": 1, "voice": "test-voice",
  "sampleRate": 48000, "startTime": 0, "endTime": 1,
  "notes": [ { "key": "n1", "start": 0, "duration": 1, "pitch": 60, "lyric": "la", "rest": false, "properties": {} } ],
  "pitch": [ { "t": 0, "v": 60 }, { "t": 1, "v": 60 } ],
  "parameters": {}
}' | head -c 300
```

## 版本演进

`version` 目前恒为 1。加性字段优先（客户端忽略未知字段；服务端也应忽略请求里的未知字段）。
破坏性变更会升 `version`，插件会在不匹配时给出明确错误而不是误解析。
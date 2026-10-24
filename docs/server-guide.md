# 搭建云端歌声合成服务端指南

本文讲**如何为 TuneLab Cloud Voice 插件搭建一个适配的云端歌声合成服务**：协议怎么实现、时间轴与音素
怎么对齐、怎么接真实模型、怎么部署与验收，以及最容易踩的坑。

配套材料：

- 协议字段字典：[cloud-protocol.md](cloud-protocol.md)
- 插件内部设计：[architecture.md](architecture.md)
- 最小参考服务（stdlib，零依赖）：[../tools/mock_cloud_server.py](../tools/mock_cloud_server.py)
- 生产骨架参考实现（FastAPI，本仓库已与 TuneLab 2.1.1 实际对接验证）：
  [../tools/server_example/app.py](../tools/server_example/app.py)
- 真实改造案例：把 OpenCyan 引擎包装成**独立服务器程序**（进程内直接加载 OpenCyan.Net + 原生引擎，
  不经 TuneLab），客户端直接用本仓库的通用云插件 TuneLab.CloudVoice 连接：
  [../demo/OpenCyanCloud/README.md](../demo/OpenCyanCloud/README.md)

## 0. 先理解三件事

1. **这是一个同步的"块合成"接口**。TuneLab 把一段乐谱（按音符间隙切成块）发给你，你返回这段时长的音频。
   没有流式、没有进度回调、没有增量 diff：每次编辑都可能重发整块。所以服务端应当是**无状态计算**
   （缓存可选），把"一次请求"做快做稳。
2. **协议只要求一个端点**：POST /synthesize。请求里已经包含"唱什么、怎么唱"：歌词、音符时长、音高曲线、
   力度/气声曲线、可选的钉死音素。响应里放音频，可选回填音素/音高/能量回显。
3. **音频不是"返回一段声音"就完事**，时间轴必须对齐：响应音频默认从请求的 startTime 开始、长度覆盖
   [startTime, endTime]；音素回显要与音频真正唱的发音一致，否则用户会看到"口型对不上声音"。

## 1. 十分钟跑通最小服务

建议先用仓库里的零依赖 mock 服务验证插件和网络配置，再替换成自己的实现：

```bash
# 终端 A：启动 mock 云服务（会打印收到的请求）
python3 tools/mock_cloud_server.py --port 18765 --log /tmp/cloud-requests.jsonl

# 终端 B：用真实 TuneLab 无头宿主验证（需要先编译 TuneLab.Cli，见 README）
cd /path/to/TuneLabRemotePlugin
TUNELAB_DATA_DIR=/tmp/tunelab-demo \
  TuneLab/TuneLab.Cli/bin/Debug/net8.0/TuneLab.Cli --headless \
  sandbox run --code '
const v = sandbox.voices().find(x => x.type === "CloudVoice" && x.id === "cloud-test");
const t = tl.currentProject().addTrack();
const p = t.addPart({pos: 0, endOffset: 1920});
p.setSoundSource({kind: "voice", type: v.Type, id: v.Id});
const n = p.addNote({pos: 0, dur: 480, pitch: 60, lyric: "la"});
print(JSON.stringify(sandbox.synthesize(p)));
print(JSON.stringify(sandbox.syllable(n)));
'
```

在 TuneLab GUI 里验证时：**设置 → 扩展 → Cloud Voice**，只填 **Service URL**（服务地址）与（可选的）
**API key**；声库目录由插件自动从 `GET /voices` 拉取（见 1.2），然后新建声源片段、输入歌词、播放。

### 1.1 换成生产骨架（FastAPI）

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/server_example/requirements.txt
.venv/bin/python tools/server_example/app.py --host 0.0.0.0 --port 8080
# 需要鉴权时：
.venv/bin/python tools/server_example/app.py --host 0.0.0.0 --port 8080 --api-key secret-token
```

它实现了本文要求的全部协议语义（校验、鉴权、并发限制、WAV 响应、音素回填、错误处理），并用加法合成
做了占位后端。替换 [app.py](../tools/server_example/app.py) 里标着

```
# PLUG IN YOUR MODEL HERE
```

的 synthesize_block() 内部实现即可接真实模型，HTTP 层与协议层不用动。

### 1.2 TuneLab 侧配置（对照表）

| TuneLab 设置 | 填什么 | 服务端对应 |
| --- | --- | --- |
| Service URL | 你的完整 POST 地址，如 https://svc.example.com/v1/synthesize | 路由挂载点（插件不会拼路径） |
| API key | 服务端要求的令牌（可留空） | `Authorization: Bearer <token>` |

插件只暴露这两个设置；采样率固定请求 48kHz、单次超时 120s、失败重试 1 次（服务端按需自行处理）。

声库目录**不需要用户填写**：插件加载时会自动请求 **`GET <endpoint 同级 /voices>`**
（`.../v1/synthesize` → `.../v1/voices`；`.../synthesize` → `.../voices`）。响应可以是
`{ "voices": [ ... ] }` 或顶层数组；每个声库至少要有 `id`，可选 `name` / `description` /
`voice`（服务端侧 voice 参数，缺省 = id）。拿不到目录时插件退回内置离线演示声库并在日志里说明。

## 2. 一次合成请求的生命周期

```
TuneLab（编辑线程）                         你的服务端
──────────────────────────────────────────────────────────────────
用户编辑一首歌
  └─ 事件标脏 → Committed 重分块
      └─ 宿主调度 peek/commit 选出一块 [startTime, endTime]
          └─ 同步前缀：物化快照（音符/音高/曲线）
              └─ worker：构造 JSON 请求
                  ├── POST /synthesize ──────────────► 校验 → 鉴权 → 排队
                  │                                      └─ 合成（模型/第三方）
                  │                                          └─ 编码音频 + 音素/曲线
                  │◄────────────── 200 JSON ──────────┘
                  ├── 解码音频（PCM/WAV/base64 或 URL）
                  └─ 在数据线程发布：音频段 + 音素/音高/能量回显
```

几个直接后果：

- **没有进度回调**。插件状态带的进度是本地阶段（发送前/收到响应/解码完成），不代表服务端生成进度。
  如果需要长时间生成，最好自己维护任务队列，让 HTTP 请求等待的这段时间可被监控。
- **同一 part 的块是串行的**，服务端延迟 = 合成总时长的一部分；多个 part / 多个客户端可以并发，
  但单客户端的并发不高。服务端应该**限制并发并快速返回 503**，而不是无上限排队把超时拖爆。
- **请求是自包含的**。你不需要保存工程状态；同一个块被重发完全正常（编辑、重试、重新打开工程都会发生）。

## 3. 必须/建议实现的 HTTP 接口

| 方法/路径 | 必需 | 说明 |
| --- | --- | --- |
| POST /synthesize（路径由用户配置，可任意） | 是 | 合成一块，返回 JSON |
| GET /health | 建议 | 负载均衡/监控探活，返回 200 即可 |
| GET /voices | 是 | 插件加载时自动拉取声库目录（响应格式见 §1.2） |
| GET /schema | 推荐 | 引擎能力声明：属性面板参数 / 可画线轨 / 回显轨 / 自动音高；没有时插件回退固定声明 |

技术要求：

- HTTP/1.1 或 HTTP/2 均可；客户端是 .NET HttpClient，默认 HTTP/1.1，会复用长连接。
- 请求体 UTF-8 JSON；忽略未知字段（协议是加性的）。
- 响应 Content-Type 无所谓（客户端按字符串读），但请返回 application/json。
- **不要依赖请求头里的 Content-Encoding 压缩**：客户端默认不发 Accept-Encoding，也不会自动解压 gzip。
  反向代理也不要强制对 /synthesize 响应 gzip。
- TLS 建议终止在反代上；插件要求 http(s) URL，自签证书需要客户端信任链（内网测试可用 http://127.0.0.1）。

### 3.1 后端能力 schema：属性面板 / 画线轨 / 自动音高

`GET /schema`（或直接放在 `/voices` 响应的 `schema` 字段里）告诉插件后端引擎能调什么：
插件据此**动态生成**属性面板、可编辑自动化轨、只读回显轨，并决定是否渲染自动音高。
没有 schema 时插件回退到 `dynamics` / `breathiness` + `energy` 的固定声明（兼容旧服务端）。

```json
{
  "pitch": { "auto": true },
  "partProperties": [
    { "id": "PitchMode", "type": "combo", "default": "predict",
      "options": [ { "value": "predict", "label": "Auto" }, { "value": "tunelab", "label": "Manual" } ] }
  ],
  "noteProperties": [
    { "id": "language", "type": "combo", "default": "",
      "options": [ { "value": "", "label": "Follow part" } ] }
  ],
  "automations": [
    { "id": "DYN", "label": "Dynamics", "min": -1, "max": 1, "default": 0, "color": "#73ACE5" }
  ],
  "synthesizedParameters": [
    { "id": "energy", "label": "Energy", "min": 0, "max": 100, "color": "#8ADF73" }
  ]
}
```

实现建议：

- part / note 属性会在请求的 `partProperties` / `notes[].properties` 里**原样回来**，按自己引擎的设置应用；
- `automations` 的 id 由后端定义（OpenCyan 用 `DYN`/`BRE`/`PIT`/…），请求 `parameters` 按同样 id 给曲线；
- `synthesizedParameters` 在响应 `parameters` 里按同样 id 返回曲线（如 `energy`）；
- `pitch.auto = true` 时，把**实际渲染出的 F0** 放在响应 `pitch` 里，插件会把它显示为合成音高；
- 用 `default` 值省略整块曲线是安全的（客户端对全默认值的轨不发送）。

完整字段见 [cloud-protocol.md](cloud-protocol.md#后端能力-schema-get-schema)。

## 4. 请求：字段、语义与校验

完整字段表见 [cloud-protocol.md](cloud-protocol.md)，这里只强调实现时**必须理解**的语义。

### 4.1 请求里已经替你算好的东西

- **notes 是排好序的**（按开始时间升序），且 note 的 start/duration 是**全局秒**，不是 tick。
- **延音（melisma）已经合并进链头 note 的 duration**。用户在 TuneLab 里用歌词 `-` 表示拖长前一个音，
  请求里不会出现这条延音 note，而是链头 note 的 duration 变长。所以服务端不需要理解 `-`，按普通长音处理即可。
- **休止 note 会以 rest=true 出现**，它占时间轴但不发声；音频对应区间应保持静音。
- **pitch 是已解析的最终音高曲线**：100Hz 采样、全局秒 → 半音（MIDI 音高浮点值）。它已经融合了用户的音高轨、
  颤音等 PitchDeviation。请求同时带 `pitchMode`：`auto`（后端声明了自动音高）或 `manual`。
  自动音高模式下把这条曲线当"用户期望/偏差参考"，返回你**实际预测并渲染出的 F0**；
  手动模式下把它当绝对 F0 目标（f = 440 * 2^((semitone-69)/12)）。
- **parameters 是 schema.automations 声明的可画线参数**，50Hz 采样，key = schema 里的 id。
  整块为默认值的轨会被省略（客户端不浪费带宽）；分段轨的空洞不会发送，所以数组可能不等间隔，读时按 t 插值。
- **partProperties / notes[].properties 是 schema 声明的 part / note 参数**（number / boolean / string），
  用户改了属性面板后原样回来；具体如何作用于你的引擎由你决定。
- **notes[].phonemes 只在用户钉死音素时出现**。有它时必须优先遵守；没有它时由服务端自行 G2P 与定时。
- **notes[].phonemes 只在用户钉死音素时出现**。有它时必须优先遵守（音素符号 + 标称时长 + 权重 +
  bodyOffset）；没有它时由服务端自行 G2P 与定时。

### 4.2 服务端必须做的校验（建议直接返回 400）

| 校验 | 原因 |
| --- | --- |
| protocol == "tunelab-cloudvoice" | 防误连 |
| version == 1 | 未来破坏性升级时给出明确错误 |
| startTime/endTime 有限且 endTime > startTime | 防除零/NaN 扩散 |
| 单块时长上限（如 600s） | 见 §6.1 / §9 的长块讨论 |
| notes 数量上限（如 256） | 防滥用 |
| sampleRate 在支持列表 | 避免生成错率音频 |
| note.key 非空、duration > 0 | 响应音素用 key 回填 |
| voice 在允许的声库表中 | 未知 voice 返回 400 并给用户可读错误 |

错误响应就该用 **4xx + {"error": "..."}**：插件对 4xx（除 408/429）不重试，直接把这条错误显示在状态带上，
用户改设置/改音符后重试。**不要用 200 + error 表示参数错误**（虽然协议支持，但会让重试语义变模糊）。

### 4.3 常见请求陷阱

- **中文歌词**：lyric 可能是汉字原样（"春"），也可能是拼音/拉丁音节（"chun"）。G2P 是你的责任，插件不归一化。
- **音素不能靠"每个音符一个元音"想当然**：拼音声母/韵母、英文辅音簇、CJK 多字都可能出现；长度也可能是
  0.05s 的短音。请以毫秒级容错处理。
- **不要修改/回显 note.key**：它是本次请求内的关联键，跨会话会变，不能当持久 id。
- **请求可能重复**：重试、重开工程、编辑都会重发同一个块。服务端应能幂等地重复合成，或者用 body 哈希做缓存。
- **块边界处不要做渐变**：块与块之间在 TuneLab 端是分离的音频段；服务端只需保证块内从静音到静音
  （或按 rest 留静音），不要在块首尾加不必要的淡入淡出，否则咬字会变糊。

## 5. 响应：音频与回显怎么构造

### 5.1 音频（必须）

三种传法，任选其一：

```json
{ "audio": { "format": "wav",       "channels": 1, "data": "<base64>" } }
{ "audio": { "format": "pcm_s16le", "channels": 1, "sampleRate": 48000, "data": "<base64>" } }
{ "audio": { "format": "wav",       "url": "https://cdn.example.com/abc.wav" } }
```

支持的格式矩阵（客户端解码器实现）：

| format | 说明 | 备注 |
| --- | --- | --- |
| wav | RIFF/WAVE，PCM 8/16/24/32 bit 或 IEEE float 32/64 | **采样率以 WAV 头为准**；不支持 WAVE_FORMAT_EXTENSIBLE (0xFFFE) |
| pcm_f32le | 原始小端 float32 | 采样率必须由响应声明（顶层 sampleRate 或 audio.sampleRate） |
| pcm_s16le | 原始小端 int16 | 同上 |
| url | 用 GET 下载上面的格式 | 跨源时**不会**带 API key（可放心用预签名 URL） |

不支持 mp3/flac/ogg。多声道会下混为单声道。客户端对采样率不一致的段会自动重采样（宿主统一做），
所以返回 44.1k/48k/模型原生率都可以，但**声明值必须与实际一致**。

时间对齐规则：

- 音频第一个采样默认对应 **startTime**；响应里的 `audioStart` 可以覆盖（一般不需要，保持 startTime 最稳）。
- 音频长度应覆盖 [startTime, endTime]。短了 → 尾部静音（不报错，但用户可听出缺音）；长了 → 会占用后续时间轴，
  建议严格按块时长返回（允许 ±10ms 误差）。
- 整块都是休止也要返回等长静音，**不要返回空音频**（空音频会被判失败）。

### 5.2 音素回显（可选，但强烈建议）

```json
"phonemes": {
  "note-runtime-id": {
    "leading": [ { "symbol": "l", "duration": 0.06, "stretchWeight": 0 } ],
    "body":    [ { "symbol": "a", "duration": 0.3,  "stretchWeight": 1 } ],
    "bodyOffset": 0.06
  }
}
```

- key 必须用请求里的 note.key；值是该 note 的**标称**音节几何，不要报绝对位置：
  - leading：核前辅音（w=0，固定长）；
  - body：核 + 尾辅音（核 w>0，可伸缩；辅音 w=0）；
  - bodyOffset：主体起点相对 note 头的有符号偏移，junction = noteStart + bodyOffset。
- 客户端拿它做编辑器的音素显示与吸附，定位/去重叠由 TuneLab 统一负责，你只管报"标称时长 + 权重"。
- **键存在 + 空列表 = 这个音算完了但不发音**（休止、或者发音全挂在链头）；**键缺失 = 没答案**（客户端会
  用内置简单 G2P 兜底，不会空白）。如果你不回填音素，编辑器会用简单规则显示（拉丁音节拆分 / 汉字整字），
  能用但不好看；上线建议回填。
- 如果音频与显示都要 WYSIWYG，音素几何必须与音频里实际发音一致；返回的音素符号也尽量贴近模型实际发音
  （例如模型用 `zh` 音素体系，就回报模型的符号）。
- 延音 note 不会出现在请求里，插件会自行补空音节，无需你处理。

### 5.3 音高回显（可选）

```json
"pitch": [ { "t": 1.5, "v": 60.0 }, { "t": 1.6, "v": 60.2 } ]
```

- 全局秒 → 半音，一条折线即可；建议 20–100Hz 采样。
- 不返回时，插件会把请求里的 pitch 原样当回显；返回时请给**实际唱出来的** F0，这样音高轨才诚实。
- 如果模型有自己的滑音/颤音处理，把最终 F0 报回来最有价值。

### 5.4 能量回显（可选）

```json
"parameters": { "energy": [ { "t": 1.5, "v": 55.0 }, { "t": 4.0, "v": 80.0 } ] }
```

- 目前只消费 energy（0..100，分段曲线）。不返回时插件用返回音频的 RMS 包络兜底。
- 想让用户看到"模型实际用的能量/气声"，可以扩展返回别的 key，但客户端当前只显示已声明的 energy 轨。

### 5.5 错误与 JSON 合法性

- HTTP 200 + `{"error": "..."}` = 本次合成失败；插件不重试，错误显示在状态带。
- 非 2xx：408/429/5xx 会被**重试**（默认最多 2 次，退避 400ms×次数，**不解析 Retry-After**）；
  其他 4xx 不重试。
- **绝对不要输出 NaN/Infinity**：Python 的 json.dumps 默认会输出 `NaN`，标准 JSON 不允许，客户端解析
  会失败（且是非重试失败）。用 `json.dumps(..., allow_nan=False)` 或先清洗曲线。
- 响应必须是完整 JSON；空 body / HTTP 204 / 非 JSON 都会被判失败。
- 内部异常请返回 500 + 简短错误，不要把堆栈/密钥写进响应。

## 6. 接入真实歌声合成后端

### 6.1 本地/自研模型（DiffSinger、NNSVS、VITS 类）

推荐把"协议适配层"和"推理层"分开：协议层（本文第 4/5 节）固定，推理层实现一个函数：

```python
def synthesize_block(request: CloudSynthesisRequest) -> dict[str, Any]:
    # 1) 文本 → 音素（G2P）
    #    若 request.notes[i].phonemes 存在，必须优先使用（用户钉死）
    # 2) 音素 → 时长（dur 模型），约束：核铺满 note，辅音固定长
    # 3) 目标 F0 曲线：request.pitch（已含 user vibrato）
    # 4) 条件（可选）：schema 声明的 part / note 参数（如 OpenCyan 的 PitchMode / Language）
    # 5) 声学模型 → mel → 声码器 → 音频（wav 16bit 44.1k/48k）
    # 6) 返回音频 + 实际音素 + 实际 F0 + energy
    ...
```

映射细节：

- **时长**：每个 note 的可用时间是 duration；钉死音素给出标称时长与权重，核需要铺满到 note 末。
  如果有跨 note 的辅音簇/压缩，内部自行处理，但要注意返回的 `phonemes` 与音频用的定时必须一致
  （要么返回你实际用的标称几何，要么干脆不返回、让编辑器用简化规则显示）。
- **音高**：request.pitch 是 100Hz 的最终半音曲线，直接作为 F0 目标（含用户颤音）。模型有自己的 F0
  预测时，把它当约束（例如做偏差学习或直接替换），并把最终实际 F0 回填 `pitch`。
- **力度/气声**：parameters 是 50Hz 的 part 级曲线，properties 是单 note 覆盖。常见做法：
  `final = clamp(part + note, 范围)`，再乘到模型条件 embedding。
- **音素体系**：TuneLab 不强制你的音素集。返回模型实际用的符号即可（例如 `zh`/`en` 或 X-SAMPA），
  编辑器只做显示与布局，不会给符号赋予语义。
- **采样率**：模型原生率可以不是 48k；返回时声明正确即可，宿主会重采样。
- **GPU 并发**：同一张卡建议 `max_concurrent=1` 或按显存压测；超出的请求用队列 + 503 快速拒绝，
  不要无限等待（客户端有超时，排队等太久只会得到超时重试）。
- **长块**：连续长句会合并成一个请求（可能几十秒到几分钟）。模型推理显存/时长要能支撑；
  支撑不了就按上限拒绝（400 并给用户可读文案），或在前端指导用户把长句拆成带休止的多段。

### 6.2 包装第三方/商用歌声合成 API

如果你只是转调一个已有的云端服务，服务端做的是"协议翻译 + 结果对齐"：

1. **voice 映射**：插件设置里的 `voice` → 第三方音色 id / model 名。给未知 voice 返回 400。
2. **输入翻译**：把 notes 的歌词/音素 + 时长 + F0 + 参数转成第三方 API 的请求。多数商用 API 只接受
   "文本 + 音高/时长"，需要你先做 G2P 和时长规划。
3. **分段对齐**：第三方对单次请求长度有上限时，在你的服务内部按句/按静音切开分别调，再按时间轴拼回
   整块音频。拼接处做 5–10ms 交叉淡化，避免爆音；**不要改变整体时间轴**。
4. **返回值翻译**：第三方音频转 WAV/PCM；如果它给音节/音素信息，映射回 note.key；给 F0 就回填 pitch。
5. **错误映射**：第三方 401/403 → 你的 4xx（非重试）；第三方 429/5xx/超时 → 你的 429/5xx（可重试）。
   注意重试会重复计费，建议在服务端做短 TTL 的 body 哈希缓存，或把重试交给插件（同一请求可能来 2–3 次）。
6. **合规**：第三方音色的使用许可、声音克隆的授权、数据留存——都在你这一侧负责；插件不做任何把关。

### 6.3 异步任务队列模式（生成较慢时）

协议是同步的，但服务端内部可以异步：

```
POST /synthesize
  ├─ 参数校验 + 鉴权
  ├─ 计算 idempotencyKey = sha256(body)（可选但推荐）
  ├─ 提交 job 到队列（GPU worker），最多等待 N 秒（N < 插件 timeout）
  │    ├─ 完成 → 编码音频 → 200
  │    └─ 超时 → 503 + Retry-After（插件会重试；重试若命中缓存直接 200）
  └─ 队列满 → 429/503 快速返回，不占住连接
```

要点：

- **等待时间要明显小于插件超时**（默认 60s，用户最多可设 600s）。例如插件超时 60s、你在内部最多等 45s，
  超时返回 503 让插件重试；配合 idempotencyKey 缓存，第二次请求直接拿结果。
- **客户端断开 = 取消**：长时间推理时定期检查连接是否还在（例如每 200ms 检查一次），断了就停止/降优先级，
  否则 GPU 会被已取消的请求占满。FastAPI 里可用 `await request.is_disconnected()` 配合协作式取消。
- **重试不保证顺序**：插件重试是串行的，但多个 part/客户端会并发，队列要能处理乱序完成。
- 如果业务上必须"提交后轮询"，不要指望插件支持；把轮询封装在服务端内部等结果即可。

## 7. 正确性检查清单

实现完先对着这份清单自测（每条都能在 TuneLab 里观察到）：

- [ ] 单音 `la`：音素显示 l + a，音频在 480 tick 内起音收尾，无爆音。
- [ ] 两个音 + 空隙：请求是两个块还是三块取决于音符间隙；每块音频都在自己的 [start,end] 内。
- [ ] 延音 `-`：第二个 note 不出现在请求里，第一个 note duration 变长；音频是连续长音，没有断口。
- [ ] 休止 rest：请求里有 rest=true 的 note，对应时间是静音。
- [ ] 钉死音素：在 TuneLab 里手动改音素后，请求里出现 phonemes 字段；你按它发音后，音频与显示一致。
- [ ] 音高曲线：在 TuneLab 画一条明显滑音，请求 pitch 曲线跟着变，音频也跟着变；回显 F0 与音频一致。
- [ ] schema 参数：属性面板出现 /schema 声明的字段；改值后请求 partProperties / notes[].properties 变化。
- [ ] 可画线轨：schema.automations 的轨出现在编辑器里；画线后请求 parameters 变化，能听出差异。
- [ ] 自动音高：schema.pitch.auto=true 且 PitchMode=predict 时，响应 pitch 是引擎预测的 F0，编辑器显示合成音高。
- [ ] 连续长句：一个请求可能几十秒，服务端不超时、内存不炸、返回音频时长正确。
- [ ] 错误：错 API key → 状态带显示 401 错误且不重试；5xx → 插件按 retries 重试。
- [ ] 取消：合成中关闭工程/停止播放，服务端连接断，GPU 任务能及时释放。
- [ ] 中文歌词、英文歌词、纯辅音歌词各唱一遍不崩。
- [ ] 返回非整数采样率 / 立体声 WAV：宿主能正常重采样/下混。

## 8. 安全、限额与多租户

- **鉴权**：至少 Bearer token；建议每个用户/团队单独签发并支持轮换。插件固定使用
  `Authorization: Bearer <token>`（设置里只有 Service URL 与 API Key；API Key 可留空，但生产环境请开启）。
- **限流**：按 key/IP 限制 QPS 与并发；超限返回 429（插件会重试）或 503，并给简短错误。
- **请求大小**：pitch 100Hz ≈ 3KB/s，parameters ≈ 3KB/s，4 分钟块约 1–2MB JSON。反代 `client_max_body_size`
  至少留 8MB；服务内部再按 notes/时长上限拒绝。
- **长度/资源上限**：单块最长时长、最大 note 数、最大并发、单请求 GPU 秒数都要有硬上限并返回可读错误。
- **日志脱敏**：不要记录 Authorization 头、API key、完整歌词（如需排查用哈希/长度）。
- **音频 URL**：返回 `audio.url` 时用预签名、短过期、只读的地址；客户端对跨源 URL 不带 API key。
- **CORS**：不需要（客户端是桌面 HttpClient）。
- **SSRF**：服务端通常不需要访问用户提供的 URL；如果你为了拉参考音频而访问外部 URL，做域名白名单。
- **多租户**：voice/model 命名空间、配额、限流、缓存隔离都要按租户维度做；不要把 A 的音频缓存返回给 B。

## 9. 性能与容量

先估一笔账（单声道）：

| 音频 | 原始大小 | base64 后 | 4 分钟块 |
| --- | --- | --- | --- |
| float32 48kHz | 192KB/s | 256KB/s | ~61MB |
| int16 48kHz | 96KB/s | 128KB/s | ~31MB |
| int16 44.1kHz | 88KB/s | 118KB/s | ~28MB |

建议：

- **短块（< 60s）**：base64 内联最简单（默认行为）。
- **长块**：返回 `audio.url`，音频落在对象存储（预签名），响应只有几百字节；避免几十 MB JSON
  在客户端和服务端各复制几份。
- 采样率选模型原生率，不要为了"统一"在服务端做无谓重采样（宿主会做一次）。
- **缓存**：对 `sha256(request body)` 做 LRU/对象存储缓存很划算——同一块经常被重发（编辑其它块、
  改音高再改回来、重开工程）。缓存命中直接返回，GPU 利用率会明显下降。
- **队列与并发**：GPU 1–2 并发 / 卡；CPU 声码器可适当提高。用信号量限制，超出快速 503。
- **观测**：记录 request 的 notes 数、音频秒数、排队时间、推理时间、RTF（音频秒/耗时）、错误分类；
  这些指标直接决定容量规划。给每个请求加 request id（可回传到日志，但协议里没有字段，不要指望客户端关联）。
- **超时预算**：用户默认 60s。若你的 P95 推理时间超过 45s，要么优化、要么在前端文档里告知用户调大 timeout。

## 10. 部署与运维

### 10.1 直接运行（FastAPI 示例）

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/server_example/requirements.txt
.venv/bin/uvicorn --app-dir tools/server_example app:create_app --factory \
  --host 0.0.0.0 --port 8080 --workers 1
# 生产建议：进程数按 GPU 数（每个 worker 独占一张卡），或 CPU 后端用 --workers N
```

### 10.2 Dockerfile 示例

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY tools/server_example/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tools/server_example/app.py .
EXPOSE 8080
# --api-key 等配置用环境变量/启动参数注入，不要写进镜像
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080", "--api-key", "change-me"]
```

### 10.3 反向代理（nginx）关键配置

```nginx
location /v1/synthesize {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    client_max_body_size 16m;       # pitch/parameters 数组可能上 MB
    proxy_read_timeout 600s;        # 必须 >= 插件 timeout，且留余量
    proxy_send_timeout 600s;
    proxy_buffering off;            # 大响应直接转发，避免代理落盘/内存翻倍
    proxy_request_buffering off;
}
```

### 10.4 运行注意

- **健康检查**：/health 返回 200；负载均衡别把健康检查打到 /synthesize。
- **优雅关闭**：收到 SIGTERM 后先停止接新请求，等在途请求完成（或超时）再退出；断连要能取消推理。
- **日志**：结构化 JSON 最好；不要把 API key、Authorization、完整歌词写入日志。
- **配置管理**：API key、模型路径、并发数用环境变量/密钥管理注入，不要提交到仓库。
- **HTTPS**：反代终止 TLS；证书有效。内网调试可以 http，但不要在生产用明文传输 key。

## 11. 测试与验收

### 11.1 协议层自测（不依赖 TuneLab）

准备一份最小请求，直接 curl：

```bash
curl -s http://127.0.0.1:8080/synthesize \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer secret-token' \
  -d '{
    "protocol": "tunelab-cloudvoice",
    "version": 1,
    "voice": "demo",
    "sampleRate": 48000,
    "startTime": 0,
    "endTime": 1,
    "notes": [
      {"key": "n1", "start": 0, "duration": 1, "pitch": 60, "lyric": "la", "rest": false,
       "properties": {"dynamics": 0.0, "breathiness": 0.2}}
    ],
    "pitch": [{"t": 0, "v": 60}, {"t": 1, "v": 60}],
    "parameters": {"dynamics": [{"t": 0, "v": 0}, {"t": 1, "v": 0}],
                   "breathiness": [{"t": 0, "v": 0.2}, {"t": 1, "v": 0.2}]}
  }' | python3 -c 'import json,sys,base64; r=json.load(sys.stdin); a=r["audio"]; b=base64.b64decode(a["data"]); print("format", a["format"], "bytes", len(b), "phonemes", list(r.get("phonemes", {})))'
```

自测用例集（建议写成 pytest，对 /synthesize 直接发请求）：

| 用例 | 期望 |
| --- | --- |
| 正常单音 | 200，音频非空，phonemes 键 = note.key |
| 休止 rest | 200，音频静音但非空 |
| 延音合并后的长 note | 200，音频长度 ≈ duration |
| 未知 voice | 400，error 可读 |
| 错误 protocol/version | 400 |
| 缺字段/负 duration | 400 或 422（FastAPI 默认） |
| 无 Authorization | 401 |
| 错误 Authorization | 401 |
| body 超限 | 413 |
| 服务内部异常 | 500 + 不含堆栈 |
| 长度超限 | 400 + 可读信息 |

### 11.2 端到端验收（真实 TuneLab）

仓库里的集成脚本已经覆盖了 mock 服务路径；换成你的服务时，直接改插件设置指向它，然后：

```bash
# 1) 无头宿主加载插件并实际合成（读回音素）
TUNELAB_DATA_DIR=/tmp/tl \
  TuneLab/TuneLab.Cli/bin/Debug/net8.0/TuneLab.Cli --headless \
  sandbox run --code '
const v = sandbox.voices().find(x => x.type === "CloudVoice" && x.id === "你的声库id");
const t = tl.currentProject().addTrack();
const p = t.addPart({pos: 0, endOffset: 1920});
p.setSoundSource({kind: "voice", type: v.Type, id: v.Id});
const n = p.addNote({pos: 0, dur: 480, pitch: 60, lyric: "la"});
const n2 = p.addNote({pos: 480, dur: 480, pitch: 62, lyric: "-"});
print(JSON.stringify(sandbox.synthesize(p)));
print(JSON.stringify(sandbox.syllable(n)));
print(JSON.stringify(sandbox.syllable(n2)));
'

# 2) 走完整混音导出，确认音频真的进了成品
#    （先建工程 + project synthesize + project export，再 export-audio，见 scripts/integration.sh）
```

验收通过的标志：

- 状态带没有红块；project synthesize 能 settle；
- 导出 WAV 有声音、时长对、音高与音符一致；
- 编辑器里音素显示与音频听感一致；
- 错误场景（401/400/超时）在状态带能看到可读原因。

### 11.3 性能验收

- 单块 P50/P95 延迟、RTF（音频秒 / 推理秒）；
- 并发 2/4/8 请求下的延迟与显存；
- 长块（30s / 60s / 120s）的延迟与内存；
- 缓存命中率（如果做了 body 哈希缓存）。

## 12. 故障排查表

| 现象 | 最可能原因 | 处理 |
| --- | --- | --- |
| 状态带红块：HTTP 401 | API key 缺失或与服务端期望不一致 | 在插件设置里填 API Key（插件固定用 `Authorization: Bearer`）；服务端日志确认收到的头 |
| HTTP 404 | Service URL 不是完整 POST 路径；反代路径重写 | 插件不会拼接路径，直接 POST 你填的 URL |
| HTTP 413 / 请求被截断 | 反代 client_max_body_size 太小 | 调到 16m 或按最长块估算 |
| 超时（无响应） | 推理太慢 / 排队太久 / 反代 read timeout 小于插件 timeout | 降低 RTF、缩短内部等待、调大反代与插件 timeout，或改音频 URL 模式 |
| 504 | 反代到上游超时 | 同上；检查 proxy_read_timeout |
| 合成成功但没声音 | 音频很短/全零；audioStart 错；采样率声明错（PCM） | 用 §11.1 curl 自测解码；核对 sampleRate 与 audioStart |
| 声音错位/串音 | 返回音频长度大于块长；audioStart 不是 startTime | 严格返回 [startTime, endTime] 长度 |
| 音素显示乱/对不上 | 返回的 phonemes 与音频实际发音不一致 | 要么返回真实音素，要么干脆不返回 |
| 音素显示成整字/拼音拆分 | 没返回 phonemes，客户端用了内置简化 G2P | 回填音素 |
| 解析失败（非重试） | 响应 JSON 含 NaN/Infinity；HTML 错误页；空 body | 输出严格 JSON；错误也返回 JSON |
| 一直重试 | 返回 5xx/429/超时 | 修复服务；重试期间同一请求会重复到达，注意幂等/计费 |
| 音高轨不动 | 没返回 pitch，客户端回显请求曲线 | 想显示实际 F0 就回填 pitch |
| 合成时 GPU 被已取消请求占满 | 没处理客户端断连 | 协作式取消/检查 is_disconnected |
| 中文歌词发音不对 | G2P 把汉字当整字 | 接真正的拼音/汉字 G2P 词典 |
| TuneLab 里声库列表没有你的声库 | 设置里 voices JSON 无效或 endpoint 为空 | 重新保存设置；检查 JSON；确认 provider=http |

## 13. 协议版本与兼容

- 当前 `version = 1`。服务端应校验 version，不匹配返回 400。
- 协议会**加性演进**：客户端会忽略响应里不认识的字段，服务端也应忽略请求里不认识的字段
  （Pydantic 的 extra="ignore" 或等价配置）。
- 破坏性变更会升 version；服务端收到 version 不等于 1 的请求建议直接 400，并给出升级提示（插件当前不校验响应里的 version，但未来会按此协商）。
- 服务端可以在响应里带 protocol/version（建议总是带），客户端当前不校验，但便于排障与未来兼容。

## 14. 附录：一次真实请求/响应（精简）

请求（来自仓库集成测试的 mock 服务日志，去掉冗长曲线）：

```json
{
  "protocol": "tunelab-cloudvoice",
  "version": 1,
  "voice": "test-voice",
  "sampleRate": 48000,
  "startTime": 0,
  "endTime": 1,
  "notes": [
    {
      "key": "2",
      "start": 0,
      "duration": 1,
      "pitch": 60,
      "lyric": "la",
      "rest": false,
      "properties": { "dynamics": 0, "breathiness": 0.2 }
    }
  ],
  "pitch": [ { "t": 0, "v": 60 }, { "t": 0.5, "v": 60 }, { "t": 1, "v": 64 } ],
  "parameters": {
    "dynamics": [ { "t": 0, "v": 0 }, { "t": 1, "v": 0 } ],
    "breathiness": [ { "t": 0, "v": 0.2 }, { "t": 1, "v": 0.2 } ]
  }
}
```

注意：这个例子里用户输入了两个 note（la 和 `-`），请求里只剩一个 duration=1 的 note —— 延音被合并了。

响应（省略 base64 内容）：

```json
{
  "protocol": "tunelab-cloudvoice",
  "version": 1,
  "sampleRate": 48000,
  "audioStart": 0,
  "audio": { "format": "wav", "channels": 1, "data": "<base64-wav>" },
  "phonemes": {
    "2": {
      "leading": [ { "symbol": "l", "duration": 0.07, "stretchWeight": 0 } ],
      "body": [ { "symbol": "a", "duration": 0.28, "stretchWeight": 1 } ],
      "bodyOffset": 0.07
    }
  },
  "pitch": [ { "t": 0, "v": 60 }, { "t": 1, "v": 64 } ],
  "parameters": { "energy": [ { "t": 0, "v": 60 }, { "t": 1, "v": 60 } ] },
  "meta": { "server": "my-svs", "rtf": 0.3 }
}
```

## 15. 参考实现导读

[tools/server_example/app.py](../tools/server_example/app.py) 的结构（约 400 行，全部可读）：

| 区块 | 作用 |
| --- | --- |
| Protocol models | Pydantic 模型 + extra="ignore"，对应 §4 |
| predict_phonemes | 演示用 G2P；接真实模型时换成你的 G2P |
| synthesize_block | **唯一的模型接入点**：输入请求，输出响应 dict |
| render_demo_sine / encode_wav16 | 演示后端的加法合成与 WAV 编码 |
| create_app | 鉴权、body 限长、400/401/413/500/503、并发信号量、线程卸载 |
| validate | 边界校验（§4.2） |
| main | 命令行参数与 uvicorn 启动 |

已验证：本仓库在 macOS 上用该实现对接 TuneLab 2.1.1 完成过——

- 单音合成（200，150ms）并读回音素 l/a；
- 两音合成 + project export-audio 导出 3 秒、529KB 的非空 WAV；
- 无 token 401 / 带正确 token 200；
- 错误 protocol 返回 400 且错误可读。

运行方式与排障见 [tools/server_example/README.md](../tools/server_example/README.md)。

## 16. 上线前检查清单

- [ ] POST /synthesize 路径与插件 Service URL 完全一致（含反代前缀）
- [ ] 鉴权开启，密钥不入库不入日志；支持轮换
- [ ] 请求校验：protocol/version/sampleRate/duration/notes 上限
- [ ] 响应音频时长、audioStart、采样率声明正确；长块走 URL
- [ ] 音素回填与音频一致（或明确不回填）
- [ ] 错误码语义正确：4xx 不可重试、408/429/5xx 可重试、200+error 不可重试
- [ ] 重试幂等：同 body 重复请求不重复计费/不产生不一致
- [ ] 并发与队列上限；超限快速 429/503
- [ ] 客户端断连可取消推理
- [ ] 反代：body 限制、read/send timeout、buffering 配置
- [ ] 健康检查、日志脱敏、指标（延迟/RTF/错误率/队列长度）
- [ ] 端到端验收：无头 sandbox synth + export-audio + 错误场景
- [ ] 文档：接口地址、鉴权方式、模型/音色映射、限制与计费说明
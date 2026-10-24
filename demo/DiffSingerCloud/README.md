# DiffSinger Cloud —— Python 服务端（OpenUTAU 格式声库）

把 **DiffSinger 推理**做成一个**独立的 Python 服务器**：读取本地 **OpenUTAU 格式声库**，
用 ONNX Runtime 完成 DiffSinger 渲染（时长预测 → 方差预测 → 声学模型 → 声码器），
通过本项目的 `tunelab-cloudvoice/1` 协议对外提供合成。TuneLab 侧**不需要任何 DiffSinger 专用插件**，
直接用通用云插件 [TuneLab.CloudVoice](../../src/TuneLab.CloudVoice) 连上即可。

- 服务端本体：Python 3.10+ / FastAPI / ONNX Runtime，**不依赖 TuneLab、不依赖 OpenUTAU 程序**；
- 声库：OpenUTAU 的 `dsconfig.yaml` + `dsdict*.yaml` + `phonemes.*` + `*.onnx` + `*.emb` 原样可用；
- 后端：`--backend cpu|cuda|mps` 参数选择（mps = Apple Silicon GPU，见下文说明；`auto` = 有 CUDA 用 CUDA、否则 CPU）；
- 协议面：`/synthesize` + `/voices` + `/schema` + `/health`，音素/音高/energy 回显；
- 歌词 G2P：中文（汉字→拼音→声库字典）、日文假名、英文词典、直接写音素、钉死音素；
- 自动音高：声库带 `dspitch` 时用模型预测 F0（`pitch.auto=true`，TuneLab 里按 part 属性 `PitchMode` 选自动/跟随画线），没有 `dspitch` 的声库自动回退 TuneLab 曲线；
- 延迟：模型常驻内存（按声库懒加载 + ONNX 会话缓存），首块加载模型后后续块很快。

## 架构

    ┌──────────────────────────────┐        POST /synthesize  (tunelab-cloudvoice/1)
    │ TuneLab + TuneLab.CloudVoice │ ─────────────────────────────────────────────┐
    │（项目自带的通用云插件）        │ ◄────────────────────────────────────────────┘
    └──────────────────────────────┘        JSON: PCM audio + phonemes + pitch + energy
                                                            │
                            ┌────────────────────────────────▼────────────────────────────────┐
                            │ DiffSinger Cloud（本目录，Python 服务器进程）                     │
                            │  HTTP 层：协议校验/鉴权/限流/并发信号量（FastAPI + uvicorn）         │
                            │  声库层：扫描 OpenUTAU/、解析 dsconfig/dsdict、G2P、dsdur 时长预测   │
                            │  推理层：CPU/CUDA/Apple GPU 后端、variance + acoustic + vocoder ONNX │
                            │  按请求块裁剪音频、回填音素/音高/energy，模型与会话常驻缓存          │
                            └─────────────────────────────────────────────────────────────────┘

## 目录结构

    demo/DiffSingerCloud/
    ├── server/
    │   ├── app.py                # FastAPI：/health /voices /schema /synthesize
    │   ├── cli.py                # 命令行参数与配置装配（python -m server）
    │   ├── config.py             # ServerConfig：默认值 / YAML / 命令行覆盖
    │   ├── protocol.py           # tunelab-cloudvoice/1 请求模型与校验
    │   ├── schema.py             # 后端能力 schema（属性面板 / 画线轨 / 回显轨）
    │   ├── audio.py              # PCM/WAV 编码、曲线采样、块裁剪、RMS energy
    │   ├── voicebank/            # OpenUTAU 声库层
    │   │   ├── types.py          # 跨层数据契约
    │   │   ├── config.py         # dsconfig.yaml / vocoder.yaml / character.yaml
    │   │   ├── discovery.py      # 扫描声库目录、构建 VoiceCatalog
    │   │   ├── dictionary.py     # dsdict*.yaml / dsdict.txt、音素类型
    │   │   ├── g2p.py            # 歌词 → 音素（中/日/英/拼音/假名/直写）
    │   │   └── phonemizer.py     # 歌词 → 带绝对时间的 Phone（含 dsdur 时长预测）
    │   └── engine/               # 推理层
    │       ├── backends.py       # CPU / CUDA / MPS(Apple GPU) 后端与会话池
    │       ├── speaker.py        # .emb speaker embedding
    │       ├── pitch.py          # dspitch 自动音高预测（pitch.auto）
    │       └── inference.py      # DiffSingerRenderer：pitch/variance/acoustic/vocoder
    ├── OpenUTAU/                 # ★ 声库目录（把 OpenUTAU 格式声库放这里）
    │   ├── Vocoders/             # 可选：内建声码器（nsf_hifigan 等）
    │   └── README.md             # 声库目录放置说明
    ├── scripts/
    │   ├── setup.sh              # 创建 .venv 并安装依赖
    │   ├── run.sh                # 启动服务端
    │   ├── make_test_voicebank.py# 生成微型合成测试声库（无需真实模型）
    │   ├── verify.py             # 一键端到端 HTTP 验证（CPU/MPS/CUDA 矩阵）
    │   └── tunelab-e2e.sh        # TuneLab 无头宿主 + CloudVoice 插件端到端验证
    ├── tests/                    # pytest：声库/G2P/后端/渲染/协议/端到端
    ├── config.example.yaml       # 配置示例
    ├── requirements*.txt
    └── README.md

## 快速开始

前置：**Python 3.10+**（本机验证用 3.12）。不需要 OpenUTAU 程序、不需要 TuneLab 才能启动服务端。

    cd demo/DiffSingerCloud

    # 1) 创建虚拟环境并安装依赖（首次）
    ./scripts/setup.sh

    # 2) 把 OpenUTAU 格式声库放到 OpenUTAU/ 下（或建符号链接，见 OpenUTAU/README.md）
    ln -s "$HOME/Library/OpenUtau/Singers/JiHeHong-diffsinger-openutau-V2" OpenUTAU/

    # 3) 看服务端识别到了哪些声库 / 声码器
    .venv/bin/python -m server --list-voices

    # 4) 启动（默认 127.0.0.1:18791）
    .venv/bin/python -m server --backend auto

    # 5) 另开终端做端到端验证
    .venv/bin/python scripts/verify.py --url http://127.0.0.1:18791

在 TuneLab 里：**设置 → 扩展 → Cloud Voice**

| 字段 | 值 |
| --- | --- |
| Service URL | `http://127.0.0.1:18791/synthesize` |
| API key | 启动时传了 `--api-key` 就填同一个值；否则留空 |

保存后插件会自动 `GET /voices` 拉取声库目录；新建声源片段，在声源列表的「云端（服务商）」分组里
选声库（多说话人声库会按 speaker 展开成多个条目，如 `JiHeHong…::jihehong`），输入歌词即可合成。
属性面板里的 `Pitch mode 音高模式`（**Auto 自动音高 / TuneLab 跟随画线**）、`Language 语言`，
note 面板的 `Phonemes 音素覆盖`，以及底部 `DYN/BRE/TEN/VOI/VEL` 画线轨都由本服务端 `/schema` 声明。
声源列表里会标出 `auto-pitch`（带 dspitch 的声库）或 `manual-pitch`。

### 在 TuneLab 里一键端到端验证（无头宿主）

    # 前置：TuneLab/ 源码已编译出 TuneLab.Cli；插件 .tlx 已打包（scripts/pack.sh）
    demo/DiffSingerCloud/scripts/tunelab-e2e.sh

脚本会：启动本服务端 → 把 CloudVoice 插件装进沙箱 TuneLab → 设置 Service URL → 断言
`source list` 拉到服务端声库 → 无头宿主合成并读回音素 → 建两音符工程 synthesize + export-audio
并断言 WAV 非空。实测（TuneLab 2.1.x + JiHeHong + CPU）：插件 Loaded/Registered，自动拉到 8 个
声库条目，单音符合成 Done=true（首块约 2.6–3.8s），两音符工程导出 3.00s / 529KB 非空 WAV。

## 推理后端：CPU / CUDA / MPS

用 `--backend` 或配置文件里的 `backend` 选择。`auto` 只在有 CUDA EP 时选 cuda，否则选 cpu；
**Apple Silicon 上不会自动选 mps**（本机实测真实 DiffSinger 模型在 CoreML EP 上通常比 CPU 慢，见下表）。

| 值 | 实现 | 说明 |
| --- | --- | --- |
| `cpu` | ONNX Runtime CPUExecutionProvider | 任何平台可用；`--threads N` 控制线程数（0 = 默认） |
| `cuda` | ONNX Runtime CUDAExecutionProvider | Linux / Windows + NVIDIA GPU；安装 `requirements-cuda.txt`（`onnxruntime-gpu`），`--device-id` 选卡 |
| `mps` | ONNX Runtime CoreMLExecutionProvider（`MLComputeUnits=CPUAndGPU`） | macOS Apple Silicon 的 GPU 加速；**ONNX Runtime 没有原生 MPS ExecutionProvider**，Apple GPU 的官方路径就是 CoreML EP（OpenUTAU 的 OnnxRunner 在 macOS 上同样用 coreml），代码/配置里叫 `mps` 是为了对齐「Apple GPU 后端」的直觉 |
| `auto` | 自动 | 有 CUDA EP → cuda；否则 cpu（Apple Silicon 也不会自动选 mps） |

- 显式指定的后端不可用时，服务端**启动即报错**并给出可用方案；`auto` 会自动降级。
- 实际生效的后端与 providers 会出现在启动日志和 `GET /health` 的 `backend` 字段里。
- CoreML EP 对动态 shape 的模型可能把部分算子回退到 CPU（正常现象，日志里能看到）；若某个模型在
  CoreML 上**运行期失败**（例如动态 resize 不支持），OnnxRunner 会自动为该模型重建 CPU session、
  重试一次并记录 warning，服务不会整体失败（该请求会多一次失败重建的耗时）。
- 本机实测（Apple M1 / 16GB / onnxruntime 1.31，JiHeHong 2s 块）：CPU 热渲染约 2.8s、冷约 3.2s；
  mps(CoreML) 热约 3.4s、冷约 6.2s（还包含运行期 CPU 回退的首块开销）。想用 Apple GPU 请显式
  `--backend mps`；追求稳定低延迟用默认 `--backend cpu`。

CUDA 机器安装示例：

    python3 -m venv .venv
    .venv/bin/pip install -r requirements-cuda.txt

## 支持的 OpenUTAU 声库格式

服务端按 OpenUTAU 的规则解析声库目录（细节见 [OpenUTAU/README.md](OpenUTAU/README.md)）：

| 文件/目录 | 必需 | 用途 |
| --- | --- | --- |
| `character.txt` / `character.yaml` | 建议 | 声库名、作者、speaker/subbank（`subbanks` 的 suffix 对应 `.emb` 文件名） |
| `dsconfig.yaml` | 必需 | 声学模型路径、音素表、`use_*_embed`、扩散参数、采样率/hop/mel 规格、`speakers` |
| `dsdict.yaml`（或 `dsdict-zh.yaml` 等） | 必需 | G2P 字典：`symbols`（vowel/fricative/semivowel/liquid…）+ `entries`（歌词→音素） |
| `phonemes.txt` / `phonemes.json` | 必需 | 声学模型 token 表（每行/每 key 一个音素） |
| `*.onnx` | 必需 | 声学模型（输出 mel） |
| `dsdur/` | 可选 | 时长预测模型（`linguistic` + `dur`），有则音素时长按模型预测；没有则用启发式 |
| `dsvariance/` | 可选 | 方差模型（energy/breathiness/voicing/tension）；声学模型需要这些输入时必需 |
| `dsvocoder/` | 可选 | 声库自带声码器（`vocoder.yaml` + 模型） |
| `*.emb` | 可选 | 多 speaker / 音色的 embedding（float32 LE，`hidden_size` 个 float） |
| `dspitch/` | 可选 | 自动音高模型（`linguistic` + `pitch`）；有则 `pitch.auto=true` 可用，没有则回退 TuneLab 曲线 |

- 内建声码器名（如 `vocoder: nsf_hifigan`）会在 `OpenUTAU/Vocoders/`、OpenUTAU 的
  `Dependencies/`、TuneLab 的 ChoristaDiffsinger `Vocoders/` 等目录里查找，也可用 `--vocoder-dir` 追加；
- 多说话人声库在 `GET /voices` 里按 speaker 展开，id 形如 `<声库>::<speaker>`；
- 多语言声库（`use_lang_id: true` + `languages.json` + 带 `zh/` `ja/` `en/` 前缀的 `phonemes.json`）
  会按词典里的语言前缀和 `partProperties.language` 选择音素；
- 旧版 `dsdict.txt`（tab 分隔）也兼容。

## 歌词 → 音素（G2P）

优先级从高到低：

1. **钉死音素**：TuneLab 里钉死的音素（协议 `notes[].phonemes`），或 note 属性 `phonemes`（空格分隔，如 `l a`）；
2. **直接音素串**：歌词本身就是空格分隔的合法音素（如 `l a`），或声库里存在的单个音素；
3. **语言 G2P**：
   - 中文：汉字用 `pypinyin` 转无调拼音（也支持直接输入 `ni3` 这类带声调数字的拼音），再查声库字典（`ba → b a`）；
   - 日文：假名转 romaji 后查字典；
   - 英文：查声库自带 `dsdict-en.yaml` / 通用字典里的整词条目（多词按空格切）；
   - 多语言声库按语言选择对应 `dsdict-<lang>.yaml`，命中带语言前缀的音素；
4. 都不命中：拼音声母/韵母拆分再查，最后退化为 `SP` 并在服务端日志里 warning。

`partProperties.language`（auto/zh/ja/en）可以强制语言；`auto` 按歌词字符判断（汉字→zh、假名→ja、
纯拉丁→en）。当前英文没有内置 CMU 词典，**未收录词**会走直接音素/兜底路径，建议英文声库使用其自带词典。

## 参数与 schema 映射

服务端 `/schema` 声明了 TuneLab 里可见的控件，值随请求原样回到后端：

| TuneLab 控件 | 请求字段 | 映射到 DiffSinger |
| --- | --- | --- |
| part 属性 `Pitch mode` | `partProperties.PitchMode` | `predict`（默认）= 有 dspitch 时自动预测 F0；`tunelab` = 跟随画线曲线 |
| part 属性 `Language` | `partProperties.language` | 选择 G2P 语言 / 多语言音素前缀 |
| note 属性 `Phonemes` | `notes[].properties.phoneme` | 该 note 的音素覆盖（空格分隔） |
| 自动化轨 `DYN` (-1..1) | `parameters.DYN` | energy 输入调整：`min(pred + DYN*100*0.12, 0)` |
| 自动化轨 `BRE` (-1..1) | `parameters.BRE` | breathiness 输入调整：`min(pred + BRE*100*0.12, 0)` |
| 自动化轨 `TEN` (-1..1) | `parameters.TEN` | tension 输入调整：`pred + TEN*100*0.05` |
| 自动化轨 `VOI` (-1..1) | `parameters.VOI` | voicing 输入调整：`min(pred + VOI*100*0.12, 0)` |
| 自动化轨 `VEL` (0..200) | `parameters.VEL` | velocity 输入：`2^((VEL-100)/100)` |
| note 音高 | `notes[].pitch` + `pitch` 曲线 | 画线曲线提供目标音高/颤音；自动音高模式下作为相对 note 音高的偏差提示 |
| 只读回显 `energy` | 响应 `parameters.energy` | 方差模型预测的 energy（无声库方差模型时用输出音频 RMS） |
| 合成音高 | 响应 `pitch` | 实际使用的 f0（约 10ms 采样）；自动音高模式下即 dspitch 预测 F0 |

`pitch.auto = true`（具备自动音高能力）：具体行为由 part 属性 `PitchMode` 决定——

- **predict（默认，Auto 自动音高）**：声库有 `dspitch` 时，按 OpenUTAU `DsPitch` 的算法跑
  linguistic encoder + pitch predictor 预测 F0，再把用户画线相对 note 音高的偏差叠加到预测曲线上；
  响应的 `pitch` 回显实际渲染的 F0，TuneLab 显示为只读的「合成音高」。
  没有 `dspitch` 的声库（JiHeHong / ChiYu 等）会自动回退为按画线渲染，不会报错。
- **tunelab**：始终把 TuneLab 已解析的 pitch 曲线当作绝对 F0，编辑器画什么就唱什么（旧行为）。

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 状态 + 实际后端/providers + 声库数 |
| GET | `/voices` | 声库目录 + schema（插件加载时自动拉取） |
| GET | `/schema` | 属性面板 / 画线轨 / 回显轨声明 |
| POST | `/synthesize` | tunelab-cloudvoice/1 合成（见 [docs/cloud-protocol.md](../../docs/cloud-protocol.md)） |

最小请求：

    curl -s http://127.0.0.1:18791/synthesize -H 'Content-Type: application/json' -d '{
      "protocol": "tunelab-cloudvoice", "version": 1,
      "voice": "JiHeHong-diffsinger-openutau-V2",
      "sampleRate": 48000, "startTime": 0, "endTime": 2,
      "notes": [ { "key": "n1", "start": 0, "duration": 2, "pitch": 69, "lyric": "啊", "rest": false } ],
      "pitch": [ { "t": 0, "v": 69 }, { "t": 2, "v": 69 } ],
      "parameters": {}
    }' | head -c 200

响应：`audio.format = pcm_f32le`（单声道，采样率 = 声库原生 44100，宿主会重采样）、`phonemes`、`pitch`、
`parameters.energy`、`meta`（实际后端、声库、耗时等）。长块建议由客户端按小节切块（协议本身也是按块合成）。

**注意 audioStart**：DiffSinger 首音节的辅音（leading）在音符头之前，服务端默认把音频窗口向前扩到
渲染起点（`include_leading: true`，可用 `--no-audio-leading` 关闭），因此响应的 `audioStart` 可能比
`startTime` 早约 0.1–0.3s；音频长度对应 `endTime - audioStart`。客户端应按 `audioStart` 放置音频
（TuneLab 插件已正确处理）。音素回填遵循 TuneLab 的 junction-anchored 契约：`bodyOffset` 是主体起点
相对 note 头的有符号偏移（通常 0），leading 辅音可以落在负时间/上一音符尾段。

## 合成流水线（一个请求做了什么）

1. 校验协议/时长/音符数/声库，按 `voice` 找到声库条目（找不到会刷新一次目录再报错）；
2. 歌词 → 音素符号（G2P / 钉死音素），有 `dsdur` 时跑时长模型：linguistic encoder + duration predictor，
   按 OpenUTAU 算法把每个 note 的音素时长缩放到 note 起止（前导辅音落在 note 头之前，body 首音素从
   note 头起）；没有 `dsdur` 用启发式分配；
3. 计算声学帧：每个音素 `round(end/frame)-round(start/frame)`，句首/句尾各补 100ms SP；
4. f0：`PitchMode=predict` 且声库有 `dspitch` 时先跑自动音高预测（linguistic + pitch 模型），
   得到每帧预测半音并叠加画线偏差；否则直接用请求 pitch 曲线。最后换算成 Hz，头尾保持端点；
5. 有 `dsvariance` 时跑方差模型（linguistic + variance）预测 energy/breathiness/voicing/tension，
   叠加 DYN/BRE/TEN/VOI 用户曲线；
6. 声学模型：tokens/durations/f0 + 可选 languages/spk_embed/gender/velocity/方差输入 +
   depth/steps（continuous acceleration）或 depth/speedup（浅扩散），输出 mel；
7. 声码器：mel + f0 → waveform（44100Hz float）；
8. 用 `audio.block_slice` 把渲染结果裁到 `[audioStart, endTime)`（默认 `audioStart` 前移到首音素
   渲染起点以保留 leading 辅音；越界补零），回填音素结构、音高回显、energy 回显、meta，返回 base64。

## 验证

    # 端到端 HTTP 验证（自动启动服务端；没放真实声库时用内建合成测试声库）
    .venv/bin/python scripts/verify.py
    .venv/bin/python scripts/verify.py --backend mps
    .venv/bin/python scripts/verify.py --backend cuda
    .venv/bin/python scripts/verify.py --voice-source real --backend cpu

    # TuneLab 宿主端到端（需要 TuneLab.Cli + 插件 .tlx，见上文）
    ./scripts/tunelab-e2e.sh

    # 生成一个微型 OpenUTAU 格式测试声库（不依赖任何真实模型）
    .venv/bin/python scripts/make_test_voicebank.py --output /tmp/TestVoice --force
    .venv/bin/python -m server --voice-dir /tmp/TestVoice --backend cpu --port 18792 --list-voices

    # pytest（声库解析/G2P/后端/渲染管线/协议/合成声库端到端）
    .venv/bin/python -m pytest tests -q
    # 打开真实声库/大模型相关用例（较慢，会加载 300MB 级 ONNX）：
    TUNELAB_TEST_REAL_MODELS=1 .venv/bin/python -m pytest tests -q

## 实测记录

本机：**Apple M1 / 16GB / macOS 26.5**，Python 3.12，onnxruntime 1.31。

**声库扫描**：OpenUTAU/ 下 3 个真实声库（JiHeHong、ChiYu、yousa 的 6 个 speaker）→ 8 个 voice 条目，
0 warning；扫描约 0.05s（字典懒加载，25MB 的 dsdict 不在启动时解析）。

**scripts/verify.py 端到端矩阵**（10 步；真实声库时含自动音高步骤；全部 PASS）：

| 场景 | backend | 总耗时 | 自动音高步骤 | 音高结论 |
| --- | --- | --- | --- | --- |
| 合成测试声库 | cpu | 0.90s | SKIP（无 dspitch） | FFT 440.0Hz，RMS 0.212 |
| 真实声库（JiHeHong + yousa） | cpu | 14.22s | PASS（yousa 9.84s） | JiHeHong FFT 440.0Hz；predict echo median 69.02 / std 0.76，音频 68.74（差 0.28） |
| 合成测试声库 | mps | 1.03s | SKIP | resolved=mps（CoreML），FFT 440.0Hz |
| 真实声库（JiHeHong + yousa） | mps | 27.02s | PASS（yousa 18.66s） | JiHeHong 基频 440.0Hz；predict echo 68.98 / std 0.92，音频 68.91（差 0.07） |
| yousa::Bright 显式 | cpu | 13.65s | PASS | predict echo 69.03 / std 0.64，音频 68.72（差 0.31）；tunelab echo std 0.000 |

**渲染延迟（2s 音频块，热缓存）**：JiHeHong + CPU 约 2.8s（冷启动约 3.2s，含加载 5 个 ONNX）；
ChiYu + CPU 约 2.8s；合成测试声库约 0.06s；单独声码器 2s：CPU 1.04s vs CoreML 0.66s（CoreML 只在
小模型热路径更快，整条 DiffSinger 管线更慢）。长块 10s：JiHeHong + CPU 约 13s。

**自动音高（dspitch，yousa::Bright，A4 / 2s，CPU）**：
- `/schema` 声明 `pitch.auto=true` + part 属性 `PitchMode`（默认 `predict` = Auto 自动音高；
  `tunelab` = 跟随画线）；
- **predict**：dspitch 预测 F0（median≈69.01 半音、std≈0.65、min 66.5 / max 69.6），响应的 `pitch`
  回显与音频自相关基频 **441.0Hz（MIDI 69.04）一致**；预测 warm 约 0.59s，整条渲染比 tunelab
  每次多约 0.6s；
- **tunelab**：`pitch` 回显恒为 69.00（std=0），音频仍 441.0Hz；
- **无 dspitch** 的 JiHeHong / ChiYu / 合成声库：predict 自动回退画线（每个声库一条 warning），不报错；
- dspitch 是随机扩散类模型：同一请求两次预测曲线会略有波动（中位数稳定），TuneLab 重合成时
  合成音高线会轻微变化，属预期。

**TuneLab 2.1.x 无头宿主端到端**（scripts/tunelab-e2e.sh）：
- CloudVoice 插件 Loaded/Registered，自动 `GET /voices` 拉到 8 个声库条目（描述里标出
  `auto-pitch` / `manual-pitch`）；
- 单音符（la / A4 / 0.5s）合成 Done=true，读回音素 l + a；
- 两音符工程 synthesize + export-audio：3.00s / 529KB WAV，RMS≈0.041–0.053，非静音；服务端日志无
  dsdur 回退 warning（时长预测走真实模型）。

**单元/集成测试**：`pytest tests` 默认 **83 passed / 7 skipped**（真实大模型用例用
`TUNELAB_TEST_REAL_MODELS=1` 打开）；合成声库端到端 6 个用例全过。

## 限制与后续

- 音高：自动音高依赖声库自带的 `dspitch`；无 `dspitch` 的声库只能用 TuneLab 画线（自动回退）；
  自动音高下用户画线作为相对 note 音高的偏差叠加，不做逐音素的手动音高编辑；
- 英文 G2P 是词典查表（依赖声库自带 `dsdict-en.yaml`），没有内置 CMUdict/神经 G2P；
- 未实现 OpenUTAU 的 rap/musical_type、音色混合（voice color 曲线）、XSY 等高级参数；
- 请求中途取消不会打断正在进行的 ONNX 推理（与 tools/server_example 相同的限制）；
- 长块返回 base64 音频较大（4 分钟 44.1kHz float ≈ 42MB），建议客户端按块合成；
- 默认 `--max-concurrent 1`：DiffSinger 推理吃内存（16GB 机器同时加载两个大模型会紧张），
  多并发需求可调大，但注意显存/内存；
- 不同声库的推理结果与 OpenUTAU 不会逐样本一致（实现对齐了算法与帧对齐，但采样/线程/后端差异会带来
  微小数值差异），听感上应接近；
- 声库与声码器模型各有自己的许可证，用生成的音频分发/商用前请自行确认授权。

## 排错

| 现象 | 处理 |
| --- | --- |
| `GET /voices` 为空 | 声库目录不对：`--voice-dir` 指向含 `dsconfig.yaml` 的目录；用 `python -m server --list-voices` 看 warnings |
| 启动报 `backend 'mps' is not available` | 非 Apple Silicon 或 onnxruntime 不带 CoreML EP；改用 `--backend cpu`/`auto` |
| 启动报 CUDA 不可用 | 安装 `requirements-cuda.txt`（onnxruntime-gpu），确认 NVIDIA 驱动；或 `--backend auto` |
| 合成报缺 `vocoder` | 声库写的是内建名但本机没装对应声码器：把 nsf_hifigan 目录放进 `OpenUTAU/Vocoders/` 或用 `--vocoder-dir` |
| 合成报 `Phoneme "xxx" isn't supported` | 歌词 G2P 出了声学模型不支持的音素：检查声库 `phonemes.txt`/`dsdict.yaml`，或改用钉死音素 |
| 合成报缺 `energy`/`breathiness`/... | 声库需要方差模型但 `dsvariance/` 缺失/损坏，检查声库完整性 |
| 首块很慢 | 第一次请求加载 ONNX 模型（大模型 1-3s），之后走会话缓存；可先合一小段预热 |
| 中文歌词全变 SP | 声库词典缺拼音条目或 `partProperties.language` 设错；看服务端日志 warning |
| 声音断续/块边界爆音 | 检查块时长是否过短（<0.1s）；本服务端按请求块渲染，块边界没有额外交叉淡化 |
| TuneLab 里没有「Pitch mode / 自动音高」选项 | 插件只在加载时拉一次 `/schema`：**先启动新版服务端，再重启 TuneLab**；确认 `GET /schema` 的 `pitch.auto` 为 `true` |
| 选了 Auto 但音高还是画线 | 该声库没有 `dspitch`（声源描述显示 `manual-pitch`）会回退画线；日志会 warning。换带 dspitch 的声库（如 yousa）或看声库目录是否完整 |
| 自动音高后音高线有细微抖动 | 正常：dspitch 预测的 F0 与画线不完全一致；响应的 `pitch` 是实际渲染 F0。想完全跟画线就选 `TuneLab 跟随画线` |

## 相关文档

- 云合成协议：[docs/cloud-protocol.md](../../docs/cloud-protocol.md)
- 从零搭服务端指南：[docs/server-guide.md](../../docs/server-guide.md)
- 通用云插件：[src/TuneLab.CloudVoice](../../src/TuneLab.CloudVoice)、[README.md](../../README.md)
- 声库放置说明：[OpenUTAU/README.md](OpenUTAU/README.md)

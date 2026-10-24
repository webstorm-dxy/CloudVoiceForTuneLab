# TuneLab Cloud Voice（云端歌声合成插件）

TuneLab 的 **V1 voice 插件**：把 TuneLab 工程里的音符、歌词、音高与参数曲线发送到**云端 HTTP(S)
歌声合成服务**，取回音频与合成产物显示到编辑器里。未配置云端服务时，内置的 **mock 声库** 可以离线
跑通整条链路（分块调度、状态带、音素回填、音高/能量回显、导出音频）。配置云端时只需要填**服务地址**与
**API Key**，声库目录会在插件加载时自动从服务商拉取。

- 宿主：TuneLab 2.1.x（`sdk-version` 1.1，目标框架 `net8.0`）
- 插件引擎 id：`CloudVoice`（写进工程文件，不可变）
- 扩展包 id：`io.github.daxingyi.tunelab.cloudvoice`
- 上游 TuneLab 源码：<https://github.com/LiuYunPlayer/TuneLab>

## 目录结构

```
.
├── TuneLab/                         # 上游 TuneLab 源码（本仓库不含：scripts/update-tunelab.sh 克隆/更新）
├── src/TuneLab.CloudVoice/          # 插件本体（net8.0 类库 + manifest.json + 介绍/图标）
│   ├── CloudVoiceEngine.cs          # 引擎：生命周期、声库目录、声明面、扩展设置
│   ├── CloudVoiceSession.cs         # 会话：分块、调度、失效、云端调用、产物发布
│   ├── Voice/                       # 设置 / 声库目录 / 轻量 G2P
│   ├── Providers/                   # ICloudVoiceProvider：mock（离线）+ http（云端）
│   ├── Protocol/                    # tunelab-cloudvoice/1 线上 DTO
│   └── Audio/                       # PCM/WAV 解码、RMS 包络
├── tests/TuneLab.CloudVoice.Tests/  # xUnit 单元测试（49 个）
├── tools/mock_cloud_server.py       # 本地 mock 云服务（零依赖，验证 HTTP 路径用）
├── tools/server_example/            # FastAPI 参考服务端（生产骨架，接真实模型从这里改）
├── demo/OpenCyanCloud/              # 演示：独立 OpenCyan 合成服务器 + 用通用 CloudVoice 插件连接
├── demo/DiffSingerCloud/            # 演示：Python DiffSinger 推理服务器（OpenUTAU 格式声库，CPU/CUDA/MPS）
├── scripts/                         # update-tunelab / build / pack / install / test / integration
├── docs/cloud-protocol.md           # 云端协议规范（字段字典）
├── docs/server-guide.md             # 服务端搭建指南（接模型 / 部署 / 验收 / 排障）
└── docs/architecture.md             # 插件内部设计（改插件看这个）
```

## 快速开始

前置：**.NET 8 SDK**、`git`、`python3`（仅集成测试的 mock 云服务需要）。

```bash
# 1. 拉取 / 更新 TuneLab 最新源码到 TuneLab/
./scripts/update-tunelab.sh

# 2. 编译 + 打包成 .tlx
./scripts/pack.sh
#    -> artifacts/tlx/TuneLab.CloudVoice-1.0.0.tlx

# 3. 安装到本机 TuneLab 扩展目录（默认按系统用户数据目录；也可在 TuneLab 里拖入 .tlx）
./scripts/install.sh

# 4. 单元测试
./scripts/test.sh

# 5. 端到端验证：真实 TuneLab 无头宿主加载插件 → 合成 → 导出音频（含本地 mock 云服务）
#    先编译一次 CLI：
dotnet build TuneLab/TuneLab.Cli/TuneLab.Cli.csproj -c Debug
./scripts/integration.sh

# 6. 宿主 SDK 兼容性校验（用本机已发布 TuneLab 的 SDK 重新编译插件，防止用到宿主没有的新 API）
./scripts/check-host-compat.sh
```

> 本仓库的 TuneLab/ 是上游最新源码，但用户机器上已发布的 TuneLab 往往更旧。插件必须只使用
> **宿主 SDK 里已有的 API**，否则扩展会加载失败（例如 `built against an incompatible SDK`）。
> 改完插件后请跑一次 `scripts/check-host-compat.sh`（默认自动找 TuneLab.app，也可传 Contents/MacOS 目录）。

> 本仓库的 `TuneLab/` 是独立的上游仓库。插件通过 `ProjectReference` 引用其中的
> `TuneLab.Foundation` / `TuneLab.SDK`（`Private=false`，不随包分发）。把本仓库挪到别处时，
> `TuneLab/` 保持同级即可。

Windows 上没有 bash 时，等价命令：

```powershell
dotnet build src/TuneLab.CloudVoice/TuneLab.CloudVoice.csproj -c Release
Compress-Archive -Path artifacts/package/* -DestinationPath artifacts/tlx/TuneLab.CloudVoice-1.0.0.zip -Force
Rename-Item artifacts/tlx/TuneLab.CloudVoice-1.0.0.zip TuneLab.CloudVoice-1.0.0.tlx
```

## 在 TuneLab 中使用

1. 安装 `.tlx`（拖进编辑器的扩展侧边栏，或 `scripts/install.sh`）。
2. **设置 → 扩展 → Cloud Voice**：只填 **Service URL**（服务商的 `.../synthesize` 端点）与
   **API key**（可选；宿主会掩码显示并存进系统钥匙串，不会进工程文件）。
   插件加载时会**自动从服务商 `GET <同级 /voices>` 拉取声库目录**，不需要手工粘贴声库列表。
   服务地址留空时显示内置的 3 个离线 mock 声库，可直接离线试用。
3. 新建声源片段，在声源选择器里挑声库：配好服务后是服务商返回的声库（`云端（服务商）` 组）；
   留空服务地址时是 `内置演示（离线）` 组。
4. 输入歌词（每个音符一个音节）后播放/编辑，状态带会逐块显示 待合成 / 合成中 / 已合成 / 失败。

> 服务商需要提供 `GET /voices` 与 `GET /schema`（插件自动拉取用）；不可达时插件退回内置演示声库/默认声明并在日志说明原因。
> 如果启动 TuneLab 时服务端还没提供 schema：**先启动服务器，再重启 TuneLab**（插件在加载时拉取 schema）。

属性面板与画线轨由服务端 `/schema` 决定（例如 OpenCyan 服务端的 part 参数 `PitchMode` / `Language`、
note 参数 `language` / `musical_type` / `phoneme_str`，以及 12 条可画线轨）。老服务端未提供 schema 时
回退到下面这组固定声明。

节目参数（无 schema 时的默认声明）：

- 可编辑自动化轨：`dynamics`（-1..1）、`breathiness`（0..1）；
- 只读回显轨：`energy`（0..100，来自云端响应，云端没给时用返回音频的 RMS 包络兜底）；
- note 面板属性：`dynamics`、`breathiness`（单音符覆盖，随请求发送）；
- 歌词 `-` = **延音**：延长前一个音符的元音（melisma），不另起音节。

要自己搭云端服务端：从 [docs/server-guide.md](docs/server-guide.md) 开始（协议实现、接真实模型、部署、
验收、排障）；协议字段字典见 [docs/cloud-protocol.md](docs/cloud-protocol.md)；插件内部设计见
[docs/architecture.md](docs/architecture.md)。

## 内置 mock 引擎

mock 引擎不是"静音占位"：它按请求里的音高曲线与歌词做加法合成（带颤音、包络、按声库不同的音色），
并回填音素、音高与 energy 回显。用途是**在没有任何云端服务时完整验证插件行为**：

- 分块与调度：按音符间隙分块，编辑后只重合成受影响的块；
- 产物链路：音频段 / 音素 / 音高 / 参数回显都能在编辑器里看到；
- 导出：`project export-audio` 能渲染出真实波形（集成测试里会断言 WAV 非空）。

## 后端能力 schema：面板、画线轨与自动音高都由服务端决定

插件加载时除了 `GET /voices` 拉声库，还会取 `schema`（`/voices` 响应的 `schema` 字段，或单独的
`GET /schema`）。schema 声明：

- `partProperties` / `noteProperties`：**属性面板里能调的参数**（slider / integer / number / combo /
  checkbox / text），part / note 两级，值随请求回到后端；
- `automations`：**用户能画线的自动化轨**（id、量程、默认值、颜色、单位都由后端定）；
- `synthesizedParameters`：只读回显轨（如 energy）；
- `pitch.auto`：后端是否有**自动音高**能力。为 true 时请求带 `pitchMode: "auto"`，后端返回的
  `pitch` 回显（实际渲染 F0）会作为编辑器的合成音高显示。

没有 schema 的旧服务端会自动回退到固定声明（dynamics / breathiness + energy），不影响兼容性。

## 演示：把 OpenCyan 变成独立云端服务器（用 CloudVoice 连接）

[demo/OpenCyanCloud](demo/OpenCyanCloud/README.md) 是一个完整的"本地引擎 → 独立云端服务器"改造示例：
服务端是一个**独立的 .NET 服务器程序**，进程内直接加载 OpenCyan 的引擎库（OpenCyan.Net + 原生
libopencyan）做真实渲染，通过本项目的 tunelab-cloudvoice/1 协议对外提供合成——**不需要 TuneLab**；
客户端**直接用本项目自带的通用云插件 [src/TuneLab.CloudVoice](src/TuneLab.CloudVoice)** 连接服务器
（不再额外打包 OpenCyan 专用插件）：插件加载时自动从服务器 `GET /voices` 拉取声库，用户只需要
填服务地址 + API Key。首块渲染约 1.1s，缓存命中约 13ms。

    demo/OpenCyanCloud/scripts/verify.sh   # 一键端到端验证

## 演示：DiffSinger 推理服务器（OpenUTAU 格式声库，Python）

[demo/DiffSingerCloud](demo/DiffSingerCloud/README.md) 是一个 **Python 服务端**，用 ONNX Runtime 做真实的
**DiffSinger 推理**，直接加载 **OpenUTAU 格式声库**（acoustic / dsdur / dsvariance / dsvocoder / .emb /
dsdict 全部按 OpenUTAU 的规则解析），通过 tunelab-cloudvoice/1 协议对通用 CloudVoice 插件提供合成。
推理后端用参数选择：`--backend cpu|cuda|mps`（mps = Apple Silicon GPU，走 ONNX Runtime CoreML EP；
`auto` 有 CUDA 用 CUDA、否则 CPU——Apple Silicon 上实测真实 DiffSinger 模型 CoreML 比 CPU 慢，
想试 Apple GPU 显式指定 `--backend mps`，CoreML 执行失败会自动回退 CPU）。音高支持自动预测：
声库带 `dspitch` 时在 TuneLab 的 part 属性 `Pitch mode` 里选「Auto 自动音高」即用模型预测 F0，
选「TuneLab 跟随画线」则完全跟随编辑器曲线；没有 dspitch 的声库自动回退画线。

    cd demo/DiffSingerCloud
    ./scripts/setup.sh                                         # 首次：创建 venv + 安装依赖
    ln -s "$HOME/Library/OpenUtau/Singers/<声库>" OpenUTAU/    # 声库放 OpenUTAU/（或直接拷贝）
    .venv/bin/python -m server --backend cpu --port 18791     # 启动服务端

声库放到 `demo/DiffSingerCloud/OpenUTAU/`；服务端自动扫描并在 `GET /voices` 里列出（多说话人声库
按 speaker 展开）。一键验证：

    demo/DiffSingerCloud/scripts/verify.py                    # HTTP 端到端（CPU/MPS 矩阵、真实/合成声库）
    demo/DiffSingerCloud/scripts/tunelab-e2e.sh               # TuneLab 无头宿主 + CloudVoice 插件端到端

实测（macOS Apple M1 / 16GB，CPU 后端，JiHeHong 真实声库）：2s 块热渲染约 2.8s（冷启动约 3.2s，
含加载 5 个 ONNX）；verify 矩阵全 PASS（CPU/MPS × 真实/合成，含 yousa 自动音高 predict/tunelab）；
TuneLab 2.1.x 无头宿主自动拉到 8 个声库条目、合成音素 l+a、导出 3.00s 非空 WAV；
pytest 83 passed / 7 skipped。

## 自己搭云端服务端

从零实现一个适配的服务端，推荐路径：

1. 先读 [docs/server-guide.md](docs/server-guide.md)：同步块合成语义、时间轴与音素对齐、接模型、
   部署、验收用例与故障排查。
2. 用 [tools/server_example/app.py](tools/server_example/app.py)（FastAPI）跑通骨架——它已实现协议校验、
   鉴权、请求体限制、并发信号量、错误语义与 WAV/音素回填，并已与 TuneLab 2.1.1 实际对接验证。
3. 只替换其中 `synthesize_block()` 的占位后端为你的模型 / 第三方 API。
4. 按指南 §11 的用例集在 TuneLab 无头宿主里做端到端验收。

只想先做最小验证时，用零依赖的 [tools/mock_cloud_server.py](tools/mock_cloud_server.py) 即可，
字段规范见 [docs/cloud-protocol.md](docs/cloud-protocol.md)。

## 已实现 / 未实现

已实现：

- `IVoiceSynthesisEngine` / `IVoiceSynthesisSession` 全部成员（含 `IsContinuation` 与音素属性声明）；
- `IExtensionSettings`：只有服务地址 + API Key 两个字段、API Key 掩码；加载时自动拉取服务商声库目录（失败退回离线演示）；
- 后端能力 schema：属性面板参数 / 可画线自动化轨 / 只读回显轨 / 自动音高全部按服务端 `/schema` 动态声明；
  会话按 schema 发送 `partProperties`、`notes[].properties`、`parameters` 与 `pitchMode`；
- 自动音高：schema `pitch.auto` + 后端返回的实际 F0 → `SynthesizedPitch`，支持 predict 与 tunelab 两种渲染模式；
- 云端 HTTP provider：协议请求、`Authorization: Bearer` 鉴权、重试（5xx/429/超时/网络）、超时、取消；
- 音频解码：base64 PCM f32le / s16le / WAV（PCM 8/16/24/32、float32/64），多声道下混；采样率交给宿主重采样；
- 分块合成 + 分级失效（歌词/时长 → 音素；音高/参数 → 音频与回显）；
- 延音 `-`（melisma）：快照域自判与合成行为一致，延音 note 不产音素、时长并入链头；
- 错误呈现：失败块进入 Failed 状态并带服务端错误文案，编辑后可重试；
- 单元测试 49 个 + 真实宿主端到端集成脚本（含 schema 驱动的 part/note 属性断言）；
- 宿主 SDK 兼容性脚本（`scripts/check-host-compat.sh`）：用本机 TuneLab 自带的旧 SDK 编译校验，避免扩展加载失败；
- 附带 FastAPI 参考服务端与 [服务端搭建指南](docs/server-guide.md)（已在 TuneLab 2.1.1 上实测对接）。

未实现 / 有意留白：

- 不做本地音素编辑的 per-phoneme 自定义属性（`GetPhonemePropertyConfigs` 返回空 map）；
- 不实现 SVC/换声类 effect（本插件只做 voice）；
- 失败块不会自动重试（避免在用户不知情时反复请求；编辑该块或改设置后重试）。

## 验证记录

在 macOS / .NET SDK 9.0.202（编译 net8.0 目标）上：

- `dotnet test`：49/49 通过；
- `scripts/integration.sh`：
  - TuneLab 2.1.1（上游 master `31740d7`，v2.1.1）无头宿主识别扩展为 `Loaded` / `Registered`；
  - mock 声库合成 `la`，合成音素 `l` + `a`；
  - `project export-audio` 渲染出 3 秒、529 KB 的非空 WAV；
  - 只配置 Service URL（声库目录由插件自动 `GET /voices` 拉取），本地 mock 云服务收到请求；
  mock `/schema` 声明的 part 属性 `style`、note 属性 `tone` 与 `pitchMode: "auto"` 都出现在请求里，
  返回的 `srv-la` 音素被回填到编辑器；
- FastAPI 参考服务端（`tools/server_example/app.py`）：
  - 单音合成 200（约 150ms），编辑器读回音素 `l` + `a`；
  - 两音合成 + `project export-audio` 导出 3 秒、529 KB 非空 WAV；
  - 无 token → 401，带正确 token → 200；错误 protocol → 400 且错误可读。
- DiffSinger 推理服务器（`demo/DiffSingerCloud`，Python + ONNX Runtime，OpenUTAU 格式声库）：
  - 扫描到 3 个真实声库 / 8 个 voice（JiHeHong、ChiYu、yousa ×6 speaker），0 warning；
  - `scripts/verify.py` 9/9 PASS：合成测试声库 CPU/MPS、JiHeHong 真实声库 CPU/MPS，FFT 主频 440Hz；
  - 真实渲染 2s 块：JiHeHong CPU 热约 2.8s；ChiYu 的损坏 dsvariance 自动回退 dsdur 模型；
  - 自动音高（dspitch）：yousa 声库 predict 模式预测 F0（median 69.01 / std 0.65），响应 pitch 回显
    与音频自相关基频 441.0Hz 一致；tunelab 模式回显恒等于画线；无 dspitch 的声库自动回退画线；
  - TuneLab 2.1.x 无头宿主 + CloudVoice 插件（`scripts/tunelab-e2e.sh`）：插件 Loaded、
    自动拉取 8 个声库、合成单音符读回 l+a、两音符工程导出 3.00s / 529KB 非空 WAV；
  - pytest 83 passed / 7 skipped（真实大模型用例用 `TUNELAB_TEST_REAL_MODELS=1` 打开）。
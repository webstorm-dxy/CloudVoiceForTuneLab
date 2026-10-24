# 插件内部设计

写给要改插件的人。宿主契约以 TuneLab 仓库的 `docs/plugin-development.zh-CN.md` 为准，本文只讲本插件的取舍。

## 分层

```
CloudVoiceEngine                 IVoiceSynthesisEngine + IExtensionSettings
├── 声库目录（Init 期构建缓存，get 立即返回）
├── 声明面：自动化轨 / 回显轨 / part / note / 音素属性（纯函数）
└── CreateSession(context)
    └── CloudVoiceSession        IVoiceSynthesisSession（每 MidiPart 一个）
        ├── 分块（按 note 间隙）与脏判定（事件 → 标脏；Committed 收口重分块）
        ├── RenderBlockAsync（worker：快照 → 协议请求 → provider → 产物值）
        └── Publish（数据线程：音频段 + 音素 / 音高 / 参数回显）

Provider（每个会话一个，随会话 Dispose）
├── MockCloudVoiceProvider       离线加法合成，无网络
└── HttpCloudVoiceProvider       tunelab-cloudvoice/1 HTTP 客户端
```

## 关键纪律

1. **活视图只在数据线程**：`IVoiceSynthesisContext` / `IVoiceSynthesisNote` 只在嵌套与 `SynthesizeNext`
   同步前缀读；`IsContinuation` 也在这里对 live note 求值并冻结成 `bool[]`。
2. **worker 只读快照**：`VoiceSynthesisSnapshot` 不可变、可跨线程；协议请求、音高/自动化采样、音频
   解码全在 worker 里做，产物是纯值 `BlockRender`。
3. **产物在数据线程发布**：`await Task.Run(...)` 回来后再 `CreateAudioSegment` / 写 / Commit，
   并替换产物引用（不可变发布）。
4. **取消是正常结局**：provider 取消时返回 null，会话把块重新标脏并正常返回；不抛
   `OperationCanceledException`。真错误才进 Failed。

## 分块与失效

- 分块：按 note 间隙切块；重叠 note（和弦）以「组内最大结束」判间隙。块集不变时复用块缓存。
- 失效分级：
  - note 时长 / 歌词 / 增删 → 音素与参数回显都失效（重做 G2P）；
  - note 音高 / 钉死音素 → 只清参数与音高回显（音素几何仍有效）；
  - note 属性 / automation / Pitch 区间变更 → 只重渲音频（参数回显按语义决定是否清）；
  - `Committed` 之前只标脏，之后统一 `Resegment`，批量编辑只重分块一次。
- 状态带：每块一个 `SynthesisStatusSegment`，Synthesizing 时带阶段文案与进度。

## 延音（melisma）

`IsContinuation` 用编辑器 `-` 约定：歌词 `-` ∧ 沿不断裂相接链能回溯到内容 note ∧ 本 note 无钉死音素。
与合成行为成对：

- 请求里延音 note **不单独发送**，其结束时间并入链头 note 的 `duration`（服务端看到的是一个长音）；
- 产物里延音 note 回报**空音节**（键在、零音素）= "本 note 不发音"，宿主据此留白但邻居照常显示。

## 设置与声库目录

- `IExtensionSettings` 的设置面板只有 **Service URL + API Key** 两个字段；API Key 用
  `TextBoxConfig.WithPassword()`（宿主存系统钥匙串）。
- 声库目录：没填服务地址时 = 3 个内置 mock 声库；填了服务地址 = Init / ApplySettings 时自动
  `GET <endpoint 同级 /voices>` 拉取服务商目录（解析失败或不可达则退回 mock 并写日志）。每项映射到
  `CloudVoiceRoute(ProviderKind, ProviderVoiceId)`；目录构建后缓存，`VoiceSourceInfos` 只读缓存、不阻塞。
- HTTP provider 固定用 `Authorization: Bearer <API key>`，不再暴露自定义头 / model / 采样率 / 超时等设置。

## 后端能力 schema

- 声库目录与 schema 一起拉：先 `GET /voices`（可能带 `schema` 字段），没有再
  `GET <endpoint 同级 /schema>`；解析成 `CloudVoiceSchema`。没有 schema / 解析失败回退
  `CloudVoiceSchema.Defaults`（= 改版前固定声明，保证旧服务端兼容）。
- 声明面全部由 schema 构建（缓存到 ApplyCatalog）：`GetPartPropertyConfig` / `GetNotePropertyConfig` /
  `GetAutomationConfigs` / `GetSynthesizedParameterConfigs`；支持 slider / integer / number / combo /
  checkbox / text 与连续/分段自动化轨。
- 会话按 schema 采请求：`partProperties`、`notes[].properties`（number / bool / string）、
  `parameters`（schema.automations 的 id）、`pitchMode`（schema.pitch.auto ? auto : manual）；
  整块为默认值的自动化轨不发送。
- 回显：schema.synthesizedParameters 的 id 从响应 `parameters` 聚合为 `SynthesizedParameters`；
  响应 `pitch` 直接作为 `SynthesizedPitch`（自动音高时就是引擎实际 F0）。

## 测试策略

- 纯逻辑 / provider / schema 用 xUnit（`tests/TuneLab.CloudVoice.Tests`）：协议序列化、WAV/PCM 解码、
- schema 解析与声明配置构建（`SchemaTests`）、旧服务端默认声明回退；
  G2P、设置 clamp 与条件 schema、声库目录、HTTP 重试/超时/取消/错误、mock 合成。
- session + 真实宿主用 `scripts/integration.sh`：无头 TuneLab 加载扩展 → `sandbox run` 合成并读
  音素 → `project export-audio` 验证音频真的进了混音 → 本地 mock 云服务验证 HTTP 路径。
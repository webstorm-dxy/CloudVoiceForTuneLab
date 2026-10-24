using TuneLab.Foundation;
using TuneLab.SDK;
using TuneLab.CloudVoice.Audio;
using TuneLab.CloudVoice.Protocol;
using TuneLab.CloudVoice.Providers;
using TuneLab.CloudVoice.Voice;

namespace TuneLab.CloudVoice;

// 一次 part 合成的有状态会话（厚插件模型）：分块、调度状态、脏判定、产物全在这里。
// 线程纪律：
//   · context / IVoiceSynthesisNote 活视图只在数据线程读写（构造函数、事件 handler、SynthesizeNext 同步前缀）；
//   · offload 后 worker 只读 VoiceSynthesisSnapshot（不可变、可跨线程），产物在 await 回来的数据线程续延里发布。
public sealed class CloudVoiceSession : IVoiceSynthesisSession
{
    public CloudVoiceSession(IVoiceSynthesisContext context, CloudVoiceSettings settings, CloudVoiceRoute route, ICloudVoiceProvider provider, CloudVoiceSchema schema, CloudVoiceEngine? engine = null)
    {
        mContext = context;
        mSettings = settings;
        mRoute = route;
        mProvider = provider;
        mSchema = schema;
        mEngine = engine;
        if (engine != null)
            engine.SchemaChanged += OnSchemaChanged;

        // 变更接线：handler 只做廉价标脏；重活（重分块）延迟到 Committed 收口。
        // 分级失效链：时长/歌词（音素几何）→ 音素；音高/钉死音素 → 参数回显；note 属性 → 仅音频。
        context.Notes.WhenAnyItem(n => n.StartTime.Modified, n => n.EndTime.Modified, n => n.Lyric.Modified)
            .Subscribe(note => MarkNoteDirty(note, phonemesStale: true, parametersStale: true), mSubscriptions);
        context.Notes.WhenAnyItem(n => n.Pitch.Modified, n => n.LeadingPhonemes.Modified, n => n.BodyPhonemes.Modified, n => n.BodyOffset.Modified)
            .Subscribe(note => MarkNoteDirty(note, phonemesStale: false, parametersStale: true), mSubscriptions);
        context.Notes.WhenAnyItem(n => n.Properties.Modified)
            .Subscribe(note => MarkNoteDirty(note, phonemesStale: false, parametersStale: false), mSubscriptions);
        context.Notes.ItemAdded.Subscribe(OnNotesStructureChanged, mSubscriptions);
        context.Notes.ItemRemoved.Subscribe(OnNotesStructureChanged, mSubscriptions);
        context.PartProperties.Modified.Subscribe(MarkAllDirtyAndResegment, mSubscriptions);
        context.Committed.Subscribe(OnCommitted, mSubscriptions);

        // 音高（绝对约束 + 加性偏差）是音高级输入 → 清下游回显。
        context.Pitch.RangeModified.Subscribe(OnPitchRangeModified, mSubscriptions);
        context.PitchDeviation.RangeModified.Subscribe(OnPitchRangeModified, mSubscriptions);

        // 声明在引擎上、故构造期 context.Automations 已含自己声明的轨（schema 决定有哪些）：
        // 订阅区间失效。
        foreach (var spec in schema.Automations)
        {
            if (mSubscribedAutomations.Add(spec.Id) && context.Automations.TryGetValue(spec.Id, out var automation))
                automation.RangeModified.Subscribe(OnAutomationRangeModified, mSubscriptions);
        }

        mNeedResegment = true;
    }

    public string DefaultLyric => "la";

    // 延音判定（编辑器 "-" 约定）：歌词 "-" ∧ 沿不断裂的相接链回溯到内容 note ∧ 本 note 无钉死音素。
    // 判定与合成行为成对：判定为延续的 note 不产音素、其时段由链头元音铺过（melisma）。
    public bool IsContinuation(IVoiceSynthesisNote note)
    {
        if (note.Lyric.Value != "-")
            return false;
        if (note.LeadingPhonemes.Value.Count > 0 || note.BodyPhonemes.Value.Count > 0)
            return false;

        var current = note;
        while (true)
        {
            var previous = current.Previous;
            if (previous == null)
                return false; // 链跑出开头，无内容 note → 孤儿
            if (previous.EndTime.Value < current.StartTime.Value)
                return false; // 空隙断链 → 孤儿
            if (previous.Lyric.Value != "-" || previous.LeadingPhonemes.Value.Count > 0 || previous.BodyPhonemes.Value.Count > 0)
                return true; // 回溯到链头
            current = previous;
        }
    }

    // —— 调度 ——

    public SynthesisRange? GetNextPendingSynthesisRange(double startTime, double endTime)
    {
        return FindNextDirtyPiece(startTime, endTime) is { } piece
            ? new SynthesisRange(piece.StartTime, piece.EndTime)
            : null;
    }

    // peek 与 commit 共用同一查找（确定性：同一数据 + 同一窗口 ⇒ commit 重算出 peek 报出的同一块）。
    Piece? FindNextDirtyPiece(double startTime, double endTime)
    {
        if (mNeedResegment)
            Resegment();

        foreach (var piece in mPieces)
        {
            if (!piece.Dirty || piece.Failed || piece.Synthesizing)
                continue;
            if (piece.EndTime < startTime || piece.StartTime > endTime)
                continue;
            return piece;
        }
        return null;
    }

    public async Task SynthesizeNext(double startTime, double endTime, CancellationToken cancellation = default)
    {
        if (FindNextDirtyPiece(startTime, endTime) is not { } piece)
            return;

        // —— 同步前缀（数据线程）：拉快照 + 在 live 数据上做延音判定，随后才 offload ——
        var snapshot = mContext.GetSnapshot(piece.Notes);
        var continuation = new bool[piece.Notes.Count];
        for (int i = 0; i < piece.Notes.Count; i++)
            continuation[i] = IsContinuation(piece.Notes[i]);

        var pieceStart = piece.StartTime;
        var pieceEnd = piece.EndTime;
        piece.Dirty = false;
        piece.Failed = false;
        piece.Synthesizing = true;
        piece.Progress = 0;
        piece.Stage = "queued";
        NotifyProducts(); // 进入合成中 → 该块产物暂时留白

        var report = new Progress<double>(progress =>
        {
            piece.Progress = Math.Clamp(progress, 0, 1);
            mStatusChanged.Invoke();
        });

        try
        {
            var render = await Task.Run(
                () => RenderBlockAsync(snapshot, continuation, pieceStart, pieceEnd, report, cancellation),
                CancellationToken.None);

            if (render == null)
            {
                // 取消是正常调度结局：保留上一版产物，把块重新标脏等下一次调度。
                piece.Dirty = true;
                return;
            }

            if (!mPieces.Contains(piece))
                return; // 分块在合成期间被整块替换（会话已重建），丢弃这次结果

            Publish(piece, render);
        }
        catch (OperationCanceledException)
        {
            piece.Dirty = true;
        }
        catch (Exception ex)
        {
            piece.Failed = true;
            piece.Error = ex.Message;
        }
        finally
        {
            piece.Synthesizing = false;
            piece.Stage = null;
            NotifyProducts();
        }
    }

    // —— 后台渲染（worker 线程，只读 snapshot / 不可变设置；cloud 调用也在此） ——

    async Task<BlockRender?> RenderBlockAsync(VoiceSynthesisSnapshot snapshot, bool[] continuation,
        double startTime, double endTime, IProgress<double> progress, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var request = BuildRequest(snapshot, continuation, startTime, endTime, out var pitchTimes, out var finalPitch);
        progress.Report(0.08);

        // 整块都是休止 / 孤儿延音：本地生成静音段，不打扰云端。产物照常回填（空音节 / 音高回显）。
        if (request.Notes.Count == 0)
            return BuildSilentBlock(snapshot, continuation, startTime, endTime, pitchTimes, finalPitch);

        progress.Report(0.2);
        var result = await mProvider.SynthesizeAsync(request, progress, cancellationToken).ConfigureAwait(false);
        if (result == null)
            return null;
        cancellationToken.ThrowIfCancellationRequested();
        if (result.Audio.Length == 0)
            throw new CloudProviderException("The cloud returned empty audio.", false);

        int sampleRate = result.SampleRate > 0 ? result.SampleRate : mSettings.SampleRate;
        var phonemes = BuildPhonemes(snapshot, continuation, result);
        var pitchSegments = result.PitchSegments.Count > 0
            ? result.PitchSegments
            : BuildPitchEcho(pitchTimes, finalPitch);
        var parameters = BuildParameterEcho(result);
        progress.Report(0.98);
        return new BlockRender(result.AudioStartTime, result.Audio, sampleRate, phonemes, pitchSegments, parameters);
    }

    BlockRender BuildSilentBlock(VoiceSynthesisSnapshot snapshot, bool[] continuation,
        double startTime, double endTime, double[] pitchTimes, double[] finalPitch)
    {
        int sampleRate = Math.Max(8000, mSettings.SampleRate);
        int sampleCount = Math.Max(1, (int)Math.Round(Math.Max(0.01, endTime - startTime) * sampleRate));
        var silence = new float[sampleCount];
        var phonemes = BuildPhonemes(snapshot, continuation, null);
        var pitchSegments = BuildPitchEcho(pitchTimes, finalPitch);
        return new BlockRender(startTime, silence, sampleRate, phonemes, pitchSegments,
            new Map<string, IReadOnlyList<IReadOnlyList<Point>>>());
    }

    CloudSynthesisRequest BuildRequest(VoiceSynthesisSnapshot snapshot, bool[] continuation,
        double startTime, double endTime, out double[] pitchTimes, out double[] finalPitch)
    {
        var request = new CloudSynthesisRequest
        {
            Voice = mRoute.ProviderVoiceId,
            Model = null,
            SampleRate = mSettings.SampleRate,
            StartTime = startTime,
            EndTime = endTime,
            // 后端声明支持自动音高时告诉它可以自行预测；后端用不用由它自己的引擎参数决定
            // （比如 OpenCyan 的 PitchMode 参数会出现在属性面板里）。
            PitchMode = mSchema.PitchAuto ? "auto" : "manual",
            PartProperties = ReadProperties(snapshot.PartProperties, mSchema.PartProperties),
        };

        // —— 音高：从快照冻结求值器按控制率采样，解析成最终半音曲线（含 PitchDeviation）——
        double duration = Math.Max(0.01, endTime - startTime);
        int controlCount = Math.Max(2, (int)Math.Round(duration * PitchControlRate) + 1);
        pitchTimes = new double[controlCount];
        for (int i = 0; i < controlCount; i++)
            pitchTimes[i] = startTime + duration * i / (controlCount - 1);

        var absolute = snapshot.Pitch.Evaluator.Evaluate(pitchTimes);
        var deviation = snapshot.PitchDeviation.Evaluator.Evaluate(pitchTimes);
        finalPitch = new double[controlCount];
        for (int i = 0; i < controlCount; i++)
        {
            double fallback = NotePitchAt(snapshot, pitchTimes[i]);
            double semitone = (double.IsNaN(absolute[i]) ? fallback : absolute[i])
                              + (double.IsNaN(deviation[i]) ? 0 : deviation[i]);
            finalPitch[i] = semitone;
            request.Pitch.Add(new CloudPoint { T = pitchTimes[i], V = semitone });
        }

        // —— 自动化轨：schema 声明的可编辑轨按 50Hz 采样 ——
        // 整块都是默认值的轨省略发送（服务端按 schema 默认值处理）；分段轨的 NaN 段不发送。
        int automationCount = Math.Max(2, (int)Math.Round(duration * AutomationControlRate) + 1);
        var automationTimes = new double[automationCount];
        for (int i = 0; i < automationCount; i++)
            automationTimes[i] = startTime + duration * i / (automationCount - 1);

        foreach (var spec in mSchema.Automations)
        {
            if (!snapshot.Automations.TryGetValue(spec.Id, out var track))
                continue;
            var values = track.Evaluator.Evaluate(automationTimes);
            var points = new List<CloudPoint>(automationCount);
            bool allDefault = spec.Default.HasValue;
            for (int i = 0; i < automationCount; i++)
            {
                double value = values[i];
                if (double.IsNaN(value))
                    continue;
                if (allDefault && Math.Abs(value - spec.Default!.Value) > 1e-6)
                    allDefault = false;
                points.Add(new CloudPoint { T = automationTimes[i], V = value });
            }
            if (points.Count == 0)
                continue;
            if (allDefault && spec.Default.HasValue)
                continue;
            request.Parameters[spec.Id] = points;
        }

        // —— 音符：延音 note 并入链头时长（melisma）；休止 note 单独发送 rest 标记 ——
        CloudRequestNote? tail = null;
        for (int i = 0; i < snapshot.Notes.Count; i++)
        {
            var note = snapshot.Notes[i];
            if (i < continuation.Length && continuation[i])
            {
                if (tail != null)
                {
                    double tailEnd = Math.Max(tail.Start + tail.Duration, note.EndTime);
                    tail.Duration = Math.Max(0.02, tailEnd - tail.Start);
                }
                continue;
            }

            bool rest = IsRest(note.Lyric);
            var requestNote = new CloudRequestNote
            {
                Key = note.Id,
                Start = note.StartTime,
                Duration = Math.Max(0.02, note.EndTime - note.StartTime),
                Pitch = note.Pitch,
                Lyric = note.Lyric ?? "",
                Rest = rest,
                Phonemes = PinnedPhonemes(note),
                Properties = ReadProperties(note.Properties, mSchema.NoteProperties),
            };
            request.Notes.Add(requestNote);
            tail = rest ? null : requestNote;
        }

        return request;
    }

    static double NotePitchAt(VoiceSynthesisSnapshot snapshot, double time)
    {
        if (snapshot.Notes.Count == 0)
            return 60;
        foreach (var note in snapshot.Notes)
        {
            if (time >= note.StartTime && time < note.EndTime)
                return note.Pitch;
        }
        return snapshot.Notes[^1].Pitch;
    }

    static bool IsRest(string? lyric)
    {
        var text = (lyric ?? "").Trim();
        return text.Length == 0 || text == "-" || text.Equals("R", StringComparison.OrdinalIgnoreCase);
    }

    static CloudPhonemes? PinnedPhonemes(VoiceSynthesisNoteSnapshot note)
    {
        if (note.LeadingPhonemes.Count == 0 && note.BodyPhonemes.Count == 0)
            return null;
        return new CloudPhonemes
        {
            Leading = note.LeadingPhonemes.Select(ToCloudPhoneme).ToList(),
            Body = note.BodyPhonemes.Select(ToCloudPhoneme).ToList(),
            BodyOffset = note.BodyOffset,
        };
    }

    static CloudPhoneme ToCloudPhoneme(VoiceSynthesisPhonemeSnapshot phoneme)
        => new() { Symbol = phoneme.Symbol, Duration = phoneme.Duration, StretchWeight = phoneme.StretchWeight };

    // 按 schema 的 noteProperties / partProperties 读值：数字 / 布尔 / 字符串三类。
    static Dictionary<string, object> ReadProperties(PropertyObject values, IReadOnlyList<CloudPropertySpec> specs)
    {
        var result = new Dictionary<string, object>(specs.Count, StringComparer.Ordinal);
        foreach (var spec in specs)
        {
            switch (spec.Kind)
            {
                case CloudPropertyKind.Checkbox:
                    result[spec.Id] = values.GetBoolean(spec.Id, spec.BooleanDefault);
                    break;
                case CloudPropertyKind.Combo:
                case CloudPropertyKind.Text:
                    result[spec.Id] = values.GetString(spec.Id, spec.TextDefault ?? "");
                    break;
                default:
                    result[spec.Id] = values.GetDouble(spec.Id, spec.NumberDefault);
                    break;
            }
        }
        return result;
    }

    // —— 产物构建（worker 线程；全部是基于快照 / 响应值的不可变值） ——

    static IReadOnlyMap<string, SynthesizedSyllable> BuildPhonemes(VoiceSynthesisSnapshot snapshot, bool[] continuation, CloudSynthesisResult? result)
    {
        var map = new Map<string, SynthesizedSyllable>();
        for (int i = 0; i < snapshot.Notes.Count; i++)
        {
            var note = snapshot.Notes[i];

            if (i < continuation.Length && continuation[i])
            {
                // 判定为延续：键在、零音素 = "本 note 不发音"（宿主留白但邻居照常显示）。
                map.Add(note.Id, EmptySyllable);
                continue;
            }

            if (note.LeadingPhonemes.Count > 0 || note.BodyPhonemes.Count > 0)
            {
                map.Add(note.Id, new SynthesizedSyllable(
                    note.LeadingPhonemes.Select(ToSynthesizedPhoneme).ToList(),
                    note.BodyPhonemes.Select(ToSynthesizedPhoneme).ToList(),
                    note.BodyOffset));
                continue;
            }

            if (result != null && result.Phonemes.TryGetValue(note.Id, out var cloudSyllable))
            {
                map.Add(note.Id, new SynthesizedSyllable(cloudSyllable.Leading, cloudSyllable.Body, cloudSyllable.BodyOffset));
                continue;
            }

            map.Add(note.Id, ToSynthesizedSyllable(LyricPhonemes.Predict(note.Lyric)));
        }
        return map;
    }

    static SynthesizedPhoneme ToSynthesizedPhoneme(VoiceSynthesisPhonemeSnapshot phoneme)
        => new() { Symbol = phoneme.Symbol, Duration = phoneme.Duration, StretchWeight = phoneme.StretchWeight };

    static SynthesizedPhoneme ToSynthesizedPhoneme(PredictedPhoneme phoneme)
        => new() { Symbol = phoneme.Symbol, Duration = phoneme.Duration, StretchWeight = phoneme.StretchWeight };

    static SynthesizedSyllable ToSynthesizedSyllable(PredictedSyllable syllable)
        => new(
            syllable.Leading.Select(ToSynthesizedPhoneme).ToList(),
            syllable.Body.Select(ToSynthesizedPhoneme).ToList(),
            syllable.BodyOffset);

    IReadOnlyMap<string, IReadOnlyList<IReadOnlyList<Point>>> BuildParameterEcho(CloudSynthesisResult result)
    {
        var map = new Map<string, IReadOnlyList<IReadOnlyList<Point>>>();
        foreach (var spec in mSchema.SynthesizedParameters)
        {
            if (result.Parameters.TryGetValue(spec.Id, out var segments))
            {
                map.Add(spec.Id, segments);
                continue;
            }

            // 云端没给 energy：从音频算 RMS 包络兜底，保证这条常见回显轨始终有内容。
            if (spec.Id == "energy")
            {
                var envelope = AudioAnalysis.RmsEnvelope(result.Audio, result.SampleRate, result.AudioStartTime);
                if (envelope.Count >= 2)
                    map.Add("energy", [envelope]);
            }
        }
        return map;
    }

    static IReadOnlyList<IReadOnlyList<Point>> BuildPitchEcho(double[] times, double[] values)
    {
        if (times.Length < 2 || values.Length != times.Length)
            return [];
        var points = new List<Point>(times.Length);
        for (int i = 0; i < times.Length; i++)
            points.Add(new Point(times[i], values[i]));
        return [points];
    }

    // —— 产物发布（数据线程；换引用 = 发布不可变值） ——

    void Publish(Piece piece, BlockRender render)
    {
        piece.Segment?.Dispose();
        piece.Segment = null;

        if (render.Audio.Length > 0)
        {
            long sampleOffset = (long)Math.Round(render.AudioStartTime * render.SampleRate);
            var segment = mContext.CreateAudioSegment(sampleOffset, render.Audio.Length, render.SampleRate);
            segment.Write(0, render.Audio);
            segment.Commit();
            piece.Segment = segment;
        }

        piece.Phonemes = render.Phonemes;
        piece.PitchSegments = render.PitchSegments;
        piece.Parameters = render.Parameters;
        piece.PhonemesStale = false;
        piece.ParametersStale = false;
        piece.Failed = false;
        piece.Error = null;
        piece.Progress = 1;

        mSynthesizedPhonemesChanged.Invoke();
        mSynthesizedPitchChanged.Invoke();
        mSynthesizedParametersChanged.Invoke();
        mStatusChanged.Invoke();
    }

    // —— 产物读取（宿主任意线程只读；每块产物按 stale / failed / 是否有段过滤） ——

    public SynthesizedPitch SynthesizedPitch
    {
        get
        {
            var segments = new List<IReadOnlyList<Point>>();
            foreach (var piece in mPieces)
            {
                if (piece.Failed || piece.Segment == null || piece.ParametersStale)
                    continue;
                segments.AddRange(piece.PitchSegments);
            }
            return new SynthesizedPitch { Segments = segments };
        }
    }

    public IReadOnlyMap<string, SynthesizedParameter> SynthesizedParameters
    {
        get
        {
            var map = new Map<string, SynthesizedParameter>();
            foreach (var spec in mSchema.SynthesizedParameters)
            {
                var segments = new List<IReadOnlyList<Point>>();
                foreach (var piece in mPieces)
                {
                    if (piece.Failed || piece.Segment == null || piece.ParametersStale)
                        continue;
                    if (piece.Parameters.TryGetValue(spec.Id, out var trackSegments))
                        segments.AddRange(trackSegments);
                }
                if (segments.Count > 0)
                    map.Add(spec.Id, new SynthesizedParameter { Segments = segments });
            }
            return map;
        }
    }

    public IReadOnlyMap<string, SynthesizedSyllable> SynthesizedPhonemes
    {
        get
        {
            var map = new Map<string, SynthesizedSyllable>();
            foreach (var piece in mPieces)
            {
                if (piece.Failed || piece.Segment == null || piece.PhonemesStale)
                    continue;
                foreach (var pair in piece.Phonemes)
                    if (!map.ContainsKey(pair.Key))
                        map.Add(pair.Key, pair.Value);
            }
            return map;
        }
    }

    public IReadOnlyList<SynthesisStatusSegment> Status
    {
        get
        {
            var status = new List<SynthesisStatusSegment>(mPieces.Count);
            foreach (var piece in mPieces)
            {
                var kind = piece.Failed ? SynthesisSegmentStatus.Failed
                    : piece.Synthesizing ? SynthesisSegmentStatus.Synthesizing
                    : piece.Dirty || piece.Segment == null ? SynthesisSegmentStatus.Pending
                    : SynthesisSegmentStatus.Synthesized;
                status.Add(new SynthesisStatusSegment
                {
                    StartTime = piece.StartTime,
                    EndTime = piece.EndTime,
                    Status = kind,
                    Message = piece.Failed ? piece.Error : piece.Synthesizing ? piece.Stage : null,
                    Progress = piece.Synthesizing ? piece.Progress : 0,
                });
            }
            return status;
        }
    }

    public IActionEvent SynthesizedPhonemesChanged => mSynthesizedPhonemesChanged;
    public IActionEvent SynthesizedParametersChanged => mSynthesizedParametersChanged;
    public IActionEvent SynthesizedPitchChanged => mSynthesizedPitchChanged;
    public IActionEvent StatusChanged => mStatusChanged;

    // —— 失效（数据线程；handler 只标脏，重活推迟到 Committed） ——

    void MarkNoteDirty(IVoiceSynthesisNote note, bool phonemesStale, bool parametersStale)
    {
        bool hit = false;
        foreach (var piece in mPieces)
        {
            if (!piece.Notes.Contains(note))
                continue;
            piece.Dirty = true;
            piece.Failed = false;
            if (phonemesStale) piece.PhonemesStale = true;
            if (parametersStale) piece.ParametersStale = true;
            hit = true;
        }
        mNeedResegment = true;
        if (hit)
            NotifyProducts();
    }

    void MarkAllDirtyAndResegment()
    {
        foreach (var piece in mPieces)
        {
            piece.Dirty = true;
            piece.Failed = false;
        }
        mNeedResegment = true;
        NotifyProducts();
    }

    void OnNotesStructureChanged(IVoiceSynthesisNote note) => mNeedResegment = true;
    void OnCommitted()
    {
        if (mNeedResegment)
            Resegment();
    }

    void OnPitchRangeModified(double startTime, double endTime) => MarkRangeDirty(startTime, endTime, parametersStale: true);
    void OnAutomationRangeModified(double startTime, double endTime) => MarkRangeDirty(startTime, endTime, parametersStale: false);

    void MarkRangeDirty(double startTime, double endTime, bool parametersStale)
    {
        bool hit = false;
        foreach (var piece in mPieces)
        {
            if (piece.EndTime < startTime || piece.StartTime > endTime)
                continue;
            piece.Dirty = true;
            piece.Failed = false;
            if (parametersStale) piece.ParametersStale = true;
            hit = true;
        }
        if (hit)
            NotifyProducts();
    }

    // 按 note 间隙分块；note 可重叠（和弦），以「组内最大结束」判间隙。
    // 复用的块保留缓存与状态（note 集等价）；不再存在的块释放段握柄。
    void Resegment()
    {
        mNeedResegment = false;

        var groups = new List<List<IVoiceSynthesisNote>>();
        List<IVoiceSynthesisNote>? current = null;
        double groupMaxEnd = 0;
        foreach (var note in mContext.Notes)
        {
            if (current == null || note.StartTime.Value > groupMaxEnd)
            {
                current = new List<IVoiceSynthesisNote>();
                groups.Add(current);
                groupMaxEnd = note.EndTime.Value;
            }
            else
            {
                groupMaxEnd = Math.Max(groupMaxEnd, note.EndTime.Value);
            }
            current.Add(note);
        }

        var newPieces = new List<Piece>(groups.Count);
        foreach (var notes in groups)
        {
            double start = notes[0].StartTime.Value;
            double end = notes.Max(note => note.EndTime.Value);
            var existing = mPieces.FirstOrDefault(piece => piece.Notes.Count == notes.Count && piece.Notes.SequenceEqual(notes));
            if (existing != null)
            {
                mPieces.Remove(existing);
                existing.StartTime = start;
                existing.EndTime = end;
                newPieces.Add(existing);
            }
            else
            {
                newPieces.Add(new Piece
                {
                    Notes = notes,
                    StartTime = start,
                    EndTime = end,
                    Dirty = true,
                });
            }
        }

        foreach (var piece in mPieces)
            piece.Segment?.Dispose();
        mPieces.Clear();
        mPieces.AddRange(newPieces);
        NotifyProducts();
    }

    void NotifyProducts()
    {
        mSynthesizedPhonemesChanged.Invoke();
        mSynthesizedPitchChanged.Invoke();
        mSynthesizedParametersChanged.Invoke();
        mStatusChanged.Invoke();
    }

    // 引擎重新拉到 schema（用户保存设置 / 服务端升级后）：更新声明并让宿主重新询问 part/note 配置。
    void OnSchemaChanged(CloudVoiceSchema schema)
    {
        mSchema = schema;
        foreach (var spec in schema.Automations)
        {
            if (mSubscribedAutomations.Add(spec.Id) && mContext.Automations.TryGetValue(spec.Id, out var automation))
                automation.RangeModified.Subscribe(OnAutomationRangeModified, mSubscriptions);
        }
        // 注意：已发布的 TuneLab 2.1.x 宿主 SDK 没有 IVoiceSynthesisContext.InvalidatePart，
        // 这里不能调用（否则运行期 MissingMethodException / 扩展加载失败）。schema 更新后，
        // 新会话/重新打开工程会用新声明；已有工程的属性面板刷新仍以重启 TuneLab 为准。
        MarkAllDirtyAndResegment();
    }

    public void Dispose()
    {
        if (mEngine != null)
            mEngine.SchemaChanged -= OnSchemaChanged;
        if (mDisposed)
            return;
        mDisposed = true;
        mSubscriptions.DisposeAll();
        foreach (var piece in mPieces)
            piece.Segment?.Dispose();
        mPieces.Clear();
        mProvider.Dispose();
    }

    // —— 内部类型 ——

    sealed class Piece
    {
        public required IReadOnlyList<IVoiceSynthesisNote> Notes { get; init; }
        public double StartTime;
        public double EndTime;
        public bool Dirty;
        public bool PhonemesStale;     // 音素几何失效（上游：时长 / 歌词 / 结构）
        public bool ParametersStale;   // 参数 / 音高回显失效（上游：音素 / 音高及以上）
        public bool Failed;
        public bool Synthesizing;
        public string? Error;
        public string? Stage;
        public double Progress;
        public IAudioSegment? Segment;
        public IReadOnlyMap<string, SynthesizedSyllable> Phonemes = new Map<string, SynthesizedSyllable>();
        public IReadOnlyList<IReadOnlyList<Point>> PitchSegments = [];
        public IReadOnlyMap<string, IReadOnlyList<IReadOnlyList<Point>>> Parameters = new Map<string, IReadOnlyList<IReadOnlyList<Point>>>();
    }

    sealed class BlockRender(double audioStartTime, float[] audio, int sampleRate,
        IReadOnlyMap<string, SynthesizedSyllable> phonemes,
        IReadOnlyList<IReadOnlyList<Point>> pitchSegments,
        IReadOnlyMap<string, IReadOnlyList<IReadOnlyList<Point>>> parameters)
    {
        public double AudioStartTime { get; } = audioStartTime;
        public float[] Audio { get; } = audio;
        public int SampleRate { get; } = sampleRate;
        public IReadOnlyMap<string, SynthesizedSyllable> Phonemes { get; } = phonemes;
        public IReadOnlyList<IReadOnlyList<Point>> PitchSegments { get; } = pitchSegments;
        public IReadOnlyMap<string, IReadOnlyList<IReadOnlyList<Point>>> Parameters { get; } = parameters;
    }

    static readonly SynthesizedSyllable EmptySyllable = new([], [], 0);

    readonly IVoiceSynthesisContext mContext;
    readonly CloudVoiceSettings mSettings;
    readonly CloudVoiceRoute mRoute;
    readonly ICloudVoiceProvider mProvider;
    readonly CloudVoiceEngine? mEngine;
    CloudVoiceSchema mSchema;
    readonly HashSet<string> mSubscribedAutomations = new(StringComparer.Ordinal);
    readonly DisposableManager mSubscriptions = new();
    readonly List<Piece> mPieces = [];
    readonly ActionEvent mSynthesizedPhonemesChanged = new();
    readonly ActionEvent mSynthesizedPitchChanged = new();
    readonly ActionEvent mSynthesizedParametersChanged = new();
    readonly ActionEvent mStatusChanged = new();
    bool mNeedResegment;
    bool mDisposed;

    const int PitchControlRate = 100;        // 音高采样率（Hz）
    const int AutomationControlRate = 50;    // 自动化轨采样率（Hz）
}
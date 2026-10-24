"""声库层：DiffSingerPhonemizer 单测。

默认只跑启发式 / 合成声库（不加载真实大模型）；真实 dsdur 集成测试需要显式开启：
    TUNELAB_TEST_REAL_MODELS=1 .venv/bin/python -m pytest tests/test_phonemizer.py -q

真实测试覆盖 demo/DiffSingerCloud/OpenUTAU/ 下的 JiHeHong / ChiYu / yousa（如果存在）。
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from server.voicebank import dictionary as dictionary_module
from server.voicebank import durations, types
from server.voicebank.discovery import VoiceCatalog
from server.voicebank.phonemizer import DiffSingerPhonemizer

DEMO_DIR = Path(__file__).resolve().parent.parent
OPENUTAU_DIR = DEMO_DIR / "OpenUTAU"
VOCODER_DIR = OPENUTAU_DIR / "Vocoders"
RUN_REAL_MODELS = os.environ.get("TUNELAB_TEST_REAL_MODELS") == "1"


# --------------------------------------------------------------------------------------
# 合成声库
# --------------------------------------------------------------------------------------
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_SYMBOLS = (
    ("SP", "vowel"),
    ("AP", "vowel"),
    ("a", "vowel"),
    ("i", "vowel"),
    ("b", "fricative"),
    ("l", "liquid"),
    ("m", "fricative"),
    ("n", "fricative"),
)


def make_voicebank(
    tmp_path: Path,
    *,
    use_lang_id: bool = False,
    dictionary_dir: Path | None = None,
) -> types.Voicebank:
    bank_dir = tmp_path / "SyntheticVoice"
    bank_dir.mkdir(parents=True, exist_ok=True)
    lines = ["symbols:"]
    for symbol, symbol_type in _SYMBOLS:
        lines.append(f"- symbol: {symbol}")
        lines.append(f"  type: {symbol_type}")
    lines.append("entries:")
    for grapheme, phones in (
        ("a", ["a"]),
        ("i", ["i"]),
        ("ba", ["b", "a"]),
        ("bi", ["b", "i"]),
        ("la", ["l", "a"]),
        ("ma", ["m", "a"]),
        ("na", ["n", "a"]),
    ):
        lines.append(f"- grapheme: {grapheme}")
        lines.append("  phonemes:")
        for phone in phones:
            lines.append(f"  - {phone}")
    _write(bank_dir / "dsdict.yaml", "\n".join(lines) + "\n")

    dictionary_path = bank_dir
    if use_lang_id:
        lang_dir = dictionary_dir or (tmp_path / "LangBank")
        lang_dir.mkdir(parents=True, exist_ok=True)
        lang_lines = [
            "symbols:",
            "- symbol: SP",
            "  type: vowel",
            "- symbol: AP",
            "  type: vowel",
            "- symbol: zh/a",
            "  type: vowel",
            "- symbol: zh/b",
            "  type: fricative",
            "entries:",
            "- grapheme: ba",
            "  phonemes:",
            "  - zh/b",
            "  - zh/a",
            "- grapheme: a",
            "  phonemes:",
            "  - zh/a",
        ]
        _write(lang_dir / "dsdict-zh.yaml", "\n".join(lang_lines) + "\n")
        dictionary_path = lang_dir

    _write(bank_dir / "phonemes.txt", "SP\nAP\na\ni\nb\nl\nm\nn\n")
    _write(
        bank_dir / "dsconfig.yaml",
        "phonemes: phonemes.txt\nacoustic: acoustic.onnx\nvocoder: test_vocoder\n",
    )
    _write(bank_dir / "character.txt", "name=Synthetic Voice\n")
    config = types.DsConfig(
        source_path=(bank_dir / "dsconfig.yaml").resolve(),
        phonemes="phonemes.txt",
    )
    phonemes = {"SP": 0, "AP": 1, "a": 2, "i": 3, "b": 4, "l": 5, "m": 6, "n": 7}
    dictionary = dictionary_module.build_dictionary(
        dictionary_path, use_lang_id=use_lang_id, default_language="auto"
    )
    return types.Voicebank(
        id="SyntheticVoice",
        bank_id="SyntheticVoice",
        name="Synthetic Voice",
        path=bank_dir,
        config=config,
        phonemes=phonemes,
        dictionary=dictionary,
    )


@pytest.fixture()
def voicebank(tmp_path: Path) -> types.Voicebank:
    return make_voicebank(tmp_path)


def phonemizer() -> DiffSingerPhonemizer:
    return DiffSingerPhonemizer(
        None, types.PhonemizerOptions(use_duration_model=False, min_phone_ms=15.0)
    )


# --------------------------------------------------------------------------------------
# 启发式路径
# --------------------------------------------------------------------------------------
def test_heuristic_two_notes(voicebank: types.Voicebank) -> None:
    notes = [
        types.NoteInput(key="n1", start_s=0.0, duration_s=1.0, pitch=60, lyric="la"),
        types.NoteInput(key="n2", start_s=1.0, duration_s=1.0, pitch=62, lyric="ma"),
    ]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert [phone.symbol for phone in phrase.phones] == ["l", "a", "m", "a"]
    assert [phone.note_index for phone in phrase.phones] == [0, 0, 1, 1]
    assert phrase.phone_index_ranges() == [(0, 2), (2, 4)]
    assert phrase.phones[0].start_ms == pytest.approx(0.0)
    assert phrase.phones[0].end_ms == pytest.approx(70.0)
    assert phrase.phones[1].end_ms == pytest.approx(1000.0)
    assert phrase.phones[2].start_ms == pytest.approx(1000.0)
    assert phrase.phones[3].end_ms == pytest.approx(2000.0)
    # 每个 note 的总时长严格等于 note duration
    for note_index in (0, 1):
        covered = sum(
            phone.duration_ms for phone in phrase.phones if phone.note_index == note_index
        )
        assert covered == pytest.approx(1000.0)
    # 元音/辅音标记、tone/speaker
    assert phrase.phones[1].vowel is True
    assert phrase.phones[0].vowel is False
    assert phrase.phones[0].tone == 60
    assert phrase.phones[2].tone == 62
    assert phrase.phones[0].speaker == ""
    # Phrase.notes 与输入等长同序
    assert len(phrase.notes) == len(notes)
    assert [note.key for note in phrase.notes] == ["n1", "n2"]
    assert [note.index for note in phrase.notes] == [0, 1]
    assert phrase.notes[0].rest is False


def test_heuristic_short_note(voicebank: types.Voicebank) -> None:
    notes = [types.NoteInput(key="n1", start_s=0.0, duration_s=0.05, pitch=60, lyric="la")]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert len(phrase.phones) == 2
    total = sum(phone.duration_ms for phone in phrase.phones)
    assert total == pytest.approx(50.0)
    assert all(phone.duration_ms >= 15.0 - 1e-9 for phone in phrase.phones)
    assert phrase.phones[-1].end_ms == pytest.approx(50.0)


def test_rest_notes_keep_gap_with_sp(voicebank: types.Voicebank) -> None:
    notes = [
        types.NoteInput(key="a", start_s=0.0, duration_s=0.5, pitch=60, lyric="la"),
        types.NoteInput(key="r", start_s=0.5, duration_s=1.0, pitch=60, rest=True, lyric="R"),
        types.NoteInput(key="b", start_s=1.5, duration_s=0.5, pitch=64, lyric="ma"),
    ]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert len(phrase.notes) == 3
    assert phrase.notes[1].rest is True
    # la + ma 四只音素 + 一个 SP 填充休止
    assert [phone.symbol for phone in phrase.phones] == ["l", "a", "SP", "m", "a"]
    assert phrase.phones[2].start_ms == pytest.approx(500.0)
    assert phrase.phones[2].end_ms == pytest.approx(1500.0)
    assert phrase.phones[2].vowel is False
    # 休止 note 的 range 为空，后续 note 的 range 包含填充 SP（phone_index_ranges 合法）
    assert phrase.phone_index_ranges() == [(0, 2), (2, 2), (2, 5)]
    # 时间轴连续（无内部空洞）
    for previous, current in zip(phrase.phones, phrase.phones[1:]):
        assert current.start_ms == pytest.approx(previous.end_ms)


def test_rest_only(voicebank: types.Voicebank) -> None:
    notes = [
        types.NoteInput(key="r1", start_s=0.0, duration_s=1.0, pitch=60, rest=True),
        types.NoteInput(key="r2", start_s=1.0, duration_s=1.0, pitch=60, rest=True),
    ]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert phrase.phones == []
    assert len(phrase.notes) == 2
    assert all(note.rest for note in phrase.notes)
    assert phrase.phone_index_ranges() == [(0, 0), (0, 0)]


def test_empty_lyric_uses_sp(voicebank: types.Voicebank) -> None:
    notes = [
        types.NoteInput(key="a", start_s=0.0, duration_s=0.5, pitch=60, lyric=""),
        types.NoteInput(key="b", start_s=0.5, duration_s=0.5, pitch=60, lyric="   "),
    ]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert [phone.symbol for phone in phrase.phones] == ["SP", "SP"]
    assert phrase.phones[0].start_ms == pytest.approx(0.0)
    assert phrase.phones[0].end_ms == pytest.approx(500.0)
    assert phrase.phones[1].end_ms == pytest.approx(1000.0)
    assert phrase.phone_index_ranges() == [(0, 1), (1, 2)]


def test_pinned_phonemes_override_lyric(voicebank: types.Voicebank) -> None:
    notes = [
        types.NoteInput(
            key="n1",
            start_s=0.0,
            duration_s=1.0,
            pitch=60,
            lyric="ma",
            pinned=["b", "a"],
        ),
    ]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert [phone.symbol for phone in phrase.phones] == ["b", "a"]


def test_invalid_pinned_falls_back_to_lyric(voicebank: types.Voicebank) -> None:
    notes = [
        types.NoteInput(
            key="n1",
            start_s=0.0,
            duration_s=1.0,
            pitch=60,
            lyric="la",
            pinned=["NOT_A_PHONE"],
        ),
    ]
    phrase = phonemizer().phonemize(voicebank, notes)
    assert [phone.symbol for phone in phrase.phones] == ["l", "a"]


def test_language_override_for_use_lang_id_bank(tmp_path: Path) -> None:
    voicebank = make_voicebank(tmp_path, use_lang_id=True)
    voicebank.phonemes = {"SP": 0, "AP": 1, "zh/a": 2, "zh/b": 3}
    notes = [
        types.NoteInput(key="n1", start_s=0.0, duration_s=1.0, pitch=60, lyric="ba"),
    ]
    phrase = phonemizer().phonemize(voicebank, notes, language="zh")
    assert [phone.symbol for phone in phrase.phones] == ["zh/b", "zh/a"]
    assert phrase.phones[1].vowel is True


def test_dsdur_failure_falls_back_to_heuristic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voicebank = make_voicebank(tmp_path)
    fake_dir = tmp_path / "dsdur"
    model_path = fake_dir / "dur.onnx"
    linguistic_path = fake_dir / "linguistic.onnx"
    _write(model_path, "x")
    _write(linguistic_path, "x")
    voicing_config = types.DsConfig(source_path=(fake_dir / "dsconfig.yaml").resolve())
    voicebank.dur = types.ModelGroup(
        kind="dur",
        dir=fake_dir,
        config=voicing_config,
        phonemes=dict(voicebank.phonemes),
        linguistic_path=linguistic_path,
        model_path=model_path,
        emb_dir=fake_dir,
    )

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError("synthetic dsdur failure")

    monkeypatch.setattr(durations, "predict_duration_frames", boom)
    runner = object()  # 只要非 None，_can_use_duration_model 就会走 dsdur 分支
    phonemizer_with_model = DiffSingerPhonemizer(
        runner, types.PhonemizerOptions(use_duration_model=True)
    )
    notes = [types.NoteInput(key="n1", start_s=0.0, duration_s=1.0, pitch=60, lyric="la")]
    phrase = phonemizer_with_model.phonemize(voicebank, notes)
    assert [phone.symbol for phone in phrase.phones] == ["l", "a"]
    assert phrase.phones[-1].end_ms == pytest.approx(1000.0)
    assert any("dsdur timing failed" in message for message in phonemizer_with_model.warnings)


# --------------------------------------------------------------------------------------
# dsdur 分配逻辑（用假 runner 验证，不加载大模型）
# --------------------------------------------------------------------------------------
class FakeDurationRunner:
    """伪造 linguistic + duration 模型输出，验证 word_div/word_dur 与 per-note 分配。"""

    TOKEN_FRAMES = {0: 1.0, 1: 1.0, 2: 4.0, 3: 4.0, 4: 1.0, 5: 1.0, 6: 2.0, 7: 2.0}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._tokens: list[int] = []

    def run(self, model_path, inputs):  # noqa: ANN001
        name = str(model_path)
        self.calls.append((name, inputs))
        if "linguistic" in name:
            tokens = np.asarray(inputs["tokens"])
            count = int(tokens.shape[1])
            self._tokens = [int(value) for value in tokens[0]]
            return {
                "encoder_out": np.zeros((1, count, 8), dtype=np.float32),
                "x_masks": np.ones((1, count), dtype=bool),
            }
        durations = [self.TOKEN_FRAMES.get(token, 1.0) for token in self._tokens]
        return {"ph_dur_pred": np.asarray([durations], dtype=np.float32)}

    def input_names(self, model_path):  # noqa: ANN001, ARG002
        return []

    def output_names(self, model_path):  # noqa: ANN001, ARG002
        return []


def _attach_fake_dsdur(voicebank: types.Voicebank, tmp_path: Path) -> None:
    fake_dir = tmp_path / "dsdur"
    _write(fake_dir / "linguistic.onnx", "x")
    _write(fake_dir / "dur.onnx", "x")
    config = types.DsConfig(source_path=(fake_dir / "dsconfig.yaml").resolve())
    voicebank.dur = types.ModelGroup(
        kind="dur",
        dir=fake_dir,
        config=config,
        phonemes=dict(voicebank.phonemes),
        linguistic_path=fake_dir / "linguistic.onnx",
        model_path=fake_dir / "dur.onnx",
        emb_dir=fake_dir,
    )


def test_dsdur_openutau_alignment_with_fake_runner(
    tmp_path: Path, voicebank: types.Voicebank
) -> None:
    _attach_fake_dsdur(voicebank, tmp_path)
    runner = FakeDurationRunner()
    phonemizer = DiffSingerPhonemizer(runner, types.PhonemizerOptions(use_duration_model=True))
    notes = [
        types.NoteInput(key="n1", start_s=0.0, duration_s=1.0, pitch=60, lyric="la"),
        types.NoteInput(key="n2", start_s=1.0, duration_s=1.0, pitch=62, lyric="ma"),
    ]
    phrase = phonemizer.phonemize(voicebank, notes)
    assert not phonemizer.warnings
    assert [phone.symbol for phone in phrase.phones] == ["l", "a", "m", "a"]
    frame_ms = 1000.0 * 512 / 44100
    # junction-anchored（OpenUTAU）：body 从 note 头正向铺，前导辅音落在 note 头之前
    assert phrase.phones[0].start_ms == pytest.approx(-frame_ms)  # l 在拍前
    assert phrase.phones[0].end_ms == pytest.approx(0.0)
    assert phrase.phones[1].start_ms == pytest.approx(0.0)        # a 从 note 头开始
    assert phrase.phones[1].end_ms == pytest.approx(1000.0 * 4 / 6)  # a: 4 帧
    assert phrase.phones[2].start_ms == pytest.approx(1000.0 * 4 / 6)  # m: 2 帧，落在上一 note 尾段
    assert phrase.phones[2].end_ms == pytest.approx(1000.0)
    assert phrase.phones[3].start_ms == pytest.approx(1000.0)
    assert phrase.phones[3].end_ms == pytest.approx(2000.0)
    assert phrase.phone_index_ranges() == [(0, 2), (2, 4)]
    _assert_openutau_alignment(phrase)
    # linguistic 输入：tokens/word_div/word_dur 与 OpenUTAU 一致
    linguistic_calls = [inputs for name, inputs in runner.calls if "linguistic" in name]
    assert len(linguistic_calls) == 1
    call = linguistic_calls[0]
    assert np.asarray(call["tokens"]).shape == (1, 5)  # SP + l + a + m + a
    assert np.asarray(call["word_div"]).tolist() == [[2, 2, 1]]
    assert np.asarray(call["word_dur"]).shape == (1, 3)
    duration_calls = [inputs for name, inputs in runner.calls if "linguistic" not in name]
    assert len(duration_calls) == 1
    assert np.asarray(duration_calls[0]["ph_midi"]).shape == (1, 5)


# --------------------------------------------------------------------------------------
# 真实 dsdur（默认 skip；TUNELAB_TEST_REAL_MODELS=1 开启）
# --------------------------------------------------------------------------------------
class MiniRunner:
    """测试用的最小 OnnxRunner（types.OnnxRunner 协议），只用于集成验证。"""

    def __init__(self) -> None:
        import onnxruntime as ort  # noqa: PLC0415 - 只有开启真实模型测试才需要

        self._ort = ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self._options = options
        self._sessions: dict[str, object] = {}

    def _session(self, model_path: Path | str):
        key = str(Path(model_path).resolve())
        if key not in self._sessions:
            self._sessions[key] = self._ort.InferenceSession(
                key, sess_options=self._options, providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def input_names(self, model_path: Path | str) -> list[str]:
        return [item.name for item in self._session(model_path).get_inputs()]

    def output_names(self, model_path: Path | str) -> list[str]:
        return [item.name for item in self._session(model_path).get_outputs()]

    def run(self, model_path: Path | str, inputs: dict[str, object]) -> dict[str, np.ndarray]:
        session = self._session(model_path)
        names = [item.name for item in session.get_inputs()]
        missing = [name for name in names if name not in inputs]
        if missing:
            raise ValueError("missing model input(s): " + ", ".join(missing))
        extra = set(inputs) - set(names)
        if extra:
            raise ValueError("unexpected model input(s): " + ", ".join(sorted(extra)))
        outputs = session.run(None, {name: inputs[name] for name in names})
        return {item.name: value for item, value in zip(session.get_outputs(), outputs)}


real_models = pytest.mark.skipif(
    not RUN_REAL_MODELS,
    reason="set TUNELAB_TEST_REAL_MODELS=1 to run real dsdur inference tests",
)


@pytest.fixture(scope="module")
def real_catalog() -> VoiceCatalog:
    if not all((OPENUTAU_DIR / name).is_dir() for name in ("JiHeHong-diffsinger-openutau-V2", "ChiYu")):
        pytest.skip("real OpenUTAU voicebanks are not available")
    catalog = VoiceCatalog([OPENUTAU_DIR], [VOCODER_DIR], max_depth=3)
    catalog.refresh()
    return catalog


def _assert_openutau_alignment(phrase: types.Phrase) -> None:
    """OpenUTAU junction-anchored 校验（TuneLab 音素契约）。

    - 音素单调、首尾相接（无空洞、无重叠）；
    - 每个非休止 note 至少有一个音素起点正好等于 note 头（body/junction 锚点）；
    - 前导辅音可以在 note 头之前（负偏移），后一个 note 的辅音可以落在前一个 note 尾部；
    - 句尾最后一个非休止 note 的末音素结束于 note.end。
    """
    phones = phrase.phones
    assert phones
    for previous, current in zip(phones, phones[1:]):
        assert current.start_ms >= previous.start_ms - 1e-6
        assert current.start_ms == pytest.approx(previous.end_ms, abs=1e-3)
    for phone in phones:
        assert phone.end_ms >= phone.start_ms
    ranges = phrase.phone_index_ranges()
    assert len(ranges) == len(phrase.notes)
    non_rest: list[tuple[types.ResolvedNote, tuple[int, int]]] = []
    for note, (begin, end) in zip(phrase.notes, ranges):
        if note.rest:
            assert begin == end  # 休止 note 不产生音素
            continue
        assert end > begin, f"note {note.key} has no phones"
        non_rest.append((note, (begin, end)))
        note_phones = phones[begin:end]
        assert any(
            abs(phone.start_ms - note.start_ms) <= 1e-3 for phone in note_phones
        ), f"note {note.key} has no phoneme starting at its head"
    if non_rest:
        note, (begin, end) = non_rest[-1]
        assert phones[end - 1].end_ms == pytest.approx(note.end_ms, abs=1e-3)


@pytest.mark.parametrize("bank_id", ["JiHeHong-diffsinger-openutau-V2", "ChiYu"])
@real_models
def test_real_dsdur_simple_notes(real_catalog: VoiceCatalog, bank_id: str) -> None:
    voicebank = real_catalog.get(bank_id)
    runner = MiniRunner()
    phonemizer = DiffSingerPhonemizer(runner, types.PhonemizerOptions(use_duration_model=True))
    notes = [
        types.NoteInput(key="n1", start_s=0.0, duration_s=1.0, pitch=60, lyric="la"),
        types.NoteInput(key="n2", start_s=1.0, duration_s=1.0, pitch=62, lyric="ma"),
    ]
    phrase = phonemizer.phonemize(voicebank, notes)
    assert not phonemizer.warnings
    assert [phone.symbol for phone in phrase.phones] == ["l", "a", "m", "a"]
    assert phrase.phone_index_ranges() == [(0, 2), (2, 4)]
    _assert_openutau_alignment(phrase)
    assert phrase.phones[-1].end_ms == pytest.approx(2000.0)
    assert [phone.note_index for phone in phrase.phones] == [0, 0, 1, 1]
    # junction-anchored：l 在拍前结束于 note 头，a 从 note 头开始；
    # 下一 note 的 m 落在上一 note 尾段、结束于下一个 note 头。
    assert phrase.phones[0].start_ms < 0.0
    assert phrase.phones[0].end_ms == pytest.approx(0.0, abs=1e-3)
    assert phrase.phones[1].start_ms == pytest.approx(0.0, abs=1e-3)
    assert phrase.phones[2].start_ms < 1000.0
    assert phrase.phones[2].end_ms == pytest.approx(1000.0, abs=1e-3)
    assert phrase.phones[3].start_ms == pytest.approx(1000.0, abs=1e-3)


@real_models
def test_real_dsdur_single_vowel(real_catalog: VoiceCatalog) -> None:
    voicebank = real_catalog.get("JiHeHong-diffsinger-openutau-V2")
    runner = MiniRunner()
    phonemizer = DiffSingerPhonemizer(runner, types.PhonemizerOptions(use_duration_model=True))
    notes = [types.NoteInput(key="n1", start_s=0.0, duration_s=2.0, pitch=69, lyric="阿")]
    phrase = phonemizer.phonemize(voicebank, notes)
    assert not phonemizer.warnings
    assert [phone.symbol for phone in phrase.phones] == ["a"]
    assert phrase.phones[0].start_ms == pytest.approx(0.0, abs=1.0)
    assert phrase.phones[0].end_ms == pytest.approx(2000.0)
    assert phrase.phones[0].tone == 69
    _assert_openutau_alignment(phrase)


@real_models
def test_real_dsdur_multilingual(real_catalog: VoiceCatalog) -> None:
    if not any(bank.bank_id == "yousaV1.51_rope" for bank in real_catalog.voices()):
        pytest.skip("yousa voicebank is not available")
    voicebank = real_catalog.get("yousaV1.51_rope::Bright")
    runner = MiniRunner()
    phonemizer = DiffSingerPhonemizer(runner, types.PhonemizerOptions(use_duration_model=True))
    notes = [
        types.NoteInput(key="n1", start_s=0.0, duration_s=1.0, pitch=60, lyric="阿"),
        types.NoteInput(key="n2", start_s=1.0, duration_s=1.0, pitch=62, lyric="ba"),
    ]
    phrase = phonemizer.phonemize(voicebank, notes)
    assert not phonemizer.warnings
    symbols = [phone.symbol for phone in phrase.phones]
    assert symbols[:2] == ["zh/a", "zh/b"]
    assert symbols[2:] == ["zh/a"]
    # range 合法：每个 note 都有自己的音素
    ranges = phrase.phone_index_ranges()
    assert ranges[0] != (ranges[0][0], ranges[0][0])
    _assert_openutau_alignment(phrase)

"""声库层单测：发现 / dsconfig 解析 / 字典 / G2P。

真实声库（demo/DiffSingerCloud/OpenUTAU/ 下的符号链接）存在时跑集成断言；
不存在时相关用例 skip。合成小目录用于覆盖扫描规则与旧版 dsdict.txt。
不加载 ONNX。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from server.voicebank import config as config_parser
from server.voicebank import dictionary as dictionary_module
from server.voicebank import types
from server.voicebank.discovery import (
    VoiceCatalog,
    emb_path_for_group,
    scan_voicebanks,
    speaker_for_group,
)
from server.voicebank.g2p import G2P, detect_language, kana_to_romaji

DEMO_DIR = Path(__file__).resolve().parent.parent
OPENUTAU_DIR = DEMO_DIR / "OpenUTAU"
VOCODER_DIR = OPENUTAU_DIR / "Vocoders"

JIHEHONG = "JiHeHong-diffsinger-openutau-V2"
CHIYU = "ChiYu"
YOUSAV = "yousaV1.51_rope"
REAL_BANKS = (JIHEHONG, CHIYU, YOUSAV)

pytestmark = pytest.mark.skipif(
    not all((OPENUTAU_DIR / name).is_dir() for name in REAL_BANKS),
    reason="real OpenUTAU voicebanks are not available under demo/DiffSingerCloud/OpenUTAU/",
)


def make_g2p(bank: types.Voicebank) -> G2P:
    dictionary = bank.dictionary
    assert dictionary is not None
    return G2P(
        dictionary,
        use_lang_id=bank.config.use_lang_id,
        default_language=getattr(dictionary, "default_language", "auto"),
    )


@pytest.fixture(scope="module")
def catalog() -> VoiceCatalog:
    result = VoiceCatalog([OPENUTAU_DIR], [VOCODER_DIR], max_depth=3)
    result.refresh()
    return result


# --------------------------------------------------------------------------------------
# 扫描 / 配置
# --------------------------------------------------------------------------------------
def test_discover_real_voicebanks(catalog: VoiceCatalog) -> None:
    voices = catalog.voices()
    bank_ids = {bank.bank_id for bank in voices}
    assert bank_ids == {JIHEHONG, CHIYU, YOUSAV}
    # 多说话人声库按 speaker 展开：JiHeHong 1 + ChiYu 1 + yousa 6
    assert len(voices) == 8
    ids = {bank.id for bank in voices}
    assert JIHEHONG in ids
    assert CHIYU in ids
    for speaker in ("Bright", "Cute", "Normal", "Whisper", "Joyful", "Classic"):
        assert f"{YOUSAV}::{speaker}" in ids
    assert len(catalog.voices()) == len(voices)
    assert catalog.get(JIHEHONG).id == JIHEHONG
    with pytest.raises(KeyError):
        catalog.get("does-not-exist")
    rescanned = scan_voicebanks([OPENUTAU_DIR], [VOCODER_DIR])
    assert {bank.id for bank in rescanned} == ids


def test_voices_are_lazy_dictionaries() -> None:
    fresh = VoiceCatalog([OPENUTAU_DIR], [VOCODER_DIR], max_depth=3)
    fresh.refresh()
    assert fresh.voices()
    for bank in fresh.voices():
        assert bank.dictionary is not None
        assert bank.dictionary.loaded is False  # 27MB 的 dsdict 不能在扫描时解析


def test_jihehong_config(catalog: VoiceCatalog) -> None:
    bank = catalog.get(JIHEHONG)
    assert bank.name == "JiHeHong DiffSinger V2"
    assert bank.config.hidden_size == 256
    assert bank.config.speakers == ["231111_hdl_ds_acoustic.jihehong"]
    assert bank.selected_speaker == "231111_hdl_ds_acoustic.jihehong"
    assert bank.config.use_key_shift_embed is True
    assert bank.config.use_speed_embed is True
    assert bank.config.use_energy_embed is True
    assert bank.config.use_breathiness_embed is True
    assert bank.config.use_variable_depth is True
    assert bank.config.max_depth == pytest.approx(300.0)
    assert bank.sample_rate == 44100
    assert {"SP", "AP", "a", "b", "l"} <= set(bank.phonemes)
    assert bank.config.resolve(bank.config.acoustic).is_file()
    assert bank.dur is not None and bank.dur.available
    assert bank.dur.model_path and bank.dur.model_path.is_file()
    assert bank.dur.linguistic_path and bank.dur.linguistic_path.is_file()
    # 子模型 .emb 命名与声学不同，discovery 按 speakers 下标映射
    dur_speaker = speaker_for_group(bank, bank.dur)
    assert dur_speaker == "231111_hdl_ds_variance.jihehong"
    assert emb_path_for_group(bank, bank.dur) is not None
    assert emb_path_for_group(bank) is not None
    if bank.vocoder is not None:
        assert bank.vocoder.name == "nsf_hifigan"
        assert bank.vocoder.model_path is not None and bank.vocoder.model_path.is_file()
        assert bank.vocoder.sample_rate == 44100
    assert getattr(bank, "character", None) is not None
    assert bank.warnings == []


def test_chiyu_config(catalog: VoiceCatalog) -> None:
    bank = catalog.get(CHIYU)
    assert bank.config.use_breathiness_embed is True
    assert bank.config.use_voicing_embed is True
    assert bank.config.use_tension_embed is True
    assert bank.config.use_energy_embed is False
    assert bank.config.use_continuous_acceleration is True
    assert bank.config.use_variable_depth is True
    assert bank.config.max_depth == pytest.approx(0.6)
    assert bank.config.mel_base == "e"
    assert bank.selected_speaker == ""
    assert bank.dur is not None and bank.dur.available
    assert bank.variance is not None and bank.variance.available
    assert bank.variance.model_path and bank.variance.model_path.is_file()
    assert getattr(bank, "character") is not None
    if bank.vocoder is not None:
        assert "nsf_hifigan" in bank.vocoder.name


def test_yousa_multilingual_config(catalog: VoiceCatalog) -> None:
    voices = [bank for bank in catalog.voices() if bank.bank_id == YOUSAV]
    assert len(voices) == 6
    assert [bank.selected_speaker for bank in voices] == [
        "Bright",
        "Cute",
        "Normal",
        "Whisper",
        "Joyful",
        "Classic",
    ]
    bright = catalog.get(f"{YOUSAV}::Bright")
    assert bright.config.use_lang_id is True
    assert bright.config.use_variable_depth is True
    assert bright.config.max_depth == pytest.approx(0.6)
    assert bright.languages == {"en": 1, "ja": 2, "zh": 3}
    assert len(bright.subbanks) == 6
    assert bright.subbanks[0].suffix == "Bright"
    assert bright.dur is not None and bright.dur.available
    assert bright.dur.languages == {"en": 1, "ja": 2, "zh": 3}
    assert bright.variance is not None and bright.variance.available
    assert bright.pitch is not None  # dspitch 存在，当前不参与渲染但应解析出来
    assert bright.vocoder is not None
    # 声库自带 dsvocoder 优先于 dsconfig 里的内建名
    assert bright.vocoder.name == "yousa_vocoder_FT_0301"
    assert bright.vocoder.mel_base == "e"
    assert emb_path_for_group(bright) is not None
    assert bright.dictionary is not None and bright.dictionary.has_language("zh")
    assert bright.dictionary.has_language("ja")
    assert bright.dictionary.has_language("en")


# --------------------------------------------------------------------------------------
# 字典
# --------------------------------------------------------------------------------------
def test_jihehong_dictionary(catalog: VoiceCatalog) -> None:
    bank = catalog.get(JIHEHONG)
    dictionary = bank.dictionary
    assert dictionary.query("ba") == ["b", "a"]
    assert dictionary.query("chi") == ["ch", "ir"]
    assert dictionary.query("a") == ["a"]
    assert dictionary.is_valid("a") is True
    assert dictionary.is_valid("b") is True
    assert dictionary.is_valid("NOT_A_PHONE") is False
    assert dictionary.is_vowel("a") is True
    assert dictionary.is_vowel("b") is False
    assert dictionary.symbol_type("a") == "vowel"
    assert {"SP", "AP", "a", "ai"} <= set(dictionary.symbols())


def test_chiyu_dictionary(catalog: VoiceCatalog) -> None:
    dictionary = catalog.get(CHIYU).dictionary
    assert dictionary.query("ba") == ["b", "a"]
    assert dictionary.is_vowel("a") is True
    assert dictionary.is_valid("zh") is True


def test_yousa_dictionary_languages(catalog: VoiceCatalog) -> None:
    bank = catalog.get(f"{YOUSAV}::Bright")
    dictionary = bank.dictionary
    assert dictionary.query_lang("zh", "ba") == ["zh/b", "zh/a"]
    assert dictionary.query_lang("zh", "a") == ["zh/a"]
    assert dictionary.query("ba") == ["en/b", "en/iy", "en/ey"]  # 基础字典是英文读音
    assert dictionary.is_valid("zh/a") is True
    assert dictionary.is_valid("en/aa") is True
    assert dictionary.is_vowel("zh/a") is True
    assert dictionary.symbol_type("zh/b") in ("fricative", "vowel")
    assert dictionary.is_glide("en/l") is True or dictionary.is_glide("zh/l") is True


# --------------------------------------------------------------------------------------
# G2P
# --------------------------------------------------------------------------------------
def test_g2p_jihehong(catalog: VoiceCatalog) -> None:
    g2p = make_g2p(catalog.get(JIHEHONG))
    assert g2p.phonemize("阿") == ["a"]
    assert g2p.phonemize("ba") == ["b", "a"]
    assert g2p.phonemize("ni3") == ["n", "i"]
    assert g2p.phonemize("l a") == ["l", "a"]
    assert g2p.phonemize("SP") == ["SP"]
    assert g2p.resolve_pinned(["b", "a"]) == ["b", "a"]
    assert g2p.resolve_pinned(["b", "NOT_A_PHONE"]) == ["b"]
    assert g2p.phonemize("") == []
    assert g2p.warnings == []


def test_g2p_chiyu(catalog: VoiceCatalog) -> None:
    g2p = make_g2p(catalog.get(CHIYU))
    assert g2p.phonemize("阿") == ["a"]
    assert g2p.phonemize("ba") == ["b", "a"]
    assert g2p.phonemize("你好") == ["n", "i", "h", "ao"]


def test_g2p_yousa_multilingual(catalog: VoiceCatalog) -> None:
    bank = catalog.get(f"{YOUSAV}::Bright")
    g2p = make_g2p(bank)
    assert g2p.phonemize("阿") == ["zh/a"]
    assert g2p.phonemize("ba") == ["zh/b", "zh/a"]
    assert g2p.phonemize("ni3") == ["zh/n", "zh/i"]
    assert g2p.phonemize("あ") == ["ja/a"]
    assert g2p.phonemize("きゃ") == ["ja/ky", "ja/a"]
    english = g2p.phonemize("hello")
    assert english and all(symbol.startswith("en/") for symbol in english)
    # 显式语言覆盖
    assert g2p.phonemize("ba", language="zh") == ["zh/b", "zh/a"]
    assert g2p.resolve_pinned(["zh/b", "zh/a"]) == ["zh/b", "zh/a"]
    assert g2p.resolve_pinned(["b", "a"]) == ["zh/b", "zh/a"]


def test_language_detection_and_kana() -> None:
    assert detect_language("阿") == "zh"
    assert detect_language("あ") == "ja"
    assert detect_language("hello") == "en"
    assert detect_language("") == "auto"
    assert kana_to_romaji("あか") == ["a", "ka"]
    assert kana_to_romaji("きゃ") == ["kya"]
    assert kana_to_romaji("アカ") == ["a", "ka"]


# --------------------------------------------------------------------------------------
# 旧版 dsdict.txt / 合成字典
# --------------------------------------------------------------------------------------
def test_legacy_dsdict_txt(tmp_path: Path) -> None:
    bank_dir = tmp_path / "Legacy"
    bank_dir.mkdir()
    (bank_dir / "dsdict.txt").write_text("ba\tb a\nchi\tch ir\n", encoding="utf-8")
    symbols_dir = tmp_path / "dsdur"
    symbols_dir.mkdir()
    (symbols_dir / "dsdict.yaml").write_text(
        "symbols:\n"
        "- symbol: SP\n"
        "  type: vowel\n"
        "- symbol: b\n"
        "  type: fricative\n"
        "- symbol: a\n"
        "  type: vowel\n"
        "- symbol: ch\n"
        "  type: fricative\n"
        "- symbol: ir\n"
        "  type: vowel\n",
        encoding="utf-8",
    )
    dictionary = dictionary_module.build_dictionary(
        bank_dir, [symbols_dir], use_lang_id=False
    )
    assert dictionary.loaded is False
    assert dictionary.query("ba") == ["b", "a"]
    assert dictionary.query("chi") == ["ch", "ir"]
    assert dictionary.is_valid("a") is True
    assert dictionary.is_vowel("a") is True
    assert dictionary.is_valid("b") is True
    assert dictionary.is_vowel("b") is False
    assert dictionary.warnings == []


def test_yaml_dictionary_replacements(tmp_path: Path) -> None:
    path = tmp_path / "dsdict.yaml"
    path.write_text(
        "symbols:\n"
        "- symbol: SP\n"
        "  type: vowel\n"
        "- symbol: a\n"
        "  type: vowel\n"
        "- symbol: b\n"
        "  type: fricative\n"
        "- symbol: l\n"
        "  type: liquid\n"
        "entries:\n"
        "- grapheme: ba\n"
        "  phonemes:\n"
        "  - b\n"
        "  - a\n"
        "- grapheme: '阿'\n"
        "  phonemes: [a]\n"
        "replacements:\n"
        "- from: b\n"
        "  to: l\n",
        encoding="utf-8",
    )
    dictionary = dictionary_module.build_dictionary(tmp_path)
    assert dictionary.query("ba") == ["l", "a"]  # replacements 生效
    assert dictionary.query("阿") == ["a"]  # 引号 + flow list
    assert dictionary.is_glide("l") is True
    assert dictionary.is_vowel("l") is False


def test_flow_style_yaml_falls_back_to_pyyaml(tmp_path: Path) -> None:
    path = tmp_path / "dsdict.yaml"
    path.write_text(
        "symbols: [{symbol: a, type: vowel}, {symbol: b, type: fricative}]\n"
        "entries: [{grapheme: ba, phonemes: [b, a]}]\n",
        encoding="utf-8",
    )
    dictionary = dictionary_module.build_dictionary(tmp_path)
    assert dictionary.query("ba") == ["b", "a"]
    assert dictionary.is_vowel("a") is True


# --------------------------------------------------------------------------------------
# 扫描规则（合成目录）
# --------------------------------------------------------------------------------------
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_synthetic_bank(root: Path, name: str = "TestBank") -> Path:
    bank = root / name
    bank.mkdir(parents=True, exist_ok=True)
    _write(bank / "character.txt", "name=Test Bank\nauthor=tester\nversion=1.0\n")
    _write(
        bank / "character.yaml",
        "singer_type: diffsinger\ndefault_phonemizer: OpenUtau.Core.DiffSinger.DiffSingerChinesePhonemizer\n",
    )
    _write(
        bank / "dsconfig.yaml",
        "phonemes: phonemes.txt\n"
        "acoustic: acoustic.onnx\n"
        "vocoder: test_vocoder\n"
        "sample_rate: 44100\n"
        "hop_size: 512\n"
        "win_size: 2048\n"
        "fft_size: 2048\n"
        "num_mel_bins: 128\n",
    )
    _write(bank / "phonemes.txt", "SP\nAP\na\nb\nl\nm\n")
    _write(
        bank / "dsdict.yaml",
        "symbols:\n"
        "- symbol: SP\n"
        "  type: vowel\n"
        "- symbol: a\n"
        "  type: vowel\n"
        "- symbol: b\n"
        "  type: fricative\n"
        "entries:\n"
        "- grapheme: ba\n"
        "  phonemes:\n"
        "  - b\n"
        "  - a\n",
    )
    (bank / "acoustic.onnx").write_bytes(b"")
    _write(
        bank / "dsvocoder" / "vocoder.yaml",
        "name: test_vocoder\nmodel: vocoder.onnx\nsample_rate: 44100\nhop_size: 512\nnum_mel_bins: 128\n",
    )
    (bank / "dsvocoder" / "vocoder.onnx").write_bytes(b"")
    return bank


def test_synthetic_scan_rules(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    root.mkdir()
    good = _make_synthetic_bank(root)

    # 嵌套 dsdur 不能被当成声库；嵌套目录也不能被单独扫出来
    _write(
        good / "dsdur" / "dsconfig.yaml",
        "phonemes: phonemes.txt\nlinguistic: linguistic.onnx\ndur: dur.onnx\n",
    )
    _write(good / "dsdur" / "phonemes.txt", "SP\na\nb\n")
    (good / "dsdur" / "linguistic.onnx").write_bytes(b"")
    (good / "dsdur" / "dur.onnx").write_bytes(b"")
    # 声库内部的嵌套目录不能被单独扫出来
    _make_synthetic_bank(good / "nested", "InnerBank")

    # 坏声库：缺 acoustic，不应导致整体失败
    broken = _make_synthetic_bank(root, "BrokenBank")
    _write(broken / "dsconfig.yaml", "phonemes: phonemes.txt\nvocoder: test_vocoder\n")

    # 深度 2 的声库
    _make_synthetic_bank(root / "sub", "NestedBank")

    # 顶层符号链接
    external = tmp_path / "external" / "LinkedBank"
    _make_synthetic_bank(tmp_path / "external", "LinkedBank")
    os.symlink(external, root / "LinkedBank")

    catalog = VoiceCatalog([root], [])
    catalog.refresh()
    ids = {bank.id for bank in catalog.voices()}
    assert "TestBank" in ids
    assert "sub/NestedBank" in ids
    assert "LinkedBank" in ids
    assert not any("InnerBank" in voice_id for voice_id in ids)
    assert not any("BrokenBank" in voice_id for voice_id in ids)
    assert any("BrokenBank" in message for message in catalog.warnings)

    test_bank = catalog.get("TestBank")
    assert test_bank.dur is not None and test_bank.dur.available
    assert test_bank.vocoder is not None
    assert test_bank.vocoder.model_path and test_bank.vocoder.model_path.is_file()
    assert test_bank.config.sample_rate == 44100
    assert test_bank.dictionary is not None and test_bank.dictionary.loaded is False
    assert test_bank.dictionary.query("ba") == ["b", "a"]

    # max_depth=1 时不应发现 sub/NestedBank
    shallow = VoiceCatalog([root], [], max_depth=1)
    shallow.refresh()
    shallow_ids = {bank.id for bank in shallow.voices()}
    assert "TestBank" in shallow_ids
    assert "sub/NestedBank" not in shallow_ids

    # refresh 能看到新建的声库
    _make_synthetic_bank(root, "NewBank")
    catalog.refresh()
    assert "NewBank" in {bank.id for bank in catalog.voices()}


def test_config_parser_character_and_vocoder(tmp_path: Path) -> None:
    directory = tmp_path / "Bank"
    directory.mkdir()
    _write(
        directory / "character.txt",
        "name=My Singer\nauthor=Me\nversion=2.0\nimage=avatar.png\n",
    )
    _write(
        directory / "character.yaml",
        "singer_type: diffsinger\n"
        "default_phonemizer: OpenUtau.Core.DiffSinger.DiffSingerJapanesePhonemizer\n"
        "subbanks:\n"
        "- color: soft\n"
        "  suffix: Soft\n"
        "  prefix: ''\n"
        "  tone_ranges:\n"
        "  - C1-B7\n",
    )
    character = config_parser.load_character(directory)
    assert character.name == "My Singer"
    assert character.author == "Me"
    assert character.version == "2.0"
    assert character.default_phonemizer.endswith("DiffSingerJapanesePhonemizer")
    assert config_parser.language_from_phonemizer(character.default_phonemizer) == "ja"
    assert len(character.subbanks) == 1
    assert character.subbanks[0].color == "soft"
    assert character.subbanks[0].suffix == "Soft"

    _write(
        directory / "vocoder.yaml",
        "name: demo_vocoder\nmodel: model.onnx\nsample_rate: 48000\nhop_size: 256\n"
        "num_mel_bins: 128\nmel_base: e\nmel_scale: htk\npitch_controllable: true\n",
    )
    (directory / "model.onnx").write_bytes(b"")
    vocoder = config_parser.parse_vocoder_yaml(directory / "vocoder.yaml")
    assert vocoder.name == "demo_vocoder"
    assert vocoder.sample_rate == 48000
    assert vocoder.hop_size == 256
    assert vocoder.mel_base == "e"
    assert vocoder.mel_scale == "htk"
    assert vocoder.pitch_controllable is True
    assert vocoder.model_path == (directory / "model.onnx").resolve()

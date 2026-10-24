"""后端能力 schema：属性面板、可画线轨、回显轨、自动音高声明。

pitch.auto = true：服务端具备自动音高能力（DiffSinger dspitch 预测）。
part 属性 PitchMode 决定每个 part 用哪种音高：
  - predict（默认）：声库有 dspitch 时预测 F0（叠加用户画线的相对偏差），响应的 pitch
    回显实际渲染 F0，编辑器显示为只读合成音高；没有 dspitch 的声库自动回退 TuneLab 曲线；
  - tunelab：始终按 TuneLab 已解析的 pitch 曲线渲染（旧行为）。
参数映射见 engine/inference.py。
"""
from __future__ import annotations

from typing import Any

SCHEMA: dict[str, Any] = {
    # 具备自动音高能力（dspitch）。具体用 predict 还是 tunelab 由 part 属性 PitchMode 决定。
    "pitch": {"auto": True},
    "partProperties": [
        {
            "id": "PitchMode",
            "label": "Pitch mode 音高模式",
            "type": "combo",
            "default": "predict",
            "options": [
                {"value": "predict", "label": "Auto 自动音高（dspitch）"},
                {"value": "tunelab", "label": "TuneLab 跟随画线"},
            ],
        },
        {
            "id": "language",
            "label": "Language 语言",
            "type": "combo",
            "default": "auto",
            "options": [
                {"value": "auto", "label": "auto（按歌词判断）"},
                {"value": "zh", "label": "mandarin 中文"},
                {"value": "ja", "label": "japanese 日文"},
                {"value": "en", "label": "english 英文"},
            ],
        },
    ],
    "noteProperties": [
        {
            "id": "phoneme",
            "label": "Phonemes 音素覆盖",
            "type": "text",
            "default": "",
        },
    ],
    "automations": [
        {"id": "DYN", "label": "Dynamics 力度", "min": -1, "max": 1, "default": 0, "color": "#73ACE5"},
        {"id": "BRE", "label": "Breathiness 气声", "min": -1, "max": 1, "default": 0, "color": "#8ADF73"},
        {"id": "TEN", "label": "Tension 张力", "min": -1, "max": 1, "default": 0, "color": "#E58A73"},
        {"id": "VOI", "label": "Voicing 声化", "min": -1, "max": 1, "default": 0, "color": "#B073E5"},
        {"id": "VEL", "label": "Velocity 速度", "min": 0, "max": 200, "default": 100, "color": "#E5D373"},
    ],
    "synthesizedParameters": [
        {"id": "energy", "label": "Energy 能量", "min": 0, "max": 100, "color": "#8ADF73"},
    ],
}


def get_schema() -> dict[str, Any]:
    return SCHEMA

"""OpenUTAU 声库层（发现、dsconfig/YAML 解析、G2P、音素定时）。

由 voicebank 负责人实现：
  - types.py:      跨层共享数据契约（lead 维护，勿重命名已有字段）
  - config.py:     dsconfig.yaml / vocoder.yaml / character.yaml 解析
  - discovery.py:  扫描 OpenUTAU/ 目录，构建声库目录
  - dictionary.py: dsdict*.yaml / dsdict.txt 与音素类型
  - g2p.py:        汉字/假名/英文等歌词转音素符号
  - phonemizer.py: 歌词转带绝对时间的 Phone 序列（含 dsdur 时长预测）
"""
from . import types  # noqa: F401  （保证契约模块最先可导入）

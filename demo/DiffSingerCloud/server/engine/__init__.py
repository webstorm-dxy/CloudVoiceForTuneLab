"""ONNX 推理层（后端选择、模型会话、DiffSinger 渲染管线）。

由 engine 负责人实现：
  - backends.py: CPU / CUDA / Apple GPU（MPS，走 ONNX Runtime CoreML EP）的 OnnxRunner
  - speaker.py:  speaker embedding（.emb）
  - inference.py: 声学 / 方差 / 声码器推理管线（DiffSingerRenderer）
"""

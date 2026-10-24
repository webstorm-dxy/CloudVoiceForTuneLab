"""ONNX Runtime 后端选择与会话池。

后端语义（与 OpenUTAU 的 Onnx.cs / Preferences.OnnxRunner 对齐）：

- ``cpu``  : ``CPUExecutionProvider``。
- ``cuda`` : ``CUDAExecutionProvider``（Linux/Windows，需要 ``onnxruntime-gpu`` 与 CUDA/cuDNN）。
- ``mps``  : Apple Silicon GPU。**ONNX Runtime 没有原生 MPS ExecutionProvider**，所以 ``mps``
  在这里映射成 ``CoreMLExecutionProvider``（``MLComputeUnits=CPUAndGPU`` + ``EnableOnSubgraphs``），
  这与 OpenUTAU 在 macOS 上的 ``OnnxRunner=coreml``（``AppendExecutionProvider_CoreML``）一致：
  ORT 的 CoreML EP 会把可支持的算子放到 Apple GPU/ANE 上，其余回落到 CPU。
  macOS 官方 onnxruntime wheel 自带 CoreML EP，无需额外安装。
- ``auto`` : 有 CUDA EP 用 ``cuda``；否则 ``cpu``。Apple Silicon 上 CoreML EP 虽然可用，
  但实测真实 DiffSinger 模型在 CoreML 上可能比 CPU 慢（见 README 实测），所以 auto 不再
  自动选 mps；需要时显式 ``--backend mps``（仍带运行时 CPU 回退）。

显式指定不可用的后端时 ``resolve_backend`` 直接抛 ``RuntimeError``（服务启动即失败，错误信息
包含可用 provider 与安装提示）；``auto`` 永不因后端不可用而失败。

``OnnxRunnerImpl`` 实现 ``server.voicebank.types.OnnxRunner`` 协议：按模型路径 LRU 缓存
``ort.InferenceSession``，线程安全；``run()`` 按 session 声明的输入签名做 dtype / 形状转换
（缺 batch 维自动补、rank-0 标量自动压平），输入名缺失/多余时抛 ``ValueError``（对齐
OpenUTAU ``Onnx.VerifyInputNames``）。若 GPU/加速 EP 建 session 失败会回退 CPU 并把实际
provider 写回 ``BackendInfo``（供 ``/health`` 暴露），不会让服务整体不可用。

注意区分三种回退（日志前缀不同）：

1. 建 session 失败 -> CPU（本模块 _record_runtime_fallback，/health 的 resolved 会变成 cpu）；
2. 运行时执行失败 -> 该模型 CPU-only 重试（本模块 _degrade_to_cpu）：session 能建、但某个
   算子/动态 shape 在 CoreML 等加速 EP 上执行报错时，为这个模型路径重建
   CPUExecutionProvider session、重试一次；成功则标记该模型 cpu-only（后续 run 不再反复
   失败），并在 BackendInfo.detail 记录 'runtime-fell back to CPU for <model>'。
   /health 的 resolved 仍是 mps（只是这个模型走 CPU），可通过 runner.cpu_fallback_models
   查看。典型触发: CoreML EP 的 'Error in dynamically resizing for sequence length'。
3. inference 层的 variance/linguistic 跨目录找可加载模型（目录文件损坏），见 inference.py
   docstring 的'模型可加载性兜底'。
"""
from __future__ import annotations

import gc
import logging
import platform
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

#: 对外暴露的 --backend 取值。
BACKENDS = ("auto", "cpu", "cuda", "mps")

#: 常见别名（--backend coreml 等价于 mps；gpu 交给 auto 嗅探）。
_BACKEND_ALIASES = {
    "coreml": "mps",
    "apple": "mps",
    "metal": "mps",
    "gpu": "auto",
    "default": "auto",
}

_CPU_EP = "CPUExecutionProvider"
_CUDA_EP = "CUDAExecutionProvider"
_COREML_EP = "CoreMLExecutionProvider"

#: ONNX Runtime 的输入类型字符串 -> numpy dtype。
_ORT_DTYPES: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(bool)": np.bool_,
}

_ort_module: Any = None
_ort_import_error: Exception | None = None


def load_onnxruntime() -> Any:
    """懒加载 onnxruntime；不可用时给出清晰的安装提示。"""
    global _ort_module, _ort_import_error
    if _ort_module is not None:
        return _ort_module
    try:
        import onnxruntime as ort  # noqa: PLC0415 - 故意懒加载，便于无 ORT 时导入本模块
    except Exception as error:  # pragma: no cover - venv 里 always 可用
        _ort_import_error = error
        raise RuntimeError(
            "onnxruntime is required for the DiffSinger engine; "
            "install it with 'pip install onnxruntime' (Linux/Windows GPU: onnxruntime-gpu)"
        ) from error
    _ort_module = ort
    return ort


# --------------------------------------------------------------------------------------
# 后端解析
# --------------------------------------------------------------------------------------
@dataclass
class BackendInfo:
    """--backend 的解析结果（HTTP ``/health`` 直接把它转成 dict 暴露）。"""

    requested: str
    resolved: str
    providers: list[str]
    detail: str = ""
    # 以下是构造 runner 时需要、但不属于线上协议契约的附加信息。
    device_id: int = 0
    threads: int = 0

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "providers": list(self.providers),
            "detail": self.detail,
        }

    @property
    def is_gpu(self) -> bool:
        """是否用到了非 CPU 的加速提供者。"""
        return any(provider != _CPU_EP for provider in self.providers)


def _normalize_requested(requested: str | None) -> str:
    raw = ("" if requested is None else str(requested)).strip().lower()
    if not raw:
        raw = "auto"
    normalized = _BACKEND_ALIASES.get(raw, raw)
    if normalized not in BACKENDS:
        raise ValueError(
            "backend must be one of " + ", ".join(BACKENDS) + " (got '" + str(requested) + "')"
        )
    return raw


def resolve_backend(
    requested: str = "auto",
    *,
    device_id: int = 0,
    threads: int = 0,
    available_providers: Sequence[str] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> BackendInfo:
    """解析 ``--backend``；显式指定不可用后端时抛 ``RuntimeError``，``auto`` 自动降级。

    ``available_providers`` / ``system`` / ``machine`` 只是给单元测试注入环境用的，
    正常运行不需要传（``available_providers`` 默认取 ``onnxruntime.get_available_providers()``）。
    """
    raw = _normalize_requested(requested)
    resolved_request = _BACKEND_ALIASES.get(raw, raw)
    ort = load_onnxruntime()
    if available_providers is None:
        available = [str(item) for item in ort.get_available_providers()]
    else:
        available = [str(item) for item in available_providers]
    available_set = set(available)
    system_name = (system or platform.system()).strip().lower()
    machine_name = (machine or platform.machine()).strip().lower()
    is_apple_silicon = system_name == "darwin" and machine_name in ("arm64", "aarch64")
    available_text = ", ".join(available) or "(none)"

    if resolved_request == "cpu":
        if _CPU_EP not in available_set:
            raise RuntimeError(
                "backend 'cpu' requires CPUExecutionProvider, but onnxruntime only offers: "
                + available_text
            )
        return BackendInfo(
            requested=raw,
            resolved="cpu",
            providers=[_CPU_EP],
            detail="CPUExecutionProvider (onnxruntime default); available: " + available_text,
            device_id=int(device_id or 0),
            threads=int(threads or 0),
        )

    if resolved_request == "cuda":
        if _CUDA_EP not in available_set:
            raise RuntimeError(
                "backend 'cuda' is not available: onnxruntime was built without CUDAExecutionProvider. "
                "Install onnxruntime-gpu (Linux/Windows) matching your CUDA/cuDNN; available providers: "
                + available_text
                + ". On macOS use --backend mps or --backend cpu."
            )
        return BackendInfo(
            requested=raw,
            resolved="cuda",
            providers=[_CUDA_EP, _CPU_EP],
            detail=(
                "CUDAExecutionProvider (device_id="
                + str(int(device_id or 0))
                + ") with CPU fallback"
            ),
            device_id=int(device_id or 0),
            threads=int(threads or 0),
        )

    if resolved_request == "mps":
        if system_name != "darwin":
            raise RuntimeError(
                "backend 'mps' (Apple GPU) is only available on macOS; detected platform '"
                + (system_name or "unknown")
                + "'. Use --backend cuda on Linux/Windows (onnxruntime-gpu) or --backend cpu."
            )
        if _COREML_EP not in available_set:
            raise RuntimeError(
                "backend 'mps' requires the CoreML execution provider (ONNX Runtime has no native "
                "MPS EP; OpenUTAU's 'coreml' runner uses CoreML EP too), but this onnxruntime build "
                "only offers: " + available_text + ". Install an official macOS onnxruntime wheel."
            )
        return BackendInfo(
            requested=raw,
            resolved="mps",
            providers=[_COREML_EP, _CPU_EP],
            detail=(
                "Apple GPU via CoreMLExecutionProvider (ONNX Runtime has no native MPS EP; "
                "OpenUTAU's coreml runner uses the same provider), MLComputeUnits=CPUAndGPU; "
                "unsupported ops/model instances fall back to CPU per session (and per model at "
                "runtime if CoreML cannot execute a dynamic shape). NOTE: measured on Apple Silicon, "
                "real DiffSinger models can be slower on CoreML than on CPU; compare --backend mps "
                "vs --backend cpu before choosing"
            ),
            device_id=int(device_id or 0),
            threads=int(threads or 0),
        )

    # auto：CUDA > CPU，永不失败。Apple Silicon 的 CoreML EP 实测在真实 DiffSinger 模型上
    # 可能比 CPU 慢（2s 块热 3.39s vs 2.75s、冷 6.21s vs 3.24s），所以 auto 不自动选 mps；
    # 想用 Apple GPU 请显式 --backend mps（带运行时 CPU 回退）。
    if _CUDA_EP in available_set:
        return BackendInfo(
            requested=raw,
            resolved="cuda",
            providers=[_CUDA_EP, _CPU_EP],
            detail="auto->cuda: CUDAExecutionProvider is available (device_id=" + str(int(device_id or 0)) + ")",
            device_id=int(device_id or 0),
            threads=int(threads or 0),
        )
    cpu_detail = "auto->cpu: no CUDA execution provider"
    if is_apple_silicon and _COREML_EP in available_set:
        cpu_detail += (
            " (Apple Silicon: CoreML EP is available and maps to --backend mps, but measured "
            "DiffSinger inference on CoreML can be slower than CPU, so auto prefers cpu; "
            "use --backend mps to force Apple GPU)"
        )
    else:
        cpu_detail += "; available: " + available_text
    return BackendInfo(
        requested=raw,
        resolved="cpu",
        providers=[_CPU_EP],
        detail=cpu_detail,
        device_id=int(device_id or 0),
        threads=int(threads or 0),
    )


# --------------------------------------------------------------------------------------
# 会话池
# --------------------------------------------------------------------------------------
class _SessionEntry:
    """一个已加载的 InferenceSession 及其元数据。"""

    __slots__ = (
        "session", "path", "input_names", "input_specs", "output_names", "refs", "stamp",
        "providers", "cpu_only", "fallback_reason",
    )

    def __init__(self, session: Any, path: Path) -> None:
        self.session = session
        self.path = path
        self.input_names = [node.name for node in session.get_inputs()]
        self.input_specs = {
            node.name: (tuple(node.shape or ()), str(node.type)) for node in session.get_inputs()
        }
        self.output_names = [node.name for node in session.get_outputs()]
        self.providers = [str(provider) for provider in session.get_providers()]
        self.refs = 0
        self.stamp = 0
        # 运行时（session.run）失败后是否已被降级为 CPU-only session。
        self.cpu_only = False
        self.fallback_reason: str | None = None


def _model_key(model_path: Path | str) -> str:
    return str(Path(model_path).expanduser().resolve())


def _provider_label(providers: Sequence[Any]) -> str:
    return ", ".join(item[0] if isinstance(item, tuple) else str(item) for item in providers)


def _convert_input(value: Any, declared_shape: Sequence[Any], type_string: str, *, path: Path, name: str) -> np.ndarray:
    """把 list/np.ndarray/scalar 转成 session 需要的 dtype 与 rank。"""
    dtype = _ORT_DTYPES.get(type_string)
    if dtype is None:
        raise ValueError(
            "ONNX model " + str(path) + " input '" + name + "' has unsupported dtype '" + type_string + "'"
        )
    array = np.asarray(value)
    declared = tuple(declared_shape)
    if len(declared) == 0:
        # rank-0 标量输入（DiffSinger 的 depth / steps / speedup 都是这种）。
        flat = array.reshape(-1)
        if flat.size != 1:
            raise ValueError(
                "ONNX model " + str(path) + " input '" + name + "' is a scalar but got array shape "
                + str(array.shape)
            )
        # 注意 np.ascontiguousarray 会把 0-d 数组变成 shape (1,)，这里用 np.array 保持 rank-0。
        return np.array(flat[0], dtype=dtype)
    # 允许调用方省略前导 batch 维（模型声明 [1, n] 时传 (n,)）。
    if array.ndim == len(declared) - 1:
        first = declared[0]
        if first == 1 or not isinstance(first, int):
            array = array.reshape((1,) + array.shape)
    if array.ndim != len(declared):
        raise ValueError(
            "ONNX model " + str(path) + " input '" + name + "' expects rank "
            + str(len(declared)) + " shape " + str(list(declared)) + ", got shape " + str(array.shape)
        )
    for axis, (dim, size) in enumerate(zip(declared, array.shape)):
        if isinstance(dim, int) and dim >= 0 and dim != size:
            raise ValueError(
                "ONNX model " + str(path) + " input '" + name + "' expects shape "
                + str(list(declared)) + " (axis " + str(axis) + "=" + str(dim) + "), got shape "
                + str(array.shape)
            )
    return np.ascontiguousarray(array.astype(dtype, copy=False))


class OnnxRunnerImpl:
    """``types.OnnxRunner`` 的 ONNX Runtime 实现（按模型路径 LRU 缓存 session）。

    同一个 runner 实例被声库层（dsdur / dsvariance）与渲染层（acoustic / vocoder）共用；
    ``max_cached_sessions`` 是软上限：正在被 ``run()`` 使用的 session 不会被关闭。
    """

    def __init__(
        self,
        backend: BackendInfo,
        *,
        max_cached_sessions: int = 12,
        threads: int = 0,
        optimization_level: str = "all",
        log_severity: int = 3,
    ) -> None:
        self.backend = backend
        self.max_cached_sessions = max(1, int(max_cached_sessions or 1))
        self.threads = max(0, int(threads or 0))
        self.optimization_level = str(optimization_level or "all").lower()
        self.log_severity = int(log_severity)
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, _SessionEntry]" = OrderedDict()
        self._stamp = 0
        self._closed = False
        self._fallback_note: str | None = None
        # 运行时降级到 CPU 的模型（path -> 原因）；/health 的 BackendInfo.detail 也会记录。
        self._cpu_only_models: dict[str, str] = {}

    # ---- 元信息 ----------------------------------------------------------------------
    @property
    def effective_providers(self) -> list[str]:
        return list(self.backend.providers)

    @property
    def fallback_note(self) -> str | None:
        """GPU/加速 provider 建 session 失败回退 CPU 时的说明（None = 未回退）。"""
        return self._fallback_note

    @property
    def cpu_fallback_models(self) -> dict[str, str]:
        """运行时（session.run 失败）被降级为 CPU-only 的模型路径 -> 原因。"""
        with self._lock:
            return dict(self._cpu_only_models)

    def cached_model_paths(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    # ---- types.OnnxRunner ------------------------------------------------------------
    def input_names(self, model_path: Path | str) -> list[str]:
        return list(self._entry(model_path).input_names)

    def output_names(self, model_path: Path | str) -> list[str]:
        return list(self._entry(model_path).output_names)

    def input_shapes(self, model_path: Path | str) -> dict[str, tuple]:
        """返回 {输入名: 声明 shape}，便于 inference 层按实际签名组装可选输入。"""
        return {name: spec[0] for name, spec in self._entry(model_path).input_specs.items()}

    def run(self, model_path: Path | str, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = Path(model_path)
        entry, key = self._acquire(path)
        try:
            prepared = self._prepare_inputs(entry, inputs)
            try:
                outputs = entry.session.run(None, prepared)
            except Exception as error:  # noqa: BLE001 - 运行时（算子执行）失败：尝试 CPU 降级
                replacement = self._degrade_to_cpu(key, path, entry, error)
                if replacement is None:
                    raise RuntimeError(
                        "ONNX inference failed for " + str(entry.path) + ": "
                        + type(error).__name__ + ": " + str(error)
                    ) from error
                entry = replacement
                prepared = self._prepare_inputs(entry, inputs)
                try:
                    outputs = entry.session.run(None, prepared)
                except Exception as second:  # noqa: BLE001 - CPU 也失败则抛清晰错误
                    raise RuntimeError(
                        "ONNX inference failed for " + str(entry.path)
                        + " even after CPU fallback: " + type(second).__name__ + ": " + str(second)
                    ) from second
        finally:
            self._release(key, entry)
        return {name: outputs[index] for index, name in enumerate(entry.output_names)}

    def close(self) -> None:
        """关闭所有缓存的 session（服务退出 / 测试释放内存时调用）。"""
        with self._lock:
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            try:
                entry.session = None
            except Exception:  # noqa: BLE001 - 关闭失败不影响退出
                pass
        gc.collect()

    def __enter__(self) -> "OnnxRunnerImpl":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - contextlib 签名
        self.close()

    # ---- 内部：session 生命周期 ------------------------------------------------------
    def _session_options(self) -> Any:
        ort = load_onnxruntime()
        options = ort.SessionOptions()
        levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        options.graph_optimization_level = levels.get(
            self.optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if self.threads > 0:
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
        options.log_severity_level = self.log_severity
        return options

    def _provider_attempts(self) -> list[list[Any]]:
        """按优先级给出 provider 列表尝试序列（最后一个总会回退 CPU）。"""
        resolved = self.backend.resolved
        device_id = int(getattr(self.backend, "device_id", 0) or 0)
        if resolved == "cuda":
            return [
                [(_CUDA_EP, {"device_id": str(device_id)}), _CPU_EP],
                [_CUDA_EP, _CPU_EP],
                [_CPU_EP],
            ]
        if resolved == "mps":
            # 首选"普通 CoreML 分区"（不放宽到 MLProgram/on-subgraph）：实测默认 CoreML 分区
            # 首跑编译开销小（0.2~1s），而 MLProgram + EnableOnSubgraphs 每次新 shape 首跑要编译
            # 5~6s（之后很快）。云端短块/块长变化场景下首块延迟更重要，所以放在第一优先；
            # 若默认分区建 session 失败再尝试 OpenUTAU 风格的 on-subgraph 配置，最后回退 CPU。
            return [
                [(_COREML_EP, {"MLComputeUnits": "CPUAndGPU"}), _CPU_EP],
                [
                    (
                        _COREML_EP,
                        {
                            "ModelFormat": "MLProgram",
                            "MLComputeUnits": "CPUAndGPU",
                            "EnableOnSubgraphs": "1",
                        },
                    ),
                    _CPU_EP,
                ],
                [_CPU_EP],
            ]
        return [[_CPU_EP]]

    def _entry(self, model_path: Path | str) -> _SessionEntry:
        key = _model_key(model_path)
        with self._lock:
            if self._closed:
                raise RuntimeError("OnnxRunnerImpl is closed")
            entry = self._entries.get(key)
            if entry is None:
                entry = self._create_entry(Path(model_path), key)
                self._entries[key] = entry
            self._touch(entry)
            self._trim(key)
            return entry

    def _acquire(self, model_path: Path | str) -> tuple[_SessionEntry, str]:
        key = _model_key(model_path)
        with self._lock:
            if self._closed:
                raise RuntimeError("OnnxRunnerImpl is closed")
            entry = self._entries.get(key)
            if entry is None:
                entry = self._create_entry(Path(model_path), key)
                self._entries[key] = entry
            self._touch(entry)
            entry.refs += 1
            return entry, key

    def _release(self, key: str, entry: _SessionEntry) -> None:
        with self._lock:
            entry.refs = max(0, entry.refs - 1)
            self._trim(key)

    def _touch(self, entry: _SessionEntry) -> None:
        self._stamp += 1
        entry.stamp = self._stamp
        try:
            self._entries.move_to_end(str(entry.path.resolve()))
        except KeyError:
            pass

    def _trim(self, keep_key: str) -> None:
        while len(self._entries) > self.max_cached_sessions:
            victim: _SessionEntry | None = None
            victim_key = ""
            for entry_key, entry in self._entries.items():
                if entry_key == keep_key or entry.refs > 0:
                    continue
                victim, victim_key = entry, entry_key
                break
            if victim is None:
                return  # 全都在用：暂时超限，等 _release 再回收
            self._entries.pop(victim_key, None)
            victim.session = None

    def _record_runtime_fallback(self, path: Path, requested: list[Any], error: Exception | None) -> None:
        note = (
            "runtime fallback to CPUExecutionProvider: failed to create session with "
            + _provider_label(requested)
            + " for "
            + path.name
            + (": " + str(error) if error is not None else "")
        )
        LOG.warning("%s", note)
        self._fallback_note = note
        self.backend.detail = (self.backend.detail + " | " if self.backend.detail else "") + note
        # 只保留实际可用的 provider，/health 会显示回退后的真实状态。
        self.backend.resolved = "cpu"
        self.backend.providers = [_CPU_EP]

    def _new_session(self, path: Path, providers: Sequence[Any]) -> Any:
        ort = load_onnxruntime()
        return ort.InferenceSession(
            str(path), sess_options=self._session_options(), providers=list(providers)
        )

    def _degrade_to_cpu(
        self,
        key: str,
        path: Path,
        failed_entry: _SessionEntry,
        error: Exception,
    ) -> _SessionEntry | None:
        """session.run 在加速 provider 上失败时，为该模型重建 CPU-only session 并返回（已 acquire）。

        与"建 session 失败回退 CPU"不同：这里 session 建得起来，是执行特定算子/动态 shape 时
        才失败（典型: CoreML EP 的动态 resize）。策略：
        - 只对该模型路径降级一次并缓存 CPU-only entry，后续 run 不再反复失败；
        - 旧 entry 从缓存移除、交给其它仍在使用的线程释放后 GC（refs 归零）；
        - 并发安全：整个替换过程持锁；若别的线程已经替换过，复用其 CPU-only entry。
        返回 None 表示"无法/无需降级"（调用方抛原错误）。
        """
        with self._lock:
            if self._closed:
                return None
            current = self._entries.get(key)
            if current is not None and current is not failed_entry:
                # 已经有别的线程替换过：只有 CPU-only 才值得复用。
                if not current.cpu_only:
                    return None
                current.refs += 1
                return current
            if failed_entry.cpu_only:
                return None
            if not any(provider != _CPU_EP for provider in failed_entry.providers):
                return None  # 本来就是 CPU session，降级没有意义
            try:
                session = self._new_session(path, [_CPU_EP])
            except Exception as cpu_error:  # noqa: BLE001 - CPU 也建不起来，只能抛原错误
                LOG.warning(
                    "runtime CPU fallback for %s failed while creating CPU session: %s",
                    path, cpu_error,
                )
                return None
            replacement = _SessionEntry(session, path)
            replacement.cpu_only = True
            replacement.fallback_reason = str(error)
            # 归还本次对失败 entry 的 acquire（其它线程仍在用时由其自己 release）。
            failed_entry.refs = max(0, failed_entry.refs - 1)
            self._entries[key] = replacement
            replacement.refs = 1
            note = (
                "runtime-fell back to CPU for " + path.name + " (model marked CPU-only): "
                + type(error).__name__ + ": " + str(error)
            )
            LOG.warning("%s", note)
            self._fallback_note = note
            self._cpu_only_models[str(path.resolve())] = str(error)
            self.backend.detail = (self.backend.detail + " | " if self.backend.detail else "") + note
            return replacement

    def _create_entry(self, path: Path, key: str) -> _SessionEntry:
        if not path.is_file():
            raise FileNotFoundError("ONNX model not found: " + str(path))
        attempts = self._provider_attempts()
        last_error: Exception | None = None
        for index, providers in enumerate(attempts):
            try:
                session = self._new_session(path, providers)
            except Exception as error:  # noqa: BLE001 - 逐个尝试不同 provider 组合
                last_error = error
                LOG.warning(
                    "failed to create ONNX session for %s with %s: %s",
                    path,
                    _provider_label(providers),
                    error,
                )
                continue
            if index > 0:
                actual = [str(item) for item in session.get_providers()]
                if actual == [_CPU_EP] and self.backend.resolved != "cpu":
                    self._record_runtime_fallback(path, list(providers), last_error)
                else:
                    note = (
                        "accelerated provider options simplified for "
                        + path.name
                        + " after: "
                        + (str(last_error) if last_error is not None else "unknown error")
                    )
                    LOG.warning("%s", note)
                    self._fallback_note = note
                    self.backend.detail = (self.backend.detail + " | " if self.backend.detail else "") + note
            entry = _SessionEntry(session, path)
            LOG.info(
                "loaded ONNX model %s providers=%s inputs=%s",
                path.name,
                list(session.get_providers()),
                ", ".join(entry.input_names),
            )
            return entry
        raise RuntimeError(
            "failed to create ONNX inference session for " + str(path) + ": " + str(last_error)
        ) from last_error

    # ---- 内部：输入转换 --------------------------------------------------------------
    def _prepare_inputs(self, entry: _SessionEntry, inputs: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        given = dict(inputs or {})
        missing = sorted(name for name in entry.input_names if name not in given)
        if missing:
            raise ValueError(
                "Missing input(s) for the inference session (" + str(entry.path) + "): " + ", ".join(missing)
            )
        unexpected = sorted(name for name in given if name not in entry.input_specs)
        if unexpected:
            raise ValueError(
                "Unexpected input(s) for the inference session (" + str(entry.path) + "): " + ", ".join(unexpected)
            )
        prepared: dict[str, np.ndarray] = {}
        for name in entry.input_names:
            declared_shape, type_string = entry.input_specs[name]
            prepared[name] = _convert_input(
                given[name], declared_shape, type_string, path=entry.path, name=name
            )
        return prepared

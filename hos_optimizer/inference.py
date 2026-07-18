#!/usr/bin/env python3
"""
HOS-Model-Optimizer 统一推理模块

支持三种推理后端：
- llama-cpp-python：GGUF 格式，CPU+GPU 混合推理，适合低显存场景
- vLLM：PagedAttention + 连续批处理，高吞吐场景首选
- SGLang：RadixAttention + 约束生成，适合结构化输出与重复前缀加速

所有后端共享统一的推理接口和性能监控体系，针对 8GB VRAM 场景做了默认优化。

使用方法：
  # 命令行推理（默认自动选择后端）
  python inference.py --model path/to/model --prompt "你好"

  # 指定后端
  python inference.py --backend vllm --model path/to/model --prompt "你好"

  # 启动 API 服务
  python inference.py --backend vllm --model path/to/model --serve

  # 运行性能基准测试
  python inference.py --backend vllm --model path/to/model --benchmark

  # 交互模式
  python inference.py --backend sglang --model path/to/model --chat
"""

import argparse
import json
import time
import logging
import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class InferenceRequest:
    """推理请求数据结构"""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: Optional[List[str]] = None
    # SGLang 约束生成：JSON Schema
    json_schema: Optional[Dict[str, Any]] = None
    # 额外参数，透传给各后端
    extra: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResult:
    """推理结果数据结构"""
    text: str
    token_ids: List[int] = field(default_factory=list)
    prompt: str = ""
    # 性能指标
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 额外元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """性能统计数据结构"""
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    # 用于计算吞吐量的时间窗口
    wall_time_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """平均延迟（毫秒）"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def throughput_tokens_per_s(self) -> float:
        """吞吐量（tokens/秒）"""
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_completion_tokens / self.wall_time_s

    @property
    def requests_per_s(self) -> float:
        """每秒请求数"""
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_requests / self.wall_time_s

    def summary(self) -> str:
        """输出可读的性能摘要"""
        lines = [
            "=" * 50,
            "性能统计摘要",
            "=" * 50,
            f"  总请求数:          {self.total_requests}",
            f"  总 Prompt tokens:  {self.total_prompt_tokens}",
            f"  总生成 tokens:     {self.total_completion_tokens}",
            f"  总耗时:            {self.wall_time_s:.2f}s",
            f"  平均延迟:          {self.avg_latency_ms:.1f}ms",
            f"  吞吐量:            {self.throughput_tokens_per_s:.1f} tokens/s",
            f"  请求速率:          {self.requests_per_s:.2f} req/s",
            f"  峰值显存占用:      {self.peak_vram_mb:.1f} MB",
            "=" * 50,
        ]
        return "\n".join(lines)


# ============================================================
# 性能监控器
# ============================================================

class PerformanceMonitor:
    """
    性能监控器，跟踪吞吐量、延迟和显存占用。

    用法：
        monitor = PerformanceMonitor()
        monitor.start()
        # ... 执行推理 ...
        monitor.record_request(prompt_tokens=10, completion_tokens=50, latency_ms=120)
        monitor.end()
        print(monitor.stats.summary())
    """

    def __init__(self):
        self._stats = PerformanceStats()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @property
    def stats(self) -> PerformanceStats:
        return self._stats

    def start(self):
        """标记监控开始时间"""
        self._start_time = time.perf_counter()
        logger.debug("性能监控已启动")

    def end(self):
        """标记监控结束时间，计算总耗时"""
        if self._start_time is not None:
            self._end_time = time.perf_counter()
            self._stats.wall_time_s = self._end_time - self._start_time
            logger.debug(f"性能监控已停止，总耗时 {self._stats.wall_time_s:.2f}s")

    def record_request(self, prompt_tokens: int, completion_tokens: int, latency_ms: float):
        """
        记录单次推理请求的性能数据。

        Args:
            prompt_tokens: 输入 token 数量
            completion_tokens: 生成 token 数量
            latency_ms: 本次请求延迟（毫秒）
        """
        self._stats.total_requests += 1
        self._stats.total_prompt_tokens += prompt_tokens
        self._stats.total_completion_tokens += completion_tokens
        self._stats.total_latency_ms += latency_ms

    def update_peak_vram(self):
        """更新峰值显存占用（需要 PyTorch 可用且有 CUDA 设备）"""
        vram_mb = get_gpu_memory_usage_mb()
        if vram_mb > self._stats.peak_vram_mb:
            self._stats.peak_vram_mb = vram_mb

    def reset(self):
        """重置所有统计数据"""
        self._stats = PerformanceStats()
        self._start_time = None
        self._end_time = None


# ============================================================
# 工具函数
# ============================================================

def get_gpu_memory_usage_mb() -> float:
    """
    获取当前 GPU 显存占用（MB）。
    如果没有可用的 GPU 或 PyTorch 未安装，返回 0.0。
    """
    try:
        import torch
        if torch.cuda.is_available():
            # 返回所有 GPU 中已分配显存的最大值
            max_allocated = 0.0
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                if allocated > max_allocated:
                    max_allocated = allocated
            return max_allocated
    except ImportError:
        pass
    return 0.0


def get_total_gpu_memory_mb() -> float:
    """
    获取 GPU 总显存（MB）。
    如果没有可用的 GPU，返回 0.0。
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
    except (ImportError, AttributeError):
        pass
    return 0.0


def detect_best_backend() -> str:
    """
    根据当前环境自动检测最优推理后端。

    优先级：vLLM > SGLang > llama-cpp
    选择逻辑：
      1. 如果有 CUDA GPU 且显存 >= 6GB，优先使用 vLLM
      2. 如果有 CUDA GPU 但显存较小，尝试 SGLang（内存管理更灵活）
      3. 否则回退到 llama-cpp（纯 CPU 或 CPU+GPU 混合）
    """
    gpu_mem_mb = get_total_gpu_memory_mb()

    # 检查各后端是否可用
    vllm_available = _check_import("vllm")
    sglang_available = _check_import("sglang")
    llama_cpp_available = _check_import("llama_cpp")

    if gpu_mem_mb >= 6144 and vllm_available:
        logger.info(f"检测到 {gpu_mem_mb:.0f}MB VRAM，选择 vLLM 后端")
        return "vllm"
    elif gpu_mem_mb > 0 and sglang_available:
        logger.info(f"检测到 {gpu_mem_mb:.0f}MB VRAM，选择 SGLang 后端")
        return "sglang"
    elif llama_cpp_available:
        logger.info("选择 llama-cpp-python 后端（CPU/GPU 混合模式）")
        return "llama_cpp"
    elif vllm_available:
        return "vllm"
    elif sglang_available:
        return "sglang"
    else:
        raise RuntimeError(
            "没有可用的推理后端！请安装以下任一依赖：\n"
            "  pip install vllm\n"
            "  pip install sglang\n"
            "  pip install llama-cpp-python"
        )


def _check_import(module_name: str) -> bool:
    """检查模块是否可导入"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ============================================================
# 推理后端抽象基类
# ============================================================

class InferenceBackend(ABC):
    """
    推理后端抽象基类。

    所有推理后端（llama-cpp、vLLM、SGLang）都必须实现此接口，
    保证上层调用逻辑与具体后端解耦。
    """

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.monitor = PerformanceMonitor()
        self._loaded = False

    @abstractmethod
    def load(self):
        """加载模型到内存/显存"""
        pass

    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResult:
        """
        执行单次推理生成。

        Args:
            request: 推理请求参数

        Returns:
            InferenceResult: 推理结果
        """
        pass

    def generate_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        批量推理（默认逐条调用，各后端可覆写以实现真正的批处理）。

        Args:
            requests: 推理请求列表

        Returns:
            List[InferenceResult]: 推理结果列表
        """
        return [self.generate(req) for req in requests]

    @abstractmethod
    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """启动 OpenAI 兼容 API 服务"""
        pass

    @abstractmethod
    def shutdown(self):
        """释放资源、关闭引擎"""
        pass

    def get_performance_stats(self) -> PerformanceStats:
        """获取当前性能统计"""
        self.monitor.update_peak_vram()
        return self.monitor.stats


# ============================================================
# llama-cpp-python 后端
# ============================================================

class LlamaCppBackend(InferenceBackend):
    """
    llama-cpp-python 推理后端。

    特性：
    - 支持 GGUF 格式模型
    - CPU + GPU 混合推理（通过 n_gpu_layers 控制 offload 层数）
    - 低显存场景友好，可灵活分配 CPU/GPU 计算比例
    - 针对 8GB VRAM 场景自动调整 offload 层数

    针对 8GB VRAM 的优化策略：
    - 自动检测可用显存，计算可 offload 的最大层数
    - 默认使用 Q4_K_M 量化（如果模型支持）
    - 保留 1GB 显存余量给 KV cache
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        # GPU offload 层数，-1 表示全部 offload
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        # 上下文长度
        self.n_ctx = kwargs.get("n_ctx", 4096)
        # 线程数（CPU 推理时使用）
        self.n_threads = kwargs.get("n_threads", None)
        # 是否使用 mmap 加速加载
        self.use_mmap = kwargs.get("use_mmap", True)
        # 是否使用 mlock 锁定内存
        self.use_mlock = kwargs.get("use_mlock", False)
        # 批处理大小
        self.n_batch = kwargs.get("n_batch", 512)
        self._model = None

    def load(self):
        """
        加载 GGUF 模型。

        自动根据可用显存计算最优的 GPU offload 层数。
        对于 8GB VRAM 场景，会保留约 1GB 给 KV cache 和运行时开销。
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python 未安装，请运行：\n"
                "  pip install llama-cpp-python\n"
                "如需 CUDA 支持，请参考：https://github.com/abetlen/llama-cpp-python"
            )

        # 自动计算 GPU offload 层数
        if self.n_gpu_layers == -1:
            self.n_gpu_layers = self._compute_optimal_gpu_layers()
            logger.info(f"自动计算 GPU offload 层数: {self.n_gpu_layers}")

        load_kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "n_batch": self.n_batch,
            "verbose": True,
        }
        if self.n_threads is not None:
            load_kwargs["n_threads"] = self.n_threads

        logger.info(f"加载 GGUF 模型: {self.model_path}")
        logger.info(f"  GPU offload 层数: {self.n_gpu_layers}")
        logger.info(f"  上下文长度: {self.n_ctx}")
        logger.info(f"  批处理大小: {self.n_batch}")

        self._model = Llama(**load_kwargs)
        self._loaded = True
        self.monitor.update_peak_vram()
        logger.info("模型加载完成")

    def _compute_optimal_gpu_layers(self) -> int:
        """
        根据可用显存计算最优的 GPU offload 层数。

        策略：
        - 获取 GPU 总显存
        - 保留 1024MB 给 KV cache 和运行时
        - 按每层约 100MB 估算（粗略值，取决于模型大小）
        - 对于小模型（< 1B），通常可以全部 offload
        """
        total_vram_mb = get_total_gpu_memory_mb()
        if total_vram_mb <= 0:
            # 没有 GPU，全部使用 CPU
            return 0

        # 保留 1024MB 给 KV cache 和系统开销
        available_mb = total_vram_mb - 1024
        if available_mb <= 0:
            return 0

        # 尝试估算模型大小（通过文件大小粗略判断）
        try:
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        except OSError:
            # 可能是 HF hub ID 而非本地路径，假设全部 offload
            return -1

        # 如果模型能完全放入可用显存，全部 offload
        if model_size_mb <= available_mb:
            return -1  # -1 表示全部 offload

        # 否则按比例估算可 offload 的层数
        # 假设模型有 32 层（常见配置），按大小比例分配
        estimated_total_layers = 32
        offload_ratio = available_mb / model_size_mb
        optimal_layers = max(1, int(estimated_total_layers * offload_ratio))
        logger.info(
            f"显存估算: 总 VRAM={total_vram_mb:.0f}MB, "
            f"模型大小≈{model_size_mb:.0f}MB, "
            f"可 offload {optimal_layers}/{estimated_total_layers} 层"
        )
        return optimal_layers

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """使用 llama-cpp 执行单次推理"""
        if not self._loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        gen_kwargs = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop or [],
        }
        if request.extra:
            gen_kwargs.update(request.extra)

        start = time.perf_counter()
        output = self._model(**gen_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        # 记录性能数据
        self.monitor.record_request(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
        )
        self.monitor.update_peak_vram()

        return InferenceResult(
            text=text,
            prompt=request.prompt,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        llama-cpp 的批量推理。
        llama-cpp-python 原生不支持真正的并行批处理，这里逐条执行。
        """
        return [self.generate(req) for req in requests]

    def serve(self, host: str = "0.0.0.0", port: int = 8080):
        """
        启动 llama-cpp 的 OpenAI 兼容 API 服务。

        使用 llama_cpp.server 模块提供 HTTP 服务。
        """
        if not self._loaded:
            self.load()

        try:
            from llama_cpp.server.app import create_app
            from llama_cpp.server.settings import Settings
        except ImportError:
            raise ImportError(
                "llama-cpp-python server 模块不可用，请安装：\n"
                "  pip install llama-cpp-python[server]"
            )

        import uvicorn

        settings = Settings(
            model=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
        )
        app = create_app(settings=settings)

        logger.info(f"启动 llama-cpp API 服务: http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

    def shutdown(self):
        """释放 llama-cpp 模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("llama-cpp 模型已释放")


# ============================================================
# vLLM 后端
# ============================================================

class VLLMBackend(InferenceBackend):
    """
    vLLM 推理后端。

    特性：
    - PagedAttention：动态 KV cache 分页管理，大幅减少显存碎片
    - Continuous Batching：连续批处理，请求无需等待批次结束即可加入
    - torch.compile 算子融合加速（可选）
    - OpenAI 兼容 API 服务

    针对 8GB VRAM 的优化策略：
    - gpu_memory_utilization 设为 0.9，保留 10% 给系统
    - max_model_len 限制为 4096（8GB 显存下的安全值）
    - max_num_seqs 限制为 128，避免并发过高导致 OOM
    - 默认使用 float16 精度（比 bfloat16 兼容性更好）
    - 不启用 tensor parallelism（单卡场景）
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        # 显存利用率，8GB 场景下建议 0.9
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.9)
        # 最大模型长度
        self.max_model_len = kwargs.get("max_model_len", 4096)
        # 最大并发序列数
        self.max_num_seqs = kwargs.get("max_num_seqs", 128)
        # 推理精度
        self.dtype = kwargs.get("dtype", "float16")
        # tensor parallel 数量
        self.tp_size = kwargs.get("tensor_parallel_size", 1)
        # 是否使用 torch.compile 加速
        self.enforce_eager = kwargs.get("enforce_eager", False)
        # 信任远程代码
        self.trust_remote_code = kwargs.get("trust_remote_code", True)
        self._llm = None

    def load(self):
        """
        加载模型到 vLLM 引擎。

        自动配置 PagedAttention 参数，针对 8GB VRAM 场景做了默认优化。
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM 未安装，请运行：\n"
                "  pip install vllm"
            )

        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        logger.info(f"加载 vLLM 引擎: {self.model_path}")
        logger.info(f"  显存利用率: {self.gpu_memory_utilization}")
        logger.info(f"  最大模型长度: {self.max_model_len}")
        logger.info(f"  最大并发序列: {self.max_num_seqs}")
        logger.info(f"  推理精度: {self.dtype}")

        self._llm = LLM(
            model=self.model_path,
            dtype=torch_dtype if torch_dtype != "auto" else "auto",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=self.tp_size,
            enforce_eager=self.enforce_eager,
            trust_remote_code=self.trust_remote_code,
        )
        self._loaded = True
        self.monitor.update_peak_vram()
        logger.info("vLLM 引擎加载完成")

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """使用 vLLM 执行单次推理"""
        if not self._loaded:
            raise RuntimeError("vLLM 引擎未加载，请先调用 load()")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )

        start = time.perf_counter()
        outputs = self._llm.generate([request.prompt], sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        output = outputs[0]
        text = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(token_ids)
        tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        self.monitor.record_request(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
        )
        self.monitor.update_peak_vram()

        return InferenceResult(
            text=text,
            token_ids=token_ids,
            prompt=request.prompt,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        vLLM 批量推理，利用 Continuous Batching 特性。

        所有请求会被 vLLM 引擎自动组织为连续批次，
        比逐条调用有显著的吞吐量提升。
        """
        if not self._loaded:
            raise RuntimeError("vLLM 引擎未加载，请先调用 load()")

        from vllm import SamplingParams

        prompts = [req.prompt for req in requests]
        # 使用第一个请求的参数作为批次的采样参数（实际场景中各请求参数可能不同）
        sampling_params = SamplingParams(
            temperature=requests[0].temperature if requests else 0.7,
            top_p=requests[0].top_p if requests else 0.9,
            top_k=requests[0].top_k if requests else 50,
            max_tokens=requests[0].max_tokens if requests else 256,
            stop=requests[0].stop,
        )

        start = time.perf_counter()
        outputs = self._llm.generate(prompts, sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for output in outputs:
            text = output.outputs[0].text
            token_ids = list(output.outputs[0].token_ids)
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(token_ids)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            results.append(InferenceResult(
                text=text,
                token_ids=token_ids,
                prompt=output.prompt,
                latency_ms=elapsed_ms,
                tokens_per_second=(completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ))

        # 批量请求作为整体记录一次性能数据
        self.monitor.record_request(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency_ms=elapsed_ms,
        )
        self.monitor.update_peak_vram()

        return results

    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        """
        启动 vLLM 的 OpenAI 兼容 API 服务。

        使用 vLLM 内置的 API server，支持 /v1/completions 和 /v1/chat/completions 端点。
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn 未安装，请运行：pip install uvicorn")

        logger.info(f"启动 vLLM API 服务: http://{host}:{port}")
        logger.info(f"模型: {self.model_path}")
        logger.info("API 端点: /v1/completions, /v1/chat/completions")

        # 通过环境变量传递 vLLM 配置
        os.environ["VLLM_MODEL"] = self.model_path
        os.environ["VLLM_DTYPE"] = self.dtype
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(self.gpu_memory_utilization)
        os.environ["VLLM_MAX_MODEL_LEN"] = str(self.max_model_len)

        uvicorn.run(
            "vllm.entrypoints.openai.api_server:app",
            host=host,
            port=port,
            log_level="info",
        )

    def shutdown(self):
        """释放 vLLM 引擎资源"""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded = False
            # 清理 CUDA 缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("vLLM 引擎已释放")


# ============================================================
# SGLang 后端
# ============================================================

class SGLangBackend(InferenceBackend):
    """
    SGLang 推理后端。

    特性：
    - RadixAttention：基于基数树的 KV cache 前缀缓存，重复前缀请求加速 3-5x
    - 约束生成：支持 JSON Schema、正则表达式等格式约束
    - 高效连续批处理
    - 适合多轮对话、结构化输出等场景

    针对 8GB VRAM 的优化策略：
    - mem_fraction_static 设为 0.9，保留 10% 给系统
    - context_length 限制为 4096
    - tp_size=1（单卡场景）
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        # 静态显存分配比例
        self.mem_fraction_static = kwargs.get("mem_fraction_static", 0.9)
        # 上下文长度
        self.context_length = kwargs.get("context_length", 4096)
        # tensor parallel 数量
        self.tp_size = kwargs.get("tp_size", 1)
        # 信任远程代码
        self.trust_remote_code = kwargs.get("trust_remote_code", True)
        self._runtime = None

    def load(self):
        """
        加载模型到 SGLang 引擎。

        RadixAttention 会自动启用，无需额外配置。
        """
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError(
                "SGLang 未安装，请运行：\n"
                "  pip install sglang"
            )

        logger.info(f"加载 SGLang 引擎: {self.model_path}")
        logger.info(f"  静态显存比例: {self.mem_fraction_static}")
        logger.info(f"  上下文长度: {self.context_length}")
        logger.info(f"  RadixAttention: 自动启用")

        self._runtime = sgl.Runtime(
            model_path=self.model_path,
            tp_size=self.tp_size,
            mem_fraction_static=self.mem_fraction_static,
            context_length=self.context_length,
            trust_remote_code=self.trust_remote_code,
        )
        self._loaded = True
        self.monitor.update_peak_vram()
        logger.info("SGLang 引擎加载完成")

    def generate(self, request: InferenceRequest) -> InferenceResult:
        """
        使用 SGLang 执行单次推理。

        支持通过 request.json_schema 传入 JSON Schema 实现约束生成。
        """
        if not self._loaded:
            raise RuntimeError("SGLang 引擎未加载，请先调用 load()")

        sampling_params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_new_tokens": request.max_tokens,
        }

        # 约束生成：如果提供了 JSON Schema，添加到采样参数中
        if request.json_schema is not None:
            sampling_params["json_schema"] = request.json_schema
            logger.debug(f"启用 JSON Schema 约束生成")

        if request.stop:
            sampling_params["stop"] = request.stop

        if request.extra:
            sampling_params.update(request.extra)

        start = time.perf_counter()
        outputs = self._runtime.generate(
            [request.prompt],
            sampling_params=sampling_params,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        output = outputs[0]
        text = output["text"]
        token_ids = output.get("token_ids", [])
        completion_tokens = len(token_ids)
        # SGLang 不直接返回 prompt token 数，通过估算
        prompt_tokens = len(request.prompt) // 4  # 粗略估算：4 字符 ≈ 1 token
        tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        self.monitor.record_request(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
        )
        self.monitor.update_peak_vram()

        return InferenceResult(
            text=text,
            token_ids=token_ids,
            prompt=request.prompt,
            latency_ms=elapsed_ms,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"constrained": request.json_schema is not None},
        )

    def generate_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        SGLang 批量推理，利用 RadixAttention 的前缀缓存优势。

        当多个请求共享相同前缀时（如相同的 system prompt），
        RadixAttention 会自动缓存和复用 KV cache，显著加速推理。
        """
        if not self._loaded:
            raise RuntimeError("SGLang 引擎未加载，请先调用 load()")

        prompts = [req.prompt for req in requests]
        sampling_params = {
            "temperature": requests[0].temperature if requests else 0.7,
            "top_p": requests[0].top_p if requests else 0.9,
            "top_k": requests[0].top_k if requests else 50,
            "max_new_tokens": requests[0].max_tokens if requests else 256,
        }

        start = time.perf_counter()
        outputs = self._runtime.generate(prompts, sampling_params=sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for i, output in enumerate(outputs):
            text = output["text"]
            token_ids = output.get("token_ids", [])
            completion_tokens = len(token_ids)
            prompt_tokens = len(requests[i].prompt) // 4 if i < len(requests) else 0
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            results.append(InferenceResult(
                text=text,
                token_ids=token_ids,
                prompt=requests[i].prompt if i < len(requests) else "",
                latency_ms=elapsed_ms,
                tokens_per_second=(completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ))

        self.monitor.record_request(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency_ms=elapsed_ms,
        )
        self.monitor.update_peak_vram()

        return results

    def serve(self, host: str = "0.0.0.0", port: int = 30000):
        """
        启动 SGLang 的 OpenAI 兼容 API 服务。

        支持 /v1/completions 和 /v1/chat/completions 端点。
        """
        if not self._loaded:
            self.load()

        logger.info(f"启动 SGLang API 服务: http://{host}:{port}")
        logger.info(f"模型: {self.model_path}")
        logger.info("特性: RadixAttention, Continuous Batching, JSON Schema 约束生成")

        try:
            self._runtime.loop()
        except KeyboardInterrupt:
            self.shutdown()
            logger.info("SGLang 服务已停止")

    def shutdown(self):
        """释放 SGLang 引擎资源"""
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("SGLang 引擎已释放")


# ============================================================
# 统一推理接口
# ============================================================

class UnifiedInferenceEngine:
    """
    统一推理引擎，封装后端选择和配置优化逻辑。

    使用方式：
        engine = UnifiedInferenceEngine(model_path="path/to/model")
        # 自动选择最优后端
        result = engine.generate("你好，请介绍一下自己")

        # 指定后端
        engine = UnifiedInferenceEngine(model_path="path/to/model", backend="vllm")
        result = engine.generate("你好")

        # 获取性能统计
        print(engine.get_stats().summary())
    """

    # 后端名称到类的映射
    BACKEND_REGISTRY = {
        "llama_cpp": LlamaCppBackend,
        "llama-cpp": LlamaCppBackend,
        "llamacpp": LlamaCppBackend,
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
    }

    # 各后端针对 8GB VRAM 的默认配置
    DEFAULT_CONFIGS = {
        "llama_cpp": {
            "n_gpu_layers": -1,       # 自动计算
            "n_ctx": 4096,
            "n_batch": 512,
            "use_mmap": True,
        },
        "vllm": {
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "max_num_seqs": 128,
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "enforce_eager": False,
        },
        "sglang": {
            "mem_fraction_static": 0.9,
            "context_length": 4096,
            "tp_size": 1,
        },
    }

    def __init__(
        self,
        model_path: str,
        backend: Optional[str] = None,
        auto_load: bool = True,
        **kwargs,
    ):
        """
        初始化统一推理引擎。

        Args:
            model_path: 模型路径或 HF 仓库 ID
            backend: 推理后端名称（llama_cpp/vllm/sglang），为 None 时自动检测
            auto_load: 是否自动加载模型（默认 True）
            **kwargs: 传递给后端的额外参数
        """
        self.model_path = model_path

        # 确定后端
        if backend is None:
            self._backend_name = detect_best_backend()
        else:
            self._backend_name = backend.lower().replace("-", "_")

        if self._backend_name not in self.BACKEND_REGISTRY:
            raise ValueError(
                f"不支持的后端: {self._backend_name}，"
                f"支持的后端: {list(set(self.BACKEND_REGISTRY.values()))}"
            )

        # 合并默认配置和用户配置（用户配置优先）
        backend_key = self._normalize_backend_key(self._backend_name)
        default_config = self.DEFAULT_CONFIGS.get(backend_key, {})
        merged_kwargs = {**default_config, **kwargs}

        # 实例化后端
        backend_cls = self.BACKEND_REGISTRY[self._backend_name]
        self._backend: InferenceBackend = backend_cls(model_path, **merged_kwargs)

        logger.info(f"统一推理引擎初始化: 后端={self._backend_name}, 模型={model_path}")

        if auto_load:
            self._backend.load()

    @staticmethod
    def _normalize_backend_key(name: str) -> str:
        """将后端名称标准化为配置键"""
        name = name.lower().replace("-", "_")
        if "llama" in name:
            return "llama_cpp"
        return name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **extra,
    ) -> InferenceResult:
        """
        统一推理接口 - 单次生成。

        Args:
            prompt: 输入提示文本
            max_tokens: 最大生成 token 数
            temperature: 采样温度（0 = 贪婪解码）
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            stop: 停止词列表
            json_schema: JSON Schema 约束（仅 SGLang 支持）
            **extra: 传递给后端的额外参数

        Returns:
            InferenceResult: 推理结果
        """
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            json_schema=json_schema,
            extra=extra if extra else None,
        )
        return self._backend.generate(request)

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **extra,
    ) -> List[InferenceResult]:
        """
        统一推理接口 - 批量生成。

        Args:
            prompts: 输入提示文本列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            **extra: 传递给后端的额外参数

        Returns:
            List[InferenceResult]: 推理结果列表
        """
        requests = [
            InferenceRequest(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra=extra if extra else None,
            )
            for p in prompts
        ]
        return self._backend.generate_batch(requests)

    def get_stats(self) -> PerformanceStats:
        """获取性能统计"""
        return self._backend.get_performance_stats()

    def shutdown(self):
        """关闭引擎，释放资源"""
        self._backend.shutdown()

    @property
    def backend_name(self) -> str:
        """当前使用的后端名称"""
        return self._backend_name


# ============================================================
# 命令行接口
# ============================================================

def run_benchmark(engine: UnifiedInferenceEngine, num_warmup: int = 2, num_runs: int = 5):
    """
    运行推理性能基准测试。

    Args:
        engine: 统一推理引擎实例
        num_warmup: 预热次数
        num_runs: 正式测试次数
    """
    test_prompts = [
        "请解释SQL注入攻击的原理和防御方法。",
        "什么是XSS跨站脚本攻击？如何防范？",
        "描述一次渗透测试的完整流程。",
        "如何使用Wireshark进行网络流量分析？",
        "解释RSA加密算法的工作原理。",
    ]

    print(f"\n{'=' * 50}")
    print(f"推理性能基准测试")
    print(f"后端: {engine.backend_name}")
    print(f"模型: {engine.model_path}")
    print(f"预热次数: {num_warmup}, 测试次数: {num_runs}")
    print(f"{'=' * 50}\n")

    # 预热阶段
    print("预热中...")
    for i in range(num_warmup):
        engine.generate(test_prompts[i % len(test_prompts)], max_tokens=64)
    print("预热完成\n")

    # 重置性能统计
    engine._backend.monitor.reset()
    engine._backend.monitor.start()

    # 正式测试
    print("开始基准测试...")
    all_results = []
    for i in range(num_runs):
        prompt = test_prompts[i % len(test_prompts)]
        start = time.perf_counter()
        result = engine.generate(prompt, max_tokens=256)
        elapsed = time.perf_counter() - start
        all_results.append(result)

        print(
            f"  请求 {i + 1}/{num_runs}: "
            f"{result.completion_tokens} tokens, "
            f"{elapsed * 1000:.0f}ms, "
            f"{result.tokens_per_second:.1f} tokens/s"
        )

    engine._backend.monitor.end()

    # 输出统计
    stats = engine.get_stats()
    print(f"\n{stats.summary()}")

    # 输出显存信息
    vram_used = get_gpu_memory_usage_mb()
    vram_total = get_total_gpu_memory_mb()
    if vram_total > 0:
        print(f"  当前显存占用:      {vram_used:.1f} / {vram_total:.1f} MB "
              f"({vram_used / vram_total * 100:.1f}%)")


def run_chat(engine: UnifiedInferenceEngine):
    """
    命令行交互模式。

    Args:
        engine: 统一推理引擎实例
    """
    print(f"\n=== 交互模式 (后端: {engine.backend_name}) ===")
    print("输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            user_input = input("用户: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            # 构造 chat template（Qwen 格式）
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

            start = time.time()
            result = engine.generate(prompt, max_tokens=512)
            elapsed = time.time() - start

            print(f"\n助手: {result.text}")
            print(
                f"[{result.completion_tokens} tokens, "
                f"{elapsed:.2f}s, "
                f"{result.tokens_per_second:.1f} tokens/s]\n"
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n推理出错: {e}\n")

    print("\n再见！")


def run_single_inference(engine: UnifiedInferenceEngine, prompt: str, **kwargs):
    """
    单次推理并输出结果。

    Args:
        engine: 统一推理引擎实例
        prompt: 输入提示
        **kwargs: 推理参数
    """
    result = engine.generate(prompt, **kwargs)

    print(f"\n{'=' * 50}")
    print(f"推理结果 (后端: {engine.backend_name})")
    print(f"{'=' * 50}")
    print(f"输入: {prompt}")
    print(f"输出: {result.text}")
    print(f"\n性能指标:")
    print(f"  延迟: {result.latency_ms:.1f}ms")
    print(f"  生成 tokens: {result.completion_tokens}")
    print(f"  吞吐量: {result.tokens_per_second:.1f} tokens/s")

    vram_used = get_gpu_memory_usage_mb()
    if vram_used > 0:
        print(f"  显存占用: {vram_used:.1f} MB")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="HOS-Model-Optimizer 统一推理模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 自动选择后端，单次推理
  python inference.py --model path/to/model --prompt "你好"

  # 使用 vLLM 后端启动 API 服务
  python inference.py --backend vllm --model path/to/model --serve

  # 运行基准测试
  python inference.py --backend vllm --model path/to/model --benchmark

  # 交互模式
  python inference.py --backend sglang --model path/to/model --chat

  # llama-cpp 推理（GGUF 模型）
  python inference.py --backend llama-cpp --model model.gguf --prompt "你好"
        """,
    )

    # 模型和后端
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 HF 仓库 ID（llama-cpp 需要 GGUF 文件路径）")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["llama-cpp", "vllm", "sglang"],
                        help="推理后端（默认自动检测最优后端）")

    # 推理模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--serve", action="store_true",
                            help="启动 OpenAI 兼容 API 服务")
    mode_group.add_argument("--chat", action="store_true",
                            help="命令行交互模式")
    mode_group.add_argument("--benchmark", action="store_true",
                            help="运行性能基准测试")
    mode_group.add_argument("--prompt", type=str, default=None,
                            help="单次推理的输入提示")

    # 服务配置
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=None,
                        help="服务端口（vLLM 默认 8000，SGLang 默认 30000，llama-cpp 默认 8080）")

    # 推理参数
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="最大生成 token 数（默认 256）")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度（默认 0.7）")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="nucleus sampling 参数（默认 0.9）")

    # 8GB VRAM 优化相关
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="GPU 显存利用率（vLLM，默认 0.9）")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="最大模型长度（默认 4096）")
    parser.add_argument("--n-gpu-layers", type=int, default=None,
                        help="GPU offload 层数（llama-cpp，-1 为全部）")

    # 日志
    parser.add_argument("--verbose", action="store_true",
                        help="启用详细日志输出")

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 构建后端参数
    backend_kwargs = {}
    if args.gpu_memory_utilization is not None:
        backend_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        backend_kwargs["max_model_len"] = args.max_model_len
    if args.n_gpu_layers is not None:
        backend_kwargs["n_gpu_layers"] = args.n_gpu_layers

    # 确定默认端口
    default_ports = {"llama-cpp": 8080, "llama_cpp": 8080, "vllm": 8000, "sglang": 30000}
    backend_key = (args.backend or "auto").lower().replace("-", "_")
    port = args.port or default_ports.get(backend_key, 8000)

    try:
        # 如果启动服务模式，直接走后端的 serve 方法
        if args.serve:
            backend_name = args.backend
            if backend_name is None:
                backend_name = detect_best_backend()
            backend_name_normalized = backend_name.lower().replace("-", "_")
            backend_cls = UnifiedInferenceEngine.BACKEND_REGISTRY[backend_name_normalized]
            backend_kwargs_merged = {**UnifiedInferenceEngine.DEFAULT_CONFIGS.get(
                UnifiedInferenceEngine._normalize_backend_key(backend_name_normalized), {}
            ), **backend_kwargs}
            backend_instance = backend_cls(args.model, **backend_kwargs_merged)
            backend_instance.load()
            backend_instance.serve(host=args.host, port=port)
            return

        # 创建统一引擎（自动加载模型）
        engine = UnifiedInferenceEngine(
            model_path=args.model,
            backend=args.backend,
            auto_load=True,
            **backend_kwargs,
        )

        if args.chat:
            run_chat(engine)
        elif args.benchmark:
            run_benchmark(engine)
        elif args.prompt:
            run_single_inference(
                engine,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            # 默认：单次推理示例
            print("请指定 --serve, --chat, --benchmark 或 --prompt 参数")
            print("使用 --help 查看帮助")

        # 输出最终性能统计
        stats = engine.get_stats()
        if stats.total_requests > 0:
            print(f"\n{stats.summary()}")

    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=args.verbose)
        sys.exit(1)
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

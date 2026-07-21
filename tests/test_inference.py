"""
推理模块单元测试

测试推理模块的所有功能，包括：
- llama-cpp 后端
- vLLM 后端
- SGLang 后端
- 统一推理接口
- 性能监控
- 数据结构

使用 mock 避免实际模型加载和推理过程。
"""

import os
import sys
import pytest
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查可选依赖是否可用
LLAMA_CPP_AVAILABLE = importlib.util.find_spec("llama_cpp") is not None
VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
SGLANG_AVAILABLE = importlib.util.find_spec("sglang") is not None

from hos_optimizer.inference import (
    InferenceRequest,
    InferenceResult,
    PerformanceStats,
    PerformanceMonitor,
    InferenceBackend,
    LlamaCppBackend,
    VLLMBackend,
    SGLangBackend,
    UnifiedInferenceEngine,
    get_gpu_memory_usage_mb,
    get_total_gpu_memory_mb,
    detect_best_backend,
)


class TestDataStructures:
    """数据结构测试"""

    def test_inference_request_creation(self):
        """测试推理请求数据结构创建"""
        request = InferenceRequest(
            prompt="测试提示",
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            stop=["END"],
            json_schema={"type": "object"},
            extra={"custom": "param"}
        )
        
        assert request.prompt == "测试提示"
        assert request.max_tokens == 100
        assert request.temperature == 0.8
        assert request.top_p == 0.95
        assert request.top_k == 50
        assert request.stop == ["END"]
        assert request.json_schema == {"type": "object"}
        assert request.extra == {"custom": "param"}

    def test_inference_request_defaults(self):
        """测试推理请求默认值"""
        request = InferenceRequest(prompt="测试")
        
        assert request.max_tokens == 256
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 50
        assert request.stop is None
        assert request.json_schema is None
        assert request.extra is None

    def test_inference_result_creation(self):
        """测试推理结果数据结构创建"""
        result = InferenceResult(
            text="生成的文本",
            token_ids=[1, 2, 3],
            prompt="提示",
            latency_ms=150.5,
            tokens_per_second=33.3,
            prompt_tokens=10,
            completion_tokens=50,
            metadata={"key": "value"}
        )
        
        assert result.text == "生成的文本"
        assert result.token_ids == [1, 2, 3]
        assert result.prompt == "提示"
        assert result.latency_ms == 150.5
        assert result.tokens_per_second == 33.3
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 50
        assert result.metadata == {"key": "value"}

    def test_performance_stats_properties(self):
        """测试性能统计属性计算"""
        stats = PerformanceStats(
            total_requests=10,
            total_prompt_tokens=100,
            total_completion_tokens=500,
            total_latency_ms=1000.0,
            peak_vram_mb=2048.0,
            wall_time_s=5.0
        )
        
        assert stats.avg_latency_ms == 100.0
        assert stats.throughput_tokens_per_s == 100.0
        assert stats.requests_per_s == 2.0

    def test_performance_stats_zero_requests(self):
        """测试零请求时的性能统计"""
        stats = PerformanceStats()
        
        assert stats.avg_latency_ms == 0.0
        assert stats.throughput_tokens_per_s == 0.0
        assert stats.requests_per_s == 0.0

    def test_performance_stats_summary(self):
        """测试性能统计摘要生成"""
        stats = PerformanceStats(
            total_requests=5,
            total_prompt_tokens=50,
            total_completion_tokens=250,
            total_latency_ms=500.0,
            peak_vram_mb=1024.0,
            wall_time_s=2.5
        )
        
        summary = stats.summary()
        
        assert "性能统计摘要" in summary
        assert "5" in summary  # total_requests
        assert "50" in summary  # prompt tokens
        assert "250" in summary  # completion tokens
        assert "1024.0" in summary  # peak VRAM


class TestPerformanceMonitor:
    """性能监控器测试"""

    def test_monitor_start_end(self):
        """测试监控器启动和结束"""
        monitor = PerformanceMonitor()
        
        monitor.start()
        time.sleep(0.1)
        monitor.end()
        
        assert monitor.stats.wall_time_s > 0
        assert monitor.stats.wall_time_s >= 0.1

    def test_monitor_record_request(self):
        """测试记录请求"""
        monitor = PerformanceMonitor()
        
        monitor.record_request(prompt_tokens=10, completion_tokens=50, latency_ms=100)
        monitor.record_request(prompt_tokens=20, completion_tokens=60, latency_ms=150)
        
        assert monitor.stats.total_requests == 2
        assert monitor.stats.total_prompt_tokens == 30
        assert monitor.stats.total_completion_tokens == 110
        assert monitor.stats.total_latency_ms == 250

    def test_monitor_update_peak_vram(self):
        """测试更新峰值显存"""
        monitor = PerformanceMonitor()
        
        with patch("hos_optimizer.inference.get_gpu_memory_usage_mb") as mock_vram:
            mock_vram.return_value = 2048.0
            monitor.update_peak_vram()
            
            assert monitor.stats.peak_vram_mb == 2048.0
            
            # 更高的值应该更新
            mock_vram.return_value = 3072.0
            monitor.update_peak_vram()
            assert monitor.stats.peak_vram_mb == 3072.0
            
            # 更低的值不应该更新
            mock_vram.return_value = 1024.0
            monitor.update_peak_vram()
            assert monitor.stats.peak_vram_mb == 3072.0

    def test_monitor_reset(self):
        """测试重置监控器"""
        monitor = PerformanceMonitor()
        monitor.start()
        monitor.record_request(10, 50, 100)
        monitor.end()
        
        monitor.reset()
        
        assert monitor.stats.total_requests == 0
        assert monitor.stats.total_latency_ms == 0.0
        assert monitor._start_time is None
        assert monitor._end_time is None


class TestGPUUtilities:
    """GPU 工具函数测试"""

    def test_get_gpu_memory_usage_no_cuda(self):
        """测试无 CUDA 时的显存使用"""
        with patch("torch.cuda.is_available", return_value=False):
            usage = get_gpu_memory_usage_mb()
            assert usage == 0.0

    def test_get_gpu_memory_usage_with_cuda(self):
        """测试有 CUDA 时的显存使用"""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.max_memory_allocated", return_value=2 * 1024 ** 3):
            
            usage = get_gpu_memory_usage_mb()
            assert usage > 0

    def test_get_total_gpu_memory_no_cuda(self):
        """测试无 CUDA 时的总显存"""
        with patch("torch.cuda.is_available", return_value=False):
            total = get_total_gpu_memory_mb()
            assert total == 0.0

    def test_get_total_gpu_memory_with_cuda(self):
        """测试有 CUDA 时的总显存"""
        mock_props = MagicMock()
        mock_props.total_mem = 8 * 1024 ** 3
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            
            total = get_total_gpu_memory_mb()
            assert total > 0


class TestBackendDetection:
    """后端检测测试"""

    def test_detect_best_backend_vllm(self):
        """测试检测到 vLLM 作为最优后端"""
        with patch("hos_optimizer.inference.get_total_gpu_memory_mb", return_value=8192), \
             patch("hos_optimizer.inference._check_import") as mock_check:
            
            mock_check.side_effect = lambda name: name == "vllm"
            
            backend = detect_best_backend()
            assert backend == "vllm"

    def test_detect_best_backend_sglang(self):
        """测试检测到 SGLang 作为最优后端"""
        with patch("hos_optimizer.inference.get_total_gpu_memory_mb", return_value=4096), \
             patch("hos_optimizer.inference._check_import") as mock_check:
            
            mock_check.side_effect = lambda name: name == "sglang"
            
            backend = detect_best_backend()
            assert backend == "sglang"

    def test_detect_best_backend_llama_cpp(self):
        """测试检测到 llama-cpp 作为最优后端"""
        with patch("hos_optimizer.inference.get_total_gpu_memory_mb", return_value=0), \
             patch("hos_optimizer.inference._check_import") as mock_check:
            
            mock_check.side_effect = lambda name: name == "llama_cpp"
            
            backend = detect_best_backend()
            assert backend == "llama_cpp"

    def test_detect_best_backend_no_backend_available(self):
        """测试没有可用后端时抛出异常"""
        with patch("hos_optimizer.inference.get_total_gpu_memory_mb", return_value=0), \
             patch("hos_optimizer.inference._check_import", return_value=False):
            
            with pytest.raises(RuntimeError) as exc_info:
                detect_best_backend()
            
            assert "没有可用的推理后端" in str(exc_info.value)


class TestLlamaCppBackend:
    """llama-cpp 后端测试"""

    def test_llama_cpp_initialization(self):
        """测试 llama-cpp 后端初始化"""
        backend = LlamaCppBackend(
            model_path="/path/to/model.gguf",
            n_gpu_layers=32,
            n_ctx=2048,
            n_threads=4
        )
        
        assert backend.model_path == "/path/to/model.gguf"
        assert backend.n_gpu_layers == 32
        assert backend.n_ctx == 2048
        assert backend.n_threads == 4
        assert backend._loaded is False

    @pytest.mark.skipif(
        not LLAMA_CPP_AVAILABLE,
        reason="llama_cpp not installed"
    )
    def test_llama_cpp_load_success(self):
        """测试 llama-cpp 模型加载成功"""
        mock_llama = MagicMock()
        mock_model = MagicMock()
        mock_llama.return_value = mock_model
        
        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama)}):
            backend = LlamaCppBackend("/path/to/model.gguf")
            
            with patch("hos_optimizer.inference.get_total_gpu_memory_mb", return_value=8192):
                backend.load()
            
            assert backend._loaded is True
            assert backend._model is not None

    def test_llama_cpp_load_import_error(self):
        """测试 llama-cpp 导入失败"""
        backend = LlamaCppBackend("/path/to/model.gguf")
        
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError) as exc_info:
                backend.load()
            
            assert "llama-cpp-python 未安装" in str(exc_info.value)

    def test_llama_cpp_generate_not_loaded(self):
        """测试未加载时生成抛出异常"""
        backend = LlamaCppBackend("/path/to/model.gguf")
        request = InferenceRequest(prompt="测试")
        
        with pytest.raises(RuntimeError) as exc_info:
            backend.generate(request)
        
        assert "模型未加载" in str(exc_info.value)

    def test_llama_cpp_generate_success(self):
        """测试 llama-cpp 生成成功"""
        backend = LlamaCppBackend("/path/to/model.gguf")
        backend._loaded = True
        backend._model = MagicMock()
        
        # Mock 模型输出
        backend._model.return_value = {
            "choices": [{"text": "生成的文本"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        request = InferenceRequest(prompt="测试", max_tokens=50)
        result = backend.generate(request)
        
        assert isinstance(result, InferenceResult)
        assert result.text == "生成的文本"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20

    def test_llama_cpp_shutdown(self):
        """测试 llama-cpp 关闭"""
        backend = LlamaCppBackend("/path/to/model.gguf")
        backend._loaded = True
        backend._model = MagicMock()
        
        backend.shutdown()
        
        assert backend._loaded is False
        assert backend._model is None


class TestVLLMBackend:
    """vLLM 后端测试"""

    def test_vllm_initialization(self):
        """测试 vLLM 后端初始化"""
        backend = VLLMBackend(
            model_path="/path/to/model",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            max_num_seqs=64
        )
        
        assert backend.model_path == "/path/to/model"
        assert backend.gpu_memory_utilization == 0.85
        assert backend.max_model_len == 2048
        assert backend.max_num_seqs == 64
        assert backend._loaded is False

    def test_vllm_load_success(self):
        """测试 vLLM 模型加载成功"""
        mock_llm = MagicMock()
        mock_engine = MagicMock()
        mock_llm.return_value = mock_engine
        
        with patch.dict("sys.modules", {"vllm": MagicMock(LLM=mock_llm)}):
            backend = VLLMBackend("/path/to/model")
            backend.load()
            
            assert backend._loaded is True
            assert backend._llm is not None

    def test_vllm_load_import_error(self):
        """测试 vLLM 导入失败"""
        backend = VLLMBackend("/path/to/model")
        
        with patch.dict("sys.modules", {"vllm": None}):
            with pytest.raises(ImportError) as exc_info:
                backend.load()
            
            assert "vLLM 未安装" in str(exc_info.value)

    def test_vllm_generate_not_loaded(self):
        """测试未加载时生成抛出异常"""
        backend = VLLMBackend("/path/to/model")
        request = InferenceRequest(prompt="测试")
        
        with pytest.raises(RuntimeError) as exc_info:
            backend.generate(request)
        
        assert "vLLM 引擎未加载" in str(exc_info.value)

    def test_vllm_generate_success(self):
        """测试 vLLM 生成成功"""
        backend = VLLMBackend("/path/to/model")
        backend._loaded = True
        backend._llm = MagicMock()
        
        # Mock 输出
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="生成的文本", token_ids=[1, 2, 3])]
        mock_output.prompt_token_ids = [10, 20, 30]
        backend._llm.generate.return_value = [mock_output]
        
        request = InferenceRequest(prompt="测试", max_tokens=50)
        
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            result = backend.generate(request)
        
        assert isinstance(result, InferenceResult)
        assert result.text == "生成的文本"
        assert result.token_ids == [1, 2, 3]
        assert result.prompt_tokens == 3

    def test_vllm_batch_generate(self):
        """测试 vLLM 批量生成"""
        backend = VLLMBackend("/path/to/model")
        backend._loaded = True
        backend._llm = MagicMock()
        
        # Mock 批量输出
        mock_output1 = MagicMock()
        mock_output1.outputs = [MagicMock(text="文本1", token_ids=[1, 2])]
        mock_output1.prompt_token_ids = [10, 20]
        mock_output1.prompt = "提示1"
        
        mock_output2 = MagicMock()
        mock_output2.outputs = [MagicMock(text="文本2", token_ids=[3, 4])]
        mock_output2.prompt_token_ids = [30, 40]
        mock_output2.prompt = "提示2"
        
        backend._llm.generate.return_value = [mock_output1, mock_output2]
        
        requests = [
            InferenceRequest(prompt="提示1"),
            InferenceRequest(prompt="提示2")
        ]
        
        with patch.dict("sys.modules", {"vllm": MagicMock()}):
            results = backend.generate_batch(requests)
        
        assert len(results) == 2
        assert results[0].text == "文本1"
        assert results[1].text == "文本2"

    def test_vllm_shutdown(self):
        """测试 vLLM 关闭"""
        backend = VLLMBackend("/path/to/model")
        backend._loaded = True
        backend._llm = MagicMock()
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache"):
            backend.shutdown()
        
        assert backend._loaded is False
        assert backend._llm is None


class TestSGLangBackend:
    """SGLang 后端测试"""

    def test_sglang_initialization(self):
        """测试 SGLang 后端初始化"""
        backend = SGLangBackend(
            model_path="/path/to/model",
            mem_fraction_static=0.85,
            context_length=2048,
            tp_size=1
        )
        
        assert backend.model_path == "/path/to/model"
        assert backend.mem_fraction_static == 0.85
        assert backend.context_length == 2048
        assert backend.tp_size == 1
        assert backend._loaded is False

    def test_sglang_load_success(self):
        """测试 SGLang 模型加载成功"""
        mock_runtime = MagicMock()
        mock_rt = MagicMock()
        mock_runtime.return_value = mock_rt
        
        with patch.dict("sys.modules", {"sglang": MagicMock(Runtime=mock_runtime)}):
            backend = SGLangBackend("/path/to/model")
            backend.load()
            
            assert backend._loaded is True
            assert backend._runtime is not None

    def test_sglang_load_import_error(self):
        """测试 SGLang 导入失败"""
        backend = SGLangBackend("/path/to/model")
        
        with patch.dict("sys.modules", {"sglang": None}):
            with pytest.raises(ImportError) as exc_info:
                backend.load()
            
            assert "SGLang 未安装" in str(exc_info.value)

    def test_sglang_generate_not_loaded(self):
        """测试未加载时生成抛出异常"""
        backend = SGLangBackend("/path/to/model")
        request = InferenceRequest(prompt="测试")
        
        with pytest.raises(RuntimeError) as exc_info:
            backend.generate(request)
        
        assert "SGLang 引擎未加载" in str(exc_info.value)

    def test_sglang_generate_success(self):
        """测试 SGLang 生成成功"""
        backend = SGLangBackend("/path/to/model")
        backend._loaded = True
        backend._runtime = MagicMock()
        
        # Mock 输出
        backend._runtime.generate.return_value = [{
            "text": "生成的文本",
            "token_ids": [1, 2, 3]
        }]
        
        request = InferenceRequest(prompt="测试", max_tokens=50)
        result = backend.generate(request)
        
        assert isinstance(result, InferenceResult)
        assert result.text == "生成的文本"
        assert result.token_ids == [1, 2, 3]

    def test_sglang_generate_with_json_schema(self):
        """测试 SGLang 带 JSON Schema 约束生成"""
        backend = SGLangBackend("/path/to/model")
        backend._loaded = True
        backend._runtime = MagicMock()
        
        backend._runtime.generate.return_value = [{
            "text": '{"key": "value"}',
            "token_ids": [1, 2, 3]
        }]
        
        request = InferenceRequest(
            prompt="测试",
            json_schema={"type": "object"}
        )
        result = backend.generate(request)
        
        assert result.metadata.get("constrained") is True

    def test_sglang_shutdown(self):
        """测试 SGLang 关闭"""
        backend = SGLangBackend("/path/to/model")
        backend._loaded = True
        backend._runtime = MagicMock()
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache"):
            backend.shutdown()
        
        assert backend._loaded is False
        assert backend._runtime is None


class TestUnifiedInferenceEngine:
    """统一推理引擎测试"""

    def test_unified_engine_initialization_auto_backend(self):
        """测试统一引擎自动选择后端"""
        with patch("hos_optimizer.inference.detect_best_backend", return_value="vllm"), \
             patch("hos_optimizer.inference.VLLMBackend") as mock_backend_cls:
            
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            
            engine = UnifiedInferenceEngine(
                model_path="/path/to/model",
                auto_load=False
            )
            
            assert engine.backend_name == "vllm"
            assert engine._backend is not None

    def test_unified_engine_initialization_specific_backend(self):
        """测试统一引擎指定后端"""
        with patch("hos_optimizer.inference.LlamaCppBackend") as mock_backend_cls:
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            
            engine = UnifiedInferenceEngine(
                model_path="/path/to/model.gguf",
                backend="llama-cpp",
                auto_load=False
            )
            
            assert engine.backend_name == "llama_cpp"

    def test_unified_engine_invalid_backend(self):
        """测试统一引擎无效后端"""
        with pytest.raises(ValueError) as exc_info:
            UnifiedInferenceEngine(
                model_path="/path/to/model",
                backend="invalid_backend",
                auto_load=False
            )
        
        assert "不支持的后端" in str(exc_info.value)

    def test_unified_engine_generate(self):
        """测试统一引擎生成"""
        mock_backend = MagicMock()
        mock_result = InferenceResult(text="结果", prompt_tokens=10, completion_tokens=20)
        mock_backend.generate.return_value = mock_result
        
        with patch.object(UnifiedInferenceEngine, 'BACKEND_REGISTRY', {"vllm": lambda *a, **kw: mock_backend}):
            engine = UnifiedInferenceEngine(
                model_path="/path/to/model",
                backend="vllm",
                auto_load=False
            )
            
            result = engine.generate("测试提示", max_tokens=100)
            
            assert result.text == "结果"
            mock_backend.generate.assert_called_once()

    def test_unified_engine_generate_batch(self):
        """测试统一引擎批量生成"""
        mock_backend = MagicMock()
        mock_results = [
            InferenceResult(text="结果1", prompt_tokens=10, completion_tokens=20),
            InferenceResult(text="结果2", prompt_tokens=15, completion_tokens=25)
        ]
        mock_backend.generate_batch.return_value = mock_results
        
        with patch.object(UnifiedInferenceEngine, 'BACKEND_REGISTRY', {"vllm": lambda *a, **kw: mock_backend}):
            engine = UnifiedInferenceEngine(
                model_path="/path/to/model",
                backend="vllm",
                auto_load=False
            )
            
            results = engine.generate_batch(["提示1", "提示2"])
            
            assert len(results) == 2
            mock_backend.generate_batch.assert_called_once()

    def test_unified_engine_get_stats(self):
        """测试统一引擎获取性能统计"""
        mock_backend = MagicMock()
        mock_stats = PerformanceStats(total_requests=5)
        mock_backend.get_performance_stats.return_value = mock_stats
        
        with patch.object(UnifiedInferenceEngine, 'BACKEND_REGISTRY', {"vllm": lambda *a, **kw: mock_backend}):
            engine = UnifiedInferenceEngine(
                model_path="/path/to/model",
                backend="vllm",
                auto_load=False
            )
            
            stats = engine.get_stats()
            
            assert stats.total_requests == 5

    def test_unified_engine_shutdown(self):
        """测试统一引擎关闭"""
        mock_backend = MagicMock()
        
        with patch.object(UnifiedInferenceEngine, 'BACKEND_REGISTRY', {"vllm": lambda *a, **kw: mock_backend}):
            engine = UnifiedInferenceEngine(
                model_path="/path/to/model",
                backend="vllm",
                auto_load=False
            )
            
            engine.shutdown()
            
            mock_backend.shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

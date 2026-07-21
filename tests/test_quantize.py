"""
量化模块单元测试

测试量化模块的所有功能，包括：
- GGUF 量化
- AWQ 量化
- GPTQ 量化
- PPL 评估
- 格式转换
- VRAM 检查和优化

使用 mock 避免实际模型加载和量化过程。
"""

import os
import sys
import pytest
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import subprocess

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查可选依赖是否可用
AWQ_AVAILABLE = importlib.util.find_spec("awq") is not None
GPTQ_AVAILABLE = importlib.util.find_spec("auto_gptq") is not None

from hos_optimizer.quantize import (
    QuantizationError,
    check_vram_availability,
    optimize_for_low_vram,
    quantize_gguf,
    quantize_awq,
    quantize_gptq,
    evaluate_perplexity,
    convert_format,
    get_model_size,
    VRAM_8GB_CONFIG,
)


class TestVRAMCheck:
    """VRAM 检查相关测试"""

    def test_check_vram_no_cuda(self):
        """测试无 CUDA 支持时的 VRAM 检查"""
        with patch("torch.cuda.is_available", return_value=False):
            result = check_vram_availability()
            
            assert result["available"] is False
            assert result["total_vram_gb"] == 0
            assert result["free_vram_gb"] == 0
            assert result["device"] == "cpu"

    def test_check_vram_with_cuda(self):
        """测试有 CUDA 支持时的 VRAM 检查"""
        mock_device = MagicMock()
        mock_device.total_memory = 8 * 1024 ** 3  # 8GB
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.get_device_properties", return_value=mock_device), \
             patch("torch.cuda.memory_allocated", return_value=2 * 1024 ** 3), \
             patch("torch.cuda.get_device_name", return_value="Test GPU"):
            
            result = check_vram_availability()
            
            assert result["available"] is True
            assert result["total_vram_gb"] == 8.0
            assert result["free_vram_gb"] == 6.0
            assert result["device"] == "Test GPU"

    def test_optimize_for_low_vram_enabled(self):
        """测试低 VRAM 优化配置启用用的情况"""
        config = {"batch_size": 4, "seq_length": 1024}
        
        with patch("hos_optimizer.quantize.check_vram_availability") as mock_check:
            mock_check.return_value = {
                "available": True,
                "free_vram_gb": 6.0,
                "total_vram_gb": 8.0,
                "device": "Test GPU"
            }
            
            optimized = optimize_for_low_vram(config)
            
            # 应该应用 8GB 优化配置
            assert optimized["max_batch_size"] == VRAM_8GB_CONFIG["max_batch_size"]
            assert optimized["max_seq_length"] == VRAM_8GB_CONFIG["max_seq_length"]
            assert optimized["gradient_checkpointing"] is True
            assert optimized["offload_to_cpu"] is True
            # 原有配置应该保留
            assert optimized["batch_size"] == 4
            assert optimized["seq_length"] == 1024

    def test_optimize_for_low_vram_disabled(self):
        """测试低 VRAM 优化配置不适用的情况"""
        config = {"batch_size": 4, "seq_length": 1024}
        
        with patch("hos_optimizer.quantize.check_vram_availability") as mock_check:
            mock_check.return_value = {
                "available": True,
                "free_vram_gb": 12.0,  # 超过 8GB
                "total_vram_gb": 16.0,
                "device": "Test GPU"
            }
            
            optimized = optimize_for_low_vram(config)
            
            # 不应该应用优化配置
            assert optimized == config


class TestGGUFQuantization:
    """GGUF 量化测试"""

    def test_quantize_gguf_success(self, tmp_dir):
        """测试 GGUF 量化成功场景"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model.gguf")
        llama_cpp_path = "/path/to/llama.cpp"

        # Mock subprocess + 使 convert.py 看起来存在
        with patch("subprocess.run") as mock_run, \
             patch("hos_optimizer.quantize._find_convert_script") as mock_find, \
             patch("tempfile.TemporaryDirectory") as mock_tmpdir:

            mock_find.return_value = os.path.join(llama_cpp_path, "convert.py")
            # 让 os.path.isfile 对 convert.py 返回 True（最低限度模拟文件存在）
            with patch("os.path.isfile", return_value=True):
                mock_run.return_value = MagicMock(returncode=0)
                mock_tmpdir.return_value.__enter__.return_value = tmp_dir

                result = quantize_gguf(
                    model_path=model_path,
                    output_path=output_path,
                    quant_type="Q4_K_M",
                    llama_cpp_path=llama_cpp_path
                )

            assert result == output_path
            # 验证调用了检查工具 + 转换 + 量化 3 次
            assert mock_run.call_count >= 2

    def test_quantize_gguf_tool_not_found(self, tmp_dir):
        """测试 GGUF 量化工具不存在的情况"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model.gguf")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with pytest.raises(QuantizationError) as exc_info:
                quantize_gguf(model_path, output_path)

            assert "找不到 llama-quantize 工具" in str(exc_info.value)

    def test_quantize_gguf_conversion_failed(self, tmp_dir):
        """测试 GGUF 量化转换失败的情况"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model.gguf")

        with patch("subprocess.run") as mock_run, \
             patch("hos_optimizer.quantize._find_convert_script") as mock_find, \
             patch("tempfile.TemporaryDirectory") as mock_tmpdir:

            mock_find.return_value = os.path.join(tmp_dir, "convert.py")
            mock_tmpdir.return_value.__enter__.return_value = tmp_dir

            # 第一次调用成功（检查工具），第二次调用失败（转换）
            with patch("os.path.isfile", return_value=True):
                mock_run.side_effect = [
                    MagicMock(returncode=0),  # 检查工具
                    subprocess.CalledProcessError(1, "convert", stderr="Conversion failed")
                ]

                with pytest.raises(QuantizationError) as exc_info:
                    quantize_gguf(model_path, output_path)

            assert "GGUF 量化失败" in str(exc_info.value)


class TestAWQQuantization:
    """AWQ 量化测试"""

    @pytest.mark.skipif(
        not AWQ_AVAILABLE,
        reason="awq not installed"
    )
    def test_quantize_awq_success(self, tmp_dir):
        """测试 AWQ 量化成功场景"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-awq")
        
        with patch("hos_optimizer.quantize.AutoAWQForCausalLM") as mock_awq_cls, \
             patch("hos_optimizer.quantize.AutoTokenizer") as mock_tokenizer_cls, \
             patch("hos_optimizer.quantize.optimize_for_low_vram") as mock_optimize:
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_awq_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_optimize.return_value = {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM"
            }
            
            result = quantize_awq(
                model_path=model_path,
                output_path=output_path,
                bits=4,
                group_size=128
            )
            
            assert result == output_path
            mock_model.quantize.assert_called_once()
            mock_model.save_quantized.assert_called_once_with(output_path)
            mock_tokenizer.save_pretrained.assert_called_once_with(output_path)

    def test_quantize_awq_missing_dependency(self, tmp_dir):
        """测试 AWQ 量化缺少依赖的情况"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-awq")
        
        with patch("hos_optimizer.quantize.AutoAWQForCausalLM") as mock_awq_cls:
            mock_awq_cls.from_pretrained.side_effect = ImportError("autoawq")
            
            with pytest.raises(QuantizationError) as exc_info:
                quantize_awq(model_path, output_path)
            
            assert "缺少依赖" in str(exc_info.value)
            assert "autoawq" in str(exc_info.value)

    @pytest.mark.skipif(
        not AWQ_AVAILABLE,
        reason="awq not installed"
    )
    def test_quantize_awq_quantization_failed(self, tmp_dir):
        """测试 AWQ 量化过程失败的情况"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-awq")
        
        with patch("hos_optimizer.quantize.AutoAWQForCausalLM") as mock_awq_cls, \
             patch("hos_optimizer.quantize.AutoTokenizer") as mock_tokenizer_cls, \
             patch("hos_optimizer.quantize.optimize_for_low_vram") as mock_optimize:
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_awq_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_optimize.return_value = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
            
            # 量化过程抛出异常
            mock_model.quantize.side_effect = Exception("Quantization failed")
            
            with pytest.raises(QuantizationError) as exc_info:
                quantize_awq(model_path, output_path)
            
            assert "AWQ 量化失败" in str(exc_info.value)


class TestGPTQQuantization:
    """GPTQ 量化测试"""

    @pytest.mark.skipif(
        not GPTQ_AVAILABLE,
        reason="auto_gptq not installed"
    )
    def test_quantize_gptq_success(self, tmp_dir):
        """测试 GPTQ 量化成功场景"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-gptq")
        
        with patch("hos_optimizer.quantize.AutoGPTQForCausalLM") as mock_gptq_cls, \
             patch("hos_optimizer.quantize.AutoTokenizer") as mock_tokenizer_cls, \
             patch("hos_optimizer.quantize.BaseQuantizeConfig") as mock_config_cls:
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_config = MagicMock()
            
            mock_gptq_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_config_cls.return_value = mock_config
            
            # Mock tokenizer 调用
            mock_tokenizer.return_value = {"input_ids": MagicMock()}
            
            result = quantize_gptq(
                model_path=model_path,
                output_path=output_path,
                bits=4,
                group_size=128,
                desc_act=False
            )
            
            assert result == output_path
            mock_model.quantize.assert_called_once()
            mock_model.save_quantized.assert_called_once_with(output_path)

    def test_quantize_gptq_invalid_bits(self, tmp_dir):
        """测试 GPTQ 量化使用无效位数的情况"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-gptq")
        
        with pytest.raises(ValueError) as exc_info:
            quantize_gptq(model_path, output_path, bits=3)
        
        assert "仅支持 4-bit 或 8-bit" in str(exc_info.value)

    def test_quantize_gptq_missing_dependency(self, tmp_dir):
        """测试 GPTQ 量化缺少依赖的情况"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-gptq")
        
        with patch("hos_optimizer.quantize.AutoGPTQForCausalLM") as mock_gptq_cls:
            mock_gptq_cls.from_pretrained.side_effect = ImportError("auto_gptq")
            
            with pytest.raises(QuantizationError) as exc_info:
                quantize_gptq(model_path, output_path, bits=4)
            
            assert "缺少依赖" in str(exc_info.value)
            assert "auto-gptq" in str(exc_info.value)


class TestPerplexityEvaluation:
    """PPL 评估测试"""

    def test_evaluate_perplexity_success(self, tmp_dir):
        """测试 PPL 评估成功场景"""
        model_path = os.path.join(tmp_dir, "model")
        
        with patch("hos_optimizer.quantize.AutoModelForCausalLM") as mock_model_cls, \
             patch("hos_optimizer.quantize.AutoTokenizer") as mock_tokenizer_cls, \
             patch("hos_optimizer.quantize.load_dataset") as mock_load_dataset, \
             patch("hos_optimizer.quantize.torch.cuda.is_available", return_value=False):
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_dataset = MagicMock()
            
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            mock_load_dataset.return_value = mock_dataset
            
            # Mock 数据集返回文本列表
            mock_dataset.__getitem__.return_value = ["text1", "text2"]
            
            # Mock tokenizer 调用返回编码（dict-like）
            mock_input_ids = MagicMock()
            # size(dim) 返回 int，size() 返回 tuple
            mock_input_ids.size.side_effect = lambda dim=None: (1, 100) if dim is None else 100
            mock_input_ids.to.return_value = mock_input_ids
            # 支持切片操作，返回自身
            mock_input_ids.__getitem__.return_value = mock_input_ids
            mock_encodings = {"input_ids": mock_input_ids}
            mock_tokenizer.return_value = mock_encodings
            
            # Mock model.to() 返回自身
            mock_model.to.return_value = mock_model
            
            # Mock 模型推理返回 loss
            mock_output = MagicMock()
            mock_output.loss.item.return_value = 2.5
            mock_model.return_value = mock_output
            
            result = evaluate_perplexity(
                model_path=model_path,
                dataset="wikitext",
                max_samples=10,
                stride=512
            )
            
            assert isinstance(result, float)
            assert result > 0

    def test_evaluate_perplexity_missing_dataset(self, tmp_dir):
        """测试 PPL 评估缺少 datasets 库的情况"""
        model_path = os.path.join(tmp_dir, "model")
        
        with patch("hos_optimizer.quantize.AutoModelForCausalLM") as mock_model_cls, \
             patch("hos_optimizer.quantize.AutoTokenizer") as mock_tokenizer_cls:
            
            mock_model_cls.from_pretrained.return_value = MagicMock()
            mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
            
            with patch("hos_optimizer.quantize.load_dataset", side_effect=ImportError("datasets")):
                with pytest.raises(QuantizationError) as exc_info:
                    evaluate_perplexity(model_path)
                
                assert "缺少依赖" in str(exc_info.value)
                assert "datasets" in str(exc_info.value)


class TestFormatConversion:
    """格式转换测试"""

    def test_convert_hf_to_gguf(self, tmp_dir):
        """测试 HuggingFace 到 GGUF 格式转换"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model.gguf")
        
        with patch("hos_optimizer.quantize.quantize_gguf") as mock_quantize:
            mock_quantize.return_value = output_path
            
            result = convert_format(
                model_path=model_path,
                output_path=output_path,
                from_format="hf",
                to_format="gguf"
            )
            
            assert result == output_path
            mock_quantize.assert_called_once()

    def test_convert_hf_to_awq(self, tmp_dir):
        """测试 HuggingFace 到 AWQ 格式转换"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-awq")
        
        with patch("hos_optimizer.quantize.quantize_awq") as mock_quantize:
            mock_quantize.return_value = output_path
            
            result = convert_format(
                model_path=model_path,
                output_path=output_path,
                from_format="hf",
                to_format="awq"
            )
            
            assert result == output_path
            mock_quantize.assert_called_once()

    def test_convert_unsupported_path(self, tmp_dir):
        """测试不支持的转换路径"""
        model_path = os.path.join(tmp_dir, "model")
        output_path = os.path.join(tmp_dir, "model-out")
        
        with pytest.raises(QuantizationError) as exc_info:
            convert_format(
                model_path=model_path,
                output_path=output_path,
                from_format="gguf",
                to_format="awq"
            )
        
        assert "不支持的转换路径" in str(exc_info.value)

    def test_convert_gguf_to_hf_not_implemented(self, tmp_dir):
        """测试 GGUF 到 HuggingFace 转换未实现"""
        model_path = os.path.join(tmp_dir, "model.gguf")
        output_path = os.path.join(tmp_dir, "model")
        
        with pytest.raises(QuantizationError) as exc_info:
            convert_format(
                model_path=model_path,
                output_path=output_path,
                from_format="gguf",
                to_format="hf"
            )
        
        assert "GGUF" in str(exc_info.value) and "转换" in str(exc_info.value)


class TestModelSize:
    """模型大小计算测试"""

    def test_get_model_size_empty_dir(self, tmp_dir):
        """测试空目录的模型大小"""
        size = get_model_size(tmp_dir)
        assert size == 0.0

    def test_get_model_size_with_files(self, tmp_dir):
        """测试包含模型文件的目录大小"""
        # 创建测试文件
        test_file = os.path.join(tmp_dir, "model.safetensors")
        with open(test_file, "wb") as f:
            f.write(b"0" * (1024 * 1024))  # 1MB
        
        size = get_model_size(tmp_dir)
        assert size > 0
        assert size < 0.01  # 应该约等于 0.001GB

    def test_get_model_size_nonexistent_dir(self):
        """测试不存在的目录（返回 0 而非抛出异常）"""
        size = get_model_size("/nonexistent/path")
        assert size == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

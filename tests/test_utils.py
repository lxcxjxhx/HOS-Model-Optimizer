"""
工具函数模块单元测试

测试工具函数模块的所有功能，包括：
- 日志配置
- 文件操作工具
- 路径解析工具

使用 pytest 框架进行测试。
"""

import os
import sys
import pytest
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hos_optimizer.utils import (
    setup_logger,
    ensure_dir,
    get_file_size_gb,
    get_dir_size_gb,
    find_model_files,
    resolve_model_path,
    is_model_path,
    get_model_format,
)


class TestSetupLogger:
    """日志配置测试"""

    def test_setup_logger_default(self):
        """测试默认日志配置"""
        logger = setup_logger()
        
        assert logger is not None
        assert logger.name == "hos_optimizer"
        assert logger.level == logging.INFO

    def test_setup_logger_custom_name(self):
        """测试自定义日志记录器名称"""
        logger = setup_logger(name="test_logger")
        
        assert logger.name == "test_logger"

    def test_setup_logger_custom_level(self):
        """测试自定义日志级别"""
        logger = setup_logger(name="debug_logger", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file(self, tmp_dir):
        """测试带文件输出的日志配置"""
        log_file = os.path.join(tmp_dir, "test.log")
        logger = setup_logger(name="file_logger", log_file=log_file)
        
        assert logger is not None
        # 应该有两个 handler：控制台和文件
        assert len(logger.handlers) >= 1

    def test_setup_logger_no_duplicate_handlers(self):
        """测试避免重复添加 handler"""
        logger1 = setup_logger(name="no_dup_logger")
        handler_count = len(logger1.handlers)
        
        # 再次调用应该返回相同的 logger，不添加新 handler
        logger2 = setup_logger(name="no_dup_logger")
        
        assert logger1 is logger2
        assert len(logger2.handlers) == handler_count

    def test_setup_logger_custom_format(self):
        """测试自定义日志格式"""
        custom_fmt = "%(levelname)s: %(message)s"
        logger = setup_logger(name="format_logger", fmt=custom_fmt)
        
        assert logger is not None
        # 验证 handler 的格式器
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            assert formatter is not None


class TestEnsureDir:
    """目录创建测试"""

    def test_ensure_dir_creates_new(self, tmp_dir):
        """测试创建新目录"""
        new_dir = os.path.join(tmp_dir, "new_subdir")
        
        result = ensure_dir(new_dir)
        
        assert result == new_dir
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)

    def test_ensure_dir_existing(self, tmp_dir):
        """测试目录已存在的情况"""
        result = ensure_dir(tmp_dir)
        
        assert result == tmp_dir
        assert os.path.exists(tmp_dir)

    def test_ensure_dir_nested(self, tmp_dir):
        """测试创建嵌套目录"""
        nested_dir = os.path.join(tmp_dir, "level1", "level2", "level3")
        
        result = ensure_dir(nested_dir)
        
        assert result == nested_dir
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)


class TestFileSize:
    """文件大小测试"""

    def test_get_file_size_gb(self, tmp_dir):
        """测试获取文件大小"""
        test_file = os.path.join(tmp_dir, "test_file.bin")
        # 创建 1MB 文件
        with open(test_file, "wb") as f:
            f.write(b"0" * (1024 * 1024))
        
        size = get_file_size_gb(test_file)
        
        assert size > 0
        assert size < 0.01  # 1MB 约等于 0.001GB

    def test_get_file_size_gb_large(self, tmp_dir):
        """测试获取较大文件大小"""
        test_file = os.path.join(tmp_dir, "large_file.bin")
        # 创建 10MB 文件
        with open(test_file, "wb") as f:
            f.write(b"0" * (10 * 1024 * 1024))
        
        size = get_file_size_gb(test_file)
        
        assert size > 0.009  # 约 0.01GB
        assert size < 0.011

    def test_get_file_size_gb_nonexistent(self):
        """测试获取不存在文件的大小"""
        with pytest.raises(Exception):
            get_file_size_gb("/nonexistent/file.bin")


class TestDirSize:
    """目录大小测试"""

    def test_get_dir_size_gb_empty(self, tmp_dir):
        """测试空目录大小"""
        size = get_dir_size_gb(tmp_dir)
        
        assert size == 0.0

    def test_get_dir_size_gb_with_files(self, tmp_dir):
        """测试包含文件的目录大小"""
        # 创建多个文件
        for i in range(3):
            test_file = os.path.join(tmp_dir, f"file_{i}.bin")
            with open(test_file, "wb") as f:
                f.write(b"0" * (1024 * 1024))  # 1MB each
        
        size = get_dir_size_gb(tmp_dir)
        
        assert size > 0
        assert size < 0.01  # 3MB 约等于 0.003GB

    def test_get_dir_size_gb_nested(self, tmp_dir):
        """测试嵌套目录大小"""
        # 创建子目录和文件
        subdir = os.path.join(tmp_dir, "subdir")
        os.makedirs(subdir)
        
        for i in range(2):
            test_file = os.path.join(subdir, f"file_{i}.bin")
            with open(test_file, "wb") as f:
                f.write(b"0" * (1024 * 1024))
        
        size = get_dir_size_gb(tmp_dir)
        
        assert size > 0


class TestFindModelFiles:
    """模型文件查找测试"""

    def test_find_model_files_empty_dir(self, tmp_dir):
        """测试空目录查找"""
        files = find_model_files(tmp_dir)
        
        assert files == []

    def test_find_model_files_safetensors(self, tmp_dir):
        """测试查找 safetensors 文件"""
        # 创建测试文件
        test_file = os.path.join(tmp_dir, "model.safetensors")
        with open(test_file, "w") as f:
            f.write("test")
        
        files = find_model_files(tmp_dir)
        
        assert len(files) == 1
        assert files[0].endswith(".safetensors")

    def test_find_model_files_bin(self, tmp_dir):
        """测试查找 bin 文件"""
        test_file = os.path.join(tmp_dir, "model.bin")
        with open(test_file, "w") as f:
            f.write("test")
        
        files = find_model_files(tmp_dir)
        
        assert len(files) == 1
        assert files[0].endswith(".bin")

    def test_find_model_files_pt(self, tmp_dir):
        """测试查找 pt 文件"""
        test_file = os.path.join(tmp_dir, "model.pt")
        with open(test_file, "w") as f:
            f.write("test")
        
        files = find_model_files(tmp_dir)
        
        assert len(files) == 1
        assert files[0].endswith(".pt")

    def test_find_model_files_gguf(self, tmp_dir):
        """测试查找 gguf 文件"""
        test_file = os.path.join(tmp_dir, "model.gguf")
        with open(test_file, "w") as f:
            f.write("test")
        
        files = find_model_files(tmp_dir)
        
        assert len(files) == 1
        assert files[0].endswith(".gguf")

    def test_find_model_files_onnx(self, tmp_dir):
        """测试查找 onnx 文件"""
        test_file = os.path.join(tmp_dir, "model.onnx")
        with open(test_file, "w") as f:
            f.write("test")
        
        files = find_model_files(tmp_dir)
        
        assert len(files) == 1
        assert files[0].endswith(".onnx")

    def test_find_model_files_multiple(self, tmp_dir):
        """测试查找多个模型文件"""
        # 创建多个不同类型的模型文件
        files_to_create = [
            "model.safetensors",
            "model-001.bin",
            "model-002.bin",
            "config.json"  # 不是模型文件
        ]
        
        for filename in files_to_create:
            test_file = os.path.join(tmp_dir, filename)
            with open(test_file, "w") as f:
                f.write("test")
        
        files = find_model_files(tmp_dir)
        
        # 应该找到 3 个模型文件（不包括 config.json）
        assert len(files) == 3
        assert all(f.endswith((".safetensors", ".bin", ".pt", ".gguf", ".onnx")) for f in files)

    def test_find_model_files_nested(self, tmp_dir):
        """测试嵌套目录中的模型文件查找"""
        # 创建子目录
        subdir = os.path.join(tmp_dir, "subdir")
        os.makedirs(subdir)
        
        # 在不同层级创建文件
        file1 = os.path.join(tmp_dir, "model1.safetensors")
        file2 = os.path.join(subdir, "model2.bin")
        
        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")
        
        files = find_model_files(tmp_dir)
        
        assert len(files) == 2
        assert any("model1.safetensors" in f for f in files)
        assert any("model2.bin" in f for f in files)

    def test_find_model_files_sorted(self, tmp_dir):
        """测试返回的文件列表已排序"""
        # 创建多个文件
        for i in [3, 1, 2]:
            test_file = os.path.join(tmp_dir, f"model_{i}.bin")
            with open(test_file, "w") as f:
                f.write("test")
        
        files = find_model_files(tmp_dir)
        
        # 应该按字母顺序排序
        assert files == sorted(files)


class TestResolveModelPath:
    """模型路径解析测试"""

    def test_resolve_model_path_absolute(self):
        """测试绝对路径解析"""
        abs_path = os.path.abspath("/absolute/path/to/model")
        result = resolve_model_path(abs_path)

        assert result == abs_path

    def test_resolve_model_path_relative(self):
        """测试相对路径解析"""
        rel_path = "./relative/path"
        result = resolve_model_path(rel_path)

        assert os.path.isabs(result)
        assert "relative" in result
        assert "path" in result

    def test_resolve_model_path_home_expansion(self):
        """测试家目录展开"""
        home_path = "~/models/test"
        result = resolve_model_path(home_path)

        assert "~" not in result
        assert os.path.isabs(result)

    def test_resolve_model_path_env_var(self):
        """测试环境变量展开"""
        # Windows 使用 %VAR% 语法，Unix 使用 $VAR
        if os.name == "nt":
            os.environ["TEST_MODEL_DIR_WIN"] = "C:\\test\\dir"
            env_path = "%TEST_MODEL_DIR_WIN%/model"
            expected_segment = "C:\\test\\dir"
        else:
            os.environ["TEST_MODEL_DIR"] = "/test/dir"
            env_path = "$TEST_MODEL_DIR/model"
            expected_segment = "/test/dir"

        result = resolve_model_path(env_path)

        assert expected_segment in result

        # 清理环境变量
        for key in ["TEST_MODEL_DIR", "TEST_MODEL_DIR_WIN"]:
            os.environ.pop(key, None)


class TestIsModelPath:
    """模型路径验证测试"""

    def test_is_model_path_existing_local(self, tmp_dir):
        """测试存在的本地路径"""
        result = is_model_path(tmp_dir)
        
        assert result is True

    def test_is_model_path_hf_hub_id(self):
        """测试 HuggingFace Hub ID"""
        hf_id = "Qwen/Qwen2.5-0.5B"
        result = is_model_path(hf_id)
        
        assert result is True

    def test_is_model_path_invalid(self):
        """测试无效路径"""
        invalid_path = "invalid_path_without_slash"
        result = is_model_path(invalid_path)
        
        # 如果路径不存在且不是 HF Hub ID 格式，应该返回 False
        if not os.path.exists(invalid_path):
            assert result is False

    def test_is_model_path_hf_format_with_numbers(self):
        """测试带数字的 HF Hub ID"""
        hf_id = "meta-llama/Llama-2-7b-hf"
        result = is_model_path(hf_id)
        
        assert result is True

    def test_is_model_path_too_many_slashes(self):
        """测试过多斜杠的路径"""
        bad_path = "org/repo/extra"
        result = is_model_path(bad_path)
        
        # 不是标准的 HF Hub ID 格式（应该只有两部分）
        if not os.path.exists(bad_path):
            assert result is False


class TestGetModelFormat:
    """模型格式推断测试"""

    def test_get_model_format_gguf_file(self, tmp_dir):
        """测试 GGUF 文件格式推断"""
        gguf_file = os.path.join(tmp_dir, "model.gguf")
        with open(gguf_file, "w") as f:
            f.write("test")
        
        result = get_model_format(gguf_file)
        
        assert result == "gguf"

    def test_get_model_format_safetensors_dir(self, tmp_dir):
        """测试包含 safetensors 的目录"""
        # 创建 safetensors 文件
        test_file = os.path.join(tmp_dir, "model.safetensors")
        with open(test_file, "w") as f:
            f.write("test")
        
        result = get_model_format(tmp_dir)
        
        assert result == "safetensors"

    def test_get_model_format_pytorch_bin(self, tmp_dir):
        """测试包含 bin 文件的目录"""
        test_file = os.path.join(tmp_dir, "model.bin")
        with open(test_file, "w") as f:
            f.write("test")
        
        result = get_model_format(tmp_dir)
        
        assert result == "pytorch"

    def test_get_model_format_pytorch_pt(self, tmp_dir):
        """测试包含 pt 文件的目录"""
        test_file = os.path.join(tmp_dir, "model.pt")
        with open(test_file, "w") as f:
            f.write("test")
        
        result = get_model_format(tmp_dir)
        
        assert result == "pytorch"

    def test_get_model_format_unknown_empty(self, tmp_dir):
        """测试空目录格式推断"""
        result = get_model_format(tmp_dir)
        
        assert result == "unknown"

    def test_get_model_format_unknown_no_model_files(self, tmp_dir):
        """测试没有模型文件的目录"""
        # 创建非模型文件
        test_file = os.path.join(tmp_dir, "config.json")
        with open(test_file, "w") as f:
            f.write("{}")
        
        result = get_model_format(tmp_dir)
        
        assert result == "unknown"

    def test_get_model_format_priority(self, tmp_dir):
        """测试格式优先级（safetensors 优先于 bin）"""
        # 同时创建 safetensors 和 bin 文件
        safetensors_file = os.path.join(tmp_dir, "model.safetensors")
        bin_file = os.path.join(tmp_dir, "model.bin")
        
        with open(safetensors_file, "w") as f:
            f.write("test")
        with open(bin_file, "w") as f:
            f.write("test")
        
        result = get_model_format(tmp_dir)
        
        # safetensors 应该优先
        assert result == "safetensors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

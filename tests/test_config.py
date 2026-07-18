"""
配置模块单元测试

测试配置模块的所有功能，包括：
- 配置文件加载
- 配置生成
- 配置验证
- 模板管理
- 配置合并

使用 pytest 框架进行测试。
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hos_optimizer.config import (
    ConfigError,
    ConfigValidationError,
    ConfigConflictError,
    TemplateNotFoundError,
    ConfigManager,
    _BUILTIN_TEMPLATES,
    _8GB_OPTIMAL_CONFIGS,
)


class TestConfigManager:
    """配置管理器测试"""

    def test_config_manager_initialization_default(self):
        """测试配置管理器默认初始化"""
        manager = ConfigManager()
        
        assert manager.config_dir is not None
        assert manager._custom_templates == {}

    def test_config_manager_initialization_custom_dir(self, tmp_dir):
        """测试配置管理器自定义目录初始化"""
        config_dir = os.path.join(tmp_dir, "configs")
        manager = ConfigManager(config_dir=config_dir)
        
        assert str(manager.config_dir) == config_dir


class TestConfigLoading:
    """配置加载测试"""

    def test_load_config_success(self, tmp_dir):
        """测试成功加载配置文件"""
        config_path = os.path.join(tmp_dir, "test_config.yaml")
        config_data = {
            "backend": "vllm",
            "model": {
                "path": "/tmp/model",
                "format": "awq"
            },
            "inference": {
                "max_model_len": 512
            }
        }
        
        # 创建 YAML 文件
        with patch("hos_optimizer.config.yaml") as mock_yaml:
            mock_yaml.safe_load.return_value = config_data
            
            manager = ConfigManager()
            result = manager.load_config(config_path)
            
            assert result == config_data

    def test_load_config_file_not_found(self, tmp_dir):
        """测试加载不存在的配置文件"""
        config_path = os.path.join(tmp_dir, "nonexistent.yaml")
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.load_config(config_path)
        
        assert "配置文件不存在" in str(exc_info.value)

    def test_load_config_not_a_file(self, tmp_dir):
        """测试加载路径不是文件的情况"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.load_config(tmp_dir)  # 目录不是文件
        
        assert "路径不是文件" in str(exc_info.value)

    def test_load_config_yaml_error(self, tmp_dir):
        """测试 YAML 解析错误"""
        config_path = os.path.join(tmp_dir, "invalid.yaml")
        
        with patch("hos_optimizer.config.yaml") as mock_yaml:
            mock_yaml.safe_load.side_effect = Exception("YAML parse error")
            mock_yaml.YAMLError = Exception
            
            manager = ConfigManager()
            
            with pytest.raises(ConfigError) as exc_info:
                manager.load_config(config_path)
            
            assert "YAML 解析失败" in str(exc_info.value)

    def test_load_config_empty_file(self, tmp_dir):
        """测试加载空配置文件"""
        config_path = os.path.join(tmp_dir, "empty.yaml")
        
        with patch("hos_optimizer.config.yaml") as mock_yaml:
            mock_yaml.safe_load.return_value = None
            
            manager = ConfigManager()
            result = manager.load_config(config_path)
            
            assert result == {}

    def test_load_config_invalid_structure(self, tmp_dir):
        """测试配置文件结构不是字典"""
        config_path = os.path.join(tmp_dir, "list_config.yaml")
        
        with patch("hos_optimizer.config.yaml") as mock_yaml:
            mock_yaml.safe_load.return_value = [1, 2, 3]  # 列表而不是字典
            
            manager = ConfigManager()
            
            with pytest.raises(ConfigError) as exc_info:
                manager.load_config(config_path)
            
            assert "配置文件顶层结构必须是字典" in str(exc_info.value)


class TestConfigSaving:
    """配置保存测试"""

    def test_save_config_success(self, tmp_dir):
        """测试成功保存配置"""
        output_path = os.path.join(tmp_dir, "output.yaml")
        config_data = {
            "backend": "vllm",
            "model": {"path": "/tmp/model"}
        }
        
        with patch("hos_optimizer.config.yaml") as mock_yaml:
            manager = ConfigManager()
            manager.save_config(config_data, output_path)
            
            mock_yaml.dump.assert_called_once()

    def test_save_config_creates_directory(self, tmp_dir):
        """测试保存配置时创建目录"""
        output_path = os.path.join(tmp_dir, "new_dir", "output.yaml")
        config_data = {"backend": "vllm"}
        
        with patch("hos_optimizer.config.yaml"):
            manager = ConfigManager()
            manager.save_config(config_data, output_path)
            
            # 验证目录被创建
            assert os.path.exists(os.path.join(tmp_dir, "new_dir"))


class TestConfigMerging:
    """配置合并测试"""

    def test_merge_configs_simple(self):
        """测试简单配置合并"""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        
        manager = ConfigManager()
        result = manager.merge_configs(base, override)
        
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_configs_nested(self):
        """测试嵌套配置合并"""
        base = {
            "model": {
                "path": "/base",
                "format": "gguf"
            },
            "inference": {
                "n_ctx": 512
            }
        }
        override = {
            "model": {
                "path": "/override"
            },
            "inference": {
                "max_model_len": 2048
            }
        }
        
        manager = ConfigManager()
        result = manager.merge_configs(base, override)
        
        assert result["model"]["path"] == "/override"
        assert result["model"]["format"] == "gguf"
        assert result["inference"]["n_ctx"] == 512
        assert result["inference"]["max_model_len"] == 2048

    def test_merge_configs_deep_copy(self):
        """测试合并配置深拷贝"""
        base = {"model": {"path": "/base"}}
        override = {"model": {"format": "gguf"}}
        
        manager = ConfigManager()
        result = manager.merge_configs(base, override)
        
        # 修改原始配置不应该影响结果
        base["model"]["path"] = "/modified"
        assert result["model"]["path"] == "/base"


class TestConfigGeneration:
    """配置生成测试"""

    def test_generate_optimal_config_inference_0_8b(self):
        """测试生成 0.8B 模型推理配置"""
        manager = ConfigManager()
        
        config = manager.generate_optimal_config(
            scenario="inference_0.8b",
            model_path="/tmp/model",
            vram_gb=8.0
        )
        
        assert config["backend"] == "llama-cpp"
        assert config["model"]["path"] == "/tmp/model"
        assert "inference" in config

    def test_generate_optimal_config_inference_7b(self):
        """测试生成 7B 模型推理配置"""
        manager = ConfigManager()
        
        config = manager.generate_optimal_config(
            scenario="inference_7b",
            model_path="/tmp/model",
            vram_gb=8.0
        )
        
        assert config["backend"] == "llama-cpp"
        assert "inference" in config

    def test_generate_optimal_config_training_0_8b(self):
        """测试生成 0.8B 模型训练配置"""
        manager = ConfigManager()
        
        config = manager.generate_optimal_config(
            scenario="training_0.8b",
            model_path="/tmp/model",
            vram_gb=8.0
        )
        
        assert config["method"] == "qlora"
        assert "lora" in config
        assert "training" in config

    def test_generate_optimal_config_invalid_scenario(self):
        """测试生成无效场景配置"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.generate_optimal_config(
                scenario="invalid_scenario",
                model_path="/tmp/model"
            )
        
        assert "不支持的场景" in str(exc_info.value)

    def test_generate_optimal_config_low_vram(self):
        """测试低 VRAM 场景配置生成"""
        manager = ConfigManager()
        
        config = manager.generate_optimal_config(
            scenario="inference_7b",
            model_path="/tmp/model",
            vram_gb=4.0  # 低 VRAM
        )
        
        # 应该调整上下文长度
        assert config["inference"]["n_ctx"] <= 512

    def test_auto_select_scenario_inference_small(self):
        """测试自动选择小模型推理场景"""
        manager = ConfigManager()
        
        config = manager.auto_select_scenario(
            model_size_b=0.5,
            task="inference",
            vram_gb=8.0
        )
        
        assert config is not None
        assert "backend" in config

    def test_auto_select_scenario_training_large(self):
        """测试自动选择大模型训练场景"""
        manager = ConfigManager()
        
        config = manager.auto_select_scenario(
            model_size_b=7.0,
            task="training",
            vram_gb=8.0
        )
        
        assert config is not None
        assert "method" in config

    def test_auto_select_scenario_model_too_large(self):
        """测试模型过大时自动选择失败"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.auto_select_scenario(
                model_size_b=13.0,  # 超出 8GB VRAM 支持范围
                task="inference",
                vram_gb=8.0
            )
        
        assert "超出 8GB VRAM 场景支持范围" in str(exc_info.value)

    def test_auto_select_scenario_invalid_task(self):
        """测试无效任务类型"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.auto_select_scenario(
                model_size_b=0.5,
                task="invalid_task",
                vram_gb=8.0
            )
        
        assert "不支持的任务类型" in str(exc_info.value)


class TestConfigValidation:
    """配置验证测试"""

    def test_validate_config_valid(self):
        """测试有效配置验证"""
        config = {
            "backend": "vllm",
            "model": {
                "path": "/tmp/model",
                "format": "awq"
            },
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9
            },
            "inference": {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 512
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 可能有警告但没有错误
        assert isinstance(issues, list)

    def test_validate_config_fp16_bf16_conflict(self):
        """测试 fp16 和 bf16 冲突"""
        config = {
            "training": {
                "fp16": True,
                "bf16": True
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到冲突
        assert any("fp16 和 bf16 不能同时启用" in issue for issue in issues)

    def test_validate_config_gguf_with_dtype(self):
        """测试 GGUF 格式设置 dtype"""
        config = {
            "model": {
                "format": "gguf",
                "dtype": "float16"
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到冲突
        assert any("GGUF 格式不支持 dtype" in issue for issue in issues)

    def test_validate_config_tensor_parallel_too_large(self):
        """测试 tensor_parallel_size 过大"""
        config = {
            "inference": {
                "tensor_parallel_size": 2
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到冲突
        assert any("tensor_parallel_size" in issue for issue in issues)

    def test_validate_config_backend_mismatch(self):
        """测试后端不匹配"""
        config = {
            "backend": "llama-cpp",
            "inference": {
                "enable_prefix_caching": True
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到冲突
        assert any("llama-cpp 后端不支持" in issue for issue in issues)

    def test_validate_config_temperature_out_of_range(self):
        """测试 temperature 超出范围"""
        config = {
            "sampling": {
                "temperature": 3.0  # 超出 [0, 2.0]
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到范围问题
        assert any("temperature" in issue and "超出合理范围" in issue for issue in issues)

    def test_validate_config_top_p_out_of_range(self):
        """测试 top_p 超出范围"""
        config = {
            "sampling": {
                "top_p": 1.5  # 超出 [0, 1.0]
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到范围问题
        assert any("top_p" in issue and "超出合理范围" in issue for issue in issues)

    def test_validate_config_gpu_memory_utilization_out_of_range(self):
        """测试 gpu_memory_utilization 超出范围"""
        config = {
            "inference": {
                "gpu_memory_utilization": 0.98  # 超出 [0.5, 0.95]
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到范围问题
        assert any("gpu_memory_utilization" in issue for issue in issues)

    def test_validate_config_learning_rate_out_of_range(self):
        """测试 learning_rate 超出范围"""
        config = {
            "training": {
                "learning_rate": 0.1  # 超出 (0, 1e-2]
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到范围问题
        assert any("learning_rate" in issue for issue in issues)

    def test_validate_config_negative_context_length(self):
        """测试负数上下文长度"""
        config = {
            "inference": {
                "n_ctx": -100
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到范围问题
        assert any("n_ctx" in issue and "必须为正整数" in issue for issue in issues)

    def test_validate_config_empty_model_path(self):
        """测试空模型路径"""
        config = {
            "model": {
                "path": ""
            }
        }
        
        manager = ConfigManager()
        issues = manager.validate_config(config)
        
        # 应该检测到必填字段问题
        assert any("model.path 未设置" in issue for issue in issues)

    def test_validate_and_raise_success(self):
        """测试验证并抛出异常成功"""
        config = {
            "model": {
                "path": "/tmp/model"
            }
        }
        
        manager = ConfigManager()
        # 不应该抛出异常
        manager.validate_and_raise(config)

    def test_validate_and_raise_conflict(self):
        """测试验证发现冲突时抛出异常"""
        config = {
            "training": {
                "fp16": True,
                "bf16": True
            }
        }
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigConflictError) as exc_info:
            manager.validate_and_raise(config)
        
        assert "配置冲突" in str(exc_info.value)

    def test_validate_and_raise_validation_error(self):
        """测试验证发现错误时抛出异常"""
        config = {
            "model": {
                "path": ""
            },
            "sampling": {
                "temperature": 3.0
            }
        }
        
        manager = ConfigManager()
        
        with pytest.raises((ConfigValidationError, ConfigConflictError)):
            manager.validate_and_raise(config)


class TestTemplateManagement:
    """模板管理测试"""

    def test_list_templates(self):
        """测试列出所有模板"""
        manager = ConfigManager()
        templates = manager.list_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        # 应该包含内置模板
        assert "llama_cpp" in templates
        assert "vllm" in templates
        assert "sglang" in templates

    def test_get_template_builtin(self):
        """测试获取内置模板"""
        manager = ConfigManager()
        
        template = manager.get_template("llama_cpp")
        
        assert template is not None
        assert isinstance(template, dict)
        assert "backend" in template

    def test_get_template_not_found(self):
        """测试获取不存在的模板"""
        manager = ConfigManager()
        
        with pytest.raises(TemplateNotFoundError) as exc_info:
            manager.get_template("nonexistent_template")
        
        assert "模板" in str(exc_info.value)
        assert "不存在" in str(exc_info.value)

    def test_register_template(self):
        """测试注册自定义模板"""
        manager = ConfigManager()
        
        custom_template = {
            "backend": "custom",
            "model": {"path": ""}
        }
        
        manager.register_template("custom_template", custom_template)
        
        # 应该能够获取注册的模板
        template = manager.get_template("custom_template")
        assert template == custom_template

    def test_register_template_invalid_type(self):
        """测试注册非字典类型模板"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.register_template("invalid", [1, 2, 3])
        
        assert "模板必须是字典类型" in str(exc_info.value)

    def test_unregister_template(self):
        """测试注销自定义模板"""
        manager = ConfigManager()
        
        # 先注册
        custom_template = {"backend": "custom"}
        manager.register_template("to_remove", custom_template)
        
        # 再注销
        manager.unregister_template("to_remove")
        
        # 应该无法获取
        with pytest.raises(TemplateNotFoundError):
            manager.get_template("to_remove")

    def test_unregister_builtin_template(self):
        """测试注销内置模板失败"""
        manager = ConfigManager()
        
        with pytest.raises(ConfigError) as exc_info:
            manager.unregister_template("llama_cpp")
        
        assert "不能注销内置模板" in str(exc_info.value)

    def test_unregister_nonexistent_template(self):
        """测试注销不存在的模板"""
        manager = ConfigManager()
        
        with pytest.raises(TemplateNotFoundError):
            manager.unregister_template("nonexistent")

    def test_export_template(self, tmp_dir):
        """测试导出模板"""
        output_path = os.path.join(tmp_dir, "exported.yaml")
        
        with patch("hos_optimizer.config.yaml"):
            manager = ConfigManager()
            manager.export_template("llama_cpp", output_path)
            
            # 验证文件被创建
            assert os.path.exists(output_path)

    def test_load_template_from_file(self, tmp_dir):
        """测试从文件加载模板"""
        config_path = os.path.join(tmp_dir, "template.yaml")
        template_data = {
            "backend": "custom",
            "model": {"path": ""}
        }
        
        with patch("hos_optimizer.config.yaml") as mock_yaml:
            mock_yaml.safe_load.return_value = template_data
            
            manager = ConfigManager()
            manager.load_template_from_file("loaded_template", config_path)
            
            # 应该能够获取加载的模板
            template = manager.get_template("loaded_template")
            assert template == template_data


class TestNestedOperations:
    """嵌套操作测试"""

    def test_get_nested_simple(self):
        """测试简单嵌套读取"""
        config = {
            "model": {
                "path": "/tmp/model"
            }
        }
        
        result = ConfigManager._get_nested(config, "model.path")
        
        assert result == "/tmp/model"

    def test_get_nested_deep(self):
        """测试深层嵌套读取"""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
        
        result = ConfigManager._get_nested(config, "level1.level2.level3.value")
        
        assert result == "deep"

    def test_get_nested_not_found(self):
        """测试读取不存在的路径"""
        config = {
            "model": {
                "path": "/tmp/model"
            }
        }
        
        result = ConfigManager._get_nested(config, "model.nonexistent")
        
        assert result is None

    def test_set_nested_simple(self):
        """测试简单嵌套设置"""
        config = {
            "model": {
                "path": "/old"
            }
        }
        
        ConfigManager._set_nested(config, "model.path", "/new")
        
        assert config["model"]["path"] == "/new"

    def test_set_nested_create_intermediate(self):
        """测试设置时创建中间层级"""
        config = {}
        
        ConfigManager._set_nested(config, "model.path", "/tmp/model")
        
        assert config["model"]["path"] == "/tmp/model"

    def test_set_nested_deep(self):
        """测试深层嵌套设置"""
        config = {
            "level1": {
                "level2": {}
            }
        }
        
        ConfigManager._set_nested(config, "level1.level2.level3.value", "deep")
        
        assert config["level1"]["level2"]["level3"]["value"] == "deep"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

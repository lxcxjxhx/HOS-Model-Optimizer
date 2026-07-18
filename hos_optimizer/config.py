#!/usr/bin/env python3
"""
HOS Model Optimizer - 配置优化模块

提供配置管理功能：
- 配置文件加载和解析（YAML 格式）
- 8GB VRAM 场景最优配置生成
- 配置验证和冲突检测
- 配置模板管理

针对 8GB VRAM 场景进行了优化，支持推理、量化、训练和部署四大配置场景。

使用方法：
  # 生成 8GB VRAM 最优配置
  python config.py --generate --vram 8

  # 加载并验证配置文件
  python config.py --validate --config my_config.yaml

  # 列出所有配置模板
  python config.py --list-templates

  # 导出指定模板
  python config.py --export-template inference_8gb --output ./my_config.yaml
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# 异常定义
# ============================================================

class ConfigError(Exception):
    """配置相关异常的基类"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证失败时抛出"""
    pass


class ConfigConflictError(ConfigError):
    """配置项之间存在冲突时抛出"""
    pass


class TemplateNotFoundError(ConfigError):
    """请求的模板不存在时抛出"""
    pass


# ============================================================
# 默认配置模板
# ============================================================

# 推理配置模板 - llama.cpp 后端
_TEMPLATE_LLAMA_CPP = {
    "backend": "llama-cpp",
    "model": {
        "path": "",
        "format": "gguf",
        "quantization": "Q4_K_M",
    },
    "inference": {
        "n_ctx": 512,
        "n_batch": 1,
        "n_threads": 4,
        "n_gpu_layers": -1,
        "rope_freq_base": 0.0,
        "rope_freq_scale": 0.0,
    },
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 256,
    },
    "memory": {
        "use_mmap": True,
        "use_mlock": False,
        "offload_kqv": True,
    },
}

# 推理配置模板 - vLLM 后端
_TEMPLATE_VLLM = {
    "backend": "vllm",
    "model": {
        "path": "",
        "format": "awq",
        "quantization": "awq",
        "dtype": "float16",
    },
    "inference": {
        "max_model_len": 512,
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 16,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
    },
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "repeat_penalty": 1.1,
        "max_tokens": 256,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_type": "openai",
    },
}

# 推理配置模板 - SGLang 后端
_TEMPLATE_SGLANG = {
    "backend": "sglang",
    "model": {
        "path": "",
        "format": "awq",
        "quantization": "awq",
        "dtype": "float16",
    },
    "inference": {
        "context_length": 512,
        "mem_fraction_static": 0.88,
        "max_running_requests": 16,
        "tp_size": 1,
        "disable_radix_cache": False,
        "enable_overlap_schedule": True,
    },
    "sampling": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_new_tokens": 256,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 30000,
        "api_type": "openai",
    },
}

# 量化配置模板
_TEMPLATE_QUANTIZE = {
    "method": "gguf",
    "model": {
        "path": "",
        "output_path": "",
    },
    "quantization": {
        "format": "gguf",
        "type": "Q4_K_M",
        "bits": 4,
        "group_size": 128,
        "symmetric": False,
    },
    "calibration": {
        "dataset": "",
        "num_samples": 128,
        "max_length": 512,
    },
    "evaluation": {
        "compute_ppl": True,
        "perplexity_stride": 512,
    },
}

# 训练配置模板
_TEMPLATE_TRAINING = {
    "method": "qlora",
    "model": {
        "path": "",
        "max_seq_length": 512,
        "load_in_4bit": True,
        "use_gradient_checkpointing": True,
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "output_dir": "./output",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "fp16": True,
        "bf16": False,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "optim": "paged_adamw_32bit",
        "max_grad_norm": 0.3,
        "seed": 42,
    },
    "dataset": {
        "path": "",
        "format": "alpaca",
        "test_split_ratio": 0.1,
    },
}

# 部署配置模板
_TEMPLATE_DEPLOY = {
    "service_type": "api",
    "model": {
        "path": "",
        "format": "gguf",
        "quantization": "Q4_K_M",
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "max_concurrent_requests": 32,
        "timeout": 300,
    },
    "backend": "llama-cpp",
    "scaling": {
        "min_replicas": 1,
        "max_replicas": 1,
        "target_gpu_utilization": 0.80,
    },
    "logging": {
        "level": "INFO",
        "access_log": True,
        "metrics_enabled": True,
    },
}


# 所有内置模板注册表
_BUILTIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "llama_cpp": _TEMPLATE_LLAMA_CPP,
    "vllm": _TEMPLATE_VLLM,
    "sglang": _TEMPLATE_SGLANG,
    "quantize": _TEMPLATE_QUANTIZE,
    "training": _TEMPLATE_TRAINING,
    "deploy": _TEMPLATE_DEPLOY,
}


# ============================================================
# 8GB VRAM 场景最优配置
# ============================================================

# 不同模型大小在 8GB VRAM 下的推荐配置
_8GB_OPTIMAL_CONFIGS = {
    "inference_0.8b": {
        "backend": "llama-cpp",
        "model": {
            "path": "",
            "format": "gguf",
            "quantization": "Q4_K_M",
        },
        "inference": {
            "n_ctx": 2048,
            "n_batch": 1,
            "n_threads": 4,
            "n_gpu_layers": -1,
        },
        "expected_performance": {
            "tokens_per_second": "100+",
            "vram_usage_gb": "1-2",
        },
    },
    "inference_7b": {
        "backend": "llama-cpp",
        "model": {
            "path": "",
            "format": "gguf",
            "quantization": "Q4_K_M",
        },
        "inference": {
            "n_ctx": 512,
            "n_batch": 1,
            "n_threads": 4,
            "n_gpu_layers": -1,
        },
        "expected_performance": {
            "tokens_per_second": "35-45",
            "vram_usage_gb": "5-6",
        },
    },
    "training_0.8b": {
        "method": "qlora",
        "model": {
            "path": "",
            "max_seq_length": 512,
            "load_in_4bit": True,
            "use_gradient_checkpointing": True,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        "training": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-4,
            "fp16": True,
            "optim": "paged_adamw_32bit",
        },
        "expected_performance": {
            "vram_usage_gb": "2-3",
        },
    },
    "training_7b": {
        "method": "qlora",
        "model": {
            "path": "",
            "max_seq_length": 512,
            "load_in_4bit": True,
            "use_gradient_checkpointing": True,
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "learning_rate": 1e-4,
            "fp16": True,
            "optim": "paged_adamw_32bit",
        },
        "expected_performance": {
            "vram_usage_gb": "6-8",
        },
    },
    "high_throughput_serving": {
        "backend": "vllm",
        "model": {
            "path": "",
            "format": "awq",
            "quantization": "awq",
            "dtype": "float16",
        },
        "inference": {
            "max_model_len": 512,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 16,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
        },
        "expected_performance": {
            "description": "高吞吐服务场景",
            "vram_usage_gb": "6-8",
        },
    },
    "multi_turn_dialogue": {
        "backend": "sglang",
        "model": {
            "path": "",
            "format": "awq",
            "quantization": "awq",
            "dtype": "float16",
        },
        "inference": {
            "context_length": 1024,
            "mem_fraction_static": 0.88,
            "disable_radix_cache": False,
        },
        "expected_performance": {
            "description": "多轮对话场景，RadixAttention 缓存命中 40-70%",
            "vram_usage_gb": "5-7",
        },
    },
}


# ============================================================
# 配置验证规则
# ============================================================

# 互斥的配置项组合（同一组内不能同时为 True 或同时存在）
_CONFLICT_RULES: List[Dict[str, Any]] = [
    {
        "description": "fp16 和 bf16 不能同时启用",
        "keys": [("training.fp16", "training.bf16")],
        "condition": "both_true",
    },
    {
        "description": "GGUF 格式不支持 dtype 参数",
        "keys": [("model.format", "model.dtype")],
        "condition": "format_gguf_with_dtype",
    },
    {
        "description": "tensor_parallel_size > 1 需要多 GPU，与 8GB VRAM 场景冲突",
        "keys": [("inference.tensor_parallel_size",)],
        "condition": "value_gt",
        "threshold": 1,
    },
    {
        "description": "llama-cpp 后端不支持 enable_prefix_caching",
        "keys": [("backend", "inference.enable_prefix_caching")],
        "condition": "backend_mismatch",
        "expected_backend": "vllm",
    },
    {
        "description": "llama-cpp 后端不支持 disable_radix_cache",
        "keys": [("backend", "inference.disable_radix_cache")],
        "condition": "backend_mismatch",
        "expected_backend": "sglang",
    },
]


# ============================================================
# 核心类
# ============================================================

class ConfigManager:
    """配置管理器，负责加载、生成、验证和管理配置"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_dir: 配置文件目录，默认为项目根目录下的 configs/
        """
        if config_dir is None:
            self.config_dir = Path(__file__).parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)

        # 用户自定义模板注册表
        self._custom_templates: Dict[str, Dict[str, Any]] = {}

    # ----------------------------------------------------------
    # 配置文件加载和解析
    # ----------------------------------------------------------

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        从 YAML 文件加载配置

        Args:
            config_path: YAML 配置文件路径

        Returns:
            解析后的配置字典

        Raises:
            ConfigError: 文件不存在或解析失败
        """
        if yaml is None:
            raise ConfigError(
                "PyYAML 未安装，请运行: pip install pyyaml>=6.0"
            )

        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")
        if not path.is_file():
            raise ConfigError(f"路径不是文件: {config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 解析失败: {e}")

        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ConfigError(
                f"配置文件顶层结构必须是字典，当前为: {type(config).__name__}"
            )

        return config

    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        将配置保存为 YAML 文件

        Args:
            config: 配置字典
            output_path: 输出文件路径

        Raises:
            ConfigError: 保存失败
        """
        if yaml is None:
            raise ConfigError(
                "PyYAML 未安装，请运行: pip install pyyaml>=6.0"
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except (IOError, yaml.YAMLError) as e:
            raise ConfigError(f"保存配置文件失败: {e}")

    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个配置字典，override 中的值覆盖 base 中的值

        Args:
            base: 基础配置
            override: 覆盖配置

        Returns:
            合并后的新配置字典
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    # ----------------------------------------------------------
    # 8GB VRAM 最优配置生成
    # ----------------------------------------------------------

    def generate_optimal_config(
        self,
        scenario: str,
        model_path: str = "",
        vram_gb: float = 8.0,
    ) -> Dict[str, Any]:
        """
        根据使用场景和硬件条件生成最优配置

        Args:
            scenario: 场景名称，可选值：
                      inference_0.8b, inference_7b, training_0.8b,
                      training_7b, high_throughput_serving, multi_turn_dialogue
            model_path: 模型路径
            vram_gb: 可用显存大小（GB）

        Returns:
            最优配置字典

        Raises:
            ConfigError: 不支持的场景或显存不足
        """
        if scenario not in _8GB_OPTIMAL_CONFIGS:
            available = ", ".join(sorted(_8GB_OPTIMAL_CONFIGS.keys()))
            raise ConfigError(
                f"不支持的场景: {scenario}，可选: {available}"
            )

        if vram_gb > 8.0:
            vram_gb = 8.0

        config = copy.deepcopy(_8GB_OPTIMAL_CONFIGS[scenario])

        # 设置模型路径
        self._set_nested(config, "model.path", model_path)

        # 根据实际 VRAM 微调参数
        config = self._adjust_for_vram(config, vram_gb)

        return config

    def auto_select_scenario(
        self,
        model_size_b: float,
        task: str = "inference",
        vram_gb: float = 8.0,
    ) -> Dict[str, Any]:
        """
        根据模型大小和任务类型自动选择最优场景并生成配置

        Args:
            model_size_b: 模型大小（十亿参数）
            task: 任务类型，可选 inference / training / serving / dialogue
            vram_gb: 可用显存（GB）

        Returns:
            自动选择的最优配置
        """
        # 根据模型大小确定档位
        if model_size_b <= 1.0:
            size_key = "0.8b"
        elif model_size_b <= 8.0:
            size_key = "7b"
        else:
            raise ConfigError(
                f"模型大小 {model_size_b}B 超出 8GB VRAM 场景支持范围（最大 ~7B）"
            )

        # 映射任务类型到场景
        task_scenario_map = {
            "inference": f"inference_{size_key}",
            "training": f"training_{size_key}",
            "serving": "high_throughput_serving",
            "dialogue": "multi_turn_dialogue",
        }

        if task not in task_scenario_map:
            available = ", ".join(sorted(task_scenario_map.keys()))
            raise ConfigError(f"不支持的任务类型: {task}，可选: {available}")

        scenario = task_scenario_map[task]
        return self.generate_optimal_config(scenario, vram_gb=vram_gb)

    def _adjust_for_vram(
        self, config: Dict[str, Any], vram_gb: float
    ) -> Dict[str, Any]:
        """
        根据实际可用 VRAM 微调配置参数

        Args:
            config: 基础配置
            vram_gb: 可用显存（GB）

        Returns:
            调整后的配置
        """
        # 推理场景：根据显存调整上下文长度
        n_ctx = self._get_nested(config, "inference.n_ctx")
        if n_ctx is not None:
            if vram_gb < 4.0:
                config["inference"]["n_ctx"] = min(n_ctx, 256)
            elif vram_gb < 6.0:
                config["inference"]["n_ctx"] = min(n_ctx, 512)
            else:
                config["inference"]["n_ctx"] = min(n_ctx, 2048)

        # 推理场景：调整 GPU 内存利用率
        gpu_util = self._get_nested(config, "inference.gpu_memory_utilization")
        if gpu_util is not None:
            if vram_gb < 6.0:
                config["inference"]["gpu_memory_utilization"] = 0.85
            else:
                config["inference"]["gpu_memory_utilization"] = min(gpu_util, 0.90)

        # 训练场景：根据显存调整 LoRA 秩和批大小
        lora_r = self._get_nested(config, "lora.r")
        if lora_r is not None:
            if vram_gb < 4.0:
                config["lora"]["r"] = 4
                config["lora"]["lora_alpha"] = 8
                config["training"]["gradient_accumulation_steps"] = 64
            elif vram_gb < 6.0:
                config["lora"]["r"] = min(lora_r, 8)
                config["training"]["gradient_accumulation_steps"] = max(
                    self._get_nested(config, "training.gradient_accumulation_steps") or 16, 16
                )

        return config

    # ----------------------------------------------------------
    # 配置验证和冲突检测
    # ----------------------------------------------------------

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置并返回所有发现的问题列表

        Args:
            config: 待验证的配置字典

        Returns:
            问题描述字符串列表，空列表表示验证通过
        """
        issues: List[str] = []

        # 1. 检查互斥规则
        issues.extend(self._check_conflicts(config))

        # 2. 检查数值范围
        issues.extend(self._check_value_ranges(config))

        # 3. 检查必填字段
        issues.extend(self._check_required_fields(config))

        return issues

    def _check_conflicts(self, config: Dict[str, Any]) -> List[str]:
        """检测配置项之间的冲突"""
        issues: List[str] = []

        for rule in _CONFLICT_RULES:
            condition = rule["condition"]

            if condition == "both_true":
                for key_pair in rule["keys"]:
                    v1 = self._get_nested(config, key_pair[0])
                    v2 = self._get_nested(config, key_pair[1])
                    if v1 is True and v2 is True:
                        issues.append(
                            f"[冲突] {rule['description']}: "
                            f"{key_pair[0]}={v1}, {key_pair[1]}={v2}"
                        )

            elif condition == "format_gguf_with_dtype":
                fmt = self._get_nested(config, "model.format")
                dtype = self._get_nested(config, "model.dtype")
                if fmt == "gguf" and dtype is not None:
                    issues.append(
                        f"[冲突] {rule['description']}: "
                        f"format=gguf 时不应设置 dtype={dtype}"
                    )

            elif condition == "value_gt":
                for key in rule["keys"]:
                    val = self._get_nested(config, key)
                    threshold = rule.get("threshold", 1)
                    if val is not None and val > threshold:
                        issues.append(
                            f"[冲突] {rule['description']}: {key}={val}"
                        )

            elif condition == "backend_mismatch":
                backend = self._get_nested(config, "backend")
                expected = rule.get("expected_backend", "")
                # 检查第二个 key 是否存在且为 True / 非默认值
                if len(rule["keys"]) > 1:
                    feature_key = rule["keys"][1]
                    feature_val = self._get_nested(config, feature_key)
                    if backend is not None and backend != expected and feature_val:
                        issues.append(
                            f"[冲突] {rule['description']}: "
                            f"backend={backend} 不支持 {feature_key}={feature_val}"
                        )

        return issues

    def _check_value_ranges(self, config: Dict[str, Any]) -> List[str]:
        """检查数值参数的合理范围"""
        issues: List[str] = []

        # temperature 范围检查
        temp = self._get_nested(config, "sampling.temperature")
        if temp is not None and (temp < 0 or temp > 2.0):
            issues.append(
                f"[范围] sampling.temperature={temp} 超出合理范围 [0, 2.0]"
            )

        # top_p 范围检查
        top_p = self._get_nested(config, "sampling.top_p")
        if top_p is not None and (top_p < 0 or top_p > 1.0):
            issues.append(
                f"[范围] sampling.top_p={top_p} 超出合理范围 [0, 1.0]"
            )

        # gpu_memory_utilization 范围检查
        gpu_util = self._get_nested(config, "inference.gpu_memory_utilization")
        if gpu_util is not None and (gpu_util < 0.5 or gpu_util > 0.95):
            issues.append(
                f"[范围] inference.gpu_memory_utilization={gpu_util} "
                f"超出安全范围 [0.5, 0.95]"
            )

        # learning_rate 合理性检查
        lr = self._get_nested(config, "training.learning_rate")
        if lr is not None and (lr <= 0 or lr > 1e-2):
            issues.append(
                f"[范围] training.learning_rate={lr} 超出合理范围 (0, 1e-2]"
            )

        # n_ctx / max_model_len / context_length 正值检查
        for key in ["inference.n_ctx", "inference.max_model_len", "inference.context_length"]:
            val = self._get_nested(config, key)
            if val is not None and val <= 0:
                issues.append(f"[范围] {key}={val} 必须为正整数")

        # batch_size 正值检查
        for key in ["inference.n_batch", "training.per_device_train_batch_size"]:
            val = self._get_nested(config, key)
            if val is not None and val <= 0:
                issues.append(f"[范围] {key}={val} 必须为正整数")

        return issues

    def _check_required_fields(self, config: Dict[str, Any]) -> List[str]:
        """检查必填字段是否存在且非空"""
        issues: List[str] = []

        # 模型路径是关键字段
        model_path = self._get_nested(config, "model.path")
        if model_path is not None and model_path == "":
            issues.append("[必填] model.path 未设置，请指定模型路径")

        return issues

    def validate_and_raise(self, config: Dict[str, Any]) -> None:
        """
        验证配置，如果发现问题则抛出异常

        Args:
            config: 待验证的配置

        Raises:
            ConfigValidationError: 存在验证问题
            ConfigConflictError: 存在配置冲突
        """
        issues = self.validate_config(config)
        if not issues:
            return

        conflicts = [i for i in issues if i.startswith("[冲突]")]
        others = [i for i in issues if not i.startswith("[冲突]")]

        if conflicts:
            raise ConfigConflictError(
                "配置冲突:\n" + "\n".join(conflicts)
            )
        if others:
            raise ConfigValidationError(
                "配置验证失败:\n" + "\n".join(others)
            )

    # ----------------------------------------------------------
    # 配置模板管理
    # ----------------------------------------------------------

    def list_templates(self) -> List[str]:
        """
        列出所有可用的配置模板名称（内置 + 自定义）

        Returns:
            模板名称列表
        """
        all_names = sorted(_BUILTIN_TEMPLATES.keys())
        all_names.extend(sorted(self._custom_templates.keys()))
        return all_names

    def get_template(self, name: str) -> Dict[str, Any]:
        """
        获取指定名称的配置模板（深拷贝）

        Args:
            name: 模板名称

        Returns:
            模板配置字典

        Raises:
            TemplateNotFoundError: 模板不存在
        """
        if name in _BUILTIN_TEMPLATES:
            return copy.deepcopy(_BUILTIN_TEMPLATES[name])
        if name in self._custom_templates:
            return copy.deepcopy(self._custom_templates[name])
        raise TemplateNotFoundError(
            f"模板 '{name}' 不存在，可用模板: {', '.join(self.list_templates())}"
        )

    def register_template(self, name: str, template: Dict[str, Any]) -> None:
        """
        注册自定义配置模板

        Args:
            name: 模板名称
            template: 模板配置字典
        """
        if not isinstance(template, dict):
            raise ConfigError("模板必须是字典类型")
        self._custom_templates[name] = copy.deepcopy(template)

    def unregister_template(self, name: str) -> None:
        """
        注销自定义配置模板（不能注销内置模板）

        Args:
            name: 模板名称

        Raises:
            ConfigError: 尝试注销内置模板
            TemplateNotFoundError: 模板不存在
        """
        if name in _BUILTIN_TEMPLATES:
            raise ConfigError(f"不能注销内置模板: {name}")
        if name not in self._custom_templates:
            raise TemplateNotFoundError(f"自定义模板 '{name}' 不存在")
        del self._custom_templates[name]

    def export_template(self, name: str, output_path: str) -> None:
        """
        将模板导出为 YAML 文件

        Args:
            name: 模板名称
            output_path: 输出文件路径
        """
        template = self.get_template(name)
        self.save_config(template, output_path)

    def load_template_from_file(self, name: str, file_path: str) -> None:
        """
        从 YAML 文件加载并注册为自定义模板

        Args:
            name: 注册时使用的模板名称
            file_path: YAML 文件路径
        """
        config = self.load_config(file_path)
        self.register_template(name, config)

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    @staticmethod
    def _get_nested(config: Dict[str, Any], dotted_key: str) -> Any:
        """
        通过点分路径读取嵌套字典的值

        Args:
            config: 配置字典
            dotted_key: 点分路径，如 "inference.n_ctx"

        Returns:
            对应值，路径不存在时返回 None
        """
        keys = dotted_key.split(".")
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    @staticmethod
    def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
        """
        通过点分路径设置嵌套字典的值

        Args:
            config: 配置字典
            dotted_key: 点分路径
            value: 要设置的值
        """
        keys = dotted_key.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


# ============================================================
# 命令行接口
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="HOS Model Optimizer - 配置优化模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 8GB VRAM 推理配置（0.8B 模型）
  python config.py --generate --scenario inference_0.8b --model-path ./my_model

  # 自动生成最优配置
  python config.py --auto --model-size 0.8 --task inference

  # 验证配置文件
  python config.py --validate --config my_config.yaml

  # 列出所有模板
  python config.py --list-templates

  # 导出模板
  python config.py --export-template llama_cpp --output ./llama_cpp.yaml
        """,
    )

    # 操作模式（互斥）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate",
        action="store_true",
        help="生成指定场景的最优配置",
    )
    group.add_argument(
        "--auto",
        action="store_true",
        help="根据模型大小和任务类型自动生成最优配置",
    )
    group.add_argument(
        "--validate",
        action="store_true",
        help="验证配置文件的有效性",
    )
    group.add_argument(
        "--list-templates",
        action="store_true",
        help="列出所有可用的配置模板",
    )
    group.add_argument(
        "--export-template",
        type=str,
        metavar="NAME",
        help="导出指定名称的配置模板",
    )

    # 通用参数
    parser.add_argument(
        "--config", type=str, help="配置文件路径（用于 --validate）"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="输出文件路径"
    )

    # 生成相关参数
    parser.add_argument(
        "--scenario",
        type=str,
        help="场景名称（用于 --generate），可选: "
        + ", ".join(sorted(_8GB_OPTIMAL_CONFIGS.keys())),
    )
    parser.add_argument(
        "--model-path", type=str, default="", help="模型路径"
    )
    parser.add_argument(
        "--vram", type=float, default=8.0, help="可用显存大小（GB），默认 8"
    )

    # 自动生成相关参数
    parser.add_argument(
        "--model-size",
        type=float,
        help="模型大小（十亿参数），用于 --auto",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="inference",
        choices=["inference", "training", "serving", "dialogue"],
        help="任务类型（用于 --auto），默认 inference",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    命令行入口函数

    Args:
        argv: 命令行参数列表，默认使用 sys.argv

    Returns:
        退出码，0 表示成功
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    manager = ConfigManager()

    try:
        # 列出模板
        if args.list_templates:
            templates = manager.list_templates()
            print("可用的配置模板:")
            for name in templates:
                source = "内置" if name in _BUILTIN_TEMPLATES else "自定义"
                print(f"  - {name}  [{source}]")
            return 0

        # 导出模板
        if args.export_template:
            if not args.output:
                print("错误: 导出模板需要指定 --output 参数", file=sys.stderr)
                return 1
            manager.export_template(args.export_template, args.output)
            print(f"模板 '{args.export_template}' 已导出到: {args.output}")
            return 0

        # 验证配置
        if args.validate:
            if not args.config:
                print("错误: 验证配置需要指定 --config 参数", file=sys.stderr)
                return 1
            config = manager.load_config(args.config)
            issues = manager.validate_config(config)
            if not issues:
                print("配置验证通过 ✓")
                return 0
            else:
                print(f"发现 {len(issues)} 个问题:")
                for issue in issues:
                    print(f"  {issue}")
                return 1

        # 生成最优配置
        if args.generate:
            if not args.scenario:
                print(
                    "错误: 生成配置需要指定 --scenario 参数",
                    file=sys.stderr,
                )
                print(
                    "可选场景: " + ", ".join(sorted(_8GB_OPTIMAL_CONFIGS.keys())),
                    file=sys.stderr,
                )
                return 1
            config = manager.generate_optimal_config(
                scenario=args.scenario,
                model_path=args.model_path,
                vram_gb=args.vram,
            )
            if args.output:
                manager.save_config(config, args.output)
                print(f"配置已保存到: {args.output}")
            else:
                if yaml is not None:
                    print(yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False))
                else:
                    import json
                    print(json.dumps(config, indent=2, ensure_ascii=False))
            return 0

        # 自动生成
        if args.auto:
            if args.model_size is None:
                print("错误: 自动生成需要指定 --model-size 参数", file=sys.stderr)
                return 1
            config = manager.auto_select_scenario(
                model_size_b=args.model_size,
                task=args.task,
                vram_gb=args.vram,
            )
            if args.model_path:
                ConfigManager._set_nested(config, "model.path", args.model_path)
            if args.output:
                manager.save_config(config, args.output)
                print(f"配置已保存到: {args.output}")
            else:
                if yaml is not None:
                    print(yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False))
                else:
                    import json
                    print(json.dumps(config, indent=2, ensure_ascii=False))
            return 0

    except ConfigError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"未知错误: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

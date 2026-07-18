#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOS Model Optimizer - 小模型优化工具包

提供模型量化、推理、训练、合并和部署的完整工具链，
针对 8GB VRAM 场景进行了优化。

主要功能：
- 量化：GGUF、AWQ、GPTQ 等多种量化方法
- 推理：支持 llama-cpp、vLLM、SGLang 三种后端
- 训练：QLoRA/LoRA 微调
- 部署：一键部署 API 服务
- 配置：智能配置管理和优化
"""

__version__ = "1.0.0"
__author__ = "HOS Team"

# 导出主要类
from .config import ConfigManager
from .inference import UnifiedInferenceEngine
from .quantize import QuantizationError
from .train import TrainingConfig
from .deploy import HardwareDetector, ConfigSelector, ServiceLauncher

__all__ = [
    "__version__",
    "__author__",
    "ConfigManager",
    "UnifiedInferenceEngine",
    "QuantizationError",
    "TrainingConfig",
    "HardwareDetector",
    "ConfigSelector",
    "ServiceLauncher",
]

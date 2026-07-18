#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOS Model Optimizer - 工具函数模块

提供通用工具函数：
- 日志配置
- 文件操作工具
- 模型路径处理
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List


# ============================================================
# 日志配置
# ============================================================

def setup_logger(
    name: str = "hos_optimizer",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    """
    配置并返回日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
        fmt: 日志格式
        datefmt: 日期格式

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# 文件操作工具
# ============================================================

def ensure_dir(path: str) -> str:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_gb(path: str) -> float:
    """
    获取文件大小（GB）

    Args:
        path: 文件路径

    Returns:
        文件大小（GB）
    """
    return os.path.getsize(path) / (1024 ** 3)


def get_dir_size_gb(path: str) -> float:
    """
    获取目录总大小（GB）

    Args:
        path: 目录路径

    Returns:
        目录总大小（GB）
    """
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def find_model_files(path: str) -> List[str]:
    """
    在目录中查找模型文件

    Args:
        path: 搜索路径

    Returns:
        模型文件路径列表
    """
    extensions = (".safetensors", ".bin", ".pt", ".gguf", ".onnx")
    model_files = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(extensions):
                model_files.append(os.path.join(dirpath, f))
    return sorted(model_files)


# ============================================================
# 模型路径处理
# ============================================================

def resolve_model_path(path: str) -> str:
    """
    解析模型路径，支持相对路径和环境变量展开

    Args:
        path: 原始路径

    Returns:
        解析后的绝对路径
    """
    expanded = os.path.expandvars(os.path.expanduser(path))
    return os.path.abspath(expanded)


def is_model_path(path: str) -> bool:
    """
    判断路径是否为有效的模型路径（本地目录或 HF Hub ID）

    Args:
        path: 路径字符串

    Returns:
        是否为有效模型路径
    """
    # 本地路径检查
    if os.path.exists(path):
        return True
    # HF Hub ID 格式检查（如 "Qwen/Qwen2.5-0.5B"）
    if "/" in path and not os.path.sep in path.replace("/", os.path.sep):
        parts = path.split("/")
        if len(parts) == 2 and all(parts):
            return True
    return False


def get_model_format(path: str) -> str:
    """
    推断模型格式

    Args:
        path: 模型路径

    Returns:
        格式字符串：gguf / safetensors / pytorch / unknown
    """
    if path.endswith(".gguf"):
        return "gguf"
    if os.path.isdir(path):
        files = find_model_files(path)
        for f in files:
            if f.endswith(".safetensors"):
                return "safetensors"
            if f.endswith((".bin", ".pt")):
                return "pytorch"
    return "unknown"

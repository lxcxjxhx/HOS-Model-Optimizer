#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOS Model Optimizer 安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
this_directory = Path(__file__).parent
long_description = ""
if (this_directory / "README.md").exists():
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# 读取 requirements
def read_requirements(filename):
    """读取依赖文件"""
    requirements = []
    filepath = this_directory / filename
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    requirements.append(line)
    return requirements

setup(
    name="hos-model-optimizer",
    version="1.0.0",
    author="HOS Team",
    author_email="hos@example.com",
    description="HOS 小模型优化工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lxcxjxhx/HOS-Model-Optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "psutil>=5.9.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "peft>=0.5.0",
        "huggingface_hub>=0.19.0",
    ],
    extras_require={
        "quantization": [
            "autoawq>=0.1.0",
            "auto-gptq>=0.5.0",
            "bitsandbytes>=0.41.0",
        ],
        "inference": [
            "llama-cpp-python>=0.2.0",
        ],
        "vllm": [
            "vllm>=0.2.0",
        ],
        "sglang": [
            "sglang>=0.1.0",
        ],
        "training": [
            "unsloth>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "all": [
            "autoawq>=0.1.0",
            "auto-gptq>=0.5.0",
            "bitsandbytes>=0.41.0",
            "llama-cpp-python>=0.2.0",
            "vllm>=0.2.0",
            "sglang>=0.1.0",
            "unsloth>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hos-quantize=hos_optimizer.cli:quantize_cmd",
            "hos-infer=hos_optimizer.cli:infer_cmd",
            "hos-train=hos_optimizer.cli:train_cmd",
            "hos-merge=hos_optimizer.cli:merge_cmd",
            "hos-deploy=hos_optimizer.cli:deploy_cmd",
            "hos-config=hos_optimizer.cli:config_cmd",
            "hos-optimizer=hos_optimizer.cli:main",
            "hos-upload=hos_optimizer.cli:upload_cmd",
            "hos-run=hos_optimizer.cli:run_cmd",
        ],
    },
    keywords="llm optimization quantization inference training deployment",
    project_urls={
        "Bug Reports": "https://github.com/lxcxjxhx/HOS-Model-Optimizer/issues",
        "Source": "https://github.com/lxcxjxhx/HOS-Model-Optimizer",
    },
)

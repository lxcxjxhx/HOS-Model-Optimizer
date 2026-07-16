# HOS Model Optimizer

HOS Model Optimizer 是一个面向小模型（0.5B-7B 参数）的完整优化工具链，针对 **8GB VRAM** 场景进行了深度优化。

## 项目介绍

本项目提供从模型量化、推理加速、微调到一键部署的全流程解决方案，帮助开发者在有限硬件资源下高效运行大语言模型。

### 核心特性

- **多种量化方案**：支持 GGUF、AWQ、GPTQ 三种主流量化格式
- **三后端推理引擎**：llama-cpp-python、vLLM、SGLang，自动选择最优后端
- **低资源微调**：QLoRA/LoRA 微调，8GB VRAM 即可训练 7B 模型
- **一键部署**：自动检测硬件并选择最优配置，快速启动 API 服务
- **智能配置**：根据场景自动生成 8GB VRAM 最优配置

## 核心功能列表

| 功能模块 | 说明 | 命令行工具 |
|---------|------|-----------|
| 量化 | GGUF/AWQ/GPTQ 量化、PPL 评估、格式转换 | `hos-quantize` |
| 推理 | 三后端统一接口、API 服务、交互模式、性能基准测试 | `hos-infer` |
| 训练 | QLoRA/LoRA 微调、数据集处理、模型合并 | `hos-train` |
| 部署 | 硬件检测、自动配置、一键启动 API 服务 | `hos-deploy` |
| 配置 | 最优配置生成、配置验证、模板管理 | `hos-config` |

## 快速开始

### 安装

```bash
# 基础安装
pip install -e .

# 安装全部依赖
pip install -e ".[all]"

# 按需安装
pip install -e ".[quantization]"   # 量化相关
pip install -e ".[inference]"      # llama-cpp 推理
pip install -e ".[vllm]"           # vLLM 推理
pip install -e ".[sglang]"         # SGLang 推理
pip install -e ".[training]"       # 训练加速
```

### 5 分钟上手

```bash
# 1. 量化模型（GGUF Q4_K_M）
hos-quantize --method gguf --model ./model --output ./model.gguf

# 2. 推理测试
hos-infer --model ./model.gguf --prompt "你好，请介绍一下自己"

# 3. 启动 API 服务
hos-infer --model ./model.gguf --serve --port 8000

# 4. 一键部署（自动检测硬件并选择最优配置）
hos-deploy --model ./model.gguf --model-size 7.0

# 5. 生成 8GB VRAM 最优配置
hos-config --generate --scenario inference_7b --model-path ./model
```

## 安装说明

### 系统要求

- Python >= 3.8
- CUDA >= 11.8（GPU 加速需要）
- 推荐 8GB+ VRAM（针对 8GB 场景优化）

### 安装方式

```bash
# 从源码安装
git clone https://github.com/hos-team/hos-model-optimizer.git
cd HOS-Model-Optimizer
pip install -e .

# 安装可选依赖
pip install -e ".[all]"
```

详细安装指南请参考 [INSTALL.md](INSTALL.md)。

## 使用示例

### 量化

```bash
# GGUF 量化（推荐 8GB VRAM 场景）
hos-quantize --method gguf --model ./model --output ./model.gguf --quant-type Q4_K_M

# AWQ 4-bit 量化
hos-quantize --method awq --bits 4 --model ./model --output ./model-awq

# GPTQ 量化
hos-quantize --method gptq --bits 4 --model ./model --output ./model-gptq

# 评估量化质量（PPL）
hos-quantize --method evaluate --model ./model-awq
```

### 推理

```bash
# 自动选择后端，单次推理
hos-infer --model ./model --prompt "什么是SQL注入？"

# 指定 vLLM 后端，启动 API 服务
hos-infer --backend vllm --model ./model --serve --port 8000

# 交互模式
hos-infer --model ./model --chat

# 性能基准测试
hos-infer --model ./model --benchmark
```

### 训练

```bash
# QLoRA 微调（8GB VRAM 可用）
hos-train --model Qwen/Qwen2.5-0.5B --dataset ./data.json --method qlora

# LoRA 微调并自动合并
hos-train --model ./model --dataset ./data.json --method lora --merge

# 自定义训练参数
hos-train --model ./model --dataset ./data.json \
    --epochs 3 --batch-size 2 --lr 2e-4 \
    --lora-rank 16 --lora-alpha 32
```

### 部署

```bash
# 部署 7B 模型
hos-deploy --model ./model.gguf --model-size 7.0

# 高并发场景
hos-deploy --model ./model.gguf --model-size 7.0 --use-case high_concurrency

# 多轮对话场景
hos-deploy --model ./model.gguf --model-size 7.0 --use-case multi_turn
```

更多示例请参考 [EXAMPLES.md](EXAMPLES.md)。

## 配置说明

### 配置文件格式

配置文件使用 YAML 格式，支持以下场景模板：

| 模板名称 | 说明 |
|---------|------|
| `llama_cpp` | llama-cpp 推理配置 |
| `vllm` | vLLM 推理配置 |
| `sglang` | SGLang 推理配置 |
| `quantize` | 量化配置 |
| `training` | 训练配置 |
| `deploy` | 部署配置 |

### 生成最优配置

```bash
# 根据场景生成
hos-config --generate --scenario inference_7b --model-path ./model

# 自动生成（根据模型大小和任务类型）
hos-config --auto --model-size 7.0 --task inference

# 验证配置文件
hos-config --validate --config my_config.yaml

# 列出所有模板
hos-config --list-templates

# 导出模板
hos-config --export-template llama_cpp --output ./llama_cpp.yaml
```

### 8GB VRAM 场景预设

| 场景 | 推荐后端 | 预期性能 |
|------|---------|---------|
| `inference_0.8b` | llama-cpp | 100+ tokens/s, 1-2GB VRAM |
| `inference_7b` | llama-cpp | 35-45 tokens/s, 5-6GB VRAM |
| `training_0.8b` | QLoRA | 2-3GB VRAM |
| `training_7b` | QLoRA | 6-8GB VRAM |
| `high_throughput_serving` | vLLM | 高吞吐服务 |
| `multi_turn_dialogue` | SGLang | 多轮对话优化 |

## 性能优化建议

### 推理优化

1. **选择合适的量化格式**
   - GGUF Q4_K_M：8GB VRAM 首选，平衡速度和精度
   - AWQ 4-bit：vLLM 后端首选，精度损失最小
   - GPTQ 4-bit：通用性好，兼容性强

2. **调整上下文长度**
   - 8GB VRAM + 7B 模型：建议 `n_ctx=512` 或 `max_model_len=512`
   - 8GB VRAM + 0.8B 模型：可使用 `n_ctx=2048`

3. **GPU Offload 策略**
   - llama-cpp：设置 `n_gpu_layers=-1` 全部 offload 到 GPU
   - 如果 OOM，逐步减少 offload 层数

4. **批处理优化**
   - vLLM：利用 Continuous Batching，设置合理的 `max_num_seqs`
   - SGLang：利用 RadixAttention，多轮对话可加速 3-5x

### 训练优化

1. **使用 QLoRA**：4-bit 量化 + LoRA，大幅降低显存需求
2. **梯度检查点**：启用 `gradient_checkpointing=True`
3. **梯度累积**：使用 `gradient_accumulation_steps` 模拟大批次
4. **Unsloth 加速**：安装 `unsloth` 可获得 2x 训练加速

### 部署优化

1. **自动配置**：使用 `hos-deploy` 自动检测硬件并选择最优配置
2. **健康检查**：部署后自动执行健康检查确保服务正常
3. **监控显存**：使用 `VRAMCallback` 监控训练过程中的显存使用

## 项目结构

```
HOS-Model-Optimizer/
├── hos_optimizer/
│   ├── __init__.py          # 包初始化，导出主要类
│   ├── cli.py               # 统一命令行接口
│   ├── config.py            # 配置管理和优化
│   ├── quantize.py          # 量化模块（GGUF/AWQ/GPTQ）
│   ├── inference.py         # 推理模块（llama-cpp/vLLM/SGLang）
│   ├── train.py             # 训练模块（QLoRA/LoRA）
│   ├── deploy.py            # 部署模块（硬件检测/自动配置）
│   └── utils.py             # 工具函数
├── docs/
│   └── architecture.md      # 架构文档
├── pyproject.toml           # 项目配置
├── requirements.txt         # 依赖列表
├── setup.py                 # 安装脚本
├── LICENSE                  # MIT 许可证
├── README.md                # 项目主文档（本文件）
├── INSTALL.md               # 安装指南
├── EXAMPLES.md              # 使用示例
└── API.md                   # API 文档
```

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发规范

- 代码风格遵循 PEP 8，使用 `black` 格式化（行宽 100）
- 提交前运行 `flake8` 检查
- 新功能需要附带测试用例
- Commit message 使用语义化格式

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black hos_optimizer/

# 代码检查
flake8 hos_optimizer/
```

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

```
MIT License

Copyright (c) 2026 HOS Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 相关链接

- [安装指南](INSTALL.md) - 详细安装步骤
- [使用示例](EXAMPLES.md) - 完整使用示例
- [API 文档](API.md) - Python API 参考
- [架构文档](docs/architecture.md) - 系统架构说明
- [问题反馈](https://github.com/hos-team/hos-model-optimizer/issues)
- [项目源码](https://github.com/hos-team/hos-model-optimizer)

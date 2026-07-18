# HOS Model Optimizer Skill

## Description
HOS Model Optimizer 是一个面向小模型（0.5B-7B 参数）的完整优化工具链，针对 8GB VRAM 场景进行了深度优化。提供从模型量化、推理加速、微调到一键部署的全流程解决方案。

## When to Use
当用户需要以下操作时自动触发此技能：
- 量化大语言模型（GGUF/AWQ/GPTQ）
- 使用推理后端（llama-cpp/vLLM/SGLang）进行推理
- 微调训练模型（QLoRA/LoRA）
- 部署模型为 API 服务
- 配置和优化模型性能

## Capabilities
- **量化**: 支持 GGUF、AWQ、GPTQ 三种量化方案
- **推理**: 三后端统一接口，自动选择最优后端
- **训练**: QLoRA/LoRA 微调，8GB VRAM 即可训练 7B 模型
- **部署**: 硬件检测、自动配置、一键启动 API 服务
- **评测**: 支持 BLEU、ROUGE、PPL 等多种评测指标

## Commands

### 量化模型
```bash
hos-quantize --method gguf --model ./model --output ./model.gguf
hos-quantize --method awq --bits 4 --model ./model --output ./model-awq
```

### 推理服务
```bash
# 单次推理
hos-infer --model ./model --prompt "你好"

# 启动 API 服务
hos-infer --model ./model --serve --port 8000

# 交互模式
hos-infer --model ./model --chat
```

### 训练模型
```bash
# QLoRA 微调
hos-train --model ./model --dataset ./data.json --method qlora

# LoRA 微调并自动合并
hos-train --model ./model --dataset ./data.json --method lora --merge
```

### 部署服务
```bash
hos-deploy --model ./model.gguf --model-size 7.0
```

### 配置管理
```bash
# 生成最优配置
hos-config --generate --scenario inference_7b --model-path ./model

# 列出所有模板
hos-config --list-templates
```

### 模型评测
```bash
hos-evaluate --model ./model --dataset ./test.json --metrics bleu rouge
```

## Installation
```bash
cd HOS-Model-Optimizer
pip install -e .
```

## Python API
```python
from hos_optimizer import (
    ConfigManager,
    UnifiedInferenceEngine,
    TrainingConfig,
    HardwareDetector,
)

# 推理
engine = UnifiedInferenceEngine(model_path="./model")
result = engine.generate("你好")

# 训练
config = TrainingConfig(
    model_name_or_path="./model",
    dataset_path="./data.json",
    finetuning_type="qlora",
)
from hos_optimizer.train import train
train(config)
```

## Notes
- 所有命令都针对 8GB VRAM 场景进行了优化
- 支持 CPU 和 GPU 混合推理
- 自动检测硬件并选择最优配置

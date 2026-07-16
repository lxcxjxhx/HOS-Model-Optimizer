# 使用示例

本文档提供 HOS Model Optimizer 的详细使用示例，涵盖量化、推理、训练、部署等核心功能。

## 目录

- [量化示例](#量化示例)
- [推理示例](#推理示例)
- [训练示例](#训练示例)
- [部署示例](#部署示例)
- [性能测试示例](#性能测试示例)
- [8GB VRAM 场景优化示例](#8gb-vram-场景优化示例)

---

## 量化示例

### GGUF 量化（推荐 8GB VRAM 场景）

GGUF 是 llama.cpp 的原生格式，支持 CPU+GPU 混合推理，是 8GB VRAM 场景的首选。

```bash
# 基础 GGUF 量化（Q4_K_M，平衡速度和精度）
hos-quantize --method gguf --model ./model --output ./model.gguf

# 指定量化类型
hos-quantize --method gguf --model ./model --output ./model-q4.gguf --quant-type Q4_K_M
hos-quantize --method gguf --model ./model --output ./model-q5.gguf --quant-type Q5_K_M
hos-quantize --method gguf --model ./model --output ./model-q8.gguf --quant-type Q8_0

# 指定 llama.cpp 路径
hos-quantize --method gguf --model ./model --output ./model.gguf \
    --llama-cpp-path /path/to/llama.cpp
```

**Python API 示例**：

```python
from hos_optimizer.quantize import quantize_gguf

# GGUF 量化
output_path = quantize_gguf(
    model_path="./model",
    output_path="./model.gguf",
    quant_type="Q4_K_M",
    llama_cpp_path=None  # 从 PATH 查找
)
print(f"量化完成: {output_path}")
```

### AWQ 量化

AWQ（Activation-aware Weight Quantization）通过保护显著权重通道实现高精度 4-bit 量化。

```bash
# AWQ 4-bit 量化
hos-quantize --method awq --bits 4 --model ./model --output ./model-awq

# 自定义分组大小
hos-quantize --method awq --bits 4 --group-size 128 --model ./model --output ./model-awq
```

**Python API 示例**：

```python
from hos_optimizer.quantize import quantize_awq

# AWQ 量化
output_path = quantize_awq(
    model_path="./model",
    output_path="./model-awq",
    bits=4,
    group_size=128
)
```

### GPTQ 量化

GPTQ 基于 Optimal Brain Quantization 框架，通过逐层量化和误差补偿实现高精度。

```bash
# GPTQ 4-bit 量化
hos-quantize --method gptq --bits 4 --model ./model --output ./model-gptq

# GPTQ 8-bit 量化（精度更高）
hos-quantize --method gptq --bits 8 --model ./model --output ./model-gptq-8bit

# 自定义参数
hos-quantize --method gptq --bits 4 --group-size 128 --model ./model --output ./model-gptq
```

**Python API 示例**：

```python
from hos_optimizer.quantize import quantize_gptq

# GPTQ 量化
output_path = quantize_gptq(
    model_path="./model",
    output_path="./model-gptq",
    bits=4,
    group_size=128,
    desc_act=False
)
```

### 量化质量评估

评估量化后模型的 PPL（Perplexity），用于选择最佳量化方案。

```bash
# 评估量化模型 PPL
hos-quantize --method evaluate --model ./model-awq

# 指定评估数据集和样本数
hos-quantize --method evaluate --model ./model-gptq
```

**Python API 示例**：

```python
from hos_optimizer.quantize import evaluate_perplexity

# 评估 PPL
ppl = evaluate_perplexity(
    model_path="./model-awq",
    dataset="wikitext",
    max_samples=100,
    stride=512
)
print(f"PPL: {ppl:.2f}")
```

### 格式转换

在不同量化格式之间转换。

```bash
# HuggingFace -> GGUF
hos-quantize --method convert --from hf --to gguf --model ./model --output ./model.gguf

# HuggingFace -> AWQ
hos-quantize --method convert --from hf --to awq --model ./model --output ./model-awq

# HuggingFace -> GPTQ
hos-quantize --method convert --from hf --to gptq --model ./model --output ./model-gptq
```

**Python API 示例**：

```python
from hos_optimizer.quantize import convert_format

# 格式转换
output_path = convert_format(
    model_path="./model",
    output_path="./model-awq",
    from_format="hf",
    to_format="awq",
    bits=4
)
```

---

## 推理示例

### llama-cpp 推理（GGUF 格式）

llama-cpp 是 8GB VRAM 场景的首选后端，支持 CPU+GPU 混合推理。

```bash
# 单次推理
hos-infer --backend llama-cpp --model ./model.gguf --prompt "什么是SQL注入？"

# 启动 API 服务
hos-infer --backend llama-cpp --model ./model.gguf --serve --port 8080

# 交互模式
hos-infer --backend llama-cpp --model ./model.gguf --chat

# 自定义参数
hos-infer --backend llama-cpp --model ./model.gguf --prompt "你好" \
    --max-tokens 256 --temperature 0.7 --top-p 0.9
```

**Python API 示例**：

```python
from hos_optimizer.inference import UnifiedInferenceEngine

# 创建引擎（llama-cpp 后端）
engine = UnifiedInferenceEngine(
    model_path="./model.gguf",
    backend="llama-cpp",
    n_gpu_layers=-1,  # 全部 offload 到 GPU
    n_ctx=512,
    n_batch=512
)

# 单次推理
result = engine.generate(
    prompt="什么是SQL注入？",
    max_tokens=256,
    temperature=0.7
)
print(result.text)

# 批量推理
prompts = ["问题1", "问题2", "问题3"]
results = engine.generate_batch(prompts, max_tokens=128)
for r in results:
    print(r.text)

# 获取性能统计
stats = engine.get_stats()
print(stats.summary())

# 关闭引擎
engine.shutdown()
```

### vLLM 推理（高吞吐场景）

vLLM 使用 PagedAttention 和 Continuous Batching，适合高吞吐场景。

```bash
# 单次推理
hos-infer --backend vllm --model ./model-awq --prompt "什么是XSS攻击？"

# 启动 API 服务
hos-infer --backend vllm --model ./model-awq --serve --port 8000

# 自定义参数
hos-infer --backend vllm --model ./model-awq --prompt "你好" \
    --gpu-memory-utilization 0.9 --max-model-len 512
```

**Python API 示例**：

```python
from hos_optimizer.inference import UnifiedInferenceEngine

# 创建引擎（vLLM 后端）
engine = UnifiedInferenceEngine(
    model_path="./model-awq",
    backend="vllm",
    gpu_memory_utilization=0.9,
    max_model_len=512,
    max_num_seqs=128,
    dtype="float16"
)

# 单次推理
result = engine.generate(
    prompt="什么是XSS攻击？",
    max_tokens=256,
    temperature=0.7
)
print(result.text)

# 批量推理（利用 Continuous Batching）
prompts = ["问题1", "问题2", "问题3"]
results = engine.generate_batch(prompts, max_tokens=128)

# 性能统计
stats = engine.get_stats()
print(f"吞吐量: {stats.throughput_tokens_per_s:.1f} tokens/s")

engine.shutdown()
```

### SGLang 推理（结构化生成）

SGLang 支持 RadixAttention 和 JSON Schema 约束生成，适合多轮对话和结构化输出。

```bash
# 单次推理
hos-infer --backend sglang --model ./model-awq --prompt "什么是渗透测试？"

# 启动 API 服务
hos-infer --backend sglang --model ./model-awq --serve --port 30000

# 交互模式
hos-infer --backend sglang --model ./model-awq --chat
```

**Python API 示例**：

```python
from hos_optimizer.inference import UnifiedInferenceEngine

# 创建引擎（SGLang 后端）
engine = UnifiedInferenceEngine(
    model_path="./model-awq",
    backend="sglang",
    mem_fraction_static=0.9,
    context_length=512,
    tp_size=1
)

# 单次推理
result = engine.generate(
    prompt="什么是渗透测试？",
    max_tokens=256,
    temperature=0.7
)
print(result.text)

# 约束生成（JSON Schema）
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}
result = engine.generate(
    prompt="生成一个用户信息",
    json_schema=json_schema,
    max_tokens=128
)
print(result.text)  # 输出符合 JSON Schema

# 批量推理（利用 RadixAttention）
prompts = [
    "系统提示：你是助手。\n用户：你好",
    "系统提示：你是助手。\n用户：再见"
]
results = engine.generate_batch(prompts)  # 共享前缀会自动缓存

engine.shutdown()
```

### 自动选择后端

不指定后端时，系统会自动检测最优后端。

```bash
# 自动选择后端
hos-infer --model ./model --prompt "你好"

# 性能基准测试
hos-infer --model ./model --benchmark
```

**Python API 示例**：

```python
from hos_optimizer.inference import UnifiedInferenceEngine

# 自动选择后端
engine = UnifiedInferenceEngine(
    model_path="./model",
    backend=None,  # 自动检测
    auto_load=True
)

print(f"使用的后端: {engine.backend_name}")

result = engine.generate("你好")
print(result.text)

engine.shutdown()
```

---

## 训练示例

### QLoRA 训练（8GB VRAM 可用）

QLoRA 使用 4-bit 量化 + LoRA，大幅降低显存需求。

```bash
# 基础 QLoRA 训练
hos-train --model Qwen/Qwen2.5-0.5B --dataset ./data.json --method qlora

# 自定义训练参数
hos-train --model ./model --dataset ./data.json --method qlora \
    --epochs 3 --batch-size 2 --lr 2e-4 \
    --lora-rank 16 --lora-alpha 32 \
    --max-seq-length 512

# 训练后自动合并
hos-train --model ./model --dataset ./data.json --method qlora --merge

# 指定数据格式
hos-train --model ./model --dataset ./data.json --method qlora \
    --format alpaca  # 或 sharegpt
```

**Python API 示例**：

```python
from hos_optimizer.train import TrainingConfig, train

# 创建训练配置
config = TrainingConfig(
    model_name_or_path="Qwen/Qwen2.5-0.5B",
    dataset_path="./data.json",
    dataset_format="alpaca",
    finetuning_type="qlora",
    use_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_seq_length=512,
    output_dir="./output",
    gradient_checkpointing=True,
    use_unsloth=True  # 如果可用
)

# 执行训练
train(config)
```

### LoRA 训练

LoRA 使用全精度 + LoRA，精度更高但显存需求更大。

```bash
# LoRA 训练
hos-train --model ./model --dataset ./data.json --method lora

# 自定义参数
hos-train --model ./model --dataset ./data.json --method lora \
    --epochs 3 --batch-size 1 --lr 1e-4 \
    --lora-rank 8 --lora-alpha 16
```

**Python API 示例**：

```python
from hos_optimizer.train import TrainingConfig, train

# LoRA 配置
config = TrainingConfig(
    model_name_or_path="./model",
    dataset_path="./data.json",
    finetuning_type="lora",
    use_4bit=False,  # 不使用 4-bit
    lora_rank=8,
    lora_alpha=16,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
    output_dir="./output"
)

train(config)
```

### 模型合并

将 LoRA adapter 合并到基础模型。

```bash
# 合并模型
hos-merge --base-model ./base --adapter ./adapter --output ./merged
```

**Python API 示例**：

```python
from hos_optimizer.train import merge_model

# 合并模型
merge_model(
    base_model_path="./base",
    adapter_path="./adapter",
    output_path="./merged",
    trust_remote_code=True
)
```

### 数据集格式

#### Alpaca 格式

```json
[
  {
    "instruction": "什么是SQL注入？",
    "input": "",
    "output": "SQL注入是一种常见的Web安全漏洞..."
  },
  {
    "instruction": "解释XSS攻击",
    "input": "请举例说明",
    "output": "XSS（跨站脚本攻击）是..."
  }
]
```

#### ShareGPT 格式

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"}
    ]
  }
]
```

---

## 部署示例

### 一键部署

自动检测硬件并选择最优配置。

```bash
# 部署 7B 模型
hos-deploy --model ./model.gguf --model-size 7.0

# 指定使用场景
hos-deploy --model ./model.gguf --model-size 7.0 --use-case general
hos-deploy --model ./model.gguf --model-size 7.0 --use-case high_concurrency
hos-deploy --model ./model.gguf --model-size 7.0 --use-case multi_turn

# 自定义端口
hos-deploy --model ./model.gguf --model-size 7.0 --port 8000

# 仅生成配置，不启动服务
hos-deploy --model ./model.gguf --model-size 7.0 --no-auto-start
```

**Python API 示例**：

```python
from hos_optimizer.deploy import deploy_model

# 一键部署
launcher = deploy_model(
    model_path="./model.gguf",
    model_size_b=7.0,
    use_case="general",
    host="0.0.0.0",
    port=8000,
    auto_start=True
)

if launcher:
    print("服务已启动")
    # 等待用户中断
    try:
        launcher.process.wait()
    except KeyboardInterrupt:
        launcher.stop()
```

### 手动配置部署

```python
from hos_optimizer.deploy import HardwareDetector, ConfigSelector, ServiceLauncher

# 1. 检测硬件
hardware = HardwareDetector.get_hardware_info()
print(f"GPU: {hardware.gpu_name}, VRAM: {hardware.gpu_memory_gb:.1f}GB")

# 2. 选择配置
config = ConfigSelector.select_config(
    hardware=hardware,
    model_size_b=7.0,
    use_case="general"
)
print(f"推荐配置: {config.recommended_for}")

# 3. 启动服务
launcher = ServiceLauncher(
    config=config,
    model_path="./model.gguf",
    host="0.0.0.0",
    port=8000
)

if launcher.start():
    print("服务已启动")
    # 健康检查
    from hos_optimizer.deploy import HealthChecker
    checker = HealthChecker(host="localhost", port=8000)
    if checker.check_model_loaded(timeout=300):
        print("模型加载完成")
```

---

## 性能测试示例

### 推理性能基准测试

```bash
# 运行基准测试
hos-infer --model ./model --benchmark

# 指定后端
hos-infer --backend vllm --model ./model --benchmark
```

**Python API 示例**：

```python
from hos_optimizer.inference import UnifiedInferenceEngine, run_benchmark

# 创建引擎
engine = UnifiedInferenceEngine(
    model_path="./model",
    backend="vllm"
)

# 运行基准测试
run_benchmark(
    engine=engine,
    num_warmup=2,  # 预热次数
    num_runs=5     # 测试次数
)

# 获取性能统计
stats = engine.get_stats()
print(stats.summary())

engine.shutdown()
```

### 显存监控

```python
from hos_optimizer.inference import get_gpu_memory_usage_mb, get_total_gpu_memory_mb

# 获取当前显存占用
used_mb = get_gpu_memory_usage_mb()
total_mb = get_total_gpu_memory_mb()

print(f"显存占用: {used_mb:.1f} / {total_mb:.1f} MB")
print(f"使用率: {used_mb / total_mb * 100:.1f}%")
```

---

## 8GB VRAM 场景优化示例

### 推理优化

```python
from hos_optimizer.config import ConfigManager

# 生成 8GB VRAM 最优配置
manager = ConfigManager()
config = manager.generate_optimal_config(
    scenario="inference_7b",
    model_path="./model",
    vram_gb=8.0
)

print(f"推荐后端: {config['backend']}")
print(f"上下文长度: {config['inference']['n_ctx']}")
print(f"预期性能: {config['expected_performance']['tokens_per_second']} tokens/s")
```

### 训练优化

```python
from hos_optimizer.train import TrainingConfig

# 8GB VRAM 训练配置
config = TrainingConfig(
    model_name_or_path="./model",
    dataset_path="./data.json",
    finetuning_type="qlora",
    use_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    lora_rank=8,  # 降低 rank 节省显存
    lora_alpha=16,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 小批次
    gradient_accumulation_steps=32,  # 大累积步数
    learning_rate=1e-4,
    max_seq_length=512,  # 限制序列长度
    gradient_checkpointing=True,  # 启用梯度检查点
    optim="paged_adamw_32bit",  # 分页优化器
    output_dir="./output"
)
```

### 配置模板

```bash
# 列出所有配置模板
hos-config --list-templates

# 导出 8GB VRAM 推理配置
hos-config --export-template llama_cpp --output ./llama_cpp_8gb.yaml

# 生成特定场景配置
hos-config --generate --scenario inference_7b --model-path ./model --vram 8.0

# 自动生成最优配置
hos-config --auto --model-size 7.0 --task inference --vram 8.0
```

### 实际场景示例

#### 场景 1：8GB VRAM + 7B 模型推理

```bash
# 1. 量化模型
hos-quantize --method gguf --model ./model --output ./model-q4.gguf --quant-type Q4_K_M

# 2. 推理测试
hos-infer --backend llama-cpp --model ./model-q4.gguf --prompt "你好" \
    --n-gpu-layers -1 --n-ctx 512

# 3. 启动服务
hos-deploy --model ./model-q4.gguf --model-size 7.0
```

#### 场景 2：8GB VRAM + 7B 模型训练

```bash
# QLoRA 训练
hos-train --model Qwen/Qwen2.5-0.5B --dataset ./data.json --method qlora \
    --epochs 3 --batch-size 1 --grad-accum 32 \
    --lora-rank 8 --lora-alpha 16 \
    --max-seq-length 512 --lr 1e-4
```

#### 场景 3：高吞吐服务部署

```bash
# 使用 vLLM 后端
hos-deploy --model ./model-awq --model-size 7.0 --use-case high_concurrency

# 或手动配置
hos-infer --backend vllm --model ./model-awq --serve --port 8000 \
    --gpu-memory-utilization 0.9 --max-model-len 512 --max-tokens 256
```

#### 场景 4：多轮对话优化

```bash
# 使用 SGLang 后端（RadixAttention）
hos-deploy --model ./model-awq --model-size 7.0 --use-case multi_turn

# 或手动配置
hos-infer --backend sglang --model ./model-awq --serve --port 30000 \
    --mem-fraction-static 0.9 --context-length 1024
```

---

## 更多资源

- [API 文档](API.md) - 详细 API 参考
- [架构文档](docs/architecture.md) - 系统设计说明
- [安装指南](INSTALL.md) - 安装和配置

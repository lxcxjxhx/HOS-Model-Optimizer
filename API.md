# API 文档

本文档提供 HOS Model Optimizer 的 Python API 参考，涵盖所有核心模块的类、方法和函数。

## 目录

- [量化模块 API](#量化模块-api)
- [推理模块 API](#推理模块-api)
- [训练模块 API](#训练模块-api)
- [部署模块 API](#部署模块-api)
- [配置模块 API](#配置模块-api)
- [工具函数 API](#工具函数-api)

---

## 量化模块 API

量化模块位于 `hos_optimizer.quantize`，提供多种量化方法和评估工具。

### 异常类

#### `QuantizationError`

量化过程中的异常。

```python
from hos_optimizer.quantize import QuantizationError

try:
    quantize_gguf(...)
except QuantizationError as e:
    print(f"量化失败: {e}")
```

### 核心函数

#### `quantize_gguf()`

GGUF 量化 - 使用 llama.cpp 工具链。

```python
def quantize_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    llama_cpp_path: Optional[str] = None
) -> str
```

**参数**：
- `model_path` (str): 输入模型路径（HuggingFace 格式）
- `output_path` (str): 输出 GGUF 文件路径
- `quant_type` (str): 量化类型，如 `Q4_K_M`, `Q5_K_M`, `Q8_0` 等，默认 `Q4_K_M`
- `llama_cpp_path` (Optional[str]): llama.cpp 安装路径，为 None 则从 PATH 查找

**返回**：`str` - 输出文件路径

**异常**：`QuantizationError` - 量化失败时抛出

**示例**：

```python
from hos_optimizer.quantize import quantize_gguf

output = quantize_gguf(
    model_path="./model",
    output_path="./model.gguf",
    quant_type="Q4_K_M"
)
```

#### `quantize_awq()`

AWQ 4-bit 量化 - 激活感知量化，精度损失最小。

```python
def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128
) -> str
```

**参数**：
- `model_path` (str): 输入模型路径
- `output_path` (str): 输出模型路径
- `bits` (int): 量化位数，默认 4
- `group_size` (int): 分组大小，默认 128

**返回**：`str` - 输出模型路径

**异常**：`QuantizationError` - 量化失败时抛出

**示例**：

```python
from hos_optimizer.quantize import quantize_awq

output = quantize_awq(
    model_path="./model",
    output_path="./model-awq",
    bits=4,
    group_size=128
)
```

#### `quantize_gptq()`

GPTQ 量化 - 基于 Optimal Brain Quantization 框架。

```python
def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False
) -> str
```

**参数**：
- `model_path` (str): 输入模型路径
- `output_path` (str): 输出模型路径
- `bits` (int): 量化位数，4 或 8
- `group_size` (int): 分组大小，默认 128
- `desc_act` (bool): 是否按激活值排序，默认 False

**返回**：`str` - 输出模型路径

**异常**：`QuantizationError` - 量化失败时抛出

**示例**：

```python
from hos_optimizer.quantize import quantize_gptq

output = quantize_gptq(
    model_path="./model",
    output_path="./model-gptq",
    bits=4,
    group_size=128
)
```

#### `evaluate_perplexity()`

评估量化模型的 PPL（Perplexity）。

```python
def evaluate_perplexity(
    model_path: str,
    dataset: str = "wikitext",
    max_samples: int = 100,
    stride: int = 512
) -> float
```

**参数**：
- `model_path` (str): 模型路径
- `dataset` (str): 评估数据集名称，默认 `wikitext`
- `max_samples` (int): 最大评估样本数
- `stride` (int): 滑动窗口步长

**返回**：`float` - PPL 值

**异常**：`QuantizationError` - 评估失败时抛出

**示例**：

```python
from hos_optimizer.quantize import evaluate_perplexity

ppl = evaluate_perplexity(
    model_path="./model-awq",
    dataset="wikitext",
    max_samples=100
)
print(f"PPL: {ppl:.2f}")
```

#### `convert_format()`

量化格式转换工具。

```python
def convert_format(
    model_path: str,
    output_path: str,
    from_format: str,
    to_format: str,
    **kwargs
) -> str
```

**参数**：
- `model_path` (str): 输入模型路径
- `output_path` (str): 输出模型路径
- `from_format` (str): 源格式 (`gguf`, `awq`, `gptq`, `hf`)
- `to_format` (str): 目标格式 (`gguf`, `awq`, `gptq`, `hf`)
- `**kwargs`: 其他参数（如 `bits`, `quant_type`）

**返回**：`str` - 输出模型路径

**异常**：`QuantizationError` - 转换失败时抛出

**示例**：

```python
from hos_optimizer.quantize import convert_format

output = convert_format(
    model_path="./model",
    output_path="./model-awq",
    from_format="hf",
    to_format="awq",
    bits=4
)
```

### 辅助函数

#### `check_vram_availability()`

检查 GPU VRAM 可用性。

```python
def check_vram_availability() -> Dict[str, Any]
```

**返回**：`Dict[str, Any]` - 包含 VRAM 信息的字典

```python
{
    "available": True,
    "total_vram_gb": 8.0,
    "free_vram_gb": 6.5,
    "device": "NVIDIA GeForce RTX 3070"
}
```

#### `optimize_for_low_vram()`

根据 VRAM 限制优化配置。

```python
def optimize_for_low_vram(config: Dict[str, Any]) -> Dict[str, Any]
```

**参数**：
- `config` (Dict[str, Any]): 原始配置

**返回**：`Dict[str, Any]` - 优化后的配置

#### `get_model_size()`

获取模型文件大小（GB）。

```python
def get_model_size(model_path: str) -> float
```

**参数**：
- `model_path` (str): 模型路径

**返回**：`float` - 模型大小（GB）

---

## 推理模块 API

推理模块位于 `hos_optimizer.inference`，提供统一的推理接口和三个后端实现。

### 数据结构

#### `InferenceRequest`

推理请求数据结构。

```python
@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: Optional[List[str]] = None
    json_schema: Optional[Dict[str, Any]] = None  # SGLang 约束生成
    extra: Optional[Dict[str, Any]] = None
```

#### `InferenceResult`

推理结果数据结构。

```python
@dataclass
class InferenceResult:
    text: str
    token_ids: List[int] = field(default_factory=list)
    prompt: str = ""
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### `PerformanceStats`

性能统计数据结构。

```python
@dataclass
class PerformanceStats:
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    wall_time_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float: ...

    @property
    def throughput_tokens_per_s(self) -> float: ...

    @property
    def requests_per_s(self) -> float: ...

    def summary(self) -> str: ...
```

### 核心类

#### `UnifiedInferenceEngine`

统一推理引擎，封装后端选择和配置优化逻辑。

```python
class UnifiedInferenceEngine:
    def __init__(
        self,
        model_path: str,
        backend: Optional[str] = None,
        auto_load: bool = True,
        **kwargs
    )
```

**参数**：
- `model_path` (str): 模型路径或 HF 仓库 ID
- `backend` (Optional[str]): 推理后端名称（`llama_cpp`/`vllm`/`sglang`），为 None 时自动检测
- `auto_load` (bool): 是否自动加载模型，默认 True
- `**kwargs`: 传递给后端的额外参数

**方法**：

##### `generate()`

单次推理生成。

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    stop: Optional[List[str]] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    **extra
) -> InferenceResult
```

**参数**：
- `prompt` (str): 输入提示文本
- `max_tokens` (int): 最大生成 token 数
- `temperature` (float): 采样温度（0 = 贪婪解码）
- `top_p` (float): nucleus sampling 参数
- `top_k` (int): top-k sampling 参数
- `stop` (Optional[List[str]]): 停止词列表
- `json_schema` (Optional[Dict[str, Any]]): JSON Schema 约束（仅 SGLang 支持）
- `**extra`: 传递给后端的额外参数

**返回**：`InferenceResult` - 推理结果

##### `generate_batch()`

批量推理生成。

```python
def generate_batch(
    self,
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **extra
) -> List[InferenceResult]
```

**参数**：
- `prompts` (List[str]): 输入提示文本列表
- `max_tokens` (int): 最大生成 token 数
- `temperature` (float): 采样温度
- `top_p` (float): nucleus sampling 参数
- `**extra`: 传递给后端的额外参数

**返回**：`List[InferenceResult]` - 推理结果列表

##### `get_stats()`

获取性能统计。

```python
def get_stats(self) -> PerformanceStats
```

**返回**：`PerformanceStats` - 性能统计

##### `shutdown()`

关闭引擎，释放资源。

```python
def shutdown(self) -> None
```

**属性**：

##### `backend_name`

当前使用的后端名称。

```python
@property
def backend_name(self) -> str
```

**示例**：

```python
from hos_optimizer.inference import UnifiedInferenceEngine

# 创建引擎
engine = UnifiedInferenceEngine(
    model_path="./model",
    backend="vllm",
    gpu_memory_utilization=0.9
)

# 单次推理
result = engine.generate("你好", max_tokens=256)
print(result.text)

# 批量推理
results = engine.generate_batch(["问题1", "问题2"])

# 性能统计
stats = engine.get_stats()
print(stats.summary())

# 关闭
engine.shutdown()
```

### 后端类

#### `LlamaCppBackend`

llama-cpp-python 推理后端。

```python
class LlamaCppBackend(InferenceBackend):
    def __init__(self, model_path: str, **kwargs)
```

**参数**：
- `model_path` (str): GGUF 模型路径
- `n_gpu_layers` (int): GPU offload 层数，-1 表示全部，默认 -1
- `n_ctx` (int): 上下文长度，默认 4096
- `n_threads` (int): CPU 线程数
- `use_mmap` (bool): 是否使用 mmap，默认 True
- `use_mlock` (bool): 是否使用 mlock，默认 False
- `n_batch` (int): 批处理大小，默认 512

**方法**：
- `load()`: 加载模型
- `generate(request: InferenceRequest) -> InferenceResult`: 单次推理
- `generate_batch(requests: List[InferenceRequest]) -> List[InferenceResult]`: 批量推理
- `serve(host: str = "0.0.0.0", port: int = 8080)`: 启动 API 服务
- `shutdown()`: 释放资源

#### `VLLMBackend`

vLLM 推理后端。

```python
class VLLMBackend(InferenceBackend):
    def __init__(self, model_path: str, **kwargs)
```

**参数**：
- `model_path` (str): 模型路径
- `gpu_memory_utilization` (float): 显存利用率，默认 0.9
- `max_model_len` (int): 最大模型长度，默认 4096
- `max_num_seqs` (int): 最大并发序列数，默认 128
- `dtype` (str): 推理精度，默认 `float16`
- `tensor_parallel_size` (int): tensor parallel 数量，默认 1
- `enforce_eager` (bool): 是否使用 eager 模式，默认 False
- `trust_remote_code` (bool): 是否信任远程代码，默认 True

**方法**：
- `load()`: 加载模型
- `generate(request: InferenceRequest) -> InferenceResult`: 单次推理
- `generate_batch(requests: List[InferenceRequest]) -> List[InferenceResult]`: 批量推理
- `serve(host: str = "0.0.0.0", port: int = 8000)`: 启动 API 服务
- `shutdown()`: 释放资源

#### `SGLangBackend`

SGLang 推理后端。

```python
class SGLangBackend(InferenceBackend):
    def __init__(self, model_path: str, **kwargs)
```

**参数**：
- `model_path` (str): 模型路径
- `mem_fraction_static` (float): 静态显存分配比例，默认 0.9
- `context_length` (int): 上下文长度，默认 4096
- `tp_size` (int): tensor parallel 数量，默认 1
- `trust_remote_code` (bool): 是否信任远程代码，默认 True

**方法**：
- `load()`: 加载模型
- `generate(request: InferenceRequest) -> InferenceResult`: 单次推理（支持 JSON Schema 约束）
- `generate_batch(requests: List[InferenceRequest]) -> List[InferenceResult]`: 批量推理
- `serve(host: str = "0.0.0.0", port: int = 30000)`: 启动 API 服务
- `shutdown()`: 释放资源

### 性能监控

#### `PerformanceMonitor`

性能监控器，跟踪吞吐量、延迟和显存占用。

```python
class PerformanceMonitor:
    def __init__(self)
    def start(self) -> None
    def end(self) -> None
    def record_request(self, prompt_tokens: int, completion_tokens: int, latency_ms: float) -> None
    def update_peak_vram(self) -> None
    def reset(self) -> None

    @property
    def stats(self) -> PerformanceStats
```

### 工具函数

#### `detect_best_backend()`

自动检测最优推理后端。

```python
def detect_best_backend() -> str
```

**返回**：`str` - 后端名称（`vllm`/`sglang`/`llama_cpp`）

#### `get_gpu_memory_usage_mb()`

获取当前 GPU 显存占用（MB）。

```python
def get_gpu_memory_usage_mb() -> float
```

**返回**：`float` - 显存占用（MB）

#### `get_total_gpu_memory_mb()`

获取 GPU 总显存（MB）。

```python
def get_total_gpu_memory_mb() -> float
```

**返回**：`float` - 总显存（MB）

---

## 训练模块 API

训练模块位于 `hos_optimizer.train`，提供 QLoRA/LoRA 微调功能。

### 核心类

#### `TrainingConfig`

训练配置类。

```python
@dataclass
class TrainingConfig:
    # 模型配置
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code: bool = True

    # 训练方法
    finetuning_type: str = "qlora"  # "qlora" 或 "lora"

    # 数据集配置
    dataset_path: str = ""
    dataset_format: str = "alpaca"  # "alpaca" 或 "sharegpt"
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4

    # QLoRA 量化配置
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA 配置
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["all"])

    # 训练参数
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # 日志和保存
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # 精度配置
    bf16: bool = True
    fp16: bool = False

    # 性能优化
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"

    # 其他
    seed: int = 42
    use_unsloth: bool = True
```

#### `DatasetProcessor`

数据集处理器。

```python
class DatasetProcessor:
    def __init__(self, tokenizer, max_seq_length: int = 2048)

    def format_alpaca(self, example: Dict) -> Dict
    def format_sharegpt(self, example: Dict) -> Dict
    def tokenize_function(self, example: Dict) -> Dict
    def process_dataset(self, dataset: Dataset, dataset_format: str = "alpaca") -> Dataset
```

### 核心函数

#### `train()`

执行训练。

```python
def train(config: TrainingConfig) -> None
```

**参数**：
- `config` (TrainingConfig): 训练配置

**示例**：

```python
from hos_optimizer.train import TrainingConfig, train

config = TrainingConfig(
    model_name_or_path="./model",
    dataset_path="./data.json",
    finetuning_type="qlora",
    num_train_epochs=3
)

train(config)
```

#### `merge_model()`

合并 LoRA 权重到基础模型。

```python
def merge_model(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool = True
) -> None
```

**参数**：
- `base_model_path` (str): 基础模型路径
- `adapter_path` (str): LoRA adapter 路径
- `output_path` (str): 输出路径
- `trust_remote_code` (bool): 是否信任远程代码

**示例**：

```python
from hos_optimizer.train import merge_model

merge_model(
    base_model_path="./base",
    adapter_path="./adapter",
    output_path="./merged"
)
```

#### `load_and_process_dataset()`

加载并处理数据集。

```python
def load_and_process_dataset(
    dataset_path: str,
    tokenizer,
    dataset_format: str = "alpaca",
    max_seq_length: int = 2048,
    test_size: float = 0.05
) -> DatasetDict
```

**参数**：
- `dataset_path` (str): 数据集路径（JSON 文件）
- `tokenizer`: 分词器
- `dataset_format` (str): 数据格式（`alpaca` 或 `sharegpt`）
- `max_seq_length` (int): 最大序列长度
- `test_size` (float): 测试集比例

**返回**：`DatasetDict` - 包含 train 和 test 的 DatasetDict

---

## 部署模块 API

部署模块位于 `hos_optimizer.deploy`，提供硬件检测、自动配置和服务启动功能。

### 数据结构

#### `HardwareInfo`

硬件信息数据类。

```python
@dataclass
class HardwareInfo:
    gpu_name: str
    gpu_memory_gb: float
    cuda_version: Optional[str]
    cpu_cores: int
    system_memory_gb: float
```

#### `DeploymentConfig`

部署配置数据类。

```python
@dataclass
class DeploymentConfig:
    backend: Backend
    quantization: Quantization
    max_model_size_gb: float
    recommended_for: str
    startup_args: Dict
```

### 核心类

#### `HardwareDetector`

硬件检测器。

```python
class HardwareDetector:
    @staticmethod
    def detect_gpu() -> Tuple[str, float, Optional[str]]
    @staticmethod
    def detect_cpu() -> int
    @staticmethod
    def detect_system_memory() -> float
    @classmethod
    def get_hardware_info(cls) -> HardwareInfo
```

**示例**：

```python
from hos_optimizer.deploy import HardwareDetector

hardware = HardwareDetector.get_hardware_info()
print(f"GPU: {hardware.gpu_name}")
print(f"VRAM: {hardware.gpu_memory_gb:.1f}GB")
print(f"CPU cores: {hardware.cpu_cores}")
```

#### `ConfigSelector`

自动配置选择器。

```python
class ConfigSelector:
    @staticmethod
    def select_config(
        hardware: HardwareInfo,
        model_size_b: float,
        use_case: str = "general"
    ) -> DeploymentConfig
```

**参数**：
- `hardware` (HardwareInfo): 硬件信息
- `model_size_b` (float): 模型大小（十亿参数）
- `use_case` (str): 使用场景（`general`, `high_concurrency`, `multi_turn`）

**返回**：`DeploymentConfig` - 推荐的部署配置

**示例**：

```python
from hos_optimizer.deploy import HardwareDetector, ConfigSelector

hardware = HardwareDetector.get_hardware_info()
config = ConfigSelector.select_config(hardware, model_size_b=7.0, use_case="general")
print(f"推荐: {config.recommended_for}")
```

#### `ServiceLauncher`

服务启动器。

```python
class ServiceLauncher:
    def __init__(self, config: DeploymentConfig, model_path: str, host: str = "0.0.0.0", port: int = 8000)
    def start(self) -> bool
    def stop(self) -> None
```

**方法**：
- `start()`: 启动服务，返回是否成功
- `stop()`: 停止服务

**示例**：

```python
from hos_optimizer.deploy import ServiceLauncher, ConfigSelector, HardwareDetector

hardware = HardwareDetector.get_hardware_info()
config = ConfigSelector.select_config(hardware, 7.0)
launcher = ServiceLauncher(config, "./model.gguf", "0.0.0.0", 8000)

if launcher.start():
    print("服务已启动")
    # ...
    launcher.stop()
```

#### `HealthChecker`

服务健康检查器。

```python
class HealthChecker:
    def __init__(self, host: str = "localhost", port: int = 8000)
    def check_health(self, timeout: int = 5) -> bool
    def check_model_loaded(self, timeout: int = 300) -> bool
    def get_service_info(self) -> Optional[Dict]
```

**方法**：
- `check_health()`: 检查服务健康状态
- `check_model_loaded()`: 检查模型是否加载完成
- `get_service_info()`: 获取服务信息

### 核心函数

#### `deploy_model()`

一键部署模型。

```python
def deploy_model(
    model_path: str,
    model_size_b: float = 7.0,
    use_case: str = "general",
    host: str = "0.0.0.0",
    port: int = 8000,
    auto_start: bool = True
) -> Optional[ServiceLauncher]
```

**参数**：
- `model_path` (str): 模型文件路径
- `model_size_b` (float): 模型大小（十亿参数）
- `use_case` (str): 使用场景
- `host` (str): 服务主机地址
- `port` (int): 服务端口
- `auto_start` (bool): 是否自动启动服务

**返回**：`Optional[ServiceLauncher]` - 服务启动器实例

**示例**：

```python
from hos_optimizer.deploy import deploy_model

launcher = deploy_model(
    model_path="./model.gguf",
    model_size_b=7.0,
    use_case="general",
    auto_start=True
)
```

---

## 配置模块 API

配置模块位于 `hos_optimizer.config`，提供配置管理和优化功能。

### 异常类

#### `ConfigError`

配置相关异常的基类。

#### `ConfigValidationError`

配置验证失败时抛出。

#### `ConfigConflictError`

配置项之间存在冲突时抛出。

#### `TemplateNotFoundError`

请求的模板不存在时抛出。

### 核心类

#### `ConfigManager`

配置管理器，负责加载、生成、验证和管理配置。

```python
class ConfigManager:
    def __init__(self, config_dir: Optional[str] = None)
```

**参数**：
- `config_dir` (Optional[str]): 配置文件目录，默认为项目根目录下的 `configs/`

**方法**：

##### `load_config()`

从 YAML 文件加载配置。

```python
def load_config(self, config_path: str) -> Dict[str, Any]
```

**参数**：
- `config_path` (str): YAML 配置文件路径

**返回**：`Dict[str, Any]` - 解析后的配置字典

**异常**：`ConfigError` - 文件不存在或解析失败

##### `save_config()`

将配置保存为 YAML 文件。

```python
def save_config(self, config: Dict[str, Any], output_path: str) -> None
```

**参数**：
- `config` (Dict[str, Any]): 配置字典
- `output_path` (str): 输出文件路径

##### `merge_configs()`

深度合并两个配置字典。

```python
def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]
```

**参数**：
- `base` (Dict[str, Any]): 基础配置
- `override` (Dict[str, Any]): 覆盖配置

**返回**：`Dict[str, Any]` - 合并后的新配置字典

##### `generate_optimal_config()`

根据使用场景和硬件条件生成最优配置。

```python
def generate_optimal_config(
    self,
    scenario: str,
    model_path: str = "",
    vram_gb: float = 8.0
) -> Dict[str, Any]
```

**参数**：
- `scenario` (str): 场景名称，可选值：
  - `inference_0.8b`
  - `inference_7b`
  - `training_0.8b`
  - `training_7b`
  - `high_throughput_serving`
  - `multi_turn_dialogue`
- `model_path` (str): 模型路径
- `vram_gb` (float): 可用显存大小（GB）

**返回**：`Dict[str, Any]` - 最优配置字典

**异常**：`ConfigError` - 不支持的场景或显存不足

**示例**：

```python
from hos_optimizer.config import ConfigManager

manager = ConfigManager()
config = manager.generate_optimal_config(
    scenario="inference_7b",
    model_path="./model",
    vram_gb=8.0
)
```

##### `auto_select_scenario()`

根据模型大小和任务类型自动选择最优场景并生成配置。

```python
def auto_select_scenario(
    self,
    model_size_b: float,
    task: str = "inference",
    vram_gb: float = 8.0
) -> Dict[str, Any]
```

**参数**：
- `model_size_b` (float): 模型大小（十亿参数）
- `task` (str): 任务类型（`inference`/`training`/`serving`/`dialogue`）
- `vram_gb` (float): 可用显存（GB）

**返回**：`Dict[str, Any]` - 自动选择的最优配置

##### `validate_config()`

验证配置并返回所有发现的问题列表。

```python
def validate_config(self, config: Dict[str, Any]) -> List[str]
```

**参数**：
- `config` (Dict[str, Any]): 待验证的配置字典

**返回**：`List[str]` - 问题描述字符串列表，空列表表示验证通过

##### `validate_and_raise()`

验证配置，如果发现问题则抛出异常。

```python
def validate_and_raise(self, config: Dict[str, Any]) -> None
```

**参数**：
- `config` (Dict[str, Any]): 待验证的配置

**异常**：
- `ConfigValidationError`: 存在验证问题
- `ConfigConflictError`: 存在配置冲突

##### `list_templates()`

列出所有可用的配置模板名称。

```python
def list_templates(self) -> List[str]
```

**返回**：`List[str]` - 模板名称列表

##### `get_template()`

获取指定名称的配置模板。

```python
def get_template(self, name: str) -> Dict[str, Any]
```

**参数**：
- `name` (str): 模板名称

**返回**：`Dict[str, Any]` - 模板配置字典

**异常**：`TemplateNotFoundError` - 模板不存在

##### `register_template()`

注册自定义配置模板。

```python
def register_template(self, name: str, template: Dict[str, Any]) -> None
```

**参数**：
- `name` (str): 模板名称
- `template` (Dict[str, Any]): 模板配置字典

##### `unregister_template()`

注销自定义配置模板。

```python
def unregister_template(self, name: str) -> None
```

**参数**：
- `name` (str): 模板名称

**异常**：
- `ConfigError`: 尝试注销内置模板
- `TemplateNotFoundError`: 模板不存在

##### `export_template()`

将模板导出为 YAML 文件。

```python
def export_template(self, name: str, output_path: str) -> None
```

**参数**：
- `name` (str): 模板名称
- `output_path` (str): 输出文件路径

##### `load_template_from_file()`

从 YAML 文件加载并注册为自定义模板。

```python
def load_template_from_file(self, name: str, file_path: str) -> None
```

**参数**：
- `name` (str): 注册时使用的模板名称
- `file_path` (str): YAML 文件路径

---

## 工具函数 API

工具函数位于 `hos_optimizer.utils`，提供日志配置、文件操作和模型路径处理等通用功能。

### 日志配置

#### `setup_logger()`

配置并返回日志记录器。

```python
def setup_logger(
    name: str = "hos_optimizer",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S"
) -> logging.Logger
```

**参数**：
- `name` (str): 日志记录器名称
- `level` (int): 日志级别
- `log_file` (Optional[str]): 日志文件路径（可选）
- `fmt` (str): 日志格式
- `datefmt` (str): 日期格式

**返回**：`logging.Logger` - 配置好的 Logger 实例

**示例**：

```python
from hos_optimizer.utils import setup_logger

logger = setup_logger(
    name="my_app",
    level=logging.DEBUG,
    log_file="app.log"
)
logger.info("应用启动")
```

### 文件操作工具

#### `ensure_dir()`

确保目录存在，不存在则创建。

```python
def ensure_dir(path: str) -> str
```

**参数**：
- `path` (str): 目录路径

**返回**：`str` - 目录路径

#### `get_file_size_gb()`

获取文件大小（GB）。

```python
def get_file_size_gb(path: str) -> float
```

**参数**：
- `path` (str): 文件路径

**返回**：`float` - 文件大小（GB）

#### `get_dir_size_gb()`

获取目录总大小（GB）。

```python
def get_dir_size_gb(path: str) -> float
```

**参数**：
- `path` (str): 目录路径

**返回**：`float` - 目录总大小（GB）

#### `find_model_files()`

在目录中查找模型文件。

```python
def find_model_files(path: str) -> List[str]
```

**参数**：
- `path` (str): 搜索路径

**返回**：`List[str]` - 模型文件路径列表

### 模型路径处理

#### `resolve_model_path()`

解析模型路径，支持相对路径和环境变量展开。

```python
def resolve_model_path(path: str) -> str
```

**参数**：
- `path` (str): 原始路径

**返回**：`str` - 解析后的绝对路径

#### `is_model_path()`

判断路径是否为有效的模型路径。

```python
def is_model_path(path: str) -> bool
```

**参数**：
- `path` (str): 路径字符串

**返回**：`bool` - 是否为有效模型路径

#### `get_model_format()`

推断模型格式。

```python
def get_model_format(path: str) -> str
```

**参数**：
- `path` (str): 模型路径

**返回**：`str` - 格式字符串（`gguf`/`safetensors`/`pytorch`/`unknown`）

---

## 更多资源

- [使用示例](EXAMPLES.md) - 完整使用示例
- [架构文档](docs/architecture.md) - 系统设计说明
- [安装指南](INSTALL.md) - 安装和配置

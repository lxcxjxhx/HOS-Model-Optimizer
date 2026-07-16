# 安装指南

本文档提供 HOS Model Optimizer 的详细安装步骤，涵盖系统要求、依赖安装、虚拟环境配置以及常见问题的解决方案。

## 目录

- [系统要求](#系统要求)
- [依赖安装步骤](#依赖安装步骤)
- [虚拟环境配置](#虚拟环境配置)
- [llama-cpp-python 安装](#llama-cpp-python-安装)
- [常见问题解决](#常见问题解决)

---

## 系统要求

### 基础要求

| 组件 | 最低版本 | 推荐版本 | 说明 |
|------|---------|---------|------|
| Python | 3.8 | 3.10+ | 必须 64 位版本 |
| CUDA | 11.8 | 12.1+ | GPU 加速必需 |
| GPU VRAM | 4GB | 8GB+ | 针对 8GB 场景优化 |
| 系统内存 | 8GB | 16GB+ | 量化/训练时需要 |
| 磁盘空间 | 10GB | 50GB+ | 模型文件较大 |

### 操作系统支持

- Windows 10/11（推荐）
- Linux（Ubuntu 20.04+）
- macOS（仅 CPU 推理）

### 硬件检测

安装前可以使用以下命令检测硬件：

```bash
# 检测 GPU 和 CUDA 版本
nvidia-smi

# 检测 Python 版本
python --version

# 检测系统内存
# Windows
systeminfo | findstr /C:"Total Physical Memory"
# Linux
free -h
```

---

## 依赖安装步骤

### 第一步：安装基础依赖

```bash
# 克隆项目
git clone https://github.com/hos-team/hos-model-optimizer.git
cd HOS-Model-Optimizer

# 基础安装（包含核心功能）
pip install -e .
```

### 第二步：安装可选依赖

根据使用场景选择安装：

```bash
# 量化相关（GGUF/AWQ/GPTQ）
pip install -e ".[quantization]"

# llama-cpp 推理（GGUF 格式）
pip install -e ".[inference]"

# vLLM 推理（高吞吐场景）
pip install -e ".[vllm]"

# SGLang 推理（结构化生成）
pip install -e ".[sglang]"

# 训练加速（Unsloth）
pip install -e ".[training]"

# 安装全部依赖
pip install -e ".[all]"
```

### 第三步：验证安装

```bash
# 检查版本
hos-optimizer --version

# 检查命令行工具
hos-quantize --help
hos-infer --help
hos-train --help
hos-deploy --help
hos-config --help

# 检查 GPU 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 依赖列表

#### 核心依赖

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| click | >=8.0.0 | 命令行框架 |
| pyyaml | >=6.0 | 配置文件解析 |
| requests | >=2.28.0 | HTTP 请求 |
| psutil | >=5.9.0 | 系统信息检测 |
| torch | >=2.0.0 | 深度学习框架 |
| transformers | >=4.35.0 | 模型加载和推理 |
| datasets | >=2.14.0 | 数据集处理 |
| peft | >=0.5.0 | LoRA/QLoRA 训练 |

#### 可选依赖

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| autoawq | >=0.1.0 | AWQ 量化 |
| auto-gptq | >=0.5.0 | GPTQ 量化 |
| bitsandbytes | >=0.41.0 | 4-bit 量化支持 |
| llama-cpp-python | >=0.2.0 | GGUF 推理后端 |
| vllm | >=0.2.0 | vLLM 推理后端 |
| sglang | >=0.1.0 | SGLang 推理后端 |
| unsloth | >=0.1.0 | 训练加速 |

---

## 虚拟环境配置

### 使用 Conda（推荐）

```bash
# 创建虚拟环境
conda create -n hos-optimizer python=3.10 -y

# 激活环境
conda activate hos-optimizer

# 安装项目
pip install -e .

# 安装全部依赖
pip install -e ".[all]"
```

### 使用 venv

```bash
# 创建虚拟环境
python -m venv hos-env

# 激活环境
# Windows
hos-env\Scripts\activate
# Linux/macOS
source hos-env/bin/activate

# 升级 pip
python -m pip install --upgrade pip

# 安装项目
pip install -e .
```

### 使用 Poetry

```bash
# 初始化项目（如果尚未初始化）
poetry init

# 添加依赖
poetry add click pyyaml requests psutil torch transformers datasets peft

# 安装可选依赖
poetry add autoawq auto-gptq bitsandbytes
poetry add llama-cpp-python
poetry add vllm
poetry add sglang

# 安装项目
poetry install
```

### 环境隔离建议

1. **为不同场景创建独立环境**

```bash
# 推理环境
conda create -n hos-infer python=3.10 -y
conda activate hos-infer
pip install -e ".[inference]"

# 训练环境
conda create -n hos-train python=3.10 -y
conda activate hos-train
pip install -e ".[training]"

# 完整环境
conda create -n hos-full python=3.10 -y
conda activate hos-full
pip install -e ".[all]"
```

2. **固定依赖版本**

```bash
# 导出依赖
pip freeze > requirements-frozen.txt

# 从固定版本安装
pip install -r requirements-frozen.txt
```

---

## llama-cpp-python 安装

llama-cpp-python 是 GGUF 格式推理的核心依赖，安装时需要特别注意 CUDA 支持。

### CPU 版本安装

```bash
# 纯 CPU 版本（无需 CUDA）
pip install llama-cpp-python
```

### CUDA 版本安装（推荐）

#### Windows

```bash
# 设置环境变量（CUDA 12.x）
set CMAKE_ARGS="-DGGML_CUDA=on"
set FORCE_CMAKE=1

# 安装
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

#### Linux

```bash
# 设置环境变量
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1

# 安装
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 指定 CUDA 版本

```bash
# CUDA 11.8
set CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"

# CUDA 12.x
set CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
```

### 验证安装

```bash
# 检查是否支持 CUDA
python -c "from llama_cpp import Llama; print('llama-cpp-python installed successfully')"

# 测试 GPU 加速
python -c "
from llama_cpp import Llama
llm = Llama(model_path='path/to/model.gguf', n_gpu_layers=-1)
print('GPU offload layers:', llm.n_gpu_layers)
"
```

### 从源码编译

如果预编译版本不可用，可以从源码编译：

```bash
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 编译（CUDA 支持）
# Windows
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Linux
mkdir build && cd build
cmake -DGGML_CUDA=ON ..
make -j$(nproc)

# 安装 llama-cpp-python
cd ../bindings/python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall --no-cache-dir .
```

---

## 常见问题解决

### 问题 1：CUDA 不可用

**症状**：`torch.cuda.is_available()` 返回 `False`

**解决方案**：

```bash
# 检查 CUDA 安装
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装 PyTorch（CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 问题 2：llama-cpp-python 安装失败

**症状**：编译错误或缺少 CUDA 支持

**解决方案**：

```bash
# 方案 1：使用预编译 wheel
pip install llama-cpp-python --prefer-binary

# 方案 2：设置正确的 CUDA 路径
# Windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Linux
export CUDA_HOME=/usr/local/cuda
export CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 问题 3：内存不足（OOM）

**症状**：训练或推理时出现 `CUDA out of memory`

**解决方案**：

```bash
# 方案 1：减小批次大小
hos-train --model ./model --dataset ./data.json --batch-size 1

# 方案 2：启用梯度检查点
# 在配置文件中设置
gradient_checkpointing: true

# 方案 3：使用 QLoRA 代替 LoRA
hos-train --model ./model --dataset ./data.json --method qlora

# 方案 4：减小上下文长度
hos-infer --model ./model --max-model-len 512
```

### 问题 4：模型加载失败

**症状**：`OSError: Model path does not exist`

**解决方案**：

```bash
# 检查模型路径
python -c "
import os
path = './model'
print(f'Path exists: {os.path.exists(path)}')
print(f'Is directory: {os.path.isdir(path)}')
if os.path.isdir(path):
    print(f'Files: {os.listdir(path)}')
"

# 使用绝对路径
hos-infer --model C:/path/to/model --prompt "你好"
```

### 问题 5：依赖冲突

**症状**：`pip install` 时报依赖冲突

**解决方案**：

```bash
# 方案 1：使用虚拟环境隔离
conda create -n hos-clean python=3.10 -y
conda activate hos-clean
pip install -e .

# 方案 2：升级 pip
python -m pip install --upgrade pip

# 方案 3：使用 --no-deps 跳过依赖检查
pip install package-name --no-deps
```

### 问题 6：vLLM 安装失败

**症状**：vLLM 编译或安装失败

**解决方案**：

```bash
# 方案 1：使用预编译版本
pip install vllm --prefer-binary

# 方案 2：检查 CUDA 版本兼容性
# vLLM 需要 CUDA 11.8+ 或 12.x
nvidia-smi

# 方案 3：从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 问题 7：Windows 路径问题

**症状**：路径中的反斜杠导致解析错误

**解决方案**：

```bash
# 使用正斜杠
hos-infer --model C:/path/to/model

# 使用双反斜杠
hos-infer --model C:\\path\\to\\model

# 使用引号包裹
hos-infer --model "C:\path\to\model"
```

### 问题 8：量化后模型精度下降

**症状**：量化后模型输出质量明显下降

**解决方案**：

```bash
# 方案 1：使用更高精度的量化
hos-quantize --method gguf --model ./model --quant-type Q5_K_M  # 5-bit
hos-quantize --method gguf --model ./model --quant-type Q8_0    # 8-bit

# 方案 2：使用 AWQ 量化（精度损失更小）
hos-quantize --method awq --bits 4 --model ./model

# 方案 3：评估 PPL 选择最佳量化方案
hos-quantize --method evaluate --model ./model-quantized
```

---

## 安装检查清单

完成安装后，运行以下检查：

```bash
# 1. 检查 Python 版本
python --version  # 应该 >= 3.8

# 2. 检查 CUDA 可用性
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. 检查核心依赖
python -c "import transformers; import datasets; import peft; print('Core deps OK')"

# 4. 检查命令行工具
hos-optimizer --version

# 5. 运行简单推理测试
hos-infer --model Qwen/Qwen2.5-0.5B --prompt "Hello" --max-tokens 10
```

如果所有检查通过，说明安装成功！

---

## 下一步

- 阅读 [使用示例](EXAMPLES.md) 了解详细用法
- 查阅 [API 文档](API.md) 了解 Python 接口
- 查看 [架构文档](docs/architecture.md) 了解系统设计

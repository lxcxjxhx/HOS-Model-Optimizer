"""
HOS 模型微调模块

提供 QLoRA 和 LoRA 微调功能，针对 8GB VRAM 场景优化。
支持 Alpaca 和 ShareGPT 数据格式，集成 Unsloth 加速（可选）。

主要功能：
- QLoRA 训练（4-bit 量化 + LoRA）
- LoRA 训练（全精度 + LoRA）
- 数据集加载和预处理
- 训练监控和日志
- 模型合并
"""

import os
import json
import logging
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import load_dataset, Dataset, DatasetDict

# 尝试导入 Unsloth（可选加速）
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logging.warning("Unsloth 未安装，将使用标准训练流程。安装命令: pip install unsloth")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置类"""
    
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
    use_unsloth: bool = True  # 如果可用，使用 Unsloth 加速


class DatasetProcessor:
    """数据集处理器"""
    
    def __init__(self, tokenizer, max_seq_length: int = 2048):
        """
        初始化数据集处理器
        
        Args:
            tokenizer: 分词器
            max_seq_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def format_alpaca(self, example: Dict) -> Dict:
        """
        格式化 Alpaca 格式数据
        
        Alpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
        
        Args:
            example: 数据样本
            
        Returns:
            格式化后的字典
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        # 构建对话格式
        if input_text:
            text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output}"
        else:
            text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"
        
        return {"text": text}
    
    def format_sharegpt(self, example: Dict) -> Dict:
        """
        格式化 ShareGPT 格式数据
        
        ShareGPT 格式: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        
        Args:
            example: 数据样本
            
        Returns:
            格式化后的字典
        """
        conversations = example.get("conversations", [])
        
        text_parts = []
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            
            if role == "human":
                text_parts.append(f"### 用户:\n{value}")
            elif role == "gpt":
                text_parts.append(f"### 助手:\n{value}")
        
        text = "\n\n".join(text_parts)
        return {"text": text}
    
    def format_messages(self, example: Dict) -> Dict:
        """
        格式化 messages 格式数据（OpenAI 风格）
        
        messages 格式: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        
        Args:
            example: 数据样本
            
        Returns:
            格式化后的字典
        """
        messages = example.get("messages", [])
        
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                text_parts.append(f"### 用户:\n{content}")
            elif role == "assistant":
                text_parts.append(f"### 助手:\n{content}")
            elif role == "system":
                text_parts.append(f"### 系统:\n{content}")
        
        text = "\n\n".join(text_parts)
        return {"text": text}
    
    def tokenize_function(self, example: Dict) -> Dict:
        """
        分词函数
        
        Args:
            example: 包含 "text" 字段的字典
            
        Returns:
            分词后的字典
        """
        text = example.get("text", "")
        
        # 分词
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        
        # 设置 labels（用于语言模型训练）
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def process_dataset(
        self,
        dataset: Dataset,
        dataset_format: str = "alpaca"
    ) -> Dataset:
        """
        处理数据集
        
        Args:
            dataset: 原始数据集
            dataset_format: 数据格式 ("alpaca", "sharegpt" 或 "messages")
            
        Returns:
            处理后的数据集
        """
        logger.info(f"开始处理数据集，格式: {dataset_format}")
        
        # 格式化数据
        if dataset_format == "alpaca":
            format_func = self.format_alpaca
        elif dataset_format == "sharegpt":
            format_func = self.format_sharegpt
        elif dataset_format == "messages":
            format_func = self.format_messages
        else:
            raise ValueError(f"不支持的数据格式: {dataset_format}，支持的格式: alpaca, sharegpt, messages")
        
        # 应用格式化
        dataset = dataset.map(format_func)
        logger.info(f"数据格式化完成，样本数: {len(dataset)}")
        
        # 分词
        dataset = dataset.map(
            self.tokenize_function,
            num_proc=self.tokenizer.num_proc if hasattr(self.tokenizer, 'num_proc') else 4,
            remove_columns=dataset.column_names,
            desc="分词处理中"
        )
        logger.info(f"分词完成，有效样本数: {len(dataset)}")
        
        return dataset


def load_and_process_dataset(
    dataset_path: str,
    tokenizer,
    dataset_format: str = "alpaca",
    max_seq_length: int = 2048,
    test_size: float = 0.05
) -> DatasetDict:
    """
    加载并处理数据集
    
    Args:
        dataset_path: 数据集路径（JSON 文件）
        tokenizer: 分词器
        dataset_format: 数据格式 ("alpaca" 或 "sharegpt")
        max_seq_length: 最大序列长度
        test_size: 测试集比例
        
    Returns:
        包含 train 和 test 的 DatasetDict
    """
    logger.info(f"加载数据集: {dataset_path}")
    
    # 加载 JSON 数据
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logger.info(f"原始数据集大小: {len(dataset)}")
    
    # 处理数据集
    processor = DatasetProcessor(tokenizer, max_seq_length)
    processed_dataset = processor.process_dataset(dataset, dataset_format)
    
    # 划分训练集和测试集
    dataset_dict = processed_dataset.train_test_split(
        test_size=test_size,
        seed=42
    )
    
    logger.info(f"训练集大小: {len(dataset_dict['train'])}")
    logger.info(f"测试集大小: {len(dataset_dict['test'])}")
    
    return dataset_dict


def get_quantization_config(config: TrainingConfig) -> Optional[BitsAndBytesConfig]:
    """
    获取量化配置
    
    Args:
        config: 训练配置
        
    Returns:
        BitsAndBytesConfig 或 None
    """
    if not config.use_4bit:
        return None
    
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    logger.info(f"启用 4-bit 量化: {config.bnb_4bit_quant_type}")
    return bnb_config


def load_model_and_tokenizer(
    config: TrainingConfig
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和分词器

    Args:
        config: 训练配置

    Returns:
        (模型, 分词器) 元组
    """
    logger.info(f"加载模型: {config.model_name_or_path}")

    # 检查是否使用 Unsloth
    if config.use_unsloth and UNSLOTH_AVAILABLE and config.finetuning_type == "qlora":
        logger.info("使用 Unsloth 加速加载模型")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name_or_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.use_4bit,
            trust_remote_code=config.trust_remote_code,
        )
        return model, tokenizer

    # 标准加载流程
    # 获取量化配置
    quantization_config = get_quantization_config(config)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    has_gpu = torch.cuda.is_available()
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }

    # 只在有 GPU 时使用 device_map
    if has_gpu:
        model_kwargs["device_map"] = "auto"

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # 根据 GPU 可用性设置 dtype
    if has_gpu:
        model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs
    )

    # CPU 训练时不需要移动模型
    if not has_gpu:
        logger.info("CPU 模式，模型保持在 CPU")
    else:
        logger.info(f"GPU 模式，模型已加载到 GPU")

    logger.info(f"模型加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer


def get_lora_config(config: TrainingConfig) -> LoraConfig:
    """
    获取 LoRA 配置
    
    Args:
        config: 训练配置
        
    Returns:
        LoraConfig
    """
    # 确定目标模块
    if "all" in config.lora_target_modules:
        # 对于 Qwen 模型，明确指定目标模块
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = config.lora_target_modules
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"LoRA 配置: rank={config.lora_rank}, alpha={config.lora_alpha}, targets={target_modules}")
    return lora_config


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    config: TrainingConfig
) -> AutoModelForCausalLM:
    """
    准备模型进行训练
    
    Args:
        model: 基础模型
        config: 训练配置
        
    Returns:
        准备好训练的模型
    """
    # 如果使用 4-bit 量化，需要准备模型
    if config.use_4bit and config.finetuning_type == "qlora":
        model = prepare_model_for_kbit_training(model)
        logger.info("模型已准备用于 k-bit 训练")
    
    # 获取 LoRA 配置
    lora_config = get_lora_config(config)
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def get_training_arguments(config: TrainingConfig, dataset_size: int = 1000) -> TrainingArguments:
    """
    获取训练参数

    Args:
        config: 训练配置
        dataset_size: 数据集大小，用于估算总步数

    Returns:
        TrainingArguments
    """
    # 检测是否有 GPU
    has_gpu = torch.cuda.is_available()

    # 估算总步数（基于数据集大小）
    steps_per_epoch = dataset_size // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_train_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    # 根据 GPU 可用性设置精度
    bf16_enabled = has_gpu and torch.cuda.is_bf16_supported()
    fp16_enabled = has_gpu and not bf16_enabled

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        gradient_checkpointing=config.gradient_checkpointing if has_gpu else False,
        optim=config.optim,
        seed=config.seed,
        report_to=[],
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    if has_gpu:
        logger.info(f"检测到 GPU，启用 GPU 加速 (bf16={bf16_enabled}, fp16={fp16_enabled})")
    else:
        logger.warning("未检测到 GPU，使用 CPU 训练（速度较慢）")

    return training_args


class ProgressCallback(TrainerCallback):
    """训练进度条回调"""
    
    def __init__(self):
        self.pbar = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时初始化进度条"""
        if state.is_world_process_zero:
            total_steps = state.max_steps
            self.pbar = tqdm(total=total_steps, desc="训练进度", unit="step")
    
    def on_step_end(self, args, state, control, **kwargs):
        """每个训练步骤后更新进度条"""
        if self.pbar is not None and state.is_world_process_zero:
            self.pbar.update(1)
            
            # 显示当前损失和 VRAM 使用情况
            if state.log_history:
                last_log = state.log_history[-1]
                if 'loss' in last_log:
                    self.pbar.set_postfix({
                        'loss': f"{last_log['loss']:.4f}",
                        'lr': f"{last_log.get('learning_rate', 0):.2e}"
                    })
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时关闭进度条"""
        if self.pbar is not None:
            self.pbar.close()


class VRAMCallback(TrainerCallback):
    """VRAM 监控回调"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在日志记录时输出 VRAM 使用情况"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            
            if logs is not None:
                logs["gpu_memory_gb"] = f"{gpu_memory:.2f}"
                logs["gpu_memory_reserved_gb"] = f"{gpu_memory_reserved:.2f}"
            
            if state.is_world_process_zero:
                logger.info(
                    f"Step {state.global_step}: "
                    f"GPU 显存: {gpu_memory:.2f}GB, "
                    f"已保留: {gpu_memory_reserved:.2f}GB"
                )


def train(config: TrainingConfig):
    """
    执行训练
    
    Args:
        config: 训练配置
    """
    logger.info("=" * 60)
    logger.info("开始训练流程")
    logger.info("=" * 60)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 加载并处理数据集
    dataset_dict = load_and_process_dataset(
        dataset_path=config.dataset_path,
        tokenizer=tokenizer,
        dataset_format=config.dataset_format,
        max_seq_length=config.max_seq_length,
    )
    
    # 准备模型进行训练
    model = prepare_model_for_training(model, config)
    
    # 获取训练参数（传递数据集大小）
    training_args = get_training_arguments(config, dataset_size=len(dataset_dict["train"]))
    logger.info(f"训练参数: {training_args}")
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        callbacks=[ProgressCallback(), VRAMCallback()],
    )
    
    # 开始训练
    logger.info("开始训练...")
    train_result = trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 保存训练结果
    trainer.save_state()
    
    # 打印训练结果
    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info(f"训练损失: {train_result.training_loss:.4f}")
    logger.info(f"训练时间: {train_result.metrics.get('train_runtime', 0):.2f} 秒")
    logger.info("=" * 60)


def backup_model(
    model_path: str,
    backup_dir: str = "./backups",
    backup_name: Optional[str] = None
) -> str:
    """
    备份模型到指定目录
    
    Args:
        model_path: 模型路径
        backup_dir: 备份目录
        backup_name: 备份名称（默认使用时间戳）
        
    Returns:
        备份路径
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 生成备份名称
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path)
        backup_name = f"{model_name}_{timestamp}"
    
    backup_path = os.path.join(backup_dir, backup_name)
    
    # 如果备份已存在，跳过
    if os.path.exists(backup_path):
        logger.info(f"备份已存在: {backup_path}")
        return backup_path
    
    logger.info(f"开始备份模型: {model_path} -> {backup_path}")
    
    # 复制模型文件
    if os.path.isdir(model_path):
        shutil.copytree(model_path, backup_path, symlinks=False, ignore_dangling_symlinks=True)
    else:
        shutil.copy2(model_path, backup_path)
    
    logger.info(f"模型备份完成: {backup_path}")
    return backup_path


def merge_model(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool = True
):
    """
    合并 LoRA 权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        adapter_path: LoRA adapter 路径
        output_path: 输出路径
        trust_remote_code: 是否信任远程代码
    """
    logger.info("=" * 60)
    logger.info("开始合并模型")
    logger.info("=" * 60)
    
    # 加载基础模型
    logger.info(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        dtype=torch.float16,
        device_map="auto",
    )
    
    # 加载 LoRA 模型
    logger.info(f"加载 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
    )
    
    # 合并权重
    logger.info("合并 LoRA 权重...")
    model = model.merge_and_unload()
    
    # 保存合并后的模型
    logger.info(f"保存合并后的模型到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    # 保存分词器
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_path)
    
    logger.info("=" * 60)
    logger.info("模型合并完成")
    logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HOS 模型微调工具")
    
    # 模型配置
    parser.add_argument("--model", type=str, required=True, help="基础模型路径或名称")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径（JSON 文件）")
    parser.add_argument("--output", type=str, default="./output", help="输出目录")
    
    # 训练方法
    parser.add_argument(
        "--method",
        type=str,
        choices=["qlora", "lora"],
        default="qlora",
        help="训练方法: qlora (4-bit) 或 lora (全精度)"
    )
    
    # 数据集配置
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "sharegpt", "messages"],
        default="alpaca",
        help="数据格式"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="最大序列长度")
    
    # LoRA 配置
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    
    # 其他选项
    parser.add_argument("--no-unsloth", action="store_true", help="禁用 Unsloth 加速")
    parser.add_argument("--merge", action="store_true", help="训练后自动合并模型")
    parser.add_argument("--merge-only", action="store_true", help="仅执行模型合并")
    parser.add_argument("--adapter-path", type=str, help="LoRA adapter 路径（用于合并）")
    
    args = parser.parse_args()
    
    # 创建训练配置
    config = TrainingConfig(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        dataset_format=args.format,
        max_seq_length=args.max_seq_length,
        finetuning_type=args.method,
        use_4bit=(args.method == "qlora"),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_unsloth=not args.no_unsloth,
    )
    
    # 执行操作
    if args.merge_only:
        if not args.adapter_path:
            raise ValueError("合并模式需要指定 --adapter-path")
        merge_model(
            base_model_path=args.model,
            adapter_path=args.adapter_path,
            output_path=args.output,
        )
    else:
        # 训练
        train(config)
        
        # 可选：自动合并
        if args.merge:
            merge_model(
                base_model_path=args.model,
                adapter_path=args.output,
                output_path=f"{args.output}_merged",
            )


if __name__ == "__main__":
    main()

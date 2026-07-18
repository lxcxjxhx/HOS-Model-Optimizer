#!/usr/bin/env python3
"""
HOS Model Optimizer - 评测模块

提供模型评测功能：
- 支持 PPL、BLEU、ROUGE、Exact Match、F1 等多种评测指标
- 支持 JSON/JSONL 格式数据集，自动检测 Alpaca/ShareGPT/Messages 格式
- 支持文本生成任务评测，集成 HuggingFace evaluate 库
- 支持多模型对比评测
- 评测结果导出为 JSON 或 Markdown 格式

针对 8GB VRAM 场景进行了优化，支持低显存推理评测。

使用方法：
  # 命令行评测
  python evaluate.py --model path/to/model --dataset data/test.json --metrics bleu rouge

  # 多模型对比
  python evaluate.py --model model_a model_b --dataset data/test.json --metrics bleu f1

  # 指定输出格式
  python evaluate.py --model path/to/model --dataset data/test.json --output result.md
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================
# 异常定义
# ============================================================

class EvaluationError(Exception):
    """评测相关异常的基类"""
    pass


class DatasetFormatError(EvaluationError):
    """数据集格式不正确时抛出"""
    pass


class MetricComputeError(EvaluationError):
    """指标计算失败时抛出"""
    pass


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class EvaluationConfig:
    """评测配置数据类"""

    # 模型配置
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = True

    # 数据集配置
    dataset_path: str = ""
    dataset_format: Optional[str] = None  # 自动检测: alpaca / sharegpt / messages
    max_samples: Optional[int] = None     # 最大评测样本数，None 表示全部
    max_seq_length: int = 512             # 最大序列长度

    # 评测指标
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge"])

    # 任务类型
    task_type: str = "text_generation"    # text_generation / perplexity

    # 生成参数
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 1                   # 8GB VRAM 场景建议 batch_size=1

    # 输出配置
    output_format: str = "json"           # json / markdown
    output_path: Optional[str] = None     # 结果输出路径

    # 8GB VRAM 优化
    load_in_4bit: bool = False            # 4-bit 量化加载以节省显存
    device_map: str = "auto"

    # 其他
    seed: int = 42
    verbose: bool = False


@dataclass
class SampleResult:
    """单条样本的评测结果"""
    index: int
    prompt: str
    reference: str
    prediction: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """评测结果数据类"""
    model_path: str
    dataset_path: str
    task_type: str
    metrics_summary: Dict[str, float] = field(default_factory=dict)
    sample_results: List[SampleResult] = field(default_factory=list)
    total_samples: int = 0
    elapsed_seconds: float = 0.0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 评测指标加载器
# ============================================================

class MetricLoader:
    """
    评测指标加载器，封装 HuggingFace evaluate 库和自定义指标。

    支持的指标：
    - ppl: 困惑度（Perplexity），基于模型 logits 计算
    - bleu: BLEU 分数（n-gram 精确率）
    - rouge: ROUGE 分数（召回率导向）
    - exact_match: 精确匹配率
    - f1: Token 级 F1 分数
    """

    # 内置指标名称列表
    SUPPORTED_METRICS = ("ppl", "bleu", "rouge", "exact_match", "f1")

    def __init__(self):
        self._evaluate_module = None

    def _load_evaluate(self):
        """懒加载 HuggingFace evaluate 模块"""
        if self._evaluate_module is not None:
            return
        try:
            import evaluate
            self._evaluate_module = evaluate
        except ImportError:
            self._evaluate_module = None
            logger.warning(
                "evaluate 库未安装，BLEU/ROUGE 等指标将使用内置实现。"
                "安装命令: pip install evaluate"
            )

    def compute(
        self,
        metric_name: str,
        predictions: List[str],
        references: List[str],
        model=None,
        tokenizer=None,
    ) -> float:
        """
        计算指定评测指标。

        Args:
            metric_name: 指标名称（ppl/bleu/rouge/exact_match/f1）
            predictions: 模型预测文本列表
            references: 参考文本列表
            model: 模型对象（PPL 计算时需要）
            tokenizer: 分词器对象（PPL 计算时需要）

        Returns:
            指标分数

        Raises:
            MetricComputeError: 指标计算失败
        """
        metric_name = metric_name.lower().strip()

        if metric_name == "ppl":
            return self._compute_perplexity(predictions, references, model, tokenizer)
        elif metric_name == "bleu":
            return self._compute_bleu(predictions, references)
        elif metric_name == "rouge":
            return self._compute_rouge(predictions, references)
        elif metric_name == "exact_match":
            return self._compute_exact_match(predictions, references)
        elif metric_name == "f1":
            return self._compute_f1(predictions, references)
        else:
            raise MetricComputeError(
                f"不支持的指标: {metric_name}，"
                f"支持的指标: {', '.join(self.SUPPORTED_METRICS)}"
            )

    def _compute_perplexity(
        self,
        predictions: List[str],
        references: List[str],
        model,
        tokenizer,
    ) -> float:
        """
        计算困惑度（Perplexity）。

        基于参考文本的 token 序列计算交叉熵损失，然后取指数。
        使用 stride 滑动窗口以支持长文本。
        """
        if model is None or tokenizer is None:
            raise MetricComputeError("计算 PPL 需要提供 model 和 tokenizer")

        try:
            import torch
        except ImportError:
            raise MetricComputeError("计算 PPL 需要 PyTorch")

        stride = 512
        all_nlls = []

        model.eval()
        device = next(model.parameters()).device

        # 使用参考文本计算 PPL
        texts = references if references else predictions

        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
                input_ids = encodings["input_ids"].to(device)
                seq_len = input_ids.size(1)

                nlls_list = []
                prev_end = 0

                for begin in range(0, seq_len, stride):
                    end = min(begin + stride, seq_len)
                    chunk = input_ids[:, begin:end]
                    target = chunk.clone()
                    # 第一个 chunk 不需要计算前面 token 的 loss
                    target[:, :max(0, prev_end - begin)] = -100

                    outputs = model(chunk, labels=target)
                    nlls = outputs.loss
                    nlls_list.append(nlls.item())
                    prev_end = end

                    if end >= seq_len:
                        break

                if nlls_list:
                    avg_nll = sum(nlls_list) / len(nlls_list)
                    all_nlls.append(avg_nll)

        if not all_nlls:
            return 0.0

        mean_nll = sum(all_nlls) / len(all_nlls)
        ppl = math.exp(mean_nll)
        return round(ppl, 4)

    def _compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """
        计算 BLEU 分数。

        优先使用 HuggingFace evaluate 库，不可用时使用内置实现。
        """
        self._load_evaluate()

        if self._evaluate_module is not None:
            try:
                bleu = self._evaluate_module.load("bleu")
                result = bleu.compute(
                    predictions=predictions,
                    references=[[ref] for ref in references],
                )
                return round(result["bleu"] * 100, 2)
            except Exception as e:
                logger.warning(f"evaluate 库 BLEU 计算失败，使用内置实现: {e}")

        # 内置 BLEU 实现（简化版，基于 n-gram 精确率）
        return self._builtin_bleu(predictions, references)

    def _builtin_bleu(self, predictions: List[str], references: List[str]) -> float:
        """内置 BLEU 分数计算（简化实现）"""
        from collections import Counter

        total_score = 0.0
        valid_count = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if not pred_tokens or not ref_tokens:
                continue

            # 计算 1-gram 到 4-gram 精确率
            precisions = []
            for n in range(1, 5):
                pred_ngrams = Counter(
                    tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)
                )
                ref_ngrams = Counter(
                    tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
                )

                clipped = sum(
                    min(count, ref_ngrams.get(ngram, 0))
                    for ngram, count in pred_ngrams.items()
                )
                total = max(len(pred_tokens) - n + 1, 1)
                precisions.append(clipped / total if clipped > 0 else 0)

            if any(p == 0 for p in precisions):
                continue

            # 几何平均
            log_avg = sum(math.log(p) for p in precisions) / len(precisions)
            geo_mean = math.exp(log_avg)

            # 简短惩罚
            bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
            score = bp * geo_mean
            total_score += score
            valid_count += 1

        if valid_count == 0:
            return 0.0
        return round(total_score / valid_count * 100, 2)

    def _compute_rouge(self, predictions: List[str], references: List[str]) -> float:
        """
        计算 ROUGE 分数（ROUGE-L F1）。

        优先使用 HuggingFace evaluate 库，不可用时使用内置实现。
        """
        self._load_evaluate()

        if self._evaluate_module is not None:
            try:
                rouge = self._evaluate_module.load("rouge")
                result = rouge.compute(
                    predictions=predictions,
                    references=references,
                    rouge_types=["rougeL"],
                )
                return round(result["rougeL"] * 100, 2)
            except Exception as e:
                logger.warning(f"evaluate 库 ROUGE 计算失败，使用内置实现: {e}")

        # 内置 ROUGE-L 实现
        return self._builtin_rouge_l(predictions, references)

    def _builtin_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """内置 ROUGE-L F1 计算"""
        total_score = 0.0
        valid_count = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if not pred_tokens or not ref_tokens:
                continue

            # 最长公共子序列长度
            lcs_len = self._lcs_length(pred_tokens, ref_tokens)

            if lcs_len == 0:
                continue

            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(ref_tokens)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                total_score += f1
                valid_count += 1

        if valid_count == 0:
            return 0.0
        return round(total_score / valid_count * 100, 2)

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(x), len(y)
        # 优化空间：只保留两行
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def _compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """
        计算精确匹配率。

        对预测和参考文本进行标准化后比较是否完全一致。
        """
        if not predictions:
            return 0.0

        match_count = 0
        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            if pred_norm == ref_norm:
                match_count += 1

        return round(match_count / len(predictions) * 100, 2)

    def _compute_f1(self, predictions: List[str], references: List[str]) -> float:
        """
        计算 Token 级 F1 分数。

        将文本拆分为 token 集合，计算集合级别的 F1。
        """
        if not predictions:
            return 0.0

        total_f1 = 0.0
        valid_count = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._normalize_text(pred).split())
            ref_tokens = set(self._normalize_text(ref).split())

            if not pred_tokens or not ref_tokens:
                continue

            common = pred_tokens & ref_tokens
            if not common:
                continue

            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            total_f1 += f1
            valid_count += 1

        if valid_count == 0:
            return 0.0
        return round(total_f1 / valid_count * 100, 2)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """文本标准化：小写、去除多余空白和标点"""
        import string
        text = text.lower().strip()
        # 去除标点
        text = text.translate(str.maketrans("", "", string.punctuation))
        # 合并空白
        text = " ".join(text.split())
        return text


# ============================================================
# 数据集加载器
# ============================================================

class DatasetLoader:
    """
    数据集加载器，支持 JSON/JSONL 格式。

    自动检测以下数据格式：
    - Alpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
    - ShareGPT 格式: {"conversations": [{"from": "human", "value": "..."}, ...]}
    - Messages 格式: {"messages": [{"role": "user", "content": "..."}, ...]}

    输出统一为 (prompt, reference) 元组列表。
    """

    # 支持的格式名称
    SUPPORTED_FORMATS = ("alpaca", "sharegpt", "messages")

    def load(
        self,
        dataset_path: str,
        dataset_format: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """
        加载数据集并解析为 (prompt, reference) 元组列表。

        Args:
            dataset_path: 数据集文件路径（JSON 或 JSONL）
            dataset_format: 数据格式名称，None 表示自动检测
            max_samples: 最大样本数，None 表示全部

        Returns:
            (prompt, reference) 元组列表

        Raises:
            DatasetFormatError: 文件格式或内容不正确
        """
        path = Path(dataset_path)
        if not path.exists():
            raise DatasetFormatError(f"数据集文件不存在: {dataset_path}")
        if not path.is_file():
            raise DatasetFormatError(f"数据集路径不是文件: {dataset_path}")

        # 读取数据
        suffix = path.suffix.lower()
        if suffix == ".json":
            raw_data = self._load_json(path)
        elif suffix == ".jsonl":
            raw_data = self._load_jsonl(path)
        else:
            # 尝试 JSON 格式
            try:
                raw_data = self._load_json(path)
            except Exception:
                try:
                    raw_data = self._load_jsonl(path)
                except Exception:
                    raise DatasetFormatError(
                        f"无法解析数据集文件: {dataset_path}，支持 .json 和 .jsonl 格式"
                    )

        if not raw_data:
            raise DatasetFormatError("数据集为空")

        # 检测格式
        if dataset_format is None:
            dataset_format = self._detect_format(raw_data[0])
            logger.info(f"自动检测到数据集格式: {dataset_format}")

        # 解析数据
        samples = self._parse_data(raw_data, dataset_format)

        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]

        logger.info(f"加载数据集完成: {len(samples)} 条样本，格式: {dataset_format}")
        return samples

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSON 文件"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 可能是 {"data": [...]} 格式
            for key in ("data", "samples", "items", "examples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise DatasetFormatError("JSON 文件顶层为字典，但未找到数据列表字段")
        if isinstance(data, list):
            return data
        raise DatasetFormatError(f"JSON 文件顶层类型不支持: {type(data).__name__}")

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSONL 文件"""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetFormatError(
                        f"JSONL 第 {line_num} 行解析失败: {e}"
                    )
        return data

    def _detect_format(self, sample: Dict[str, Any]) -> str:
        """
        根据第一条样本自动检测数据格式。

        检测逻辑：
        1. 包含 "conversations" 字段 → ShareGPT
        2. 包含 "messages" 字段 → Messages
        3. 包含 "instruction" 字段 → Alpaca
        4. 包含 "prompt"/"input"/"output" 字段 → Alpaca（简化版）
        """
        if "conversations" in sample:
            return "sharegpt"
        if "messages" in sample:
            return "messages"
        if "instruction" in sample:
            return "alpaca"
        # 简化格式：直接包含 prompt/input 和 output/response
        if ("prompt" in sample or "input" in sample) and (
            "output" in sample or "response" in sample
        ):
            return "alpaca"
        raise DatasetFormatError(
            f"无法自动检测数据集格式，样本字段: {list(sample.keys())}。"
            f"请使用 --format 参数指定格式（支持: {', '.join(self.SUPPORTED_FORMATS)}）"
        )

    def _parse_data(
        self, raw_data: List[Dict[str, Any]], fmt: str
    ) -> List[Tuple[str, str]]:
        """根据格式解析数据为 (prompt, reference) 元组列表"""
        if fmt == "alpaca":
            return self._parse_alpaca(raw_data)
        elif fmt == "sharegpt":
            return self._parse_sharegpt(raw_data)
        elif fmt == "messages":
            return self._parse_messages(raw_data)
        else:
            raise DatasetFormatError(f"不支持的数据集格式: {fmt}")

    def _parse_alpaca(self, data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        解析 Alpaca 格式数据。

        格式: {"instruction": "...", "input": "...", "output": "..."}
        prompt = instruction + input（如有）
        reference = output 或 response
        """
        samples = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "") or item.get("response", "")

            if not output:
                logger.warning("跳过缺少 output 字段的样本")
                continue

            # 构建 prompt
            if input_text:
                prompt = f"{instruction}\n\n输入：{input_text}"
            else:
                prompt = instruction

            samples.append((prompt, output))
        return samples

    def _parse_sharegpt(self, data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        解析 ShareGPT 格式数据。

        格式: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]}
        prompt = 第一条 human 消息
        reference = 紧随其后的 gpt 回复
        """
        samples = []
        for item in data:
            conversations = item.get("conversations", [])
            if not conversations:
                continue

            # 提取最后一轮对话作为评测样本
            prompt = ""
            reference = ""
            for i, turn in enumerate(conversations):
                role = turn.get("from", "") or turn.get("role", "")
                value = turn.get("value", "") or turn.get("content", "")
                if role in ("human", "user") and i + 1 < len(conversations):
                    prompt = value
                    next_turn = conversations[i + 1]
                    next_role = next_turn.get("from", "") or next_turn.get("role", "")
                    if next_role in ("gpt", "assistant"):
                        reference = next_turn.get("value", "") or next_turn.get("content", "")
                        break

            if prompt and reference:
                samples.append((prompt, reference))
        return samples

    def _parse_messages(self, data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        解析 Messages 格式数据。

        格式: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
        prompt = 最后一条 user 消息
        reference = 紧随其后的 assistant 回复
        """
        samples = []
        for item in data:
            messages = item.get("messages", [])
            if not messages:
                continue

            # 找到最后一条 user 消息和对应的 assistant 回复
            prompt = ""
            reference = ""
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant" and i > 0:
                    reference = messages[i].get("content", "")
                    # 向前找 user 消息
                    for j in range(i - 1, -1, -1):
                        if messages[j].get("role") == "user":
                            prompt = messages[j].get("content", "")
                            break
                    break

            if prompt and reference:
                samples.append((prompt, reference))
        return samples


# ============================================================
# 评测执行引擎
# ============================================================

class EvaluationEngine:
    """
    评测执行引擎，负责加载模型并执行推理生成。

    针对 8GB VRAM 场景的优化：
    - 支持 4-bit 量化加载（load_in_4bit）
    - 默认 batch_size=1，避免显存溢出
    - 使用 device_map="auto" 自动分配设备
    - 支持梯度检查点减少显存占用
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._metric_loader = MetricLoader()

    def load_model(self):
        """
        加载模型和分词器。

        针对 8GB VRAM 场景，支持 4-bit 量化加载。
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise EvaluationError(
                "评测需要 transformers 和 torch，请安装: pip install transformers torch"
            )

        model_path = self.config.model_path
        tokenizer_path = self.config.tokenizer_path or model_path

        logger.info(f"加载模型: {model_path}")
        logger.info(f"加载分词器: {tokenizer_path}")

        # 加载分词器
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 构建模型加载参数
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
        }

        # 4-bit 量化加载（8GB VRAM 优化）
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info("启用 4-bit 量化加载（NF4 + 双重量化）")
        else:
            model_kwargs["torch_dtype"] = torch.float16

        # 加载模型
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._model.eval()

        logger.info("模型加载完成")

    def generate_predictions(
        self, samples: List[Tuple[str, str]]
    ) -> List[str]:
        """
        对数据集样本执行推理生成。

        Args:
            samples: (prompt, reference) 元组列表

        Returns:
            模型生成的文本列表
        """
        if self._model is None or self._tokenizer is None:
            raise EvaluationError("模型未加载，请先调用 load_model()")

        try:
            import torch
        except ImportError:
            raise EvaluationError("推理生成需要 PyTorch")

        predictions = []
        device = next(self._model.parameters()).device

        logger.info(f"开始推理生成: 共 {len(samples)} 条样本，batch_size={self.config.batch_size}")

        for batch_start in range(0, len(samples), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(samples))
            batch_prompts = [s[0] for s in samples[batch_start:batch_end]]

            # 编码输入
            inputs = self._tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(device)

            # 生成
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # 解码（仅保留新生成的部分）
            input_len = inputs["input_ids"].shape[1]
            for i, output in enumerate(outputs):
                new_tokens = output[input_len:]
                text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                predictions.append(text.strip())

            if (batch_end - batch_start) % 10 == 0 or batch_end == len(samples):
                logger.info(f"推理进度: {batch_end}/{len(samples)}")

        return predictions

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        计算所有配置的评测指标。

        Args:
            predictions: 模型预测文本列表
            references: 参考文本列表

        Returns:
            指标名称到分数的映射字典
        """
        results = {}
        for metric_name in self.config.metrics:
            logger.info(f"计算指标: {metric_name}")
            try:
                score = self._metric_loader.compute(
                    metric_name=metric_name,
                    predictions=predictions,
                    references=references,
                    model=self._model,
                    tokenizer=self._tokenizer,
                )
                results[metric_name] = score
                logger.info(f"  {metric_name} = {score}")
            except MetricComputeError as e:
                logger.error(f"指标 {metric_name} 计算失败: {e}")
                results[metric_name] = 0.0

        return results

    def shutdown(self):
        """释放模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("评测引擎资源已释放")


# ============================================================
# 结果导出器
# ============================================================

class ResultExporter:
    """评测结果导出器，支持 JSON 和 Markdown 格式"""

    @staticmethod
    def export_json(result: EvaluationResult, output_path: str) -> None:
        """
        导出评测结果为 JSON 格式。

        Args:
            result: 评测结果
            output_path: 输出文件路径
        """
        data = {
            "model_path": result.model_path,
            "dataset_path": result.dataset_path,
            "task_type": result.task_type,
            "timestamp": result.timestamp,
            "elapsed_seconds": result.elapsed_seconds,
            "total_samples": result.total_samples,
            "metrics_summary": result.metrics_summary,
            "metadata": result.metadata,
            "sample_results": [
                {
                    "index": sr.index,
                    "prompt": sr.prompt,
                    "reference": sr.reference,
                    "prediction": sr.prediction,
                    "metrics": sr.metrics,
                }
                for sr in result.sample_results
            ],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"评测结果已导出为 JSON: {output_path}")

    @staticmethod
    def export_markdown(result: EvaluationResult, output_path: str) -> None:
        """
        导出评测结果为 Markdown 格式。

        Args:
            result: 评测结果
            output_path: 输出文件路径
        """
        lines = [
            "# 模型评测报告",
            "",
            "## 基本信息",
            "",
            f"| 项目 | 值 |",
            f"|------|------|",
            f"| 模型路径 | `{result.model_path}` |",
            f"| 数据集路径 | `{result.dataset_path}` |",
            f"| 任务类型 | {result.task_type} |",
            f"| 评测时间 | {result.timestamp} |",
            f"| 评测样本数 | {result.total_samples} |",
            f"| 耗时 | {result.elapsed_seconds:.1f}s |",
            "",
            "## 评测指标汇总",
            "",
            "| 指标 | 分数 |",
            "|------|------|",
        ]

        for metric_name, score in sorted(result.metrics_summary.items()):
            lines.append(f"| {metric_name.upper()} | {score} |")

        lines.extend([
            "",
            "## 样本评测详情",
            "",
        ])

        # 最多展示前 20 条样本详情
        max_display = min(len(result.sample_results), 20)
        for sr in result.sample_results[:max_display]:
            lines.extend([
                f"### 样本 #{sr.index + 1}",
                "",
                f"**Prompt:** {sr.prompt[:200]}{'...' if len(sr.prompt) > 200 else ''}",
                "",
                f"**参考:** {sr.reference[:200]}{'...' if len(sr.reference) > 200 else ''}",
                "",
                f"**预测:** {sr.prediction[:200]}{'...' if len(sr.prediction) > 200 else ''}",
                "",
                f"**指标:** {', '.join(f'{k}={v}' for k, v in sr.metrics.items())}",
                "",
            ])

        if len(result.sample_results) > max_display:
            lines.append(f"> 仅展示前 {max_display} 条样本，共 {len(result.sample_results)} 条")
            lines.append("")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"评测结果已导出为 Markdown: {output_path}")

    @classmethod
    def export(cls, result: EvaluationResult, output_path: str, fmt: str = "json") -> None:
        """
        根据格式导出评测结果。

        Args:
            result: 评测结果
            output_path: 输出文件路径
            fmt: 输出格式（json / markdown）
        """
        if fmt == "json":
            cls.export_json(result, output_path)
        elif fmt == "markdown":
            cls.export_markdown(result, output_path)
        else:
            raise EvaluationError(f"不支持的输出格式: {fmt}，支持: json, markdown")


# ============================================================
# 主评测接口
# ============================================================

def evaluate_model(config: EvaluationConfig) -> EvaluationResult:
    """
    统一评测接口，执行完整的评测流程。

    流程：
    1. 加载数据集
    2. 加载模型
    3. 执行推理生成
    4. 计算评测指标
    5. 组装评测结果
    6. 导出结果（如指定输出路径）

    Args:
        config: 评测配置

    Returns:
        EvaluationResult: 完整评测结果

    Raises:
        EvaluationError: 评测过程中出错
    """
    from datetime import datetime

    logger.info("=" * 60)
    logger.info("开始模型评测")
    logger.info(f"  模型: {config.model_path}")
    logger.info(f"  数据集: {config.dataset_path}")
    logger.info(f"  指标: {', '.join(config.metrics)}")
    logger.info(f"  任务: {config.task_type}")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. 加载数据集
    dataset_loader = DatasetLoader()
    samples = dataset_loader.load(
        dataset_path=config.dataset_path,
        dataset_format=config.dataset_format,
        max_samples=config.max_samples,
    )

    if not samples:
        raise EvaluationError("数据集中没有有效样本")

    references = [s[1] for s in samples]

    # 2. 加载模型
    engine = EvaluationEngine(config)
    try:
        engine.load_model()

        # 3. 执行推理生成
        predictions = engine.generate_predictions(samples)

        # 4. 计算评测指标
        metrics_summary = engine.compute_metrics(predictions, references)

    finally:
        engine.shutdown()

    elapsed = time.time() - start_time

    # 5. 组装评测结果
    sample_results = []
    for i, (prompt, reference) in enumerate(samples):
        pred = predictions[i] if i < len(predictions) else ""
        # 计算单条样本的指标（仅 exact_match 和 f1 有意义）
        single_metrics = {}
        for metric_name in config.metrics:
            if metric_name in ("exact_match", "f1"):
                try:
                    single_metrics[metric_name] = MetricLoader().compute(
                        metric_name=metric_name,
                        predictions=[pred],
                        references=[reference],
                    )
                except Exception:
                    single_metrics[metric_name] = 0.0

        sample_results.append(SampleResult(
            index=i,
            prompt=prompt,
            reference=reference,
            prediction=pred,
            metrics=single_metrics,
        ))

    result = EvaluationResult(
        model_path=config.model_path,
        dataset_path=config.dataset_path,
        task_type=config.task_type,
        metrics_summary=metrics_summary,
        sample_results=sample_results,
        total_samples=len(samples),
        elapsed_seconds=round(elapsed, 2),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metadata={
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "load_in_4bit": config.load_in_4bit,
        },
    )

    # 6. 导出结果
    if config.output_path:
        ResultExporter.export(result, config.output_path, config.output_format)

    logger.info("=" * 60)
    logger.info("评测完成")
    logger.info(f"  总样本数: {result.total_samples}")
    logger.info(f"  耗时: {result.elapsed_seconds}s")
    for metric_name, score in result.metrics_summary.items():
        logger.info(f"  {metric_name.upper()}: {score}")
    logger.info("=" * 60)

    return result


# ============================================================
# 多模型对比评测
# ============================================================

def compare_models(
    model_paths: List[str],
    config: EvaluationConfig,
) -> List[EvaluationResult]:
    """
    多模型对比评测。

    使用相同的数据集和指标对多个模型进行评测，返回对比结果。

    Args:
        model_paths: 模型路径列表
        config: 评测配置（model_path 字段会被忽略，以 model_paths 为准）

    Returns:
        各模型的评测结果列表
    """
    results = []

    for model_path in model_paths:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"评测模型: {model_path}")
        logger.info(f"{'=' * 60}\n")

        model_config = EvaluationConfig(
            model_path=model_path,
            tokenizer_path=config.tokenizer_path,
            trust_remote_code=config.trust_remote_code,
            dataset_path=config.dataset_path,
            dataset_format=config.dataset_format,
            max_samples=config.max_samples,
            max_seq_length=config.max_seq_length,
            metrics=list(config.metrics),
            task_type=config.task_type,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            batch_size=config.batch_size,
            output_format=config.output_format,
            output_path=None,  # 对比模式下不单独输出
            load_in_4bit=config.load_in_4bit,
            device_map=config.device_map,
            seed=config.seed,
            verbose=config.verbose,
        )

        result = evaluate_model(model_config)
        results.append(result)

    # 输出对比摘要
    _print_comparison_summary(results)

    # 导出对比结果
    if config.output_path:
        _export_comparison(results, config.output_path, config.output_format)

    return results


def _print_comparison_summary(results: List[EvaluationResult]) -> None:
    """打印多模型对比摘要"""
    if len(results) <= 1:
        return

    print(f"\n{'=' * 60}")
    print("多模型对比摘要")
    print(f"{'=' * 60}")

    # 收集所有指标名称
    all_metrics = set()
    for r in results:
        all_metrics.update(r.metrics_summary.keys())
    all_metrics = sorted(all_metrics)

    # 表头
    header = f"{'指标':<15}"
    for r in results:
        model_name = Path(r.model_path).name or r.model_path
        header += f"  {model_name:<20}"
    print(header)
    print("-" * len(header))

    # 各指标
    for metric in all_metrics:
        row = f"{metric.upper():<15}"
        for r in results:
            score = r.metrics_summary.get(metric, 0.0)
            row += f"  {str(score):<20}"
        print(row)

    print(f"{'=' * 60}\n")


def _export_comparison(
    results: List[EvaluationResult], output_path: str, fmt: str
) -> None:
    """导出多模型对比结果"""
    if fmt == "json":
        data = {
            "comparison": [
                {
                    "model_path": r.model_path,
                    "metrics_summary": r.metrics_summary,
                    "total_samples": r.total_samples,
                    "elapsed_seconds": r.elapsed_seconds,
                    "timestamp": r.timestamp,
                }
                for r in results
            ]
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"对比结果已导出为 JSON: {output_path}")

    elif fmt == "markdown":
        lines = [
            "# 多模型对比评测报告",
            "",
            "## 指标对比",
            "",
        ]

        all_metrics = set()
        for r in results:
            all_metrics.update(r.metrics_summary.keys())
        all_metrics = sorted(all_metrics)

        # 表头
        header = "| 指标 |"
        separator = "|------|"
        for r in results:
            model_name = Path(r.model_path).name or r.model_path
            header += f" {model_name} |"
            separator += "------|"
        lines.extend([header, separator])

        for metric in all_metrics:
            row = f"| {metric.upper()} |"
            for r in results:
                score = r.metrics_summary.get(metric, 0.0)
                row += f" {score} |"
            lines.append(row)

        lines.extend(["", "## 各模型详情", ""])
        for r in results:
            lines.extend([
                f"### {r.model_path}",
                "",
                f"- 评测样本数: {r.total_samples}",
                f"- 耗时: {r.elapsed_seconds}s",
                f"- 评测时间: {r.timestamp}",
                "",
            ])

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"对比结果已导出为 Markdown: {output_path}")


# ============================================================
# 命令行接口
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="HOS Model Optimizer - 评测模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型评测
  python evaluate.py --model path/to/model --dataset data/test.json --metrics bleu rouge

  # 多模型对比
  python evaluate.py --model model_a model_b --dataset data/test.json --metrics bleu f1

  # 指定输出格式和路径
  python evaluate.py --model path/to/model --dataset data/test.json --output result.md

  # 使用 4-bit 量化加载（8GB VRAM 优化）
  python evaluate.py --model path/to/model --dataset data/test.json --load-in-4bit

  # 指定数据格式和最大样本数
  python evaluate.py --model path/to/model --dataset data/test.jsonl --format sharegpt --max-samples 100
        """,
    )

    # 模型参数
    parser.add_argument(
        "--model", type=str, nargs="+", required=True,
        help="模型路径（支持多个模型进行对比评测）",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="分词器路径（默认与模型路径相同）",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=True,
        help="信任远程代码（默认 True）",
    )

    # 数据集参数
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="评测数据集路径（JSON/JSONL 格式）",
    )
    parser.add_argument(
        "--format", type=str, default=None,
        choices=["alpaca", "sharegpt", "messages"],
        help="数据集格式（默认自动检测）",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="最大评测样本数（默认全部）",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="最大序列长度（默认 512）",
    )

    # 评测指标
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["bleu", "rouge"],
        help="评测指标列表（默认: bleu rouge），支持: ppl bleu rouge exact_match f1",
    )

    # 任务类型
    parser.add_argument(
        "--task", type=str, default="text_generation",
        choices=["text_generation", "perplexity"],
        help="任务类型（默认: text_generation）",
    )

    # 生成参数
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="最大生成 token 数（默认 256）",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="采样温度（默认 0.7）",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="nucleus sampling 参数（默认 0.9）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="推理批大小（8GB VRAM 建议 1，默认 1）",
    )

    # 输出配置
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="结果输出路径（支持 .json 和 .md 格式）",
    )
    parser.add_argument(
        "--output-format", type=str, default="json",
        choices=["json", "markdown"],
        help="输出格式（默认: json）",
    )

    # 8GB VRAM 优化
    parser.add_argument(
        "--load-in-4bit", action="store_true", default=False,
        help="使用 4-bit 量化加载模型（节省显存，适合 8GB VRAM 场景）",
    )

    # 其他
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="启用详细日志输出",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    命令行入口函数。

    Args:
        argv: 命令行参数列表，默认使用 sys.argv

    Returns:
        退出码，0 表示成功
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 自动推断输出格式
    output_format = args.output_format
    if args.output:
        if args.output.endswith(".md"):
            output_format = "markdown"
        elif args.output.endswith(".json"):
            output_format = "json"

    # 构建评测配置
    config = EvaluationConfig(
        model_path=args.model[0] if len(args.model) == 1 else "",
        tokenizer_path=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        dataset_path=args.dataset,
        dataset_format=args.format,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        metrics=args.metrics,
        task_type=args.task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        output_format=output_format,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        verbose=args.verbose,
    )

    try:
        if len(args.model) > 1:
            # 多模型对比评测
            compare_models(args.model, config)
        else:
            # 单模型评测
            evaluate_model(config)
        return 0

    except EvaluationError as e:
        logger.error(f"评测错误: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("评测已中断")
        return 1
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())

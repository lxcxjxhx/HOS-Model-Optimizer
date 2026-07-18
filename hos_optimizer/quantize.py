#!/usr/bin/env python3
"""
HOS Model Optimizer - 量化模块

提供多种量化方法支持：
- GGUF 量化：使用 llama.cpp 工具链
- AWQ 4-bit 量化：激活感知量化，精度损失最小
- GPTQ 4/8-bit 量化：基于 GPU 的 Post-Training 量化
- 量化质量评估：PPL (Perplexity) 计算
- 格式转换工具：支持不同量化格式间的转换

针对 8GB VRAM 场景进行了优化配置。

使用方法：
  # GGUF 量化
  python quantize.py --method gguf --model ./model_path --output ./output_path

  # AWQ 4-bit 量化
  python quantize.py --method awq --bits 4 --model ./model_path

  # GPTQ 量化
  python quantize.py --method gptq --bits 4 --model ./model_path

  # 评估量化质量
  python quantize.py --method evaluate --model ./quantized_model_path

  # 格式转换
  python quantize.py --method convert --from gguf --to awq --model ./model_path
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import time


# 8GB VRAM 优化配置
VRAM_8GB_CONFIG = {
    "max_batch_size": 1,
    "max_seq_length": 512,
    "gradient_checkpointing": True,
    "offload_to_cpu": True,
    "calibration_samples": 128,
}


class QuantizationError(Exception):
    """量化过程中的异常"""
    pass


def check_vram_availability() -> Dict[str, Any]:
    """检查 GPU VRAM 可用性
    
    Returns:
        包含 VRAM 信息的字典
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "total_vram_gb": 0,
            "free_vram_gb": 0,
            "device": "cpu"
        }
    
    device = torch.cuda.current_device()
    total_vram = torch.cuda.get_device_properties(device).total_memory
    free_vram = total_vram - torch.cuda.memory_allocated(device)
    
    return {
        "available": True,
        "total_vram_gb": total_vram / (1024 ** 3),
        "free_vram_gb": free_vram / (1024 ** 3),
        "device": torch.cuda.get_device_name(device)
    }


def optimize_for_low_vram(config: Dict[str, Any]) -> Dict[str, Any]:
    """根据 VRAM 限制优化配置
    
    Args:
        config: 原始配置
        
    Returns:
        优化后的配置
    """
    vram_info = check_vram_availability()
    
    if not vram_info["available"] or vram_info["free_vram_gb"] <= 8:
        print(f"⚠️  检测到 VRAM <= 8GB ({vram_info['free_vram_gb']:.2f}GB 可用)")
        print("   启用低 VRAM 优化配置...")
        
        # 合并优化配置
        optimized = config.copy()
        optimized.update(VRAM_8GB_CONFIG)
        
        return optimized
    
    return config


def quantize_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    llama_cpp_path: Optional[str] = None
) -> str:
    """GGUF 量化 - 使用 llama.cpp 工具链
    
    Args:
        model_path: 输入模型路径（HuggingFace 格式）
        output_path: 输出 GGUF 文件路径
        quant_type: 量化类型，如 Q4_K_M, Q5_K_M, Q8_0 等
        llama_cpp_path: llama.cpp 安装路径，如果为 None 则从 PATH 查找
        
    Returns:
        输出文件路径
        
    Raises:
        QuantizationError: 量化失败时抛出
    """
    print(f"=== GGUF {quant_type} 量化 ===")
    print(f"输入模型: {model_path}")
    print(f"输出路径: {output_path}")
    
    # 检查 llama-quantize 工具
    quantize_tool = "llama-quantize"
    if llama_cpp_path:
        quantize_tool = os.path.join(llama_cpp_path, "llama-quantize")
    
    try:
        # 检查工具是否存在
        result = subprocess.run(
            [quantize_tool, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise QuantizationError(
            "找不到 llama-quantize 工具。请确保已安装 llama.cpp 并将其添加到 PATH，"
            "或通过 --llama-cpp-path 指定安装路径。\n"
            "安装指南: https://github.com/ggerganov/llama.cpp"
        )
    
    # 转换模型为 GGUF 格式
    print("步骤 1/2: 转换模型为 GGUF 格式...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存为 GGUF 兼容格式
            temp_model_path = os.path.join(tmpdir, "model")
            model.save_pretrained(temp_model_path)
            tokenizer.save_pretrained(temp_model_path)
            
            # 使用 llama.cpp 转换工具
            convert_script = "convert.py"
            if llama_cpp_path:
                convert_script = os.path.join(llama_cpp_path, "convert.py")
            
            f16_path = os.path.join(tmpdir, "model-f16.gguf")
            
            print("  转换为 FP16 GGUF...")
            subprocess.run(
                [sys.executable, convert_script, temp_model_path, "--outfile", f16_path],
                check=True,
                capture_output=True,
                text=True
            )
            
            # 量化
            print(f"步骤 2/2: 量化为 {quant_type}...")
            subprocess.run(
                [quantize_tool, f16_path, output_path, quant_type],
                check=True,
                capture_output=True,
                text=True
            )
    
    except subprocess.CalledProcessError as e:
        raise QuantizationError(f"GGUF 量化失败: {e.stderr}")
    except Exception as e:
        raise QuantizationError(f"GGUF 量化过程中发生错误: {str(e)}")
    
    print("✓ GGUF 量化完成！")
    return output_path


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128
) -> str:
    """AWQ 4-bit 量化
    
    AWQ (Activation-aware Weight Quantization) 通过保护显著权重通道
    实现高精度 4-bit 量化，适合小模型。
    
    Args:
        model_path: 输入模型路径
        output_path: 输出模型路径
        bits: 量化位数，默认 4
        group_size: 分组大小，默认 128
        
    Returns:
        输出模型路径
        
    Raises:
        QuantizationError: 量化失败时抛出
    """
    print(f"=== AWQ {bits}-bit 量化 ===")
    print(f"输入模型: {model_path}")
    print(f"输出路径: {output_path}")
    
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        
        # 检查 VRAM 并优化配置
        config = {
            "zero_point": True,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "GEMM",
        }
        config = optimize_for_low_vram(config)
        
        # 加载模型和分词器
        print("加载模型...")
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        print(f"开始 AWQ {bits}-bit 量化...")
        print(f"  配置: zero_point={config['zero_point']}, "
              f"group_size={config['q_group_size']}, version={config['version']}")
        
        model.quantize(tokenizer, quant_config=config)
        
        # 保存量化模型
        print(f"保存量化模型到: {output_path}")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("✓ AWQ 量化完成！")
        return output_path
        
    except ImportError as e:
        raise QuantizationError(
            f"缺少依赖: {e.name}\n"
            f"请安装: pip install autoawq"
        )
    except Exception as e:
        raise QuantizationError(f"AWQ 量化失败: {str(e)}")


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False
) -> str:
    """GPTQ 量化
    
    GPTQ 基于 Optimal Brain Quantization 框架，通过逐层量化
    和误差补偿实现高精度量化。
    
    Args:
        model_path: 输入模型路径
        output_path: 输出模型路径
        bits: 量化位数，4 或 8
        group_size: 分组大小，默认 128
        desc_act: 是否按激活值排序，默认 False
        
    Returns:
        输出模型路径
        
    Raises:
        QuantizationError: 量化失败时抛出
    """
    if bits not in [4, 8]:
        raise ValueError(f"GPTQ 仅支持 4-bit 或 8-bit 量化，收到: {bits}")
    
    print(f"=== GPTQ {bits}-bit 量化 ===")
    print(f"输入模型: {model_path}")
    print(f"输出路径: {output_path}")
    
    try:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        # GPTQ 量化配置
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            damp_percent=0.1,
        )
        
        print("加载模型...")
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # 准备校准数据
        print("准备校准数据...")
        calibration_texts = [
            "信息安全是保护计算机系统和网络免受未经授权的访问、攻击或破坏的实践。",
            "SQL注入是一种常见的Web安全漏洞，攻击者通过注入恶意SQL代码来操纵数据库。",
            "渗透测试是一种模拟真实攻击者的方法来评估系统安全性的技术。",
            "防火墙是网络安全的第一道防线，用于监控和控制进出网络的流量。",
            "加密技术用于保护数据在传输和存储过程中的机密性和完整性。",
            "身份验证是确认用户身份的过程，通常使用密码、生物特征或多因素认证。",
            "网络监控用于检测和分析网络流量，以识别潜在的安全威胁。",
            "数据备份是防止数据丢失的重要措施，应定期进行并存储在安全位置。",
        ]
        
        calibration_data = [
            tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            for text in calibration_texts
        ]
        
        print(f"开始 GPTQ {bits}-bit 量化...")
        print(f"  配置: bits={bits}, group_size={group_size}, desc_act={desc_act}")
        
        model.quantize(calibration_data)
        
        print(f"保存量化模型到: {output_path}")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("✓ GPTQ 量化完成！")
        return output_path
        
    except ImportError as e:
        raise QuantizationError(
            f"缺少依赖: {e.name}\n"
            f"请安装: pip install auto-gptq"
        )
    except Exception as e:
        raise QuantizationError(f"GPTQ 量化失败: {str(e)}")


def evaluate_perplexity(
    model_path: str,
    dataset: str = "wikitext",
    max_samples: int = 100,
    stride: int = 512
) -> float:
    """评估量化模型的 PPL (Perplexity)
    
    Args:
        model_path: 模型路径
        dataset: 评估数据集名称，默认 wikitext
        max_samples: 最大评估样本数
        stride: 滑动窗口步长
        
    Returns:
        PPL 值
        
    Raises:
        QuantizationError: 评估失败时抛出
    """
    print(f"=== 评估模型 PPL ===")
    print(f"模型: {model_path}")
    print(f"数据集: {dataset}")
    
    try:
        # 导入评测模块
        from hos_optimizer.evaluate import EvaluationEngine, EvaluationConfig, MetricLoader
        
        # 创建评测配置
        config = EvaluationConfig(
            model_path=model_path,
            dataset_path="",  # PPL 计算不需要数据集文件
            metrics=["ppl"],
            task_type="perplexity",
            max_samples=max_samples,
            batch_size=1,
            device_map="auto",
        )
        
        # 创建评测引擎
        engine = EvaluationEngine(config)
        
        # 加载模型
        print("加载模型...")
        engine.load_model()
        
        # 加载评估数据集
        print(f"加载评估数据: {dataset}...")
        from datasets import load_dataset
        
        test_data = load_dataset(dataset, "wikitext-2-raw-v1", split="test")
        
        # 使用参考文本计算 PPL
        references = []
        for i, text in enumerate(test_data["text"]):
            if text and text.strip():
                references.append(text)
            if len(references) >= max_samples:
                break
        
        if not references:
            raise QuantizationError("数据集中没有有效的文本样本")
        
        # 计算 PPL
        print("计算 PPL...")
        metrics_result = engine.compute_metrics(
            predictions=references,
            references=references
        )
        
        ppl_value = metrics_result.get("ppl", 0.0)
        
        # 释放资源
        engine.shutdown()
        
        print(f"✓ PPL 评估完成: {ppl_value:.2f}")
        return ppl_value
        
    except ImportError as e:
        raise QuantizationError(
            f"缺少依赖: {e.name}\n"
            f"请安装: pip install datasets evaluate"
        )
    except Exception as e:
        raise QuantizationError(f"PPL 评估失败: {str(e)}")


def convert_format(
    model_path: str,
    output_path: str,
    from_format: str,
    to_format: str,
    **kwargs
) -> str:
    """量化格式转换工具
    
    Args:
        model_path: 输入模型路径
        output_path: 输出模型路径
        from_format: 源格式 (gguf, awq, gptq, hf)
        to_format: 目标格式 (gguf, awq, gptq, hf)
        **kwargs: 其他参数
        
    Returns:
        输出模型路径
        
    Raises:
        QuantizationError: 转换失败时抛出
    """
    print(f"=== 格式转换: {from_format} -> {to_format} ===")
    print(f"输入: {model_path}")
    print(f"输出: {output_path}")
    
    # 支持的转换路径
    conversion_map = {
        ("hf", "gguf"): lambda: quantize_gguf(model_path, output_path, **kwargs),
        ("hf", "awq"): lambda: quantize_awq(model_path, output_path, **kwargs),
        ("hf", "gptq"): lambda: quantize_gptq(model_path, output_path, **kwargs),
        ("gguf", "hf"): lambda: _convert_gguf_to_hf(model_path, output_path),
        ("awq", "hf"): lambda: _convert_awq_to_hf(model_path, output_path),
        ("gptq", "hf"): lambda: _convert_gptq_to_hf(model_path, output_path),
    }
    
    conversion_key = (from_format.lower(), to_format.lower())
    
    if conversion_key not in conversion_map:
        raise QuantizationError(
            f"不支持的转换路径: {from_format} -> {to_format}\n"
            f"支持的转换: {', '.join([f'{k[0]}->{k[1]}' for k in conversion_map.keys()])}"
        )
    
    try:
        result = conversion_map[conversion_key]()
        print(f"✓ 格式转换完成: {from_format} -> {to_format}")
        return result
    except Exception as e:
        raise QuantizationError(f"格式转换失败: {str(e)}")


def _convert_gguf_to_hf(model_path: str, output_path: str) -> str:
    """GGUF 转 HuggingFace 格式"""
    # 这里需要实现具体的转换逻辑
    # 通常需要使用 llama.cpp 的转换工具
    raise NotImplementedError("GGUF -> HF 转换尚未实现")


def _convert_awq_to_hf(model_path: str, output_path: str) -> str:
    """AWQ 转 HuggingFace 格式"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("加载 AWQ 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"保存到: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path


def _convert_gptq_to_hf(model_path: str, output_path: str) -> str:
    """GPTQ 转 HuggingFace 格式"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("加载 GPTQ 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    print(f"保存到: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path


def get_model_size(model_path: str) -> float:
    """获取模型文件大小 (GB)
    
    Args:
        model_path: 模型路径
        
    Returns:
        模型大小 (GB)
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(model_path):
        for f in filenames:
            if f.endswith((".safetensors", ".bin", ".pt", ".gguf")):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="HOS Model Optimizer - 量化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # GGUF 量化
  python quantize.py --method gguf --model ./model --output ./model.gguf --quant-type Q4_K_M

  # AWQ 4-bit 量化
  python quantize.py --method awq --bits 4 --model ./model --output ./model-awq

  # GPTQ 8-bit 量化
  python quantize.py --method gptq --bits 8 --model ./model --output ./model-gptq

  # 评估 PPL
  python quantize.py --method evaluate --model ./quantized-model

  # 格式转换
  python quantize.py --method convert --from hf --to awq --model ./model --output ./model-awq
        """
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["gguf", "awq", "gptq", "evaluate", "convert"],
        help="量化方法"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="输入模型路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出路径（默认自动生成）"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位数（默认: 4）"
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="Q4_K_M",
        help="GGUF 量化类型（默认: Q4_K_M）"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="量化分组大小（默认: 128）"
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        default=None,
        help="llama.cpp 安装路径（用于 GGUF 量化）"
    )
    parser.add_argument(
        "--from",
        dest="from_format",
        type=str,
        choices=["gguf", "awq", "gptq", "hf"],
        help="源格式（用于格式转换）"
    )
    parser.add_argument(
        "--to",
        dest="to_format",
        type=str,
        choices=["gguf", "awq", "gptq", "hf"],
        help="目标格式（用于格式转换）"
    )
    
    args = parser.parse_args()
    
    # 自动生成输出路径
    if args.output is None:
        model_name = os.path.basename(args.model)
        if args.method == "gguf":
            args.output = f"{model_name}-{args.quant_type}.gguf"
        elif args.method == "convert":
            args.output = f"{model_name}-{args.to_format}"
        elif args.method == "evaluate":
            print("错误: 评估模式需要指定 --output 参数")
            sys.exit(1)
        else:
            args.output = f"{model_name}-{args.method}-{args.bits}bit"
    
    # 显示 VRAM 信息
    vram_info = check_vram_availability()
    if vram_info["available"]:
        print(f"GPU: {vram_info['device']}")
        print(f"VRAM: {vram_info['free_vram_gb']:.2f}GB / {vram_info['total_vram_gb']:.2f}GB")
    else:
        print("⚠️  未检测到 GPU，将使用 CPU（速度较慢）")
    
    # 显示原始模型大小
    if os.path.exists(args.model):
        original_size = get_model_size(args.model)
        print(f"原始模型大小: {original_size:.2f} GB")
    
    # 执行量化
    try:
        if args.method == "gguf":
            result = quantize_gguf(
                args.model,
                args.output,
                quant_type=args.quant_type,
                llama_cpp_path=args.llama_cpp_path
            )
        elif args.method == "awq":
            result = quantize_awq(
                args.model,
                args.output,
                bits=args.bits,
                group_size=args.group_size
            )
        elif args.method == "gptq":
            result = quantize_gptq(
                args.model,
                args.output,
                bits=args.bits,
                group_size=args.group_size
            )
        elif args.method == "evaluate":
            ppl = evaluate_perplexity(args.model)
            print(f"\n最终 PPL: {ppl:.2f}")
            result = None
        elif args.method == "convert":
            if not args.from_format or not args.to_format:
                print("错误: 格式转换需要指定 --from 和 --to 参数")
                sys.exit(1)
            result = convert_format(
                args.model,
                args.output,
                args.from_format,
                args.to_format,
                bits=args.bits,
                quant_type=args.quant_type
            )
        
        # 显示量化后模型大小
        if result and os.path.exists(result):
            quantized_size = get_model_size(result)
            print(f"\n量化后模型大小: {quantized_size:.2f} GB")
            if os.path.exists(args.model):
                compression = (1 - quantized_size / original_size) * 100
                print(f"压缩率: {compression:.1f}%")
                
    except QuantizationError as e:
        print(f"\n❌ 量化失败: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

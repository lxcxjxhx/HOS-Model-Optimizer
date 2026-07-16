#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOS Model Optimizer - 统一命令行接口

提供统一的命令行入口，整合所有功能模块：
- hos-quantize: 模型量化
- hos-infer: 推理服务
- hos-train: 微调训练
- hos-merge: 模型合并
- hos-deploy: 部署服务
- hos-config: 配置管理
"""

import click
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@click.group()
@click.version_option(version="1.0.0", prog_name="HOS Model Optimizer")
def cli():
    """HOS Model Optimizer - 小模型优化工具集

    提供模型量化、推理、训练、合并和部署的完整工具链。
    针对 8GB VRAM 场景进行了优化。

    使用示例：
      hos-quantize --method gguf --model ./model --output ./model.gguf
      hos-infer --model ./model --prompt "你好"
      hos-train --model ./model --dataset ./data.json
      hos-deploy --model ./model --port 8000
    """
    pass


# ============================================================
# hos-quantize: 模型量化
# ============================================================

@cli.command("hos-quantize")
@click.option("--method", type=click.Choice(["gguf", "awq", "gptq", "evaluate", "convert"]),
              required=True, help="量化方法")
@click.option("--model", type=str, required=True, help="输入模型路径")
@click.option("--output", type=str, default=None, help="输出路径")
@click.option("--bits", type=int, default=4, help="量化位数（默认: 4）")
@click.option("--quant-type", type=str, default="Q4_K_M", help="GGUF 量化类型")
@click.option("--group-size", type=int, default=128, help="量化分组大小")
@click.option("--llama-cpp-path", type=str, default=None, help="llama.cpp 安装路径")
@click.option("--from", "from_format", type=click.Choice(["gguf", "awq", "gptq", "hf"]),
              help="源格式（用于格式转换）")
@click.option("--to", "to_format", type=click.Choice(["gguf", "awq", "gptq", "hf"]),
              help="目标格式（用于格式转换）")
def quantize_cmd(method, model, output, bits, quant_type, group_size, llama_cpp_path,
                 from_format, to_format):
    """量化模型

    支持多种量化方法：GGUF、AWQ、GPTQ，以及格式转换和质量评估。

    示例：
      # GGUF 量化
      hos-quantize --method gguf --model ./model --output ./model.gguf

      # AWQ 4-bit 量化
      hos-quantize --method awq --bits 4 --model ./model

      # 评估 PPL
      hos-quantize --method evaluate --model ./quantized-model
    """
    from hos_optimizer.quantize import main as quantize_main

    # 构建参数列表
    args = ["--method", method, "--model", model]
    if output:
        args.extend(["--output", output])
    if method in ["awq", "gptq"]:
        args.extend(["--bits", str(bits)])
    if method == "gguf":
        args.extend(["--quant-type", quant_type])
    if method in ["awq", "gptq"]:
        args.extend(["--group-size", str(group_size)])
    if llama_cpp_path:
        args.extend(["--llama-cpp-path", llama_cpp_path])
    if method == "convert":
        if from_format:
            args.extend(["--from", from_format])
        if to_format:
            args.extend(["--to", to_format])

    sys.argv = ["quantize"] + args
    quantize_main()


# ============================================================
# hos-infer: 推理服务
# ============================================================

@cli.command("hos-infer")
@click.option("--model", type=str, required=True, help="模型路径")
@click.option("--backend", type=click.Choice(["llama-cpp", "vllm", "sglang"]),
              default=None, help="推理后端（默认自动检测）")
@click.option("--prompt", type=str, default=None, help="单次推理的输入提示")
@click.option("--serve", is_flag=True, help="启动 API 服务")
@click.option("--chat", is_flag=True, help="交互模式")
@click.option("--benchmark", is_flag=True, help="性能基准测试")
@click.option("--host", type=str, default="0.0.0.0", help="服务监听地址")
@click.option("--port", type=int, default=None, help="服务端口")
@click.option("--max-tokens", type=int, default=256, help="最大生成 token 数")
@click.option("--temperature", type=float, default=0.7, help="采样温度")
@click.option("--top-p", type=float, default=0.9, help="nucleus sampling 参数")
def infer_cmd(model, backend, prompt, serve, chat, benchmark, host, port,
              max_tokens, temperature, top_p):
    """推理服务

    支持三种推理后端：llama-cpp、vLLM、SGLang，自动选择最优后端。

    示例：
      # 单次推理
      hos-infer --model ./model --prompt "你好"

      # 启动 API 服务
      hos-infer --model ./model --serve --port 8000

      # 交互模式
      hos-infer --model ./model --chat
    """
    from hos_optimizer.inference import main as infer_main

    args = ["--model", model]
    if backend:
        args.extend(["--backend", backend])
    if serve:
        args.append("--serve")
    if chat:
        args.append("--chat")
    if benchmark:
        args.append("--benchmark")
    if prompt:
        args.extend(["--prompt", prompt])
    args.extend(["--host", host])
    if port:
        args.extend(["--port", str(port)])
    args.extend(["--max-tokens", str(max_tokens)])
    args.extend(["--temperature", str(temperature)])
    args.extend(["--top-p", str(top_p)])

    sys.argv = ["inference"] + args
    infer_main()


# ============================================================
# hos-train: 微调训练
# ============================================================

@cli.command("hos-train")
@click.option("--model", type=str, required=True, help="基础模型路径")
@click.option("--dataset", type=str, required=True, help="数据集路径（JSON 文件）")
@click.option("--output", type=str, default="./output", help="输出目录")
@click.option("--method", type=click.Choice(["qlora", "lora"]), default="qlora",
              help="训练方法")
@click.option("--format", "dataset_format", type=click.Choice(["alpaca", "sharegpt"]),
              default="alpaca", help="数据格式")
@click.option("--max-seq-length", type=int, default=2048, help="最大序列长度")
@click.option("--lora-rank", type=int, default=16, help="LoRA rank")
@click.option("--lora-alpha", type=int, default=32, help="LoRA alpha")
@click.option("--epochs", type=int, default=3, help="训练轮数")
@click.option("--batch-size", type=int, default=2, help="批次大小")
@click.option("--lr", type=float, default=2e-4, help="学习率")
@click.option("--merge", is_flag=True, help="训练后自动合并模型")
def train_cmd(model, dataset, output, method, dataset_format, max_seq_length,
              lora_rank, lora_alpha, epochs, batch_size, lr, merge):
    """微调训练

    支持 QLoRA 和 LoRA 微调，针对 8GB VRAM 场景优化。

    示例：
      # QLoRA 训练
      hos-train --model ./model --dataset ./data.json --method qlora

      # LoRA 训练并自动合并
      hos-train --model ./model --dataset ./data.json --method lora --merge
    """
    from hos_optimizer.train import main as train_main

    args = ["--model", model, "--dataset", dataset]
    args.extend(["--output", output])
    args.extend(["--method", method])
    args.extend(["--format", dataset_format])
    args.extend(["--max-seq-length", str(max_seq_length)])
    args.extend(["--lora-rank", str(lora_rank)])
    args.extend(["--lora-alpha", str(lora_alpha)])
    args.extend(["--epochs", str(epochs)])
    args.extend(["--batch-size", str(batch_size)])
    args.extend(["--lr", str(lr)])
    if merge:
        args.append("--merge")

    sys.argv = ["train"] + args
    train_main()


# ============================================================
# hos-merge: 合并模型
# ============================================================

@cli.command("hos-merge")
@click.option("--base-model", type=str, required=True, help="基础模型路径")
@click.option("--adapter", type=str, required=True, help="LoRA adapter 路径")
@click.option("--output", type=str, required=True, help="输出路径")
def merge_cmd(base_model, adapter, output):
    """合并模型

    将 LoRA adapter 合并到基础模型中。

    示例：
      hos-merge --base-model ./base --adapter ./adapter --output ./merged
    """
    from hos_optimizer.train import merge_model

    click.echo(f"合并模型: {base_model} + {adapter} -> {output}")
    merge_model(
        base_model_path=base_model,
        adapter_path=adapter,
        output_path=output,
    )
    click.echo("模型合并完成！")


# ============================================================
# hos-deploy: 部署服务
# ============================================================

@cli.command("hos-deploy")
@click.option("--model", type=str, required=True, help="模型文件路径")
@click.option("--model-size", type=float, default=7.0, help="模型大小（十亿参数）")
@click.option("--use-case", type=click.Choice(["general", "high_concurrency", "multi_turn"]),
              default="general", help="使用场景")
@click.option("--host", type=str, default="0.0.0.0", help="服务主机地址")
@click.option("--port", type=int, default=8000, help="服务端口")
@click.option("--no-auto-start", is_flag=True, help="不自动启动服务")
def deploy_cmd(model, model_size, use_case, host, port, no_auto_start):
    """部署服务

    自动检测硬件并选择最优配置，一键部署 API 服务。

    示例：
      # 部署 7B 模型
      hos-deploy --model ./model.gguf --model-size 7.0

      # 高并发场景
      hos-deploy --model ./model --use-case high_concurrency
    """
    from hos_optimizer.deploy import main as deploy_main

    args = ["--model-path", model]
    args.extend(["--model-size", str(model_size)])
    args.extend(["--use-case", use_case])
    args.extend(["--host", host])
    args.extend(["--port", str(port)])
    if no_auto_start:
        args.append("--no-auto-start")

    sys.argv = ["deploy"] + args
    deploy_main()


# ============================================================
# hos-config: 配置管理
# ============================================================

@cli.command("hos-config")
@click.option("--generate", is_flag=True, help="生成最优配置")
@click.option("--scenario", type=str, help="场景名称")
@click.option("--model-path", type=str, default="", help="模型路径")
@click.option("--vram", type=float, default=8.0, help="可用显存（GB）")
@click.option("--validate", is_flag=True, help="验证配置文件")
@click.option("--config", type=str, help="配置文件路径")
@click.option("--list-templates", is_flag=True, help="列出所有模板")
@click.option("--export-template", type=str, help="导出模板")
@click.option("--output", "-o", type=str, help="输出文件路径")
def config_cmd(generate, scenario, model_path, vram, validate, config,
               list_templates, export_template, output):
    """配置管理

    生成 8GB VRAM 最优配置，验证配置文件，管理配置模板。

    示例：
      # 生成推理配置
      hos-config --generate --scenario inference_7b --model-path ./model

      # 验证配置
      hos-config --validate --config my_config.yaml

      # 列出模板
      hos-config --list-templates
    """
    from hos_optimizer.config import main as config_main

    args = []
    if generate:
        args.append("--generate")
        if scenario:
            args.extend(["--scenario", scenario])
        args.extend(["--model-path", model_path])
        args.extend(["--vram", str(vram)])
    elif validate:
        args.append("--validate")
        if config:
            args.extend(["--config", config])
    elif list_templates:
        args.append("--list-templates")
    elif export_template:
        args.extend(["--export-template", export_template])

    if output:
        args.extend(["--output", output])

    sys.argv = ["config"] + args
    config_main()


def main():
    """主入口函数"""
    cli()


if __name__ == "__main__":
    main()

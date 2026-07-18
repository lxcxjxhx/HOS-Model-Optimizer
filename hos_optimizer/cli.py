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
- hos-evaluate: 模型评测
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

@cli.command("quantize")
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
    from hos_optimizer.quantize import quantize_gguf, quantize_awq, quantize_gptq, evaluate_perplexity, convert_format

    if method == "gguf":
        quantize_gguf(model, output or f"{model}.gguf", quant_type, llama_cpp_path)
    elif method == "awq":
        quantize_awq(model, output or f"{model}-awq", bits, group_size)
    elif method == "gptq":
        quantize_gptq(model, output or f"{model}-gptq", bits, group_size)
    elif method == "evaluate":
        evaluate_perplexity(model)
    elif method == "convert":
        if not from_format or not to_format:
            click.echo("错误: 格式转换需要指定 --from 和 --to 参数", err=True)
            sys.exit(1)
        convert_format(model, output or f"{model}-converted", from_format, to_format)


# ============================================================
# hos-infer: 推理服务
# ============================================================

@cli.command("infer")
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
    from hos_optimizer.inference import UnifiedInferenceEngine, run_chat, run_benchmark, run_single_inference
    
    # 确定默认端口
    default_ports = {"llama-cpp": 8080, "llama_cpp": 8080, "vllm": 8000, "sglang": 30000}
    backend_key = (backend or "auto").lower().replace("-", "_")
    actual_port = port or default_ports.get(backend_key, 8000)
    
    try:
        if serve:
            # 启动服务模式
            from hos_optimizer.inference import UnifiedInferenceEngine
            engine = UnifiedInferenceEngine(model_path=model, backend=backend, auto_load=True)
            engine._backend.serve(host=host, port=actual_port)
        else:
            # 创建推理引擎
            engine = UnifiedInferenceEngine(
                model_path=model,
                backend=backend,
                auto_load=True,
            )
            
            if chat:
                run_chat(engine)
            elif benchmark:
                run_benchmark(engine)
            elif prompt:
                run_single_inference(
                    engine,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                click.echo("请指定 --serve, --chat, --benchmark 或 --prompt 参数")
                sys.exit(1)
    except Exception as e:
        click.echo(f"推理错误: {e}", err=True)
        sys.exit(1)


# ============================================================
# hos-train: 微调训练
# ============================================================

@cli.command("train")
@click.option("--model", type=str, required=True, help="基础模型路径")
@click.option("--dataset", type=str, required=True, help="数据集路径（JSON 文件）")
@click.option("--output", type=str, default="./output", help="输出目录")
@click.option("--method", type=click.Choice(["qlora", "lora"]), default="qlora",
              help="训练方法")
@click.option("--format", "dataset_format", type=click.Choice(["alpaca", "sharegpt", "messages"]),
              default="alpaca", help="数据格式")
@click.option("--max-seq-length", type=int, default=2048, help="最大序列长度")
@click.option("--lora-rank", type=int, default=16, help="LoRA rank")
@click.option("--lora-alpha", type=int, default=32, help="LoRA alpha")
@click.option("--epochs", type=int, default=3, help="训练轮数")
@click.option("--batch-size", type=int, default=2, help="批次大小")
@click.option("--grad-accum", type=int, default=8, help="梯度累积步数")
@click.option("--lr", type=float, default=2e-4, help="学习率")
@click.option("--no-unsloth", is_flag=True, help="禁用 Unsloth 加速")
@click.option("--merge", is_flag=True, help="训练后自动合并模型")
def train_cmd(model, dataset, output, method, dataset_format, max_seq_length,
              lora_rank, lora_alpha, epochs, batch_size, grad_accum, lr, no_unsloth, merge):
    """微调训练

    支持 QLoRA 和 LoRA 微调，针对 8GB VRAM 场景优化。

    示例：
      # QLoRA 训练
      hos-train --model ./model --dataset ./data.json --method qlora

      # LoRA 训练并自动合并
      hos-train --model ./model --dataset ./data.json --method lora --merge
    """
    from hos_optimizer.train import TrainingConfig, train, merge_model
    
    # 创建训练配置
    config = TrainingConfig(
        model_name_or_path=model,
        dataset_path=dataset,
        dataset_format=dataset_format,
        max_seq_length=max_seq_length,
        finetuning_type=method,
        use_4bit=(method == "qlora"),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        use_unsloth=not no_unsloth,
    )
    
    try:
        # 执行训练
        train(config)
        
        # 可选：自动合并
        if merge:
            click.echo("开始合并模型...")
            merge_model(
                base_model_path=model,
                adapter_path=output,
                output_path=f"{output}_merged",
            )
            click.echo("模型合并完成！")
    except Exception as e:
        click.echo(f"训练错误: {e}", err=True)
        sys.exit(1)


# ============================================================
# hos-merge: 合并模型
# ============================================================

@cli.command("merge")
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

@cli.command("deploy")
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
    from hos_optimizer.deploy import HardwareDetector, ConfigSelector, ServiceLauncher

    try:
        # 检测硬件
        hw_info = HardwareDetector.get_hardware_info()
        click.echo(f"检测到硬件: {hw_info.gpu_name}, {hw_info.gpu_memory_gb:.1f}GB VRAM")

        # 选择配置
        config = ConfigSelector.select_config(hw_info, model_size, use_case)
        click.echo(f"选择配置: {config.backend.value}, {config.quantization.value}")
        click.echo(f"推荐场景: {config.recommended_for}")

        # 启动服务
        launcher = ServiceLauncher(config, model, host, port)
        if not no_auto_start:
            if launcher.start():
                click.echo(f"✓ 服务已启动，监听: {host}:{port}")
            else:
                click.echo("错误: 服务启动失败", err=True)
                sys.exit(1)
        else:
            click.echo("服务未自动启动（--no-auto-start）")
    except Exception as e:
        click.echo(f"部署错误: {e}", err=True)
        sys.exit(1)


# ============================================================
# hos-config: 配置管理
# ============================================================

@cli.command("config")
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
    from hos_optimizer.config import ConfigManager
    
    manager = ConfigManager()
    
    try:
        if list_templates:
            templates = manager.list_templates()
            click.echo("可用的配置模板:")
            for name in templates:
                click.echo(f"  - {name}")
        elif export_template:
            if not output:
                click.echo("错误: 导出模板需要指定 --output 参数", err=True)
                sys.exit(1)
            manager.export_template(export_template, output)
            click.echo(f"模板 '{export_template}' 已导出到: {output}")
        elif validate:
            if not config:
                click.echo("错误: 验证配置需要指定 --config 参数", err=True)
                sys.exit(1)
            cfg = manager.load_config(config)
            issues = manager.validate_config(cfg)
            if not issues:
                click.echo("配置验证通过 ✓")
            else:
                click.echo(f"发现 {len(issues)} 个问题:")
                for issue in issues:
                    click.echo(f"  {issue}")
                sys.exit(1)
        elif generate:
            if not scenario:
                click.echo("错误: 生成配置需要指定 --scenario 参数", err=True)
                sys.exit(1)
            cfg = manager.generate_optimal_config(
                scenario=scenario,
                model_path=model_path,
                vram_gb=vram,
            )
            if output:
                manager.save_config(cfg, output)
                click.echo(f"配置已保存到: {output}")
            else:
                import yaml
                click.echo(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        else:
            click.echo("请指定 --generate, --validate, --list-templates 或 --export-template")
            sys.exit(1)
    except Exception as e:
        click.echo(f"配置错误: {e}", err=True)
        sys.exit(1)


# ============================================================
# hos-evaluate: 模型评测
# ============================================================

@cli.command("evaluate")
@click.option("--model", type=str, required=True, multiple=True, help="模型路径（支持多个模型对比评测）")
@click.option("--dataset", type=str, required=True, help="评测数据集路径（JSON/JSONL 格式）")
@click.option("--metrics", type=str, multiple=True, default=["bleu", "rouge"],
              help="评测指标（默认: bleu rouge），支持: ppl bleu rouge exact_match f1")
@click.option("--task", type=click.Choice(["text_generation", "perplexity"]),
              default="text_generation", help="任务类型（默认: text_generation）")
@click.option("--output", "-o", type=str, default=None, help="结果输出路径（支持 .json 和 .md 格式）")
@click.option("--format", "dataset_format", type=click.Choice(["alpaca", "sharegpt", "messages"]),
              default=None, help="数据集格式（默认自动检测）")
@click.option("--max-samples", type=int, default=None, help="最大评测样本数（默认全部）")
@click.option("--max-seq-length", type=int, default=512, help="最大序列长度（默认 512）")
@click.option("--max-new-tokens", type=int, default=256, help="最大生成 token 数（默认 256）")
@click.option("--temperature", type=float, default=0.7, help="采样温度（默认 0.7）")
@click.option("--top-p", type=float, default=0.9, help="nucleus sampling 参数（默认 0.9）")
@click.option("--batch-size", type=int, default=1, help="推理批大小（8GB VRAM 建议 1，默认 1）")
@click.option("--output-format", type=click.Choice(["json", "markdown"]),
              default="json", help="输出格式（默认: json）")
@click.option("--load-in-4bit", is_flag=True, default=False,
              help="使用 4-bit 量化加载模型（节省显存，适合 8GB VRAM 场景）")
@click.option("--verbose", is_flag=True, default=False, help="启用详细日志输出")
def evaluate_cmd(model, dataset, metrics, task, output, dataset_format, max_samples,
                 max_seq_length, max_new_tokens, temperature, top_p, batch_size,
                 output_format, load_in_4bit, verbose):
    """模型评测

    支持多种评测指标：PPL、BLEU、ROUGE、Exact Match、F1 等。
    支持多模型对比评测，结果可导出为 JSON 或 Markdown 格式。

    示例：
      # 单模型评测
      hos-evaluate --model ./model --dataset ./test.json --metrics bleu rouge

      # 多模型对比
      hos-evaluate --model ./model_a --model ./model_b --dataset ./test.json --metrics bleu f1

      # 指定输出格式
      hos-evaluate --model ./model --dataset ./test.json --output result.md

      # 使用 4-bit 量化加载（8GB VRAM 优化）
      hos-evaluate --model ./model --dataset ./test.json --load-in-4bit
    """
    from hos_optimizer.evaluate import evaluate_model, compare_models, EvaluationConfig

    # 自动推断输出格式
    actual_output_format = output_format
    if output:
        if output.endswith(".md"):
            actual_output_format = "markdown"
        elif output.endswith(".json"):
            actual_output_format = "json"

    # 构建评测配置
    config = EvaluationConfig(
        model_path=model[0] if len(model) == 1 else "",
        dataset_path=dataset,
        dataset_format=dataset_format,
        max_samples=max_samples,
        max_seq_length=max_seq_length,
        metrics=list(metrics),
        task_type=task,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        output_format=actual_output_format,
        output_path=output,
        load_in_4bit=load_in_4bit,
        verbose=verbose,
    )

    try:
        if len(model) > 1:
            compare_models(list(model), config)
        else:
            evaluate_model(config)
    except Exception as e:
        click.echo(f"评测错误: {e}", err=True)
        sys.exit(1)


# ============================================================
# hos-upload: 上传模型到 HuggingFace Hub
# ============================================================

@cli.command("upload")
@click.option("--model", type=str, required=True, help="模型目录路径")
@click.option("--repo-id", type=str, required=True, help="HuggingFace 仓库 ID（如 user/repo）")
@click.option("--private", is_flag=True, default=False, help="创建私有仓库")
def upload_cmd(model, repo_id, private):
    """上传模型到 HuggingFace Hub

    将本地模型目录逐文件上传到 HuggingFace Hub 仓库。

    示例：
      # 上传模型到公开仓库
      hos-optimizer upload --model ./merged --repo-id user/my-model

      # 上传到私有仓库
      hos-optimizer upload --model ./merged --repo-id user/my-model --private
    """
    # 在导入 upload 模块前设置环境变量，抑制警告以确保进度条正常显示
    import os
    import warnings
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore", category=UserWarning)

    from hos_optimizer.upload import upload_to_huggingface

    try:
        upload_to_huggingface(
            model_path=model,
            repo_id=repo_id,
            private=private,
        )
    except RuntimeError as e:
        click.echo(f"认证错误: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"路径错误: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"上传错误: {e}", err=True)
        sys.exit(1)


# ============================================================
# hos-run: 统一工作流执行
# ============================================================

@cli.command("run")
@click.option("--config", type=str, required=True, help="Pipeline 配置文件路径 (YAML)")
@click.option("--dry-run", is_flag=True, default=False, help="仅显示执行计划，不实际执行")
def run_cmd(config, dry_run):
    """执行统一工作流

    通过 YAML 配置文件定义完整的工作流程，支持训练、合并、上传、评测、部署等多步骤编排。

    示例：
      # 查看执行计划
      hos-optimizer run --config pipeline.yaml --dry-run

      # 执行完整流程
      hos-optimizer run --config pipeline.yaml
    """
    from hos_optimizer.pipeline import PipelineConfig, PipelineExecutor

    try:
        pipeline_config = PipelineConfig.from_yaml(config)
        executor = PipelineExecutor(pipeline_config)
        result = executor.execute(dry_run=dry_run)

        if result.get("status") == "failed":
            click.echo(f"\n流程执行失败: {result.get('error')}", err=True)
            sys.exit(1)
    except FileNotFoundError:
        click.echo(f"配置文件不存在: {config}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"执行错误: {e}", err=True)
        sys.exit(1)


def main():
    """主入口函数"""
    cli()


if __name__ == "__main__":
    main()

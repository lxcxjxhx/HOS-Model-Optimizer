#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署模块 - HOS Model Optimizer
提供硬件检测、自动配置选择、服务启动和健康检查功能
"""

import os
import sys
import json
import time
import subprocess
import requests
import psutil
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Backend(Enum):
    """推理后端枚举"""
    LLAMA_CPP = "llama-cpp"
    VLLM = "vllm"
    SGLANG = "sglang"


class Quantization(Enum):
    """量化方案枚举"""
    GGUF_Q4_K_M = "q4_k_m"
    GGUF_Q5_K_M = "q5_k_m"
    AWQ_4BIT = "awq-4bit"
    GPTQ_4BIT = "gptq-4bit"
    FP16 = "fp16"


@dataclass
class HardwareInfo:
    """硬件信息数据类"""
    gpu_name: str
    gpu_memory_gb: float
    cuda_version: Optional[str]
    cpu_cores: int
    system_memory_gb: float


@dataclass
class DeploymentConfig:
    """部署配置数据类"""
    backend: Backend
    quantization: Quantization
    max_model_size_gb: float
    recommended_for: str
    startup_args: Dict


class HardwareDetector:
    """硬件检测器"""
    
    @staticmethod
    def detect_gpu() -> Tuple[str, float, Optional[str]]:
        """
        检测GPU信息
        
        Returns:
            Tuple[str, float, Optional[str]]: (GPU名称, 显存大小GB, CUDA版本)
        """
        try:
            # 尝试使用nvidia-smi检测GPU
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = [p.strip() for p in lines[0].split(',')]
                    gpu_name = parts[0]
                    gpu_memory_gb = float(parts[1]) / 1024  # MB to GB
                    driver_version = parts[2] if len(parts) > 2 else None
                    
                    # 获取CUDA版本
                    cuda_result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    # 尝试获取CUDA版本
                    cuda_version = None
                    try:
                        cuda_check = subprocess.run(
                            ['nvcc', '--version'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if cuda_check.returncode == 0:
                            for line in cuda_check.stdout.split('\n'):
                                if 'release' in line.lower():
                                    cuda_version = line.split('release')[-1].strip().split(',')[0]
                                    break
                    except Exception:
                        cuda_version = f"driver-{driver_version}"
                    
                    return gpu_name, gpu_memory_gb, cuda_version
            
        except FileNotFoundError:
            logger.warning("未找到nvidia-smi，可能没有安装NVIDIA驱动")
        except Exception as e:
            logger.warning(f"GPU检测失败: {e}")
        
        # 如果没有GPU，返回CPU信息
        return "CPU", 0.0, None
    
    @staticmethod
    def detect_cpu() -> int:
        """
        检测CPU核心数
        
        Returns:
            int: CPU核心数
        """
        return psutil.cpu_count(logical=True)
    
    @staticmethod
    def detect_system_memory() -> float:
        """
        检测系统内存
        
        Returns:
            float: 系统内存大小GB
        """
        memory = psutil.virtual_memory()
        return memory.total / (1024 ** 3)  # 转换为GB
    
    @classmethod
    def get_hardware_info(cls) -> HardwareInfo:
        """
        获取完整硬件信息
        
        Returns:
            HardwareInfo: 硬件信息对象
        """
        gpu_name, gpu_memory_gb, cuda_version = cls.detect_gpu()
        cpu_cores = cls.detect_cpu()
        system_memory_gb = cls.detect_system_memory()
        
        return HardwareInfo(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cuda_version=cuda_version,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory_gb
        )


class ConfigSelector:
    """自动配置选择器"""
    
    @staticmethod
    def select_config(
        hardware: HardwareInfo,
        model_size_b: float,
        use_case: str = "general"
    ) -> DeploymentConfig:
        """
        根据硬件和使用场景选择最优配置
        
        Args:
            hardware: 硬件信息
            model_size_b: 模型大小（十亿参数）
            use_case: 使用场景 ("general", "high_concurrency", "multi_turn")
        
        Returns:
            DeploymentConfig: 推荐的部署配置
        """
        gpu_memory = hardware.gpu_memory_gb
        
        # 8GB VRAM 场景优化
        if 7.5 <= gpu_memory <= 8.5:
            # 小模型（0.8B-3B）
            if model_size_b <= 3:
                return DeploymentConfig(
                    backend=Backend.LLAMA_CPP,
                    quantization=Quantization.GGUF_Q4_K_M,
                    max_model_size_gb=4.0,
                    recommended_for="8GB VRAM + 小模型场景",
                    startup_args={
                        "n_gpu_layers": -1,  # 全部层放到GPU
                        "n_ctx": 4096,
                        "n_batch": 512,
                        "threads": max(1, hardware.cpu_cores - 2)
                    }
                )
            
            # 中等模型（7B-8B）
            elif model_size_b <= 8:
                return DeploymentConfig(
                    backend=Backend.LLAMA_CPP,
                    quantization=Quantization.GGUF_Q4_K_M,
                    max_model_size_gb=6.0,
                    recommended_for="8GB VRAM + 7B模型场景",
                    startup_args={
                        "n_gpu_layers": -1,
                        "n_ctx": 2048,  # 降低上下文长度以节省显存
                        "n_batch": 256,
                        "threads": max(1, hardware.cpu_cores - 2)
                    }
                )
            
            # 大模型（>8B）- 需要CPU offload
            else:
                return DeploymentConfig(
                    backend=Backend.LLAMA_CPP,
                    quantization=Quantization.GGUF_Q4_K_M,
                    max_model_size_gb=8.0,
                    recommended_for="8GB VRAM + 大模型（部分CPU offload）",
                    startup_args={
                        "n_gpu_layers": 20,  # 部分层放到GPU
                        "n_ctx": 1024,
                        "n_batch": 128,
                        "threads": hardware.cpu_cores
                    }
                )
        
        # 高并发场景
        elif use_case == "high_concurrency" and gpu_memory >= 16:
            return DeploymentConfig(
                backend=Backend.VLLM,
                quantization=Quantization.AWQ_4BIT,
                max_model_size_gb=gpu_memory * 0.7,
                recommended_for="高并发场景",
                startup_args={
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9,
                    "max_num_batched_tokens": 8192,
                    "max_num_seqs": 256
                }
            )
        
        # 多轮对话场景
        elif use_case == "multi_turn" and gpu_memory >= 12:
            return DeploymentConfig(
                backend=Backend.SGLANG,
                quantization=Quantization.GGUF_Q4_K_M,
                max_model_size_gb=gpu_memory * 0.75,
                recommended_for="多轮对话场景（RadixAttention优化）",
                startup_args={
                    "tp": 1,
                    "mem_fraction-static": 0.9,
                    "context-length": 8192
                }
            )
        
        # 大显存场景（>16GB）
        elif gpu_memory >= 16:
            return DeploymentConfig(
                backend=Backend.VLLM,
                quantization=Quantization.AWQ_4BIT,
                max_model_size_gb=gpu_memory * 0.8,
                recommended_for="大显存通用场景",
                startup_args={
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096
                }
            )
        
        # 默认配置（CPU或低显存）
        else:
            return DeploymentConfig(
                backend=Backend.LLAMA_CPP,
                quantization=Quantization.GGUF_Q4_K_M,
                max_model_size_gb=4.0,
                recommended_for="CPU或低显存场景",
                startup_args={
                    "n_gpu_layers": 0,  # 纯CPU推理
                    "n_ctx": 2048,
                    "n_batch": 256,
                    "threads": hardware.cpu_cores
                }
            )


class ServiceLauncher:
    """服务启动器"""
    
    def __init__(self, config: DeploymentConfig, model_path: str, host: str = "0.0.0.0", port: int = 8000):
        """
        初始化服务启动器
        
        Args:
            config: 部署配置
            model_path: 模型路径
            host: 服务主机地址
            port: 服务端口
        """
        self.config = config
        self.model_path = model_path
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
    
    def _build_llama_cpp_command(self) -> List[str]:
        """构建llama.cpp服务启动命令"""
        cmd = [
            "llama-server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--n-gpu-layers", str(self.config.startup_args.get("n_gpu_layers", 0)),
            "--ctx-size", str(self.config.startup_args.get("n_ctx", 2048)),
            "--batch-size", str(self.config.startup_args.get("n_batch", 256)),
            "--threads", str(self.config.startup_args.get("threads", 4))
        ]
        return cmd
    
    def _build_vllm_command(self) -> List[str]:
        """构建vLLM服务启动命令"""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.config.startup_args.get("tensor_parallel_size", 1)),
            "--gpu-memory-utilization", str(self.config.startup_args.get("gpu_memory_utilization", 0.9)),
            "--max-model-len", str(self.config.startup_args.get("max_model_len", 4096)),
            "--quantization", "awq"
        ]
        
        if "max_num_batched_tokens" in self.config.startup_args:
            cmd.extend(["--max-num-batched-tokens", str(self.config.startup_args["max_num_batched_tokens"])])
        
        if "max_num_seqs" in self.config.startup_args:
            cmd.extend(["--max-num-seqs", str(self.config.startup_args["max_num_seqs"])])
        
        return cmd
    
    def _build_sglang_command(self) -> List[str]:
        """构建SGLang服务启动命令"""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tp", str(self.config.startup_args.get("tp", 1)),
            "--mem-fraction-static", str(self.config.startup_args.get("mem_fraction_static", 0.9)),
            "--context-length", str(self.config.startup_args.get("context_length", 4096))
        ]
        return cmd
    
    def start(self) -> bool:
        """
        启动服务
        
        Returns:
            bool: 启动是否成功
        """
        try:
            # 根据后端类型构建命令
            if self.config.backend == Backend.LLAMA_CPP:
                cmd = self._build_llama_cpp_command()
            elif self.config.backend == Backend.VLLM:
                cmd = self._build_vllm_command()
            elif self.config.backend == Backend.SGLANG:
                cmd = self._build_sglang_command()
            else:
                logger.error(f"不支持的后端: {self.config.backend}")
                return False
            
            logger.info(f"启动服务命令: {' '.join(cmd)}")
            
            # 启动进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待服务启动
            time.sleep(3)
            
            # 检查进程是否还在运行
            if self.process.poll() is None:
                logger.info(f"服务已启动，PID: {self.process.pid}")
                return True
            else:
                # 进程已退出，读取错误信息
                stdout, stderr = self.process.communicate()
                logger.error(f"服务启动失败:\n{stderr}")
                return False
                
        except FileNotFoundError as e:
            logger.error(f"找不到可执行文件: {e}")
            logger.error("请确保已安装相应的推理后端")
            return False
        except Exception as e:
            logger.error(f"启动服务时出错: {e}")
            return False
    
    def stop(self):
        """停止服务"""
        if self.process:
            logger.info(f"停止服务，PID: {self.process.pid}")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("服务未能在5秒内停止，强制终止")
                self.process.kill()
            self.process = None


class HealthChecker:
    """服务健康检查器"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        初始化健康检查器
        
        Args:
            host: 服务主机地址
            port: 服务端口
        """
        self.base_url = f"http://{host}:{port}"
    
    def check_health(self, timeout: int = 5) -> bool:
        """
        检查服务健康状态
        
        Args:
            timeout: 请求超时时间（秒）
        
        Returns:
            bool: 服务是否健康
        """
        try:
            # 尝试多个常见的健康检查端点
            endpoints = ["/health", "/v1/health", "/api/health", "/"]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(
                        f"{self.base_url}{endpoint}",
                        timeout=timeout
                    )
                    if response.status_code == 200:
                        logger.info(f"健康检查通过: {endpoint}")
                        return True
                except requests.RequestException:
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    def check_model_loaded(self, timeout: int = 300) -> bool:
        """
        检查模型是否加载完成
        
        Args:
            timeout: 最大等待时间（秒）
        
        Returns:
            bool: 模型是否加载完成
        """
        logger.info("等待模型加载...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # 尝试获取模型信息
                response = requests.get(
                    f"{self.base_url}/v1/models",
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        logger.info("模型加载完成")
                        return True
                
            except requests.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error(f"模型加载超时（等待了{timeout}秒）")
        return False
    
    def get_service_info(self) -> Optional[Dict]:
        """
        获取服务信息
        
        Returns:
            Optional[Dict]: 服务信息字典，失败返回None
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            
        except Exception as e:
            logger.error(f"获取服务信息失败: {e}")
        
        return None


def deploy_model(
    model_path: str,
    model_size_b: float = 7.0,
    use_case: str = "general",
    host: str = "0.0.0.0",
    port: int = 8000,
    auto_start: bool = True
) -> Optional[ServiceLauncher]:
    """
    一键部署模型
    
    Args:
        model_path: 模型文件路径
        model_size_b: 模型大小（十亿参数）
        use_case: 使用场景
        host: 服务主机地址
        port: 服务端口
        auto_start: 是否自动启动服务
    
    Returns:
        Optional[ServiceLauncher]: 服务启动器实例
    """
    # 1. 检测硬件
    logger.info("正在检测硬件...")
    hardware = HardwareDetector.get_hardware_info()
    logger.info(f"硬件信息: GPU={hardware.gpu_name}, 显存={hardware.gpu_memory_gb:.1f}GB, "
                f"CPU核心={hardware.cpu_cores}, 系统内存={hardware.system_memory_gb:.1f}GB")
    
    # 2. 选择配置
    logger.info("正在选择最优配置...")
    config = ConfigSelector.select_config(hardware, model_size_b, use_case)
    logger.info(f"推荐配置: {config.recommended_for}")
    logger.info(f"  后端: {config.backend.value}")
    logger.info(f"  量化: {config.quantization.value}")
    
    # 3. 启动服务
    if auto_start:
        logger.info("正在启动服务...")
        launcher = ServiceLauncher(config, model_path, host, port)
        
        if not launcher.start():
            logger.error("服务启动失败")
            return None
        
        # 4. 健康检查
        logger.info("执行健康检查...")
        checker = HealthChecker(host.replace("0.0.0.0", "localhost"), port)
        
        if not checker.check_model_loaded(timeout=300):
            logger.error("模型加载失败")
            launcher.stop()
            return None
        
        logger.info(f"服务已成功启动: http://{host}:{port}")
        return launcher
    else:
        logger.info("配置已生成，但未自动启动服务")
        return ServiceLauncher(config, model_path, host, port)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="HOS Model Optimizer - 部署工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 部署7B模型到8GB显存
  python deploy.py --model-path ./model.gguf --model-size 7.0
  
  # 高并发场景部署
  python deploy.py --model-path ./model --model-size 7.0 --use-case high_concurrency
  
  # 多轮对话场景部署
  python deploy.py --model-path ./model --model-size 7.0 --use-case multi_turn
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型文件路径"
    )
    parser.add_argument(
        "--model-size",
        type=float,
        default=7.0,
        help="模型大小（十亿参数，默认: 7.0）"
    )
    parser.add_argument(
        "--use-case",
        type=str,
        choices=["general", "high_concurrency", "multi_turn"],
        default="general",
        help="使用场景（默认: general）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务主机地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="不自动启动服务，仅显示推荐配置"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 部署模型
    launcher = deploy_model(
        model_path=args.model_path,
        model_size_b=args.model_size,
        use_case=args.use_case,
        host=args.host,
        port=args.port,
        auto_start=not args.no_auto_start
    )
    
    if launcher is None:
        sys.exit(1)
    
    # 如果启动了服务，等待用户中断
    if not args.no_auto_start and launcher.process:
        logger.info("服务正在运行，按 Ctrl+C 停止...")
        try:
            launcher.process.wait()
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止服务...")
            launcher.stop()
            logger.info("服务已停止")


if __name__ == "__main__":
    main()

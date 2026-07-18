"""Pipeline execution module for unified workflow management"""
import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration"""
    model: str
    dataset: str
    output: str = "./output"
    method: str = "qlora"
    format: str = "alpaca"
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    no_unsloth: bool = False


@dataclass
class MergeConfig:
    """Merge configuration"""
    base_model: str
    adapter: str
    output: str
    after_train: bool = False


@dataclass
class UploadConfig:
    """Upload configuration"""
    model: str
    repo_id: str
    private: bool = False
    after_merge: bool = False


@dataclass
class EvaluateConfig:
    """Evaluation configuration"""
    model: str
    dataset: str
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge"])
    task: str = "text_generation"
    output: Optional[str] = None
    format: str = "alpaca"
    max_samples: Optional[int] = None
    max_seq_length: int = 512
    max_new_tokens: int = 256
    batch_size: int = 1
    load_in_4bit: bool = False


@dataclass
class DeployConfig:
    """Deployment configuration"""
    model: str
    model_size: float = 7.0
    use_case: str = "general"
    host: str = "0.0.0.0"
    port: int = 8000
    no_auto_start: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    train: Optional[TrainConfig] = None
    merge: Optional[MergeConfig] = None
    upload: Optional[UploadConfig] = None
    evaluate: Optional[EvaluateConfig] = None
    deploy: Optional[DeployConfig] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Parse train config
        if 'train' in data and data['train']:
            train_data = data['train']
            config.train = TrainConfig(
                model=train_data['model'],
                dataset=train_data['dataset'],
                output=train_data.get('output', './output'),
                method=train_data.get('method', 'qlora'),
                format=train_data.get('format', 'alpaca'),
                max_seq_length=train_data.get('max_seq_length', 2048),
                lora_rank=train_data.get('lora_rank', 16),
                lora_alpha=train_data.get('lora_alpha', 32),
                epochs=train_data.get('epochs', 3),
                batch_size=train_data.get('batch_size', 2),
                grad_accum=train_data.get('grad_accum', 8),
                lr=train_data.get('lr', 2e-4),
                no_unsloth=train_data.get('no_unsloth', False)
            )
        
        # Parse merge config
        if 'merge' in data and data['merge']:
            merge_data = data['merge']
            config.merge = MergeConfig(
                base_model=merge_data['base_model'],
                adapter=merge_data['adapter'],
                output=merge_data['output'],
                after_train=merge_data.get('after_train', False)
            )
        
        # Parse upload config
        if 'upload' in data and data['upload']:
            upload_data = data['upload']
            config.upload = UploadConfig(
                model=upload_data['model'],
                repo_id=upload_data['repo_id'],
                private=upload_data.get('private', False),
                after_merge=upload_data.get('after_merge', False)
            )
        
        # Parse evaluate config
        if 'evaluate' in data and data['evaluate']:
            eval_data = data['evaluate']
            config.evaluate = EvaluateConfig(
                model=eval_data['model'],
                dataset=eval_data['dataset'],
                metrics=eval_data.get('metrics', ['bleu', 'rouge']),
                task=eval_data.get('task', 'text_generation'),
                output=eval_data.get('output'),
                format=eval_data.get('format', 'alpaca'),
                max_samples=eval_data.get('max_samples'),
                max_seq_length=eval_data.get('max_seq_length', 512),
                max_new_tokens=eval_data.get('max_new_tokens', 256),
                batch_size=eval_data.get('batch_size', 1),
                load_in_4bit=eval_data.get('load_in_4bit', False)
            )
        
        # Parse deploy config
        if 'deploy' in data and data['deploy']:
            deploy_data = data['deploy']
            config.deploy = DeployConfig(
                model=deploy_data['model'],
                model_size=deploy_data.get('model_size', 7.0),
                use_case=deploy_data.get('use_case', 'general'),
                host=deploy_data.get('host', '0.0.0.0'),
                port=deploy_data.get('port', 8000),
                no_auto_start=deploy_data.get('no_auto_start', False)
            )
        
        return config
    
    def get_execution_plan(self) -> List[str]:
        """Get the execution plan based on configuration"""
        plan = []
        
        if self.train:
            plan.append('train')
        
        if self.merge:
            if self.merge.after_train and self.train:
                plan.append('merge')
            elif not self.merge.after_train:
                plan.append('merge')
        
        if self.upload:
            if self.upload.after_merge and 'merge' in plan:
                plan.append('upload')
            elif not self.upload.after_merge:
                plan.append('upload')
        
        if self.evaluate:
            plan.append('evaluate')
        
        if self.deploy:
            plan.append('deploy')
        
        return plan


class PipelineExecutor:
    """Execute pipeline based on configuration"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}
    
    def execute(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute the pipeline"""
        plan = self.config.get_execution_plan()
        
        if dry_run:
            print("\n=== Execution Plan (Dry Run) ===")
            for i, step in enumerate(plan, 1):
                print(f"{i}. {step}")
            print("\nNo actual execution in dry-run mode.\n")
            return {"status": "dry_run", "plan": plan}
        
        print(f"\n=== Starting Pipeline Execution ===")
        print(f"Steps to execute: {', '.join(plan)}\n")
        
        try:
            for i, step in enumerate(plan, 1):
                print(f"\n[{i}/{len(plan)}] Executing: {step}")
                print("-" * 50)
                
                if step == 'train':
                    self._execute_train()
                elif step == 'merge':
                    self._execute_merge()
                elif step == 'upload':
                    self._execute_upload()
                elif step == 'evaluate':
                    self._execute_evaluate()
                elif step == 'deploy':
                    self._execute_deploy()
                
                self.results[step] = "success"
                print(f"✓ {step} completed successfully")
            
            print("\n=== Pipeline Execution Completed ===")
            return {"status": "success", "results": self.results}
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.results[step] = f"failed: {str(e)}"
            return {"status": "failed", "error": str(e), "results": self.results}
    
    def _execute_train(self):
        """Execute training step"""
        if not self.config.train:
            raise ValueError("Train configuration not found")
        
        from hos_optimizer.train import TrainingConfig, train
        
        cfg = self.config.train
        train_config = TrainingConfig(
            model_name_or_path=cfg.model,
            dataset_path=cfg.dataset,
            dataset_format=cfg.format,
            max_seq_length=cfg.max_seq_length,
            finetuning_type=cfg.method,
            use_4bit=(cfg.method == "qlora"),
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            output_dir=cfg.output,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.lr,
            use_unsloth=not cfg.no_unsloth
        )
        
        train(train_config)
    
    def _execute_merge(self):
        """Execute merge step"""
        if not self.config.merge:
            raise ValueError("Merge configuration not found")
        
        from hos_optimizer.train import merge_model
        
        cfg = self.config.merge
        
        # If after_train is True and train was executed, use train output as adapter
        if cfg.after_train and self.config.train and 'train' in self.results:
            adapter_path = self.config.train.output
            logger.info(f"Using train output as adapter: {adapter_path}")
        else:
            adapter_path = cfg.adapter
        
        merge_model(
            base_model_path=cfg.base_model,
            adapter_path=adapter_path,
            output_path=cfg.output
        )
    
    def _execute_upload(self):
        """Execute upload step"""
        if not self.config.upload:
            raise ValueError("Upload configuration not found")
        
        from hos_optimizer.upload import upload_to_huggingface
        
        cfg = self.config.upload
        
        # If after_merge is True and merge was executed, use merge output as model
        if cfg.after_merge and self.config.merge and 'merge' in self.results:
            model_path = self.config.merge.output
            logger.info(f"Using merge output as model: {model_path}")
        else:
            model_path = cfg.model
        
        upload_to_huggingface(
            model_path=model_path,
            repo_id=cfg.repo_id,
            private=cfg.private
        )
    
    def _execute_evaluate(self):
        """Execute evaluation step"""
        if not self.config.evaluate:
            raise ValueError("Evaluate configuration not found")
        
        from hos_optimizer.evaluate import evaluate_model, EvaluationConfig
        
        cfg = self.config.evaluate
        eval_config = EvaluationConfig(
            model_path=cfg.model,
            dataset_path=cfg.dataset,
            dataset_format=cfg.format,
            max_samples=cfg.max_samples,
            max_seq_length=cfg.max_seq_length,
            metrics=cfg.metrics,
            task_type=cfg.task,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.batch_size,
            output_format="json",
            output_path=cfg.output,
            load_in_4bit=cfg.load_in_4bit
        )
        
        evaluate_model(eval_config)
    
    def _execute_deploy(self):
        """Execute deployment step"""
        if not self.config.deploy:
            raise ValueError("Deploy configuration not found")
        
        from hos_optimizer.deploy import HardwareDetector, ConfigSelector, ServiceLauncher
        
        cfg = self.config.deploy
        
        # Detect hardware
        hw_info = HardwareDetector.get_hardware_info()
        logger.info(f"Detected hardware: {hw_info.gpu_name}, {hw_info.gpu_memory_gb:.1f}GB VRAM")
        
        # Select configuration
        config = ConfigSelector.select_config(hw_info, cfg.model_size, cfg.use_case)
        logger.info(f"Selected config: {config.backend.value}, {config.quantization.value}")
        
        # Start service
        launcher = ServiceLauncher(config, cfg.model, cfg.host, cfg.port)
        if not cfg.no_auto_start:
            if launcher.start():
                logger.info(f"Service started at {cfg.host}:{cfg.port}")
            else:
                raise RuntimeError("Failed to start service")
        else:
            logger.info("Service not auto-started (no_auto_start=True)")

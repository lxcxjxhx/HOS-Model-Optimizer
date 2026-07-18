"""
训练模块单元测试

测试训练模块的所有功能，包括：
- QLoRA 配置
- LoRA 配置
- 数据集加载和处理
- 模型合并
- 训练流程

使用 mock 避免实际训练和模型加载。
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hos_optimizer.train import (
    TrainingConfig,
    DatasetProcessor,
    load_and_process_dataset,
    get_quantization_config,
    load_model_and_tokenizer,
    get_lora_config,
    prepare_model_for_training,
    get_training_arguments,
    VRAMCallback,
    train,
    merge_model,
)


class TestTrainingConfig:
    """训练配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        assert config.model_name_or_path == "Qwen/Qwen2.5-0.5B"
        assert config.finetuning_type == "qlora"
        assert config.use_4bit is True
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-4

    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            model_name_or_path="custom/model",
            finetuning_type="lora",
            use_4bit=False,
            lora_rank=32,
            lora_alpha=64,
            num_train_epochs=5,
            learning_rate=1e-4
        )
        
        assert config.model_name_or_path == "custom/model"
        assert config.finetuning_type == "lora"
        assert config.use_4bit is False
        assert config.lora_rank == 32
        assert config.lora_alpha == 64
        assert config.num_train_epochs == 5
        assert config.learning_rate == 1e-4


class TestDatasetProcessor:
    """数据集处理器测试"""

    def test_format_alpaca_with_input(self):
        """测试 Alpaca 格式带输入的处理"""
        mock_tokenizer = MagicMock()
        processor = DatasetProcessor(mock_tokenizer, max_seq_length=512)
        
        example = {
            "instruction": "什么是网络安全？",
            "input": "请举例说明",
            "output": "网络安全是指保护计算机网络免受未经授权的访问。"
        }
        
        result = processor.format_alpaca(example)
        
        assert "### 指令:" in result["text"]
        assert "什么是网络安全？" in result["text"]
        assert "### 输入:" in result["text"]
        assert "请举例说明" in result["text"]
        assert "### 回答:" in result["text"]
        assert "保护计算机网络" in result["text"]

    def test_format_alpaca_without_input(self):
        """测试 Alpaca 格式不带输入的处理"""
        mock_tokenizer = MagicMock()
        processor = DatasetProcessor(mock_tokenizer, max_seq_length=512)
        
        example = {
            "instruction": "什么是网络安全？",
            "input": "",
            "output": "网络安全是指保护计算机网络。"
        }
        
        result = processor.format_alpaca(example)
        
        assert "### 指令:" in result["text"]
        assert "什么是网络安全？" in result["text"]
        assert "### 输入:" not in result["text"]
        assert "### 回答:" in result["text"]

    def test_format_sharegpt(self):
        """测试 ShareGPT 格式的处理"""
        mock_tokenizer = MagicMock()
        processor = DatasetProcessor(mock_tokenizer, max_seq_length=512)
        
        example = {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"},
            ]
        }
        
        result = processor.format_sharegpt(example)
        
        assert "### 用户:" in result["text"]
        assert "你好" in result["text"]
        assert "### 助手:" in result["text"]
        assert "帮助你的吗" in result["text"]

    def test_tokenize_function(self):
        """测试分词函数"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
        
        processor = DatasetProcessor(mock_tokenizer, max_seq_length=512)
        
        example = {"text": "测试文本"}
        result = processor.tokenize_function(example)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["labels"] == result["input_ids"]

    def test_process_dataset_alpaca(self):
        """测试处理 Alpaca 格式数据集"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        mock_tokenizer.num_proc = 4
        
        processor = DatasetProcessor(mock_tokenizer, max_seq_length=512)
        
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.column_names = ["instruction", "input", "output"]
        
        # Mock map 方法
        def mock_map(func, **kwargs):
            return mock_dataset
        
        mock_dataset.map = mock_map
        
        result = processor.process_dataset(mock_dataset, dataset_format="alpaca")
        
        assert result is not None

    def test_process_dataset_invalid_format(self):
        """测试处理无效格式数据集"""
        mock_tokenizer = MagicMock()
        processor = DatasetProcessor(mock_tokenizer, max_seq_length=512)
        
        mock_dataset = MagicMock()
        
        with pytest.raises(ValueError) as exc_info:
            processor.process_dataset(mock_dataset, dataset_format="invalid_format")
        
        assert "不支持的数据格式" in str(exc_info.value)


class TestLoadAndProcessDataset:
    """数据集加载和处理测试"""

    def test_load_dataset_file_not_found(self):
        """测试加载不存在的数据集文件"""
        mock_tokenizer = MagicMock()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_and_process_dataset(
                dataset_path="/nonexistent/dataset.json",
                tokenizer=mock_tokenizer
            )
        
        assert "数据集文件不存在" in str(exc_info.value)

    def test_load_dataset_success(self, tmp_dir):
        """测试成功加载数据集"""
        # 创建测试数据集
        dataset_path = os.path.join(tmp_dir, "dataset.json")
        data = [
            {
                "instruction": "测试指令",
                "input": "",
                "output": "测试输出"
            }
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
        mock_tokenizer.num_proc = 4
        
        with patch("hos_optimizer.train.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 10
            mock_dataset.column_names = ["instruction", "input", "output"]
            mock_dataset.map.return_value = mock_dataset
            mock_dataset.train_test_split.return_value = {
                "train": MagicMock(__len__=MagicMock(return_value=9)),
                "test": MagicMock(__len__=MagicMock(return_value=1))
            }
            mock_load.return_value = mock_dataset
            
            result = load_and_process_dataset(
                dataset_path=dataset_path,
                tokenizer=mock_tokenizer,
                dataset_format="alpaca",
                max_seq_length=512,
                test_size=0.1
            )
            
            assert "train" in result
            assert "test" in result


class TestQuantizationConfig:
    """量化配置测试"""

    def test_get_quantization_config_disabled(self):
        """测试禁用量化配置"""
        config = TrainingConfig(use_4bit=False)
        
        result = get_quantization_config(config)
        
        assert result is None

    def test_get_quantization_config_enabled(self):
        """测试启用量化配置"""
        config = TrainingConfig(
            use_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16"
        )
        
        with patch("hos_optimizer.train.BitsAndBytesConfig") as mock_bnb:
            mock_config = MagicMock()
            mock_bnb.return_value = mock_config
            
            result = get_quantization_config(config)
            
            assert result is not None
            mock_bnb.assert_called_once()


class TestLoadModelAndTokenizer:
    """模型和分词器加载测试"""

    def test_load_model_standard(self):
        """测试标准模型加载"""
        config = TrainingConfig(
            model_name_or_path="test/model",
            use_unsloth=False,
            use_4bit=False
        )
        
        with patch("hos_optimizer.train.AutoTokenizer") as mock_tokenizer_cls, \
             patch("hos_optimizer.train.AutoModelForCausalLM") as mock_model_cls:
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            mock_model = MagicMock()
            mock_model.num_parameters.return_value = 1e9
            mock_model_cls.from_pretrained.return_value = mock_model
            
            model, tokenizer = load_model_and_tokenizer(config)
            
            assert model is not None
            assert tokenizer is not None
            assert tokenizer.pad_token == "<eos>"

    def test_load_model_with_unsloth(self):
        """测试使用 Unsloth 加载模型"""
        config = TrainingConfig(
            model_name_or_path="test/model",
            use_unsloth=True,
            finetuning_type="qlora"
        )
        
        with patch("hos_optimizer.train.UNSLOTH_AVAILABLE", True), \
             patch("hos_optimizer.train.FastLanguageModel") as mock_fast:
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_fast.from_pretrained.return_value = (mock_model, mock_tokenizer)
            
            model, tokenizer = load_model_and_tokenizer(config)
            
            assert model is not None
            assert tokenizer is not None
            mock_fast.from_pretrained.assert_called_once()

    def test_load_model_with_quantization(self):
        """测试带量化的模型加载"""
        config = TrainingConfig(
            model_name_or_path="test/model",
            use_unsloth=False,
            use_4bit=True
        )
        
        with patch("hos_optimizer.train.get_quantization_config") as mock_get_quant, \
             patch("hos_optimizer.train.AutoTokenizer") as mock_tokenizer_cls, \
             patch("hos_optimizer.train.AutoModelForCausalLM") as mock_model_cls:
            
            mock_quant_config = MagicMock()
            mock_get_quant.return_value = mock_quant_config
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            mock_model = MagicMock()
            mock_model.num_parameters.return_value = 1e9
            mock_model_cls.from_pretrained.return_value = mock_model
            
            model, tokenizer = load_model_and_tokenizer(config)
            
            assert model is not None
            # 验证量化配置被传递
            call_kwargs = mock_model_cls.from_pretrained.call_args[1]
            assert "quantization_config" in call_kwargs


class TestLoraConfig:
    """LoRA 配置测试"""

    def test_get_lora_config_all_modules(self):
        """测试所有模块的 LoRA 配置"""
        config = TrainingConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["all"]
        )
        
        with patch("hos_optimizer.train.LoraConfig") as mock_lora:
            mock_config = MagicMock()
            mock_lora.return_value = mock_config
            
            result = get_lora_config(config)
            
            assert result is not None
            mock_lora.assert_called_once()
            # 验证 target_modules 为 None（表示所有模块）
            call_kwargs = mock_lora.call_args[1]
            assert call_kwargs["target_modules"] is None

    def test_get_lora_config_specific_modules(self):
        """测试特定模块的 LoRA 配置"""
        config = TrainingConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "v_proj"]
        )
        
        with patch("hos_optimizer.train.LoraConfig") as mock_lora:
            mock_config = MagicMock()
            mock_lora.return_value = mock_config
            
            result = get_lora_config(config)
            
            assert result is not None
            call_kwargs = mock_lora.call_args[1]
            assert call_kwargs["target_modules"] == ["q_proj", "v_proj"]


class TestPrepareModelForTraining:
    """模型训练准备测试"""

    def test_prepare_model_qlora(self):
        """测试 QLoRA 模式的模型准备"""
        config = TrainingConfig(
            use_4bit=True,
            finetuning_type="qlora"
        )
        
        mock_model = MagicMock()
        
        with patch("hos_optimizer.train.prepare_model_for_kbit_training") as mock_prepare, \
             patch("hos_optimizer.train.get_lora_config") as mock_get_lora, \
             patch("hos_optimizer.train.get_peft_model") as mock_get_peft:
            
            mock_prepare.return_value = mock_model
            mock_lora_config = MagicMock()
            mock_get_lora.return_value = mock_lora_config
            mock_get_peft.return_value = mock_model
            
            result = prepare_model_for_training(mock_model, config)
            
            assert result is not None
            mock_prepare.assert_called_once()
            mock_get_peft.assert_called_once()

    def test_prepare_model_lora(self):
        """测试 LoRA 模式的模型准备"""
        config = TrainingConfig(
            use_4bit=False,
            finetuning_type="lora"
        )
        
        mock_model = MagicMock()
        
        with patch("hos_optimizer.train.get_lora_config") as mock_get_lora, \
             patch("hos_optimizer.train.get_peft_model") as mock_get_peft:
            
            mock_lora_config = MagicMock()
            mock_get_lora.return_value = mock_lora_config
            mock_get_peft.return_value = mock_model
            
            result = prepare_model_for_training(mock_model, config)
            
            assert result is not None
            mock_get_peft.assert_called_once()


class TestTrainingArguments:
    """训练参数测试"""

    def test_get_training_arguments(self):
        """测试获取训练参数"""
        config = TrainingConfig(
            output_dir="./test_output",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            learning_rate=1e-4
        )
        
        with patch("hos_optimizer.train.TrainingArguments") as mock_args:
            mock_training_args = MagicMock()
            mock_args.return_value = mock_training_args
            
            result = get_training_arguments(config)
            
            assert result is not None
            mock_args.assert_called_once()


class TestVRAMCallback:
    """VRAM 回调测试"""

    def test_vram_callback_on_log(self):
        """测试 VRAM 回调的日志记录"""
        callback = VRAMCallback()
        
        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.global_step = 100
        mock_state.is_world_process_zero = True
        mock_control = MagicMock()
        logs = {}
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.max_memory_allocated", return_value=4 * 1024 ** 3), \
             patch("torch.cuda.memory_reserved", return_value=6 * 1024 ** 3):
            
            callback.on_log(mock_args, mock_state, mock_control, logs=logs)
            
            assert "gpu_memory_gb" in logs
            assert "gpu_memory_reserved_gb" in logs


class TestTrain:
    """训练流程测试"""

    def test_train_success(self, tmp_dir):
        """测试成功训练流程"""
        # 创建测试数据集
        dataset_path = os.path.join(tmp_dir, "dataset.json")
        data = [{"instruction": "测试", "input": "", "output": "输出"}]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        
        config = TrainingConfig(
            model_name_or_path="test/model",
            dataset_path=dataset_path,
            output_dir=os.path.join(tmp_dir, "output")
        )
        
        with patch("hos_optimizer.train.load_model_and_tokenizer") as mock_load, \
             patch("hos_optimizer.train.load_and_process_dataset") as mock_load_dataset, \
             patch("hos_optimizer.train.prepare_model_for_training") as mock_prepare, \
             patch("hos_optimizer.train.get_training_arguments") as mock_get_args, \
             patch("hos_optimizer.train.Trainer") as mock_trainer_cls, \
             patch("hos_optimizer.train.DataCollatorForSeq2Seq"):
            
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            mock_dataset_dict = {
                "train": MagicMock(),
                "test": MagicMock()
            }
            mock_load_dataset.return_value = mock_dataset_dict
            
            mock_prepare.return_value = mock_model
            
            mock_training_args = MagicMock()
            mock_get_args.return_value = mock_training_args
            
            mock_trainer = MagicMock()
            mock_train_result = MagicMock()
            mock_train_result.training_loss = 0.5
            mock_train_result.metrics = {"train_runtime": 100.0}
            mock_trainer.train.return_value = mock_train_result
            mock_trainer_cls.return_value = mock_trainer
            
            train(config)
            
            mock_trainer.train.assert_called_once()
            mock_trainer.save_model.assert_called_once()


class TestMergeModel:
    """模型合并测试"""

    def test_merge_model_success(self, tmp_dir):
        """测试成功合并模型"""
        base_model_path = "/path/to/base"
        adapter_path = "/path/to/adapter"
        output_path = os.path.join(tmp_dir, "merged")
        
        with patch("hos_optimizer.train.AutoModelForCausalLM") as mock_model_cls, \
             patch("hos_optimizer.train.PeftModel") as mock_peft, \
             patch("hos_optimizer.train.AutoTokenizer") as mock_tokenizer_cls, \
             patch("os.makedirs"):
            
            mock_base_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_base_model
            
            mock_peft_model = MagicMock()
            mock_merged_model = MagicMock()
            mock_peft.from_pretrained.return_value = mock_peft_model
            mock_peft_model.merge_and_unload.return_value = mock_merged_model
            
            mock_tokenizer = MagicMock()
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            merge_model(
                base_model_path=base_model_path,
                adapter_path=adapter_path,
                output_path=output_path
            )
            
            mock_peft_model.merge_and_unload.assert_called_once()
            mock_merged_model.save_pretrained.assert_called_once_with(output_path)
            mock_tokenizer.save_pretrained.assert_called_once_with(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

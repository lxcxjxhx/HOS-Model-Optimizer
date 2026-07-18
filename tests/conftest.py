"""
pytest 全局配置和共享 fixtures

提供所有测试模块共用的 fixture，包括临时目录、mock 对象等。
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# 确保 hos_optimizer 包可被导入
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_dir():
    """创建临时目录，测试结束后自动清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_yaml_config(tmp_dir):
    """创建一个示例 YAML 配置文件"""
    config_path = os.path.join(tmp_dir, "test_config.yaml")
    content = (
        "backend: vllm\n"
        "model:\n"
        "  path: /tmp/test_model\n"
        "  format: awq\n"
        "  dtype: float16\n"
        "inference:\n"
        "  max_model_len: 512\n"
        "  gpu_memory_utilization: 0.9\n"
        "sampling:\n"
        "  temperature: 0.7\n"
        "  top_p: 0.9\n"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return config_path


@pytest.fixture
def sample_alpaca_dataset(tmp_dir):
    """创建一个示例 Alpaca 格式数据集文件"""
    dataset_path = os.path.join(tmp_dir, "dataset.json")
    data = [
        {
            "instruction": "什么是网络安全？",
            "input": "",
            "output": "网络安全是指保护计算机网络免受未经授权的访问。"
        },
        {
            "instruction": "解释SQL注入",
            "input": "请举例说明",
            "output": "SQL注入是通过在输入中插入恶意SQL代码来攻击数据库。"
        },
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return dataset_path


@pytest.fixture
def sample_sharegpt_dataset(tmp_dir):
    """创建一个示例 ShareGPT 格式数据集文件"""
    dataset_path = os.path.join(tmp_dir, "sharegpt_dataset.json")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "你好"},
                {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"},
            ]
        },
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return dataset_path


@pytest.fixture
def mock_torch():
    """Mock torch 模块，避免实际加载模型"""
    with patch.dict("sys.modules", {
        "torch": MagicMock(),
        "torch.cuda": MagicMock(),
    }):
        yield sys.modules["torch"]


@pytest.fixture
def mock_transformers():
    """Mock transformers 库"""
    with patch.dict("sys.modules", {
        "transformers": MagicMock(),
        "transformers.AutoModelForCausalLM": MagicMock(),
        "transformers.AutoTokenizer": MagicMock(),
        "transformers.BitsAndBytesConfig": MagicMock(),
        "transformers.TrainingArguments": MagicMock(),
        "transformers.Trainer": MagicMock(),
        "transformers.DataCollatorForSeq2Seq": MagicMock(),
    }):
        yield sys.modules["transformers"]

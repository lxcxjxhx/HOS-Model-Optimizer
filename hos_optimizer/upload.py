"""上传模型到 HuggingFace Hub 的工具模块"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

logger = logging.getLogger(__name__)


def upload_to_huggingface(
    model_path: str,
    repo_id: str,
    private: bool = False,
) -> None:
    """上传模型目录到 HuggingFace Hub。

    Args:
        model_path: 模型目录路径。
        repo_id: HuggingFace 仓库 ID（如 "user/repo"）。
        private: 是否创建私有仓库，默认 False。

    Raises:
        RuntimeError: 未登录 HuggingFace 时抛出。
        FileNotFoundError: model_path 不存在时抛出。
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    api = HfApi()

    # 认证检查
    try:
        whoami = api.whoami()
        logger.info("已登录 HuggingFace: %s", whoami.get("name", "unknown"))
    except Exception as exc:
        raise RuntimeError(
            "未登录 HuggingFace，请先执行 huggingface-cli login"
        ) from exc

    # 创建仓库
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        logger.info("仓库已就绪: https://huggingface.co/%s", repo_id)
    except HfHubHTTPError as exc:
        logger.error("创建仓库失败: %s", exc)
        raise

    # 收集所有文件（排除缓存和临时文件）
    files = sorted(
        f for f in model_path.rglob("*")
        if f.is_file() and "__pycache__" not in str(f) and not f.name.startswith(".")
    )
    total = len(files)
    if total == 0:
        logger.warning("模型目录为空，无文件可上传: %s", model_path)
        return

    logger.info("共发现 %d 个文件待上传", total)
    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"\n开始上传 {total} 个文件 ({total_size_mb:.1f} MB) 到 {repo_id}...\n", flush=True)

    # 逐个文件上传并显示进度
    for idx, file_path in enumerate(tqdm(files, desc="上传进度", unit="file"), 1):
        rel_path = file_path.relative_to(model_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # 显示当前文件信息
        print(f"[{idx}/{total}] 上传: {rel_path} ({file_size_mb:.2f} MB)...", flush=True)
        
        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(rel_path),
                repo_id=repo_id,
                repo_type="model",
            )
            print("  ✓ 完成", flush=True)
        except Exception as exc:
            logger.error("上传文件失败 %s: %s", rel_path, exc)
            raise

    print(f"\n✓ 上传完成: https://huggingface.co/{repo_id}", flush=True)

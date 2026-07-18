#!/usr/bin/env python3
"""Upload HOS-Model-Optimizer to HuggingFace"""
import os
import sys
from pathlib import Path

# 使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 使用虚拟环境的包
sys.path.insert(0, '/home/s/MODEL/ACT-DASESS/train_env/lib/python3.12/site-packages')

from huggingface_hub import HfApi

def main():
    project_path = Path("/home/s/MODEL/ACT-DASESS/HOS-Model-Optimizer")
    repo_id = "lxcxjxhx/HOS-Model-Optimizer"

    print(f"项目路径: {project_path}")
    print(f"仓库 ID: {repo_id}")
    print(f"镜像站: {os.environ['HF_ENDPOINT']}")

    api = HfApi(endpoint="https://hf-mirror.com")

    # 检查认证
    whoami = api.whoami()
    print(f"已登录: {whoami['name']}")

    # 创建仓库（如果不存在）
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"仓库已就绪: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"创建仓库失败: {e}")
        raise

    # 要上传的文件和目录
    files_to_upload = [
        "README.md",
        "LICENSE",
        "INSTALL.md",
        "EXAMPLES.md",
        "API.md",
        "SKILL.md",
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
    ]

    # 上传根目录文件
    for filename in files_to_upload:
        filepath = project_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"上传 {filename} ({size_mb:.2f} MB)...", flush=True)
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )

    # 上传目录
    dirs_to_upload = [
        "hos_optimizer",
        "tests",
        "docs",
        "examples",
        ".github",
    ]

    for dirname in dirs_to_upload:
        dirpath = project_path / dirname
        if dirpath.exists() and dirpath.is_dir():
            print(f"\n上传目录 {dirname}/...")
            files = [f for f in dirpath.rglob("*") if f.is_file() and "__pycache__" not in str(f)]
            for i, fp in enumerate(files, 1):
                rel = fp.relative_to(project_path)
                size_mb = fp.stat().st_size / (1024 * 1024)
                print(f"  [{i}/{len(files)}] {rel} ({size_mb:.2f} MB)...", flush=True)
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=str(rel),
                    repo_id=repo_id,
                    repo_type="model",
                )

    print(f"\n✓ 上传成功! https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()

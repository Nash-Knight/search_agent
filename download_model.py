"""
快速下载脚本（跨平台）
在已激活的 Python 环境中运行：
> python download_model.py

会把模型下载到 ./Qwen3-8B（默认），并跳过 *.msgpack 文件。
"""
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import fnmatch
import sys
import os # 导入 os 模块

REPO_ID = "Qwen/Qwen3-8B"
LOCAL_DIR = Path("./Qwen3-8B")
EXCLUDE = "*.msgpack"

# ========== 最小浮动修改：设置环境变量 ==========
# 在 Python 脚本中设置环境变量，等同于在命令行中执行 export HF_ENDPOINT=...
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("已设置 HF_ENDPOINT 为 https://hf-mirror.com")
# =================================================

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
api = HfApi()

print(f"开始下载：{REPO_ID} -> {LOCAL_DIR}")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

for f in files:
    if fnmatch.fnmatch(f, EXCLUDE):
        print(f"跳过：{f}")
        continue
    try:
        # 使用 hf_hub_download 保证兼容性（在较新/常见的 huggingface_hub 版本中可用）
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="model")
        p = Path(downloaded)
        target = LOCAL_DIR / f
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.resolve() != target.resolve():
            import shutil
            shutil.move(str(p), str(target))
        print(f"下载完成: {f}")
    except Exception as e:
        print(f"下载失败 {f}: {e}", file=sys.stderr)

print("完成")
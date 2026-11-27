#!/bin/bash
# Run on the cloud machine (Ubuntu + NVIDIA driver, e.g. the AWS "Deep Learning OSS Nvidia Driver AMI GPU PyTorch").
set -e
cd "$(dirname "$0")/../.."
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/.venv-recon}  # venv on local disk, not the network filesystem
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source ~/.local/bin/env; }
uv sync
uv run python - <<'PY'
import os, torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
print("cpus:", os.cpu_count())
PY
uv run pytest -q tests | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "setup ok. next: bash scripts/cloud/speedtest.sh"

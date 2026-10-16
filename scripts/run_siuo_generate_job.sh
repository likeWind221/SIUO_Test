#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_BASE:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
conda activate siuo-test
cd "${PACKAGE_ROOT}"

echo "host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch

available = torch.cuda.is_available()
print("PyTorch:", torch.__version__, "CUDA available:", available)
if not available:
    raise SystemExit("CUDA is unavailable on this node; check the GPU allocation and PyTorch installation")
PY

PYTHON_BIN="$(command -v python)" bash scripts/run_siuo_generate.sh

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NEW4DGS_PYTHON="${NEW4DGS_PYTHON:-/venv/main/bin/python}"
if ! "$NEW4DGS_PYTHON" -c 'import torch; assert torch.__version__.startswith("2.7.1")' 2>/dev/null; then
  uv pip install --python "$NEW4DGS_PYTHON" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install --python "$NEW4DGS_PYTHON" -r requirements-core-lock.txt
if [[ ! -d ../4DGaussians ]]; then
  git clone https://github.com/hustvl/4DGaussians.git ../4DGaussians
  git -C ../4DGaussians checkout 843d5ac636c37e4b611242287754f3d4ed150144
  git -C ../4DGaussians submodule update --init --recursive
fi
# Preserve any pre-existing upstream checkout; require manual path choice on mismatch.
[[ "$(git -C ../4DGaussians rev-parse HEAD)" == 843d5ac636c37e4b611242287754f3d4ed150144 ]] || { echo 'Existing 4DGaussians has another commit; preserve it and use a separate pinned checkout.' >&2; exit 1; }
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}" NVCC_PREPEND_FLAGS='-include cfloat -include cstdint' 
export MAX_JOBS TORCH_CUDA_ARCH_LIST NVCC_PREPEND_FLAGS
uv pip install --python "$NEW4DGS_PYTHON" --no-build-isolation ../4DGaussians/submodules/depth-diff-gaussian-rasterization ../4DGaussians/submodules/simple-knn
"$NEW4DGS_PYTHON" -m pytest -q
"$NEW4DGS_PYTHON" scripts/check_cuda.py
"$NEW4DGS_PYTHON" scripts/check_baseline.py

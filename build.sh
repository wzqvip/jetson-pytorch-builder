#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
SRC_DIR="$ROOT_DIR/src"
LOG_DIR="$ROOT_DIR/logs"
WHEEL_DIR="$ROOT_DIR/wheels"

mkdir -p "$SRC_DIR" "$LOG_DIR" "$WHEEL_DIR"

detect_cuda_arch_list() {
  local cap=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    if [[ "$cap" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      echo "$cap"
      return
    fi
  fi

  if [[ -r /proc/device-tree/model ]]; then
    local model
    model="$(tr -d '\0' < /proc/device-tree/model)"
    if grep -qi "thor" <<<"$model"; then
      echo "11.0"
      return
    fi
    if grep -qi "orin" <<<"$model"; then
      echo "8.7"
      return
    fi
  fi

  echo "8.7"
}

write_host_summary() {
  {
    echo "===== Host summary ($(date -Iseconds)) ====="
    uname -a || true
    if command -v lsb_release >/dev/null 2>&1; then
      lsb_release -a || true
    fi
    if [[ -r /etc/nv_tegra_release ]]; then
      cat /etc/nv_tegra_release
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv || true
    else
      echo "nvidia-smi: not available (likely stock Jetson user-space)"
    fi
    echo "CUDA_HOME=$CUDA_HOME"
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    echo "MAX_JOBS=$MAX_JOBS"
    echo "==========================================="
  } | tee "$LOG_FILE"
}

PY_VERSION="${1:-3.10}"
PYTORCH_REPO="${PYTORCH_REPO:-https://github.com/pytorch/pytorch.git}"
PYTORCH_BRANCH="${PYTORCH_BRANCH:-v2.4.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
DETECTED_CUDA_ARCH="$(detect_cuda_arch_list)"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$DETECTED_CUDA_ARCH}"
USE_NCCL="${USE_NCCL:-0}"
USE_DISTRIBUTED="${USE_DISTRIBUTED:-0}"
USE_MKLDNN="${USE_MKLDNN:-0}"
USE_NNPACK="${USE_NNPACK:-0}"
USE_QNNPACK="${USE_QNNPACK:-0}"
USE_NVTX="${USE_NVTX:-0}"
USE_PRIORITIZED_TEXT_FOR_LD="${USE_PRIORITIZED_TEXT_FOR_LD:-0}"
EXTRA_CUDA_INCLUDE_PATHS=""
CUDA_INCLUDE_CANDIDATES=(
  "$CUDA_HOME/include"
  "$CUDA_HOME/targets/sbsa-linux/include"
  "$CUDA_HOME/targets/aarch64-linux/include"
  "$CUDA_HOME/targets/x86_64-linux/include"
)
for candidate in "${CUDA_INCLUDE_CANDIDATES[@]}"; do
  if [[ -d "$candidate" ]]; then
    if [[ -z "$EXTRA_CUDA_INCLUDE_PATHS" ]]; then
      EXTRA_CUDA_INCLUDE_PATHS="$candidate"
    elif [[ ":$EXTRA_CUDA_INCLUDE_PATHS:" != *":$candidate:"* ]]; then
      EXTRA_CUDA_INCLUDE_PATHS="$EXTRA_CUDA_INCLUDE_PATHS:$candidate"
    fi
  fi
done
export EXTRA_CUDA_INCLUDE_PATHS

ENV_NAME="torch-py${PY_VERSION//./}"
REPO_DIR="$SRC_DIR/pytorch"
LOG_FILE="$LOG_DIR/pytorch-py${PY_VERSION}-$(date +%Y%m%d-%H%M%S).log"
DEST_DIR="$WHEEL_DIR/py${PY_VERSION//./}"

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return
  fi

  local conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
  if [[ -f "$conda_sh" ]]; then
    # shellcheck source=/dev/null
    source "$conda_sh"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda or adjust the script." >&2
    exit 1
  fi
}

ensure_conda

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk 'NF>0 && $1 !~ /^#/' | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env $ENV_NAME (Python $PY_VERSION)..."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Cloning PyTorch ($PYTORCH_BRANCH) into $REPO_DIR..."
  git clone --recursive --branch "$PYTORCH_BRANCH" "$PYTORCH_REPO" "$REPO_DIR"
else
  echo "Updating existing PyTorch checkout..."
  git -C "$REPO_DIR" reset --hard >/dev/null
  git -C "$REPO_DIR" fetch --prune "$PYTORCH_REPO"
  git -C "$REPO_DIR" checkout "$PYTORCH_BRANCH"
  git -C "$REPO_DIR" pull --rebase "$PYTORCH_REPO" "$PYTORCH_BRANCH"
fi

git -C "$REPO_DIR" submodule sync --recursive
git -C "$REPO_DIR" submodule update --init --recursive --force

echo "Syncing Python dependencies for $ENV_NAME..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
conda run -n "$ENV_NAME" python -m pip install -r "$REPO_DIR/requirements.txt"
conda run -n "$ENV_NAME" python -m pip install ninja cmake typing_extensions sympy pytest requests

echo "Cleaning previous build artefacts..."
git -C "$REPO_DIR" clean -fdx

# Apply any local patches (e.g., Jetson/Thor specific fixes)
PATCH_DIR="$ROOT_DIR/patches"
if [[ -d "$PATCH_DIR" ]]; then
  shopt -s nullglob
  for patch_file in "$PATCH_DIR"/*.patch; do
    echo "Applying $(basename "$patch_file")..."
    git -C "$REPO_DIR" apply "$patch_file"
  done
  shopt -u nullglob
fi

mkdir -p "$DEST_DIR"

write_host_summary

if [[ "${TORCH_CUDA_ARCH_LIST}" != "${DETECTED_CUDA_ARCH}" ]]; then
  echo "Using user-provided TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (auto-detected $DETECTED_CUDA_ARCH)" | tee -a "$LOG_FILE"
else
  echo "Auto-detected TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" | tee -a "$LOG_FILE"
fi

echo "Starting PyTorch build for Python $PY_VERSION (log: $LOG_FILE)..." | tee -a "$LOG_FILE"
if ! conda run -n "$ENV_NAME" bash -c "
  set -euo pipefail
  cd \"$REPO_DIR\"
  export CUDA_HOME=\"$CUDA_HOME\"
  if [[ -n \"${EXTRA_CUDA_INCLUDE_PATHS:-}\" ]]; then
    : \"\${CPATH:=}\"
    : \"\${CPLUS_INCLUDE_PATH:=}\"
    export CPATH=\"${EXTRA_CUDA_INCLUDE_PATHS}\${CPATH:+:\$CPATH}\"
    export CPLUS_INCLUDE_PATH=\"${EXTRA_CUDA_INCLUDE_PATHS}\${CPLUS_INCLUDE_PATH:+:\$CPLUS_INCLUDE_PATH}\"
  fi
  export USE_CUDA=1
  export USE_CUDNN=1
  export USE_NCCL=\"$USE_NCCL\"
  export USE_DISTRIBUTED=\"$USE_DISTRIBUTED\"
  export USE_MKLDNN=\"$USE_MKLDNN\"
  export USE_NNPACK=\"$USE_NNPACK\"
  export USE_QNNPACK=\"$USE_QNNPACK\"
  export USE_NVTX=\"$USE_NVTX\"
  export USE_PRIORITIZED_TEXT_FOR_LD=\"$USE_PRIORITIZED_TEXT_FOR_LD\"
  export BUILD_TEST=0
  export MAX_JOBS=\"$MAX_JOBS\"
  export TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\"
  export CMAKE_ARGS=\"${CMAKE_ARGS:-} -DCMAKE_POLICY_VERSION=3.25 -DCMAKE_POLICY_VERSION_MINIMUM=3.5\"
  python setup.py bdist_wheel
" |& tee -a "$LOG_FILE"; then
  echo "Build failed. See $LOG_FILE for details." >&2
  exit 1
fi

LATEST_WHEEL="$(ls -t "$REPO_DIR"/dist/torch-*.whl 2>/dev/null | head -n 1 || true)"
if [[ -z "$LATEST_WHEEL" ]]; then
  echo "Build finished but no torch-*.whl found in dist/. Check the log." >&2
  exit 1
fi

cp "$LATEST_WHEEL" "$DEST_DIR/"
echo "Copied $(basename "$LATEST_WHEEL") to $DEST_DIR"

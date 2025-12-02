# PyTorch from Source on Jetson Orin & Thor

This workspace automates building stock [PyTorch](https://github.com/pytorch/pytorch#from-source) with CUDA enabled for Python 3.8–3.12 on both Jetson Orin (Ampere, JetPack 6.x) and Jetson AGX Thor (Blackwell, JetPack 7.x). It codifies the upstream PyTorch instructions and the NVIDIA Developer Forum guidance for [Orin builds](https://forums.developer.nvidia.com/t/native-build-of-pytorch-for-jetson/71842) and [Thor/JetPack 7 builds](https://forums.developer.nvidia.com/t/pytorch-2-4-build-jetson-orin/291219). NVIDIA 官方 pip 仓库只提供少量预编译版本，这里可以自编译带 CUDA 的版本以匹配需求。

## Prerequisites

- JetPack 6.x (Orin) or JetPack 7.x (Thor) with CUDA `/usr/local/cuda` and cuDNN already installed.
- At least 32 GB of free disk (more if you plan to keep all 3 wheels at once) and large swap (builds routinely spill >16 GB RAM).
- System packages:

  ```bash
  sudo apt update
  sudo apt install -y build-essential git cmake ninja-build \
      libopenblas-dev libopenmpi-dev openmpi-bin libatlas-base-dev libprotobuf-dev \
      protobuf-compiler libssl-dev zlib1g-dev libffi-dev
  ```
- `~/miniconda3` (already present on this machine) or any conda distribution. The scripts will create isolated envs per Python version.

Thor-specific sanity checks (taken from this devkit, JetPack 7.0 / Ubuntu 24.04):

```bash
uname -a
# Linux thor-taco 6.8.12-tegra ... aarch64 GNU/Linux
cat /etc/nv_tegra_release
# R38.2.2 ... BOARD: generic (AGX Thor)
nvidia-smi --query-gpu=name,compute_cap,driver_version,cuda_version --format=csv
# NVIDIA Thor, 11.0, 580.00, 13.0
```

The `build.sh` script auto-detects the compute capability (`TORCH_CUDA_ARCH_LIST`) via `nvidia-smi` when present, falling back to `/proc/device-tree/model`. On this Thor devkit it resolves to `11.0`; on Orin it defaults to `8.7`.

> ℹ️ Jetson builds cannot currently use NVIDIA's binary NCCL. Following the forum advice above, the scripts default to `USE_NCCL=0`, `USE_DISTRIBUTED=0`, `USE_MKLDNN=0`, and `USE_NNPACK=0`. Override them if you have working alternatives.

## Layout

- `build.sh` — clones PyTorch (once), prepares the requested Python env, and runs `python setup.py bdist_wheel` with Jetson-friendly defaults (auto-detected CUDA arch, NCCL disabled unless you opt in, etc.).
- `build-all.sh` — convenience wrapper that invokes `build.sh` for 3.10, 3.11, and 3.12 (or any list of versions you pass).
- `src/` — source tree managed by the scripts (`src/pytorch` is the git checkout).
- `logs/` — timestamped build logs per Python version.
- `wheels/` — collected `.whl` artefacts per Python version (`wheels/py310`, `wheels/py311`, ...).

## Quick start

```bash
cd ~/jetson-pytorch-builder
chmod +x build.sh build-all.sh
# Build all supported versions (3.8–3.12)
./build-all.sh
# OR build one at a time
#    ^ Python version  ^ optional PyTorch git ref/tag
./build.sh 3.11 v2.4.1
```

Each run:

1. Creates/updates `src/pytorch` (defaults to upstream tag `v2.4.0`, override with `PYTORCH_BRANCH=<tag>` or pass a second argument such as `./build.sh 3.12 main`).
2. Creates a matching conda env (`torch-py310`, `torch-py311`, `torch-py312`) if it does not exist yet.
3. Installs PyTorch's Python build requirements into the env.
4. Cleans the repo tree (`git clean -fdx`) to avoid cross-version contamination.
5. Compiles PyTorch with CUDA enabled, targeting the detected GPU (`TORCH_CUDA_ARCH_LIST` auto-detects to 8.7 for Orin, 11.0 for Thor; override via env var to cross-compile).
6. Copies the newest `torch-*.whl` into `wheels/pyNNN`.

Successful builds print the wheel path at the end and log everything to `logs/pytorch-py<version>-<timestamp>.log`.

## Current Support

* [X] Pytorch 2.4.0
* [ ] Pytorch 2.9.1

* ### Jetson Orin

  * [ ] Python 3.8
  * [ ] Python 3.9
  * [ ] Python 3.10
  * [ ] Python 3.11
  * [ ] Python 3.12
* ### Jetson Thor

  * [X] Python 3.8
  * [X] Python 3.9
  * [X] Python 3.10
  * [X] Python 3.11
  * [X] Python 3.12


## Customisation

All relevant knobs can be changed through environment variables:

| Variable                                                                           | Default                                      | Meaning                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PYTORCH_BRANCH`                                                                 | `v2.4.0`                                   | Upstream tag/branch to checkout (can also pass as the second argument to `build.sh`).                                                                                                     |
| `PYTORCH_REPO`                                                                   | `https://github.com/pytorch/pytorch.git`   | Clone source.                                                                                                                                                                               |
| `TORCH_CUDA_ARCH_LIST`                                                           | auto (`11.0` on Thor, `8.7` on Orin)     | Target GPU architectures. Override to cross-compile.                                                                                                                                        |
| `MAX_JOBS`                                                                       | `$(nproc)`                                 | Parallel compilation jobs. Tune to control RAM usage.                                                                                                                                       |
| `CUDA_HOME`                                                                      | `/usr/local/cuda`                          | CUDA root.                                                                                                                                                                                  |
| `USE_NCCL`, `USE_DISTRIBUTED`, `USE_MKLDNN`, `USE_NNPACK`, `USE_QNNPACK` | Jetson defaults set in `build.sh`.         |                                                                                                                                                                                             |
| `TORCH_VERSION_OVERRIDE`                                                         | auto from tag (e.g.,`v2.4.0` → `2.4.0`) | Forces `TORCH_BUILD_VERSION` so the wheel filename/metadata advertises your custom build. Set empty to keep upstream git-style versions or to supply your own (e.g., `2.4.0-jetson.1`). |
| `TORCH_BUILD_NUMBER_OVERRIDE`                                                    | `1`                                        | Optional build number passed along when `TORCH_VERSION_OVERRIDE` is set.                                                                                                                  |

Example:

```bash
TORCH_CUDA_ARCH_LIST="8.7;8.9" USE_NCCL=1 MAX_JOBS=8 ./build.sh 3.12
```

## Installing the wheels

Once a build finishes, install it inside any target environment (conda, system Python, etc.):

```bash
pip install ~/jetson-pytorch-builder/wheels/py312/torch-*.whl
```

Copy the wheel to other Jetson nodes as needed. Keep the logs handy for support/bug reports.

### Versioning and torchvision / torchaudio compatibility

By default PyTorch's build system emits versions like `2.4.0a0+git<sha>`. This repo now **auto-sets `TORCH_BUILD_VERSION` to the numeric part of your tag** (e.g., `v2.4.0` → `2.4.0`), so the wheel name/metadata matches what torchvision/torchaudio expect. For non-tag refs (e.g., `main`), no override is applied unless you set it explicitly.

Two ways to stay sane:

1. **Set an explicit version for your wheel.**

   ```bash
   TORCH_VERSION_OVERRIDE="2.4.0-jetson.1" ./build.sh 3.11 v2.4.0
   ```

   The resulting wheel becomes `torch-2.4.0-jetson.1-...whl`, making it easy to match dependencies.
2. **Install torchvision without re-resolving torch.**

   If you keep the default `2.4.0a0+git...` version, install the matching source release and skip dependency checks:

   ```bash
   pip install torchvision==0.19.0 --no-deps
   pip install torchaudio==2.4.0 --no-deps   # adjust to the PyTorch series you built
   ```

   This mirrors the PyTorch instructions for source builds where `torch` is already present.

## Thor (Blackwell / JetPack 7) notes

- JetPack 7 ships CUDA 13.0 and driver 580; make sure host packages and `CUDA_HOME` point to `/usr/local/cuda-13.0` (symlinked by default). The script logs the resolved path for traceability.
- Blackwell support in PyTorch is still evolving; stick to PyTorch v2.4+ (default `v2.4.0`) or nightly master for proper `sm_110` kernels. You can change `PYTORCH_BRANCH` to `main` when you need bleeding-edge fixes.
- NCCL is still unavailable on Jetson, so distributed training remains disabled.
- If you parallelize with `MAX_JOBS > 8`, ensure Thor's LPDDR memory controller has enough headroom or the build may thrash swap.

## Troubleshooting notes

- Add swap with `sudo fallocate -l 32G /swapfile && sudo mkswap /swapfile ...` if the compiler OOMs.
- Ensure `nvcc --version` matches your JetPack CUDA (`nvcc --version` should report 13.0 on Thor, 12.x on Orin). If not, export `CUDA_HOME` explicitly.
- `python setup.py clean` is implicitly handled by `git clean -fdx`; remove `build/` manually if you pause/resume by hand.
- Refer to the PyTorch source build doc and NVIDIA forum threads listed at the top for more edge-case fixes (e.g., building with TensorRT, CUTLASS tuning, FlashAttention patches, etc.).

Happy compiling!

Special notes: The complete repo is written by Codex GPT-5.1 medium. I do not garentee this will work on your machine. Merge requests welcomed.
Tested on:
Jetson Thor: Linux thor-taco 6.8.12-tegra #1 SMP PREEMPT Thu Sep 25 15:19:42 PDT 2025 aarch64 aarch64 aarch64 GNU/Linux
  Soc: tegra264
  CUDA Arch BIN: 13.0
  L4T: 38.2.2
  Jetpack: 7.0
  CUDA: 13.0.48
  cuDNN: 9.12.0
  TensorRT: 10.13.3.9

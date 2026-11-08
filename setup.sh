#!/usr/bin/env bash
# One-shot environment setup for LadderMan (Linux, NVIDIA GPU).
#
#   bash setup.sh
#
# What it does:
#   1. uses your conda (or installs Miniconda into ./thirdparty/miniconda3 if conda is not found)
#   2. creates the conda env "$ENV_NAME" (default: ladderman) with Python 3.10
#   3. installs Isaac Sim 4.5.0 (pip) and Isaac Lab v2.2.0 (cloned into ./thirdparty/IsaacLab)
#   4. installs this repository in editable mode
#
# Environment variables:
#   ENV_NAME       name of the conda env            (default: ladderman)
#   ISAACLAB_DIR   where to clone / find Isaac Lab   (default: ./thirdparty/IsaacLab)
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ENV_NAME="${ENV_NAME:-ladderman}"
THIRDPARTY_DIR="${REPO_DIR}/thirdparty"
ISAACLAB_DIR="${ISAACLAB_DIR:-${THIRDPARTY_DIR}/IsaacLab}"
ISAACSIM_VERSION="4.5.0"
ISAACLAB_VERSION="v2.2.0"

log() { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[setup]\033[0m $*"; }

for tool in git curl; do
    command -v "$tool" > /dev/null || { echo "'$tool' is required but not installed."; exit 1; }
done
for tool in cmake g++; do
    command -v "$tool" > /dev/null || warn "'$tool' not found. Isaac Lab may need it (Ubuntu: sudo apt install cmake build-essential)."
done

# ---------------------------------------------------------------------------
# 1. conda
# ---------------------------------------------------------------------------
if command -v conda > /dev/null; then
    CONDA_BASE="$(conda info --base)"
else
    CONDA_BASE="${THIRDPARTY_DIR}/miniconda3"
    if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
        log "conda not found, installing Miniconda into ${CONDA_BASE}"
        mkdir -p "${THIRDPARTY_DIR}"
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${THIRDPARTY_DIR}/miniconda.sh"
        bash "${THIRDPARTY_DIR}/miniconda.sh" -b -p "${CONDA_BASE}"
        rm "${THIRDPARTY_DIR}/miniconda.sh"
    fi
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. conda env
# ---------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "conda env '${ENV_NAME}' already exists, reusing it"
else
    log "creating conda env '${ENV_NAME}' (python 3.10)"
    conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
# flatdict (an Isaac Lab dependency) fails to build from source with recent setuptools
conda install -y -c conda-forge flatdict=4.0.1

# ---------------------------------------------------------------------------
# 3. Isaac Sim + Isaac Lab
# ---------------------------------------------------------------------------
log "installing Isaac Sim ${ISAACSIM_VERSION}"
pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com

if [[ ! -d "${ISAACLAB_DIR}" ]]; then
    log "cloning Isaac Lab ${ISAACLAB_VERSION} into ${ISAACLAB_DIR}"
    git clone --depth 1 --branch "${ISAACLAB_VERSION}" https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi
log "installing Isaac Lab"
(cd "${ISAACLAB_DIR}" && ./isaaclab.sh --install none)

# ---------------------------------------------------------------------------
# 4. LadderMan
# ---------------------------------------------------------------------------
log "installing LadderMan"
pip install -e "${REPO_DIR}"

python - <<'EOF'
import torch, torchrl, tensordict, warp
print(f"torch {torch.__version__} (cuda available: {torch.cuda.is_available()}), "
      f"torchrl {torchrl.__version__}, tensordict {tensordict.__version__}, warp {warp.config.version}")
EOF

log "done. Activate the environment with:"
echo "    source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"

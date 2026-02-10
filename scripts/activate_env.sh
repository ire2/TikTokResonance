#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/opt/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="resonance"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda not found at $CONDA_SH"
  echo "Install Anaconda/Miniconda or update CONDA_SH in scripts/activate_env.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

echo "Activated conda env: $ENV_NAME"

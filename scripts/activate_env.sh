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

PROJECT_BIN="$(pwd)/scripts/bin"
if [[ -d "$PROJECT_BIN" ]]; then
  export PATH="$PROJECT_BIN:$PATH"
fi

echo "Activated conda env: $ENV_NAME"
echo "Shortcut available: upload <creator> <TikTok URL or video id>"

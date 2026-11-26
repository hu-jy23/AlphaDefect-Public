#!/usr/bin/env bash
set -euo pipefail
ALPHA_ROOT=/root/autodl-tmp/alphadefect
export DOTNET_ROOT="$ALPHA_ROOT/dotnet"
export PATH="$DOTNET_ROOT:/root/miniconda3/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 PYTHONUNBUFFERED=1
ulimit -c 0
cd "$ALPHA_ROOT/src/game-v0.114.0"
exec "$ALPHA_ROOT/venv/bin/python" -I python/train_group_policy.py --config "$1"

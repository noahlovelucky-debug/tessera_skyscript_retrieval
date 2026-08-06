#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"
export TMPDIR="$ROOT/.cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

python3 -m venv --system-site-packages "$ROOT/.venv"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.ustc.edu.cn}"
"$ROOT/.venv/bin/python" -m pip install \
  --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
  --no-build-isolation -e "$ROOT[test]"
echo "Environment ready: source $ROOT/.venv/bin/activate"

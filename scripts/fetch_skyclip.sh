#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/third_party/SkyScript"
ZIP="$ROOT/checkpoints/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip"
URLS=(
  "https://opendatasharing.s3.amazonaws.com/SkyScript/ckpt/$(basename "$ZIP")"
  "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/$(basename "$ZIP")"
  "https://s3.us-west-2.amazonaws.com/opendatasharing/SkyScript/ckpt/$(basename "$ZIP")"
  "https://opendatasharing.s3-us-west-2.amazonaws.com/SkyScript/ckpt/$(basename "$ZIP")"
)
COMMIT="b16d2e76c5a0cdd644e2422a4446fb092d2dc1e4"
EXPECTED_SIZE="4486300088"
ARIA_PROXY=()
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  ARIA_PROXY=("--all-proxy=${HTTPS_PROXY}")
fi

mkdir -p "$ROOT/third_party" "$ROOT/checkpoints"
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/wangzhecheng/SkyScript.git "$REPO"
fi
git -C "$REPO" checkout --detach "$COMMIT"

LOCAL_ARIA="$ROOT/.tools/usr/bin/aria2c"
LOCAL_LIB="$ROOT/.tools/usr/lib/x86_64-linux-gnu"
if command -v aria2c >/dev/null 2>&1; then
  aria2c "${ARIA_PROXY[@]}" --continue=true --max-connection-per-server=16 --split=64 \
    --min-split-size=1M --file-allocation=none --dir="$(dirname "$ZIP")" \
    --out="$(basename "$ZIP")" "${URLS[@]}"
elif [[ -x "$LOCAL_ARIA" ]]; then
  LD_LIBRARY_PATH="$LOCAL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LOCAL_ARIA" "${ARIA_PROXY[@]}" --continue=true --max-connection-per-server=16 --split=64 \
    --min-split-size=1M --file-allocation=none --dir="$(dirname "$ZIP")" \
    --out="$(basename "$ZIP")" "${URLS[@]}"
else
  curl --location --fail --continue-at - --output "$ZIP" "${URLS[0]}"
fi
ACTUAL_SIZE="$(stat -c '%s' "$ZIP")"
if [[ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]]; then
  echo "Unexpected checkpoint size: $ACTUAL_SIZE (expected $EXPECTED_SIZE)" >&2
  exit 1
fi

DEST="${ZIP%.zip}"
unzip -o "$ZIP" -d "$(dirname "$DEST")"
find "$DEST" -type f -name '*.pt' -print

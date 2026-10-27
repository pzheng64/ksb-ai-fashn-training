#!/usr/bin/env bash
# Download Linux wheels for FASHN VTON training. Reuses wheels already in this
# tree and, when present, C:/dev/ksb-ai/offline.
set -euo pipefail

FASHN_OFFLINE="$(cd "$(dirname "$0")" && pwd)"
OFFLINE_ROOT="$(cd "$FASHN_OFFLINE/.." && pwd)"
# shellcheck source=../lib-paths.sh
source "$OFFLINE_ROOT/lib-paths.sh"

COMMON_WHEELS="$COMMON_OFFLINE/wheels"
FASHN_WHEELS="$FASHN_OFFLINE/wheels"
DEBS="$COMMON_OFFLINE/debs"
PY="${PY:-python3.12}"
KSB_AI_OFFLINE="${KSB_AI_OFFLINE:-/c/dev/ksb-ai/offline}"

mkdir -p "$COMMON_WHEELS" "$FASHN_WHEELS" "$DEBS"

if [[ "${FORCE:-0}" != "1" \
  && -f "$COMMON_WHEELS/torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
  && -f "$FASHN_WHEELS/fashn_human_parser-0.1.1-py3-none-any.whl" \
  && -f "$FASHN_WHEELS/onnxruntime_gpu-1.26.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" ]]; then
  echo "Training wheels already present. Set FORCE=1 to download again."
  du -sh "$COMMON_WHEELS" "$FASHN_WHEELS" "$DEBS" 2>/dev/null || true
  exit 0
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found: $PY" >&2
  exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"
FIND_LINKS=(--find-links "$COMMON_WHEELS" --find-links "$FASHN_WHEELS")
if [[ -d "$KSB_AI_OFFLINE/common/wheels" ]]; then
  FIND_LINKS+=(--find-links "$KSB_AI_OFFLINE/common/wheels")
fi
if [[ -d "$KSB_AI_OFFLINE/dwpose/wheels" ]]; then
  FIND_LINKS+=(--find-links "$KSB_AI_OFFLINE/dwpose/wheels")
fi
if [[ -d "$KSB_AI_OFFLINE/schp/wheels" ]]; then
  FIND_LINKS+=(--find-links "$KSB_AI_OFFLINE/schp/wheels")
fi

PIP_DOWNLOAD=( "$PY" -m pip download --retries 30 --resume-retries 100 --timeout 120 "${OFFLINE_PIP_DOWNLOAD_LINUX_ARGS[@]}" )

echo "==> Training requirements → common/wheels + fashn/wheels"
"${PIP_DOWNLOAD[@]}" \
  --dest "$FASHN_WHEELS" \
  "${FIND_LINKS[@]}" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  -c "$COMMON_OFFLINE/constraints-torch-cu128.txt" \
  -r "$FASHN_OFFLINE/requirements-fashn-gpu-py312.txt"

echo "==> Ubuntu debs → common/debs"
if compgen -G "$DEBS/"'*.deb' >/dev/null; then
  echo "  already present; skipping"
elif offline_linux_cross_host && [[ -f "$OFFLINE_ROOT/download-debs-mirror.py" ]]; then
  "$PY" "$OFFLINE_ROOT/download-debs-mirror.py" --out "$DEBS"
elif command -v apt-get >/dev/null 2>&1; then
  (
    cd "$DEBS"
    PKGS=(python3.12 python3.12-venv python3.12-dev python3-pip build-essential ninja-build git libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1)
    for p in "${PKGS[@]}"; do
      apt-get download "$p" 2>/dev/null || echo "  (skip $p)"
    done
  )
fi

echo "Done."
du -sh "$COMMON_WHEELS" "$FASHN_WHEELS" "$DEBS" 2>/dev/null || true

#!/usr/bin/env bash
set -euo pipefail

FASHN_OFFLINE="$(cd "$(dirname "$0")" && pwd)"
OFFLINE_ROOT="${OFFLINE_ROOT:-$(cd "$FASHN_OFFLINE/.." && pwd)}"
# shellcheck source=../lib-paths.sh
source "$OFFLINE_ROOT/lib-paths.sh"

FASHN_ROOT="${FASHN_ROOT:-$(cd "$FASHN_OFFLINE/../../fashn-vton-1.5" && pwd)}"
PY="${PY:-python3.12}"

FIND_LINKS=()
while IFS= read -r link; do
  FIND_LINKS+=(--find-links "$link")
done < <(resolve_wheel_find_links_fashn)

if ((${#FIND_LINKS[@]} == 0)); then
  echo "Missing wheels under $COMMON_OFFLINE/wheels and/or $FASHN_OFFLINE/wheels" >&2
  exit 1
fi

DEBS="$(resolve_debs_dir)"
if [[ "${SKIP_DEBS:-0}" != "1" && -n "$DEBS" ]] && command -v dpkg >/dev/null 2>&1; then
  echo "==> Installing .deb packages from $DEBS ..."
  if sudo -n true 2>/dev/null; then
    sudo dpkg -i "$DEBS"/*.deb 2>/dev/null || sudo apt-get install -y -f
  else
    echo "  (skipped sudo; run: sudo dpkg -i $DEBS/*.deb && sudo apt-get install -f)"
  fi
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Install Python 3.12 first (see offline/common/debs)." >&2
  exit 1
fi

if [[ ! -d "$FASHN_ROOT/.venv" ]]; then
  "$PY" -m venv "$FASHN_ROOT/.venv" 2>/dev/null || uv venv "$FASHN_ROOT/.venv" --python "$PY"
fi

PYBIN="$FASHN_ROOT/.venv/bin/python"
"$PYBIN" -m pip install --no-index "${FIND_LINKS[@]}" --upgrade pip setuptools wheel
"$PYBIN" -m pip install --no-index "${FIND_LINKS[@]}" \
  -c "$COMMON_OFFLINE/constraints-torch-cu128.txt" \
  -r "$FASHN_OFFLINE/requirements-fashn-gpu-py312.txt"
"$PYBIN" -m pip install --no-index --no-deps -e "$FASHN_ROOT"

echo "Installed into $FASHN_ROOT/.venv"
"$PYBIN" - <<'PY'
import torch
import onnxruntime as ort
import fashn_human_parser
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("onnxruntime", ort.__version__, ort.get_available_providers())
print("fashn_human_parser", getattr(fashn_human_parser, "__version__", "ok"))
PY

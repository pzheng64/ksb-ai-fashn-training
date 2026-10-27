# Shared offline bundle layout.
#   offline/common  — PyTorch cu128, Ubuntu debs, wheels shared with training
#   offline/fashn   — FASHN VTON training wheels + install/download scripts

_offline_lib="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_ROOT="$_offline_lib"
COMMON_OFFLINE="$OFFLINE_ROOT/common"
FASHN_OFFLINE="$OFFLINE_ROOT/fashn"

_offline_has_wheels() {
  local dir="$1"
  [[ -n "$dir" && -d "$dir/wheels" ]] && compgen -G "$dir/wheels/"'*.whl' >/dev/null
}

_offline_has_debs() {
  local dir="$1"
  [[ -n "$dir" && -d "$dir/debs" ]] && compgen -G "$dir/debs/"'*.deb' >/dev/null
}

resolve_wheel_find_links_fashn() {
  local out=()
  _offline_has_wheels "$COMMON_OFFLINE" && out+=("$COMMON_OFFLINE/wheels")
  _offline_has_wheels "$FASHN_OFFLINE" && out+=("$FASHN_OFFLINE/wheels")
  printf '%s\n' "${out[@]}"
}

resolve_debs_dir() {
  if _offline_has_debs "$COMMON_OFFLINE"; then
    echo "$COMMON_OFFLINE/debs"
  else
    echo ""
  fi
}

offline_linux_cross_host() {
  [[ "${OFFLINE_TARGET_OS:-}" == "linux" ]] && [[ "$(uname -s 2>/dev/null)" != "Linux" ]]
}

# Cross-download manylinux wheels from macOS/Windows (OFFLINE_TARGET_OS=linux).
# Older glibc tags are listed explicitly: pip does not expand them from manylinux_2_28.
OFFLINE_PIP_DOWNLOAD_LINUX_ARGS=()
if offline_linux_cross_host; then
  OFFLINE_PIP_DOWNLOAD_LINUX_ARGS=(
    --platform manylinux_2_28_x86_64
    --platform manylinux_2_27_x86_64
    --platform manylinux_2_25_x86_64
    --platform manylinux_2_24_x86_64
    --platform manylinux_2_18_x86_64
    --platform manylinux_2_17_x86_64
    --platform manylinux2014_x86_64
    --platform manylinux_2_12_x86_64
    --platform manylinux2010_x86_64
    --platform manylinux1_x86_64
    --platform manylinux_2_5_x86_64
    --python-version 3.12
    --implementation cp
    --abi cp312
    --abi abi3
    --only-binary=:all:
  )
fi

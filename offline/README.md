# Offline bundle

Linux only. These packages are Python 3.12 manylinux wheels for x86_64, PyTorch `2.11.0+cu128` (CUDA 12.8), and Ubuntu 24.04 amd64 `.deb` files. They install on Ubuntu with Python 3.12 and an NVIDIA driver that supports CUDA 12.8.

Shared PyTorch, NVIDIA, and Ubuntu packages are not stored in this repo. Copy them from `C:\dev\ksb-ai\offline\common`. The file list is in [common/README.md](common/README.md). `fashn-human-parser` and the other training-only wheels are in `fashn/wheels`.

## Structure

| Directory | Contents | How it is installed |
|-----------|----------|---------------------|
| `common/` | Note listing the shared wheels and debs to copy from `ksb-ai/offline/common`, plus `constraints-torch-cu128.txt` | Pulled in by the FASHN installer after those files are copied in. Nothing in `common/` is installed by itself. |
| `fashn/` | Training-only wheels and `install-offline.sh` | `bash offline/fashn/install-offline.sh` from `fashn-vton-1.5/` |

```
offline/
  common/
    README.md   which wheels and debs to copy from ksb-ai/offline/common
    debs/       empty until those .deb files are copied in
    wheels/     empty until those wheels are copied in
    constraints-torch-cu128.txt
  fashn/
    wheels/     einops, matplotlib, onnxruntime-gpu, transformers, fashn-human-parser
    requirements-fashn-gpu-py312.txt
    install-offline.sh
    download-wheels.sh
```

## Install

From the repo root, with `python3.12` on `PATH`:

```bash
cd fashn-vton-1.5
bash ../offline/fashn/install-offline.sh
source .venv/bin/activate
```

The script installs `common/debs` when passwordless `sudo` works, creates a `.venv` with Python 3.12, and runs `pip install --no-index` using `common/wheels` plus `fashn/wheels`. It then installs this repo in editable mode.

Move the bundle, or skip the Ubuntu packages:

```bash
export OFFLINE_ROOT=/path/to/offline
SKIP_DEBS=1 bash ../offline/fashn/install-offline.sh
```

If `sudo` is not passwordless, install the debs first:

```bash
sudo dpkg -i offline/common/debs/*.deb
sudo apt-get install -f
```

# Shared packages are not stored here

Use the copies already downloaded in the ksb-ai offline bundle:

`C:\dev\ksb-ai\offline\common`

Copy them into this tree before `offline/fashn/install-offline.sh`:

- `C:\dev\ksb-ai\offline\common\wheels\` → `offline/common/wheels/`
- `C:\dev\ksb-ai\offline\common\debs\` → `offline/common/debs/`

`constraints-torch-cu128.txt` in this directory is the version pin file. It stays here.

## Wheels to use

Copy only these CUDA 12.8 / Python 3.12 files from `ksb-ai/offline/common/wheels`. Skip the other wheels in that folder (`*cu13*`, `cuda-toolkit-13*`, `cuda-bindings-13*`, `triton-3.8*`, `setuptools-84*`, `pillow-11*`).

- `certifi-2026.10.23-py3-none-any.whl`
- `charset_normalizer-3.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl`
- `contourpy-1.4.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `cuda_bindings-12.9.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`
- `cuda_pathfinder-1.8.4-py3-none-any.whl`
- `cuda_toolkit-12.8.1-py2.py3-none-any.whl`
- `cycler-0.12.1-py3-none-any.whl`
- `filelock-4.2.4-py3-none-any.whl`
- `flatbuffers-25.12.19-py2.py3-none-any.whl`
- `fonttools-4.67.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `fsspec-2026.9.0-py3-none-any.whl`
- `hf_xet-1.7.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `huggingface_hub-0.36.2-py3-none-any.whl`
- `idna-3.20-py3-none-any.whl`
- `jinja2-3.1.6-py3-none-any.whl`
- `kiwisolver-1.5.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `markupsafe-3.0.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl`
- `mpmath-1.3.0-py3-none-any.whl`
- `networkx-3.7-py3-none-any.whl`
- `numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl`
- `nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl`
- `nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `nvidia_cudnn_cu12-9.19.0.56-py3-none-manylinux_2_27_x86_64.whl`
- `nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl`
- `nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl`
- `nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl`
- `nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl`
- `nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl`
- `nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`
- `opencv_python-5.0.0.93-cp37-abi3-manylinux_2_28_x86_64.whl`
- `packaging-26.3-py3-none-any.whl`
- `pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `pip-26.2.2-py3-none-any.whl`
- `protobuf-7.36.2-cp310-abi3-manylinux2014_x86_64.whl`
- `pyparsing-3.3.3-py3-none-any.whl`
- `python_dateutil-3.0.0-py2.py3-none-any.whl`
- `pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl`
- `regex-2026.9.29-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl`
- `requests-2.34.2-py3-none-any.whl`
- `setuptools-81.0.0-py3-none-any.whl`
- `six-1.17.0-py2.py3-none-any.whl`
- `sympy-1.14.0-py3-none-any.whl`
- `torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl`
- `torchvision-0.26.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl`
- `tqdm-4.70.1-py3-none-any.whl`
- `triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `typing_extensions-4.16.0-py3-none-any.whl`
- `tzdata-2026.5-py2.py3-none-any.whl`
- `urllib3-2.8.0-py3-none-any.whl`
- `wheel-0.48.0-py3-none-any.whl`

Training-only wheels stay in `offline/fashn/wheels` (`einops`, `matplotlib`, `onnxruntime-gpu`, `safetensors`, `tokenizers`, `transformers`, `fashn-human-parser`).

## Debs to use

Copy every `.deb` in `C:\dev\ksb-ai\offline\common\debs`.

That set is Ubuntu 24.04 amd64 for `python3.12`, `python3.12-venv`, `python3.12-dev`, `python3-pip`, `build-essential`, `ninja-build`, `git`, `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`, `libgomp1`, and their dependencies.

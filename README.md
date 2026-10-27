# ksb-ai-fashn

FASHN VTON v1.5 training setup. Model code is in `fashn-vton-1.5/`. Linux packages for an offline install are in `offline/`.

Remote: https://github.com/pzheng64/ksb-ai-fashn-training

## What the training code does

`scripts/train.py` fine-tunes `TryOnModel` from `model.safetensors`. `scripts/train_from_scratch.py` starts from random weights. Both read a CSV of person, garment, and target photos (`person_image`, `garment_image`, `target_image`, `category`).

DWPose and the human parser stay frozen. They build the same inputs inference uses: a clothing-agnostic person image, the garment image, pose maps, and the garment category. The target photo is the image the model is trained to reach.

The loss is rectified flow. A random timestep mixes Gaussian noise with the target, and the model predicts the straight-line velocity from that noise to the target. The objective is mean squared error. About 10% of steps drop the condition so classifier-free guidance still works at inference. The optimizer is AdamW with a warmup-cosine learning rate. On CUDA the weights run in bfloat16. From-scratch training also keeps an exponential moving average of the weights (decay 0.9999).

Saved files go in `--output-dir`: `train_state_*.pt` can resume a run, and `model_finetune_last.safetensors` or `model_scratch_last.safetensors` is the weight file.

## What was set up

- Collected the Ubuntu and Python packages the training code needs: Python 3.12, PyTorch `2.11.0+cu128`, and the CUDA 12.8 wheels. Packages already present in `C:\dev\ksb-ai\offline` were reused. The only new download was `fashn-human-parser` 0.1.1.
- Left the shared wheels and `.deb` files out of this repo. `offline/common/README.md` lists the exact files to copy from `C:\dev\ksb-ai\offline\common` into `offline/common/wheels` and `offline/common/debs` before installing.
- Kept the training-only wheels in `offline/fashn/wheels` (`einops`, `matplotlib`, `onnxruntime-gpu`, `safetensors`, `tokenizers`, `transformers`, `fashn-human-parser`). `onnxruntime-gpu` is about 277 MB and is stored with Git LFS.
- Initialized this directory as the git root, removed the nested `.git` inside `fashn-vton-1.5`, and ignored `tools/` (a local Windows Python install, not used by training).
- Removed the downloaded `data/` dataset and the example collage image from `fashn-vton-1.5/README.md`.
- Pushed `master` as commit `0156b88`.

## Train

Shared packages must be copied in first. See [offline/common/README.md](offline/common/README.md), then:

```bash
cd fashn-vton-1.5
bash ../offline/fashn/install-offline.sh
source .venv/bin/activate
python scripts/train.py --manifest data/train.csv --aux-weights-dir ./weights --output-dir ./runs/finetune
```

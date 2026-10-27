#!/usr/bin/env python3
"""
Train FASHN VTON (fine-tune or from-scratch pretraining).

Manifest CSV columns:
  person_image, garment_image, target_image, category [, garment_photo_type]

Fine-tune (loads model.safetensors):
  python scripts/train.py \\
    --manifest data/train.csv \\
    --aux-weights-dir ./weights \\
    --output-dir ./runs/finetune

From scratch (random init; only DWPose/parser aux weights required):
  python scripts/train_from_scratch.py \\
    --manifest data/train.csv \\
    --aux-weights-dir ./weights \\
    --output-dir ./runs/scratch

Download aux-only weights (no TryOn checkpoint):
  python scripts/download_weights.py --weights-dir ./weights --aux-only
"""

import argparse
import json
import random

import numpy as np
import torch

from fashn_vton.training.trainer import TrainSettings, Trainer
from fashn_vton.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FASHN VTON (rectified flow)")
    parser.add_argument("--manifest", type=str, required=True, help="CSV manifest of training triplets")
    parser.add_argument(
        "--aux-weights-dir",
        "--weights-dir",
        dest="aux_weights_dir",
        type=str,
        required=True,
        help="Directory with dwpose/ (and model.safetensors for fine-tune)",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Checkpoints and logs")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Randomly initialize TryOnModel (do not load model.safetensors)",
    )
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Fine-tune: optional TryOn weight path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from train_state_*.pt checkpoint")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--time-shift-mu", type=float, default=1.5)
    parser.add_argument("--no-time-shift", action="store_true")
    parser.add_argument("--segmentation-free", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="Enable EMA when set (e.g. 0.9999; recommended for from-scratch)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_settings(args: argparse.Namespace) -> TrainSettings:
    lr, warmup_steps, save_every = TrainSettings.apply_mode_defaults(
        args.from_scratch,
        args.lr,
        args.warmup_steps,
        args.save_every,
    )
    ema_decay = args.ema_decay
    if ema_decay is None and args.from_scratch:
        ema_decay = 0.9999

    return TrainSettings(
        manifest=args.manifest,
        aux_weights_dir=args.aux_weights_dir,
        output_dir=args.output_dir,
        from_scratch=args.from_scratch,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=lr,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=warmup_steps,
        cfg_dropout=args.cfg_dropout,
        time_shift_mu=None if args.no_time_shift else args.time_shift_mu,
        segmentation_free=args.segmentation_free,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
        save_every=save_every,
        log_every=args.log_every,
        ema_decay=ema_decay,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    settings = build_settings(args)
    logger = setup_logger("train", level=20)
    logger.info("Settings: %s", json.dumps(settings.__dict__, indent=2))

    trainer = Trainer(settings, logger)
    trainer.run()


if __name__ == "__main__":
    main()

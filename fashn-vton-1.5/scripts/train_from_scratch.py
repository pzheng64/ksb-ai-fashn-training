#!/usr/bin/env python3
"""Entry point for pretraining TryOnModel from random initialization."""

import sys

from train import build_settings, parse_args, set_seed
from fashn_vton.utils.logger import setup_logger


def main() -> None:
    if "--from-scratch" not in sys.argv:
        sys.argv.append("--from-scratch")

    args = parse_args()
    set_seed(args.seed)
    settings = build_settings(args)
    logger = setup_logger("train_scratch", level=20)
    logger.info("From-scratch pretraining (no pretrained TryOn weights).")

    from fashn_vton.training.trainer import Trainer

    Trainer(settings, logger).run()


if __name__ == "__main__":
    main()

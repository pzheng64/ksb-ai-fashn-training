"""Training loop for fine-tuning or from-scratch pretraining."""

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fashn_vton.training.checkpointing import load_training_checkpoint, save_training_checkpoint
from fashn_vton.training.dataset import VtonTripletDataset
from fashn_vton.training.ema import ModelEMA
from fashn_vton.training.flow import rectified_flow_loss
from fashn_vton.training.initialize import init_tryon_model
from fashn_vton.training.preprocess import VtonTrainingPreprocessor
from fashn_vton.training.schedule import build_warmup_cosine_scheduler
from fashn_vton.tryon_mmdit import TryOnModel
from fashn_vton.utils.checkpoint import load_checkpoint, save_checkpoint


def collate_training_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([sample[key] for sample in batch], dim=0) for key in batch[0]}


@dataclass
class TrainSettings:
    manifest: str
    aux_weights_dir: str
    output_dir: str
    from_scratch: bool = False
    init_checkpoint: str | None = None
    resume: str | None = None
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    cfg_dropout: float = 0.1
    time_shift_mu: float | None = 1.5
    segmentation_free: bool = True
    device: str | None = None
    seed: int = 42
    num_workers: int = 0
    amp: bool = True
    save_every: int = 500
    log_every: int = 10
    ema_decay: float | None = None

    @staticmethod
    def apply_mode_defaults(from_scratch: bool, lr: float | None, warmup_steps: int | None, save_every: int | None) -> tuple[float, int, int]:
        if from_scratch:
            return (
                lr if lr is not None else 1e-4,
                warmup_steps if warmup_steps is not None else 2000,
                save_every if save_every is not None else 1000,
            )
        return (
            lr if lr is not None else 1e-5,
            warmup_steps if warmup_steps is not None else 500,
            save_every if save_every is not None else 500,
        )


class Trainer:
    def __init__(self, settings: TrainSettings, logger: logging.Logger):
        self.settings = settings
        self.logger = logger
        self.device = torch.device(
            settings.device if settings.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.train_dtype = torch.float32
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.train_dtype = torch.bfloat16

    def _build_model(self) -> TryOnModel:
        model = TryOnModel()

        if self.settings.from_scratch:
            self.logger.info("Initializing TryOnModel weights from scratch")
            init_tryon_model(model)
        else:
            init_path = self.settings.init_checkpoint or os.path.join(self.settings.aux_weights_dir, "model.safetensors")
            self.logger.info("Loading pretrained TryOn weights from %s", init_path)
            state_dict = load_checkpoint(init_path, device=str(self.device))
            model.load_state_dict(state_dict)

        model.to(device=self.device, dtype=self.train_dtype)
        model.train()
        return model

    def _build_dataloader(self) -> DataLoader:
        preprocessor = VtonTrainingPreprocessor(
            self.settings.aux_weights_dir,
            device=str(self.device),
            segmentation_free=self.settings.segmentation_free,
            logger=self.logger,
        )
        dataset = VtonTripletDataset(self.settings.manifest, preprocessor=preprocessor)
        return DataLoader(
            dataset,
            batch_size=self.settings.batch_size,
            shuffle=True,
            num_workers=self.settings.num_workers,
            collate_fn=collate_training_batch,
            pin_memory=self.device.type == "cuda",
        )

    def run(self) -> Path:
        output_dir = Path(self.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._build_model()
        loader = self._build_dataloader()
        steps_per_epoch = max(len(loader), 1)
        total_steps = self.settings.max_steps
        if total_steps is None:
            total_steps = steps_per_epoch * self.settings.epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.settings.lr, weight_decay=self.settings.weight_decay)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=self.settings.warmup_steps,
            total_steps=total_steps,
        )

        ema: ModelEMA | None = None
        if self.settings.ema_decay is not None:
            ema = ModelEMA(model, decay=self.settings.ema_decay, device=self.device)

        start_epoch = 0
        global_step = 0
        if self.settings.resume:
            resume_info = load_training_checkpoint(
                self.settings.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
            )
            global_step = resume_info["global_step"]
            start_epoch = resume_info["epoch"]
            if ema is not None and resume_info.get("ema") is not None:
                ema.ema_model.load_state_dict(resume_info["ema"])
            self.logger.info("Resumed from %s at step=%s epoch=%s", self.settings.resume, global_step, start_epoch)

        use_amp = self.settings.amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and self.train_dtype == torch.float16)

        self.logger.info(
            "Training mode=%s device=%s dtype=%s steps=%s",
            "scratch" if self.settings.from_scratch else "finetune",
            self.device,
            self.train_dtype,
            total_steps,
        )

        stop_training = False
        last_epoch = start_epoch
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(start_epoch, self.settings.epochs):
            last_epoch = epoch
            for batch in loader:
                batch = {k: v.to(device=self.device, dtype=self.train_dtype) for k, v in batch.items()}
                cond_mask = torch.rand(batch["target"].shape[0], device=self.device) > self.settings.cfg_dropout

                with torch.autocast(device_type=self.device.type, dtype=self.train_dtype, enabled=use_amp):
                    loss, metrics = rectified_flow_loss(
                        model,
                        batch["target"],
                        ca_images=batch["ca_images"],
                        garment_images=batch["garment_images"],
                        person_poses=batch["person_poses"],
                        garment_poses=batch["garment_poses"],
                        garment_categories=batch["garment_categories"],
                        cond_mask=cond_mask,
                        time_shift_mu=self.settings.time_shift_mu,
                    )
                    loss = loss / self.settings.grad_accum

                scaler.scale(loss).backward()

                if (global_step + 1) % self.settings.grad_accum == 0:
                    if self.settings.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.settings.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)

                if global_step % self.settings.log_every == 0:
                    lr = scheduler.get_last_lr()[0] if global_step > 0 else self.settings.lr
                    self.logger.info(
                        "epoch=%s step=%s loss=%.6f t_mean=%.4f lr=%.2e",
                        epoch,
                        global_step,
                        metrics["loss"],
                        metrics["t_mean"],
                        lr,
                    )

                global_step += 1
                if global_step % self.settings.save_every == 0:
                    ckpt_path = output_dir / f"train_state_step_{global_step}.pt"
                    save_training_checkpoint(
                        ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        args=asdict(self.settings),
                        ema_state=ema.state_dict() if ema is not None else None,
                    )
                    self.logger.info("Saved %s", ckpt_path)

                if global_step >= total_steps:
                    stop_training = True
                    break

            if stop_training:
                break

        final_model_path = output_dir / ("model_scratch_last.safetensors" if self.settings.from_scratch else "model_finetune_last.safetensors")
        weights = ema.state_dict() if ema is not None else model.state_dict()
        save_checkpoint(weights, str(final_model_path))

        save_training_checkpoint(
            output_dir / "train_state_last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=last_epoch,
            args=asdict(self.settings),
            ema_state=ema.state_dict() if ema is not None else None,
        )
        self.logger.info("Training finished. Weights: %s", final_model_path)
        return final_model_path

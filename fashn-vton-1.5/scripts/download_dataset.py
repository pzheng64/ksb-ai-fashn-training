#!/usr/bin/env python3
"""
Download a public virtual try-on dataset and build a training manifest.

FASHN does not publish their 18M-pair training corpus. This script downloads the
ungated Hugging Face mirror of the VITON-HD *test* split (2,032 top try-on pairs,
research use) and writes images + train.csv for scripts/train.py.

Full VITON-HD train (11k pairs) is available from the official Google Drive link in
the VITON-HD paper repo, or the gated HF archive zhengchong/VITON-HD-Ready (accept
terms on Hugging Face, then use --dataset vitonhd-full --hf-token $HF_TOKEN).

Usage:
  pip install datasets pillow huggingface_hub
  python scripts/download_dataset.py --output-dir ../data/vitonhd-test

License: review VITON-HD terms (research only) before commercial use.
"""

import argparse
import csv
import os
from pathlib import Path

from PIL import Image


def export_vitonhd_hf(split_dir: Path, max_samples: int | None, parquet_glob: str | None = None) -> int:
    from datasets import load_dataset

    if parquet_glob:
        import glob

        files = sorted(glob.glob(parquet_glob))
        if not files:
            raise FileNotFoundError(f"No parquet files matched: {parquet_glob}")
        print(f"Loading {len(files)} local parquet shard(s)...")
        ds = load_dataset("parquet", data_files=files, split="train")
    else:
        print("Loading SaffalPoosh/VITON-HD-test from Hugging Face...")
        ds = load_dataset("SaffalPoosh/VITON-HD-test", split="train")

    person_dir = split_dir / "person"
    garment_dir = split_dir / "garment"
    target_dir = split_dir / "target"
    for d in (person_dir, garment_dir, target_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = split_dir / "train.csv"
    rows: list[dict[str, str]] = []
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    for idx in range(n):
        row = ds[idx]
        stem = f"{idx:06d}"
        person_path = person_dir / f"{stem}.jpg"
        garment_path = garment_dir / f"{stem}.jpg"
        target_path = target_dir / f"{stem}.jpg"

        # VITON-HD test rows: `image` is the person wearing `cloth` (ground truth).
        # Training uses the GT photo as the person source; preprocessing strips the
        # garment region for the agnostic person channel.
        row["image"].convert("RGB").save(person_path, quality=95)
        row["cloth"].convert("RGB").save(garment_path, quality=95)
        row["image"].convert("RGB").save(target_path, quality=95)

        rows.append(
            {
                "person_image": f"person/{stem}.jpg",
                "garment_image": f"garment/{stem}.jpg",
                "target_image": f"target/{stem}.jpg",
                "category": "tops",
                "garment_photo_type": "model",
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"  exported {idx + 1}/{n}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["person_image", "garment_image", "target_image", "category", "garment_photo_type"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {n} samples under {split_dir}")
    print(f"Manifest: {manifest_path}")
    return n


def export_vitonhd_full_tar(archive_path: Path, output_dir: Path, max_samples: int | None) -> int:
    import tarfile

    print(f"Extracting {archive_path} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    # Expected layout after extract: train/image, train/cloth, test/...
    train_root = output_dir / "train"
    if not train_root.is_dir():
        candidates = list(output_dir.glob("**/train"))
        train_root = candidates[0] if candidates else None
    if train_root is None or not (train_root / "cloth").is_dir():
        raise FileNotFoundError(
            "Could not find train/cloth after extract. Place VITON-HD.tar.gz from "
            "https://huggingface.co/datasets/zhengchong/VITON-HD-Ready"
        )

    pairs_file = train_root / "train_pairs.txt"
    if not pairs_file.is_file():
        pairs_file = train_root.parent / "train_pairs.txt"
    if not pairs_file.is_file():
        raise FileNotFoundError("train_pairs.txt not found in extracted VITON-HD archive")

    person_dir = output_dir / "manifest_export" / "person"
    garment_dir = output_dir / "manifest_export" / "garment"
    target_dir = output_dir / "manifest_export" / "target"
    for d in (person_dir, garment_dir, target_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_dir = train_root / "image"
    cloth_dir = train_root / "cloth"
    rows: list[dict[str, str]] = []

    with open(pairs_file, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if max_samples is not None:
        lines = lines[:max_samples]

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            continue
        person_name, cloth_name = parts[0], parts[1]
        stem = f"{idx:06d}"

        person_src = image_dir / person_name
        cloth_src = cloth_dir / cloth_name
        target_src = image_dir / person_name
        if not person_src.is_file() or not cloth_src.is_file():
            continue

        person_path = person_dir / f"{stem}.jpg"
        garment_path = garment_dir / f"{stem}.jpg"
        target_path = target_dir / f"{stem}.jpg"

        Image.open(person_src).convert("RGB").save(person_path, quality=95)
        Image.open(cloth_src).convert("RGB").save(garment_path, quality=95)
        Image.open(target_src).convert("RGB").save(target_path, quality=95)

        rows.append(
            {
                "person_image": f"person/{stem}.jpg",
                "garment_image": f"garment/{stem}.jpg",
                "target_image": f"target/{stem}.jpg",
                "category": "tops",
                "garment_photo_type": "model",
            }
        )

    manifest_path = output_dir / "manifest_export" / "train.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["person_image", "garment_image", "target_image", "category", "garment_photo_type"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} training rows to {manifest_path}")
    return len(rows)


def download_vitonhd_full_archive(output_dir: Path, hf_token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="zhengchong/VITON-HD-Ready",
        filename="VITON-HD.tar.gz",
        repo_type="dataset",
        local_dir=str(output_dir),
        token=hf_token,
    )
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public VTON dataset for training manifests")
    parser.add_argument(
        "--dataset",
        choices=("vitonhd-test", "vitonhd-full"),
        default="vitonhd-test",
        help="vitonhd-test: 2k HF test split (ungated). vitonhd-full: gated 4GB tar on HF.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Where to store images and train.csv")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for smoke tests")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="For gated vitonhd-full")
    parser.add_argument(
        "--archive-path",
        type=str,
        default=None,
        help="Local VITON-HD.tar.gz (skip HF download for vitonhd-full)",
    )
    parser.add_argument(
        "--parquet-glob",
        type=str,
        default=None,
        help="Use local parquet shards instead of re-downloading vitonhd-test (e.g. ../data/vitonhd-test/hf_parquet/*.parquet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "vitonhd-test":
        export_vitonhd_hf(output_dir, args.max_samples, args.parquet_glob)
        return

    archive = Path(args.archive_path) if args.archive_path else download_vitonhd_full_archive(output_dir, args.hf_token)
    export_vitonhd_full_tar(archive, output_dir, args.max_samples)


if __name__ == "__main__":
    main()

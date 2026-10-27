"""Dataset for (person, garment, try-on target) triplets."""

import csv
import os
from dataclasses import dataclass
from typing import Literal, Optional

from PIL import Image
from torch.utils.data import Dataset

Category = Literal["tops", "bottoms", "one-pieces"]
GarmentPhotoType = Literal["model", "flat-lay"]

VALID_CATEGORIES = ("tops", "bottoms", "one-pieces")
VALID_GARMENT_TYPES = ("model", "flat-lay")


@dataclass(frozen=True)
class VtonSample:
    person_path: str
    garment_path: str
    target_path: str
    category: Category
    garment_photo_type: GarmentPhotoType = "model"


def _resolve_path(path: str, root: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError("Empty path in manifest")
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def load_manifest(manifest_path: str) -> list[VtonSample]:
    """
    Load a CSV manifest.

    Required columns: person_image, garment_image, target_image, category
    Optional column: garment_photo_type (default model)
    """
    manifest_path = os.path.abspath(manifest_path)
    root = os.path.dirname(manifest_path)
    samples: list[VtonSample] = []

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"person_image", "garment_image", "target_image", "category"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest must include columns: {sorted(required)}")

        for row_idx, row in enumerate(reader, start=2):
            category = row["category"].strip()
            if category not in VALID_CATEGORIES:
                raise ValueError(f"Row {row_idx}: invalid category '{category}'")

            garment_type = row.get("garment_photo_type", "model").strip() or "model"
            if garment_type not in VALID_GARMENT_TYPES:
                raise ValueError(f"Row {row_idx}: invalid garment_photo_type '{garment_type}'")

            samples.append(
                VtonSample(
                    person_path=_resolve_path(row["person_image"], root),
                    garment_path=_resolve_path(row["garment_image"], root),
                    target_path=_resolve_path(row["target_image"], root),
                    category=category,
                    garment_photo_type=garment_type,
                )
            )

    if not samples:
        raise ValueError(f"No samples found in {manifest_path}")

    return samples


class VtonTripletDataset(Dataset):
    """Lazy-loading triplet dataset; heavy preprocessing runs in the collate/training loop."""

    def __init__(
        self,
        manifest_path: str,
        *,
        preprocessor: Optional[object] = None,
    ):
        self.samples = load_manifest(manifest_path)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        person = Image.open(sample.person_path).convert("RGB")
        garment = Image.open(sample.garment_path).convert("RGB")
        target = Image.open(sample.target_path).convert("RGB")

        if self.preprocessor is not None:
            return self.preprocessor(person, garment, target, sample.category, sample.garment_photo_type)

        return {
            "person": person,
            "garment": garment,
            "target": target,
            "category": sample.category,
            "garment_photo_type": sample.garment_photo_type,
        }

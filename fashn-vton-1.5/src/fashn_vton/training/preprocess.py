"""Training-time preprocessing (matches `TryOnPipeline` tensor layout)."""

import logging
import os
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
from PIL import Image

from fashn_vton.dwpose import DWposeDetector, draw_pose
from fashn_vton.pipeline import TryOnPipeline
from fashn_vton.preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)
from fashn_vton.utils import get_dummy_dw_keypoints, normalize_uint8_to_neg1_1, numpy_to_torch, setup_logger


class VtonTrainingPreprocessor:
    """
    Builds model inputs for training from (person, garment, target) PIL images.

    Pose and parsing models are frozen; only `TryOnModel` is trained.
    """

    def __init__(
        self,
        weights_dir: str,
        *,
        input_shape: tuple[int, int] = (864, 576),
        device: Optional[str] = None,
        segmentation_free: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.weights_dir = os.path.abspath(weights_dir)
        self.segmentation_free = segmentation_free
        self.logger = logger or setup_logger("VtonTrainingPreprocessor", level=logging.INFO)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.category_to_label = TryOnPipeline.CATEGORY_TO_LABEL

        max_dim = max(input_shape)
        self.pre_resize = AspectPreserveResize(target_size=(max_dim, max_dim), mode="fit", backend="pil")
        h, w = input_shape
        self.resize_pad_fn = ResizePad((w, h), backend="opencv")

        self._setup_pose_model()
        self._setup_hp_model()

    def _setup_pose_model(self):
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        dwpose_device = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
        self.pose_model = DWposeDetector(checkpoints_dir=dwpose_dir, device=dwpose_device)

    def _setup_hp_model(self):
        hp_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.hp_model = FashnHumanParser(device=hp_device)

    def _to_model_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = numpy_to_torch(img).unsqueeze(0)
        t = normalize_uint8_to_neg1_1(t)
        return t.to(self.device)

    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        target_image: Image.Image,
        category: Literal["tops", "bottoms", "one-pieces"],
        garment_photo_type: Literal["model", "flat-lay"] = "model",
    ) -> dict[str, torch.Tensor]:
        person_image = self.pre_resize(person_image, allow_upsampling=False)
        garment_image = self.pre_resize(garment_image, allow_upsampling=False)
        target_image = self.pre_resize(target_image, allow_upsampling=False)

        person_np = np.array(person_image)
        garment_np = np.array(garment_image)
        target_np = np.array(target_image)

        person_pose = self.pose_model(person_np[..., ::-1])
        garment_pose = (
            get_dummy_dw_keypoints()
            if garment_photo_type == "flat-lay"
            else self.pose_model(garment_np[..., ::-1])
        )

        person_pose_img = draw_pose(person_pose, person_np.shape[0], person_np.shape[1], grayscale=True)
        garment_pose_img = draw_pose(garment_pose, garment_np.shape[0], garment_np.shape[1], grayscale=True)

        person_seg = self.hp_model.predict(person_np)
        garment_seg = self.hp_model.predict(garment_np)

        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        labels_to_segment = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
        labels_to_segment_indices = [FASHN_LABELS_TO_IDS[label] for label in labels_to_segment]

        ca_image = create_clothing_agnostic_image(
            img_np=person_np.copy(),
            seg_pred=person_seg.copy(),
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            body_coverage=body_coverage,
            disable_masking=self.segmentation_free,
            logger=self.logger,
        )
        garment_processed = create_garment_image(
            img_np=garment_np,
            seg_pred=garment_seg,
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            disable_masking=garment_photo_type == "flat-lay",
        )

        ca_image = self.resize_pad_fn(ca_image, mem_padding=True)
        target_processed = self.resize_pad_fn(target_np, mem_padding=True)
        garment_processed = self.resize_pad_fn(garment_processed)
        person_pose_img = self.resize_pad_fn(person_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)
        garment_pose_img = self.resize_pad_fn(garment_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)

        garment_categories = torch.tensor(self.category_to_label[category], device=self.device, dtype=torch.long)

        return {
            "target": self._to_model_tensor(target_processed),
            "ca_images": self._to_model_tensor(ca_image),
            "garment_images": self._to_model_tensor(garment_processed),
            "person_poses": self._to_model_tensor(person_pose_img),
            "garment_poses": self._to_model_tensor(garment_pose_img),
            "garment_categories": garment_categories.unsqueeze(0),
        }

import logging
from typing import Any

import pandas as pd
import torch
from ultralytics import YOLO

from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOMulticlass(ImageLevelModule):
    """YOLO detector that can emit multiple semantic classes (player, ball, etc.)."""

    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "category_name",
        "role",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
        self.min_confidence = float(cfg.min_confidence)
        self.class_map = dict(cfg.class_map)
        self._next_id = 0

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        results_by_image = self.model(images, verbose=False)
        emitted = []

        for results, shape, (_, metadata) in zip(results_by_image, shapes, metadatas.iterrows()):
            class_names = results.names if isinstance(results.names, dict) else {}
            for bbox in results.boxes.cpu().numpy():
                conf = float(bbox.conf[0])
                if conf < self.min_confidence:
                    continue

                cls_id = int(bbox.cls)
                cls_name = str(class_names.get(cls_id, cls_id))
                role = self.class_map.get(cls_name)
                if role is None:
                    continue

                # Keep distinct numeric ids by semantic role to avoid collisions with dataset ids.
                if role == "ball":
                    category_id = 4
                elif role == "goalkeeper":
                    category_id = 2
                else:
                    category_id = 1

                emitted.append(
                    pd.Series(
                        dict(
                            image_id=metadata.name,
                            video_id=metadata.video_id,
                            category_id=category_id,
                            category_name=cls_name,
                            role=role,
                            bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                            bbox_conf=conf,
                        ),
                        name=self._next_id,
                    )
                )
                self._next_id += 1

        return emitted

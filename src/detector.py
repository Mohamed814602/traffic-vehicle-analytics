"""
Vehicle detector wrapper around Ultralytics YOLO.

Trained on UA-DETRAC (or COCO-pretrained as a fallback baseline).
Returns detections in `supervision.Detections` format so they plug
directly into the tracker and annotators.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

# COCO class ids that correspond to vehicles — used as a fallback when
# running with stock COCO weights instead of a UA-DETRAC fine-tuned model.
COCO_VEHICLE_CLASS_IDS = {2: "car", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, weights_path: str = "yolo11n.pt", conf_threshold: float = 0.35):
        """
        Args:
            weights_path: path to a .pt file. Either a fine-tuned UA-DETRAC
                model (produced by train/train_yolo.py) or a stock COCO
                checkpoint like 'yolo11n.pt' for a quick baseline.
            conf_threshold: minimum confidence to keep a detection.
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.is_coco_model = Path(weights_path).stem in {
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",  # kept for backward compatibility
        }

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a single BGR frame, return sv.Detections."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if self.is_coco_model:
            # Filter down to vehicle classes only when using stock COCO weights
            mask = np.isin(detections.class_id, list(COCO_VEHICLE_CLASS_IDS.keys()))
            detections = detections[mask]

        return detections

    def class_name(self, class_id: int) -> str:
        if self.is_coco_model:
            return COCO_VEHICLE_CLASS_IDS.get(class_id, str(class_id))
        return self.model.names.get(class_id, str(class_id))

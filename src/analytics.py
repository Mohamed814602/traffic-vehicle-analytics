"""
Downstream analytics built on top of tracked detections:
  - per-vehicle speed estimation (using ground-plane positions over time)
  - unique vehicle counting via line-crossing

This is the layer that turns "detection + tracking" into something a
traffic-ops use case actually cares about.
"""
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

import numpy as np
import supervision as sv

from src.perspective import PerspectiveTransformer


class SpeedEstimator:
    def __init__(self, transformer: PerspectiveTransformer, fps: int, window_seconds: float = 1.0):
        """
        Args:
            transformer: maps pixel -> real-world meter coordinates
            fps: video frame rate
            window_seconds: how much history (in seconds) to keep per
                track for computing a smoothed speed estimate
        """
        self.transformer = transformer
        self.fps = fps
        self.window_size = max(2, int(window_seconds * fps))
        # tracker_id -> deque of (frame_idx, x_m, y_m)
        self.history: Dict[int, Deque[Tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, detections: sv.Detections, frame_idx: int) -> Dict[int, float]:
        """Feed current-frame tracked detections, get back
        {tracker_id: speed_kmh} for every track we have enough history for."""
        if detections.tracker_id is None:
            return {}

        anchor_points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ground_points = self.transformer.transform_points(anchor_points)

        speeds = {}
        for tracker_id, (x_m, y_m) in zip(detections.tracker_id, ground_points):
            self.history[tracker_id].append((frame_idx, x_m, y_m))
            hist = self.history[tracker_id]
            if len(hist) >= 2:
                (f0, x0, y0), (f1, x1, y1) = hist[0], hist[-1]
                dt = (f1 - f0) / self.fps
                if dt > 0:
                    dist_m = float(np.hypot(x1 - x0, y1 - y0))
                    speed_mps = dist_m / dt
                    speeds[tracker_id] = speed_mps * 3.6  # m/s -> km/h
        return speeds


class LineCounter:
    """Counts unique tracker IDs crossing a virtual line, split by direction."""

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.line_zone = sv.LineZone(start=sv.Point(*start), end=sv.Point(*end))

    def update(self, detections: sv.Detections) -> Tuple[int, int]:
        """Returns (count_in, count_out) cumulative totals."""
        self.line_zone.trigger(detections)
        return self.line_zone.in_count, self.line_zone.out_count

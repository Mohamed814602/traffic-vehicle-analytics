"""
Multi-object tracking layer on top of raw detections.

Uses ByteTrack (via supervision) to assign a persistent ID to each
vehicle across frames, so downstream logic (counting, speed) can
reason about "this specific vehicle" rather than "a detection in frame N".
"""
import numpy as np
import supervision as sv


class VehicleTracker:
    def __init__(self, frame_rate: int = 25):
        self.tracker = sv.ByteTrack(frame_rate=frame_rate)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Feed one frame's detections in, get back detections with
        a stable `tracker_id` field populated for each box."""
        return self.tracker.update_with_detections(detections)

    def reset(self):
        self.tracker.reset()

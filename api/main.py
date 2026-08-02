"""
FastAPI service for the vehicle detection/tracking/speed pipeline.

Endpoints:
    POST /analyze   - upload a video (+ optional calibration.json), get back
                       counts + per-vehicle speed summary as JSON
    GET  /health     - basic liveness check

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Note: this endpoint returns summary JSON (fast, easy to test/demo).
A second endpoint that streams back the annotated video is a natural
extension once the JSON path is validated end-to-end.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import supervision as sv
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.analytics import SpeedEstimator, LineCounter
from src.perspective import PerspectiveTransformer, example_calibration

app = FastAPI(title="Traffic Vehicle Analytics API", version="0.1.0")

WEIGHTS_PATH = "yolo11n.pt"  # swap for a fine-tuned UA-DETRAC checkpoint in production


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), calibration: Optional[UploadFile] = File(None)):
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Upload a .mp4, .avi, or .mov video file")

    with tempfile.NamedTemporaryFile(suffix=Path(video.filename).suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    try:
        video_info = sv.VideoInfo.from_video_path(tmp_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded video")

    calibration_used = False
    if calibration is not None:
        calib_data = json.loads((await calibration.read()).decode("utf-8"))
        transformer = PerspectiveTransformer(
            source=calib_data["source_points"],
            target=calib_data["target_points_meters"],
        )
        calibration_used = True
    else:
        transformer = example_calibration()

    detector = VehicleDetector(weights_path=WEIGHTS_PATH)
    tracker = VehicleTracker(frame_rate=video_info.fps)
    speed_estimator = SpeedEstimator(transformer, fps=video_info.fps)

    h, w = video_info.height, video_info.width
    line_counter = LineCounter(start=(0, int(h * 0.6)), end=(w, int(h * 0.6)))

    max_speed_per_track = {}
    class_counts = {}
    frames_processed = 0

    for frame_idx, frame in enumerate(sv.get_video_frames_generator(tmp_path)):
        frames_processed = frame_idx + 1
        detections = detector.detect(frame)
        detections = tracker.update(detections)
        speeds = speed_estimator.update(detections, frame_idx)
        line_counter.update(detections)

        for i in range(len(detections)):
            tid = detections.tracker_id[i] if detections.tracker_id is not None else None
            cname = detector.class_name(int(detections.class_id[i]))
            class_counts[cname] = class_counts.get(cname, 0) + 1
            if tid is not None and tid in speeds:
                max_speed_per_track[tid] = max(max_speed_per_track.get(tid, 0), speeds[tid])

    Path(tmp_path).unlink(missing_ok=True)

    return {
        "frames_processed": frames_processed,
        "vehicles_in": line_counter.line_zone.in_count,
        "vehicles_out": line_counter.line_zone.out_count,
        "unique_tracks": len(max_speed_per_track),
        "max_speed_kmh_per_track": {int(k): round(v, 1) for k, v in max_speed_per_track.items()},
        "detection_class_counts": class_counts,
        "calibration_used": calibration_used,
        "speed_warning": None if calibration_used else "No calibration uploaded — speed values use PLACEHOLDER pixel-to-meter scale and are not real.",
    }

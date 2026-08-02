"""
End-to-end video pipeline: detection -> tracking -> speed + counting -> annotation.

Usage:
    python -m src.pipeline --source data/sample.mp4 --output outputs/annotated.mp4 --calibration calibration.json

If --calibration is omitted, falls back to a placeholder calibration
(src/perspective.py's example_calibration()) and prints a loud warning,
since speed numbers from the placeholder are not real. Run
tools/calibrate.py once per camera/video to generate a real one.
"""
import argparse
from pathlib import Path

import cv2
import supervision as sv

from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.analytics import SpeedEstimator, LineCounter
from src.perspective import PerspectiveTransformer, example_calibration


def load_transformer(calibration_path: str | None) -> PerspectiveTransformer:
    if calibration_path is None:
        print(
            "\n[WARNING] No --calibration file provided — using PLACEHOLDER "
            "pixel-to-meter values. Speed numbers will NOT be real.\n"
            "Run: python tools/calibrate.py --video <your_video> --output calibration.json\n"
        )
        return example_calibration()
    return PerspectiveTransformer.from_json(calibration_path)


def run(source_path: str, output_path: str, weights: str = "yolo11n.pt", conf: float = 0.35,
        calibration_path: str | None = None):
    video_info = sv.VideoInfo.from_video_path(source_path)
    fps = video_info.fps

    detector = VehicleDetector(weights_path=weights, conf_threshold=conf)
    tracker = VehicleTracker(frame_rate=fps)
    transformer = load_transformer(calibration_path)
    speed_estimator = SpeedEstimator(transformer, fps=fps)

    h, w = video_info.height, video_info.width
    line_counter = LineCounter(start=(0, int(h * 0.6)), end=(w, int(h * 0.6)))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with sv.VideoSink(output_path, video_info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_path)):
            detections = detector.detect(frame)
            detections = tracker.update(detections)

            speeds = speed_estimator.update(detections, frame_idx)
            in_count, out_count = line_counter.update(detections)

            labels = []
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else -1
                class_name = detector.class_name(int(class_id))
                speed = speeds.get(tracker_id)
                label = f"#{tracker_id} {class_name}"
                if speed is not None:
                    label += f" {speed:.0f}km/h"
                labels.append(label)

            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            cv2.putText(
                annotated, f"In: {in_count}  Out: {out_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )

            sink.write_frame(annotated)

    print(f"Done. In: {in_count}, Out: {out_count}. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle detection, tracking, speed & counting pipeline")
    parser.add_argument("--source", required=True, help="path to input video")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="path to save annotated video")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights: COCO baseline or fine-tuned .pt")
    parser.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    parser.add_argument("--calibration", default=None, help="path to calibration.json from tools/calibrate.py (omit for placeholder values)")
    args = parser.parse_args()
    run(args.source, args.output, args.weights, args.conf, args.calibration)

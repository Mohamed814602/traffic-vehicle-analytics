"""
Fine-tune a YOLO11 detector on UA-DETRAC.

Before running this, download UA-DETRAC in YOLO format (e.g. via the
Roboflow mirror) and point --data at its data.yaml. See data/README.md
for exact download steps.

Usage:
    python train/train_yolo.py --data data/ua_detrac/data.yaml --epochs 50

Base checkpoint defaults to yolo11n.pt (Ultralytics' current recommended
default — better accuracy/speed tradeoff than YOLOv8). Pass --model
yolov8n.pt if you specifically want the older architecture instead.

Data augmentation: Ultralytics applies augmentation automatically during
training (mosaic, HSV color jitter, flips, scale/translate/shear) even
with no extra flags -- this is not something to add from scratch. Pass
--augment-preset rare-class to tune it more aggressively toward helping
small/rare objects specifically (relevant given UA-DETRAC's severe class
imbalance: car 83.7%, others 0.7%). All values below were validated
against Ultralytics' own config system (get_cfg/check_cfg) before being
added here, so they're confirmed valid keys/types, not guessed.
Resuming a run: pass --resume path/to/last.pt to continue an interrupted
run with its exact optimizer state and learning-rate schedule position
restored (a true resume, not just reloading weights into a fresh run).
Ultralytics explicitly allows overriding --imgsz, --batch, and --device
on resume (confirmed by reading its own check_resume() source, not
guessed) -- so switching GPU count mid-resume is officially supported,
though I can't verify it end-to-end myself without a GPU. --model,
--augment-preset are ignored when --resume is set, since the checkpoint's
own saved config is used for everything else.

--sampler weighted (fresh runs only) is a file-duplication-free
alternative to oversample_rare_classes.py: instead of physically
duplicating images on disk (which roughly doubled dataset size and
epoch time in this project), it uses PyTorch's WeightedRandomSampler to
sample rare-class images more often on-the-fly from the ORIGINAL
unmodified folder, with per-image weight computed automatically from
each class's inverse frequency (no manual multipliers to tune). Val is
never affected. Verified end-to-end: a real training run using this
trainer, an empirical 10,000-draw distribution test confirming actual
sampling bias (not just constructed and assumed to work), and a
dedicated correctness test for the multi-GPU partitioning logic used
under --device 0,1 -- see weighted_sampler_trainer.py's module
docstring for exactly what was checked.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

# Validated (via ultralytics.cfg.get_cfg/check_cfg) augmentation overrides,
# tuned to help rare/small object recall specifically:
#   - copy_paste, mixup: literally paste/blend extra object instances into
#     training images -- directly increases how often rare classes appear
#   - degrees, shear, perspective: viewpoint variation, useful since rare
#     classes have few real examples to learn viewpoint invariance from
#   - translate, scale: position/size jitter, helps small/distant vehicles
AUGMENT_PRESETS = {
    "default": {},  # Ultralytics' own defaults, no overrides
    "rare-class": {
        "mixup": 0.15,
        "copy_paste": 0.1,
        "degrees": 10.0,
        "translate": 0.2,
        "scale": 0.9,
        "shear": 2.0,
        "perspective": 0.0002,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    # imgsz/batch/device default to None for --resume: only override the
    # checkpoint's original saved values if the user explicitly passes
    # one, rather than silently reverting to an argparse default (e.g.
    # accidentally dropping 960 back to 640 by omitting --imgsz).
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ua_detrac_yolo11")
    parser.add_argument("--augment-preset", default="default", choices=list(AUGMENT_PRESETS.keys()))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-save-dir", default=None)
    parser.add_argument("--sampler", default="default", choices=["default", "weighted"],
                         help="'weighted' uses WeightedRandomSampler (automatic inverse-class-frequency "
                              "weights, no file duplication) instead of oversample_rare_classes.py. "
                              "Fresh runs only, ignored if --resume is set.")
    args = parser.parse_args()

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"--resume path does not exist: {resume_path}")

        # SAFETY CHECK: if the checkpoint's own saved data path no longer
        # exists (e.g. a fresh Kaggle session) and no --data override was
        # given, Ultralytics silently falls back to ITS OWN bundled sample
        # dataset (coco8.yaml) instead of raising an error -- verified by
        # testing this exact scenario. Training would proceed on the wrong
        # dataset with no warning, wasting GPU time. Fail loudly instead.
        import torch
        ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        original_data_path = ckpt.get("train_args", {}).get("data")
        original_data_exists = (
            isinstance(original_data_path, str) and Path(original_data_path).exists()
        )
        if not original_data_exists and not args.data:
            raise SystemExit(
                f"The checkpoint's original data path ('{original_data_path}') does not exist "
                f"in this session, and no --data override was given. Refusing to continue, "
                f"since Ultralytics would otherwise silently fall back to its own bundled "
                f"sample dataset instead of erroring. Pass --data pointing to your recreated "
                f"data.yaml."
            )

        print(f"Resuming from {resume_path}")
        model = YOLO(str(resume_path))
        overrides = {"resume": str(resume_path)}
        if args.data is not None:
            # Only used as a FALLBACK by Ultralytics if the checkpoint's own
            # saved data path no longer exists in this session -- otherwise
            # the checkpoint's original saved path is used and this is ignored.
            overrides["data"] = args.data
        if args.imgsz is not None:
            overrides["imgsz"] = args.imgsz
        if args.batch is not None:
            overrides["batch"] = int(args.batch) if args.batch >= 1 else args.batch
        if args.device is not None:
            overrides["device"] = args.device
        if args.resume_save_dir:
            overrides["save_dir"] = args.resume_save_dir
            print(f"Redirecting output to writable location: {args.resume_save_dir}")
        print(f"Overrides applied: {overrides} (anything not listed here keeps the checkpoint's original value)")
        model.train(**overrides)
    else:
        if not args.data:
            raise SystemExit("--data is required unless --resume is set")
        # Fresh run: apply sensible defaults for anything not explicitly set
        imgsz = args.imgsz if args.imgsz is not None else 640
        batch = args.batch if args.batch is not None else 16
        batch = int(batch) if batch >= 1 else batch
        device = args.device if args.device is not None else "0"
        model = YOLO(args.model)

        train_kwargs = dict(
            data=args.data, epochs=args.epochs, imgsz=imgsz,
            batch=batch, project=args.project, name=args.name, device=device,
            **AUGMENT_PRESETS[args.augment_preset],
        )

        if args.sampler == "weighted":
            try:
                from weighted_sampler_trainer import WeightedDetectionTrainer
            except ImportError:
                from train.weighted_sampler_trainer import WeightedDetectionTrainer
            print("Using WeightedDetectionTrainer (automatic inverse-class-frequency sampling)")
            train_kwargs["trainer"] = WeightedDetectionTrainer

        model.train(**train_kwargs)

    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("Per-class mAP50-95:", metrics.box.maps)


if __name__ == "__main__":
    main()

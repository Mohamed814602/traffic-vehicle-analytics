"""
Undersample the majority class (car) by removing some car-ONLY images
from the TRAIN split -- images that contain a rare class (bus/van/others)
are NEVER removed, even if they also contain a car, since removing them
would throw away the exact rare-class examples oversampling is trying
to preserve.

Removed images are MOVED (not deleted) to train/removed_images and
train/removed_labels, so this is reversible -- nothing is permanently
lost, you can just copy them back if you change your mind.

Combine with oversample_rare_classes.py for a stronger balancing effect
than either technique alone: undersampling shrinks the majority class,
oversampling grows the minority classes, working from both directions.

Usage:
    python train/undersample_car.py \
        --yolo-dir /kaggle/working/ua_detrac_yolo \
        --remove-fraction 0.5 \
        --seed 42

--remove-fraction 0.5 means: of all images containing ONLY car and no
other class, randomly remove half of them from the active train set.
"""
import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["car", "bus", "van", "others"]
CAR_CLASS_ID = 0


def classes_in_label_file(label_path: Path) -> set[int]:
    classes = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            classes.add(int(line.split()[0]))
    return classes


def count_class_boxes(label_files: list[Path]) -> dict[int, int]:
    counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for label_path in label_files:
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if line:
                counts[int(line.split()[0])] += 1
    return counts


def undersample(yolo_dir: Path, remove_fraction: float, seed: int) -> None:
    train_images = yolo_dir / "train" / "images"
    train_labels = yolo_dir / "train" / "labels"
    removed_images_dir = yolo_dir / "train" / "removed_images"
    removed_labels_dir = yolo_dir / "train" / "removed_labels"
    removed_images_dir.mkdir(parents=True, exist_ok=True)
    removed_labels_dir.mkdir(parents=True, exist_ok=True)

    all_label_files = sorted(train_labels.glob("*.txt"))
    if not all_label_files:
        raise SystemExit(f"No label files found in {train_labels}")

    before_counts = count_class_boxes(all_label_files)

    # Only images containing car AND NOTHING ELSE are eligible for removal --
    # this guarantees no rare-class instance is ever lost to undersampling.
    car_only_files = [
        lp for lp in all_label_files
        if classes_in_label_file(lp) == {CAR_CLASS_ID}
    ]

    random.seed(seed)
    shuffled = car_only_files[:]
    random.shuffle(shuffled)
    n_to_remove = int(len(shuffled) * remove_fraction)
    to_remove = shuffled[:n_to_remove]

    for label_path in to_remove:
        stem = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = train_images / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        shutil.move(str(image_path), str(removed_images_dir / image_path.name))
        shutil.move(str(label_path), str(removed_labels_dir / label_path.name))

    remaining_label_files = sorted(train_labels.glob("*.txt"))
    after_counts = count_class_boxes(remaining_label_files)

    print(f"Car-only images found: {len(car_only_files)}")
    print(f"Removed (moved to train/removed_*): {len(to_remove)}")
    print(f"Remaining train images: {len(remaining_label_files)} (was {len(all_label_files)})\n")

    print(f"{'Class':<10}{'Before':>10}{'After':>10}{'Change':>10}")
    for class_id, name in enumerate(CLASS_NAMES):
        b, a = before_counts[class_id], after_counts[class_id]
        print(f"{name:<10}{b:>10}{a:>10}{a - b:>+10}")

    total_before = sum(before_counts.values())
    total_after = sum(after_counts.values())
    print(f"\n{'Class':<10}{'Before %':>10}{'After %':>10}")
    for class_id, name in enumerate(CLASS_NAMES):
        pct_before = 100 * before_counts[class_id] / total_before if total_before else 0
        pct_after = 100 * after_counts[class_id] / total_after if total_after else 0
        print(f"{name:<10}{pct_before:>9.1f}%{pct_after:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Undersample car-only images to reduce majority-class dominance")
    parser.add_argument("--yolo-dir", required=True, help="path to converted YOLO dataset (with train/val subfolders)")
    parser.add_argument("--remove-fraction", type=float, required=True,
                         help="fraction of CAR-ONLY images to remove, e.g. 0.5 removes half of them")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 <= args.remove_fraction <= 1.0):
        raise SystemExit("--remove-fraction must be between 0.0 and 1.0")

    undersample(Path(args.yolo_dir), args.remove_fraction, args.seed)


if __name__ == "__main__":
    main()

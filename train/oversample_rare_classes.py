"""
Oversample images containing rare classes (bus, van, others) in the
TRAIN split only, to help the model see them more often per epoch.

This does NOT try to fully equalize classes -- with car at 83.7% and
others at 0.7% of boxes, full equalization would mean duplicating a
tiny handful of "others" images 100+ times each, which causes severe
overfitting on those specific images rather than genuine improvement.
Moderate multipliers (e.g. 3-8x) are standard practice instead.

Only the train split is touched -- val must stay as the original,
unduplicated distribution, or validation metrics become misleading
(the model would be "tested" on images it saw duplicated during training).

Usage:
    python train/oversample_rare_classes.py \
        --yolo-dir /kaggle/working/ua_detrac_yolo \
        --multipliers bus=3,van=2,others=8

If an image contains MORE THAN ONE rare class (e.g. both a bus and an
"others" vehicle), the largest applicable multiplier is used for that
image -- not the product of both -- to avoid runaway duplication counts
on images that happen to be doubly-rare.
"""
import argparse
import shutil
from pathlib import Path

CLASS_NAMES = ["car", "bus", "van", "others"]
NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_multipliers(spec: str) -> dict[int, int]:
    """Parse 'bus=3,van=2,others=8' into {1: 3, 2: 2, 3: 8}."""
    result = {}
    for pair in spec.split(","):
        name, value = pair.split("=")
        name = name.strip().lower()
        if name not in NAME_TO_ID:
            raise ValueError(f"Unknown class name '{name}'. Valid: {CLASS_NAMES}")
        result[NAME_TO_ID[name]] = int(value)
    return result


def classes_in_label_file(label_path: Path) -> set[int]:
    classes = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            classes.add(int(line.split()[0]))
    return classes


def oversample(yolo_dir: Path, multipliers: dict[int, int]) -> None:
    train_images = yolo_dir / "train" / "images"
    train_labels = yolo_dir / "train" / "labels"

    label_files = sorted(train_labels.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No label files found in {train_labels}")

    before_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for label_path in label_files:
        for class_id in classes_in_label_file(label_path):
            before_counts[class_id] += 1

    duplicated_image_count = 0
    duplicated_box_counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for label_path in label_files:
        present_classes = classes_in_label_file(label_path)
        applicable_multipliers = [multipliers[c] for c in present_classes if c in multipliers]
        if not applicable_multipliers:
            continue

        # Use the largest multiplier among rare classes present in this
        # image, not the product -- avoids runaway duplication for images
        # that happen to contain multiple rare classes at once.
        multiplier = max(applicable_multipliers)
        extra_copies = multiplier - 1  # multiplier=3 means 2 EXTRA copies (3 total)
        if extra_copies <= 0:
            continue

        stem = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = train_images / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"  [skip] no matching image found for label {label_path.name}")
            continue

        for dup_idx in range(1, extra_copies + 1):
            new_stem = f"{stem}_dup{dup_idx}"
            shutil.copy(image_path, train_images / f"{new_stem}{image_path.suffix}")
            shutil.copy(label_path, train_labels / f"{new_stem}.txt")
            duplicated_image_count += 1
            for class_id in present_classes:
                duplicated_box_counts[class_id] += 1

    after_counts = {k: v + duplicated_box_counts[k] for k, v in before_counts.items()}

    print(f"Created {duplicated_image_count} duplicate image+label pairs.\n")
    print(f"{'Class':<10}{'Before':>10}{'After':>10}{'Change':>10}")
    for class_id, name in enumerate(CLASS_NAMES):
        b, a = before_counts[class_id], after_counts[class_id]
        print(f"{name:<10}{b:>10}{a:>10}{f'+{a-b}':>10}")


def main():
    parser = argparse.ArgumentParser(description="Oversample rare-class images in the train split")
    parser.add_argument("--yolo-dir", required=True, help="path to converted YOLO dataset (with train/val subfolders)")
    parser.add_argument(
        "--multipliers", required=True,
        help="comma-separated class=multiplier pairs, e.g. 'bus=3,van=2,others=8'. "
             "A multiplier of 3 means the image ends up appearing 3 times total in train.",
    )
    args = parser.parse_args()

    yolo_dir = Path(args.yolo_dir)
    multipliers = parse_multipliers(args.multipliers)
    oversample(yolo_dir, multipliers)


if __name__ == "__main__":
    main()

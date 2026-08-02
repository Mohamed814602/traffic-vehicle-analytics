"""
Add REAL rare-class images from UA-DETRAC's TEST split into the training
set -- a genuine alternative to oversample_rare_classes.py's duplication
approach. This adds images the model has never seen, rather than
repeating existing ones, which should combat the overfitting pattern
observed after epoch 6 in the oversampled+augmented run (the model
peaked then declined, most likely because heavy duplication meant it
was memorizing the same few rare-class images rather than generalizing).

Only images containing at least one rare class (bus/van/others) are
added -- car-only test-split images are skipped entirely, since pulling
in the whole test set would just reintroduce the same car-dominance
problem this whole effort is trying to fix.

IMPORTANT HONEST CAVEAT: UA-DETRAC's train/test split exists specifically
so results can be compared against the published benchmark. Using test
images for training means your numbers are no longer directly comparable
to other UA-DETRAC benchmark results -- disclose this clearly in any
writeup ("custom split, not the official benchmark protocol") rather
than implying a standard comparison. For a portfolio project this is a
reasonable, disclosable tradeoff, not something to hide.

This does NOT touch your existing val split at all -- new images go
into train/images and train/labels only.

Usage:
    python train/add_test_split_rare_data.py \
        --xml-dir /path/to/DETRAC-Test-Annotations-XML \
        --img-dir /path/to/DETRAC-Test-Images \
        --train-dir /kaggle/working/ua_detrac_yolo/train
"""
import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

sys.path.insert(0, ".")
try:
    # Flat-file layout (e.g. Kaggle: all scripts in the same working dir)
    from convert_detrac_xml_to_yolo import CLASS_TO_ID, normalize_vehicle_type  # noqa: E402
except ImportError:
    # Package layout (this repo: train/ is a proper package)
    from train.convert_detrac_xml_to_yolo import CLASS_TO_ID, normalize_vehicle_type  # noqa: E402

RARE_CLASSES = {CLASS_TO_ID["bus"], CLASS_TO_ID["van"], CLASS_TO_ID["others"]}
CLASS_NAMES = ["car", "bus", "van", "others"]


def count_class_boxes(label_files: list[Path]) -> dict[int, int]:
    counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for label_path in label_files:
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if line:
                counts[int(line.split()[0])] += 1
    return counts


def add_rare_frames_from_sequence(xml_path: Path, img_dir: Path, out_images: Path, out_labels: Path) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence_name = root.attrib.get("name", xml_path.stem)
    sequence_img_dir = img_dir / sequence_name

    if not sequence_img_dir.exists():
        print(f"  [skip] no image folder for '{sequence_name}' at {sequence_img_dir}")
        return 0

    added = 0
    for frame in root.findall("frame"):
        frame_num = int(frame.attrib["num"])
        img_filename = f"img{frame_num:05d}.jpg"
        img_path = sequence_img_dir / img_filename
        if not img_path.exists():
            continue

        target_list = frame.find("target_list")
        if target_list is None:
            continue

        lines = []
        classes_present = set()
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        for target in target_list.findall("target"):
            box = target.find("box")
            attribute = target.find("attribute")
            if box is None:
                continue

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            w = float(box.attrib["width"])
            h = float(box.attrib["height"])

            vehicle_type = "others"
            if attribute is not None:
                vehicle_type = normalize_vehicle_type(attribute.attrib.get("vehicle_type", "others"))
            class_id = CLASS_TO_ID[vehicle_type]
            classes_present.add(class_id)

            x_center = min(max((left + w / 2) / img_w, 0.0), 1.0)
            y_center = min(max((top + h / 2) / img_h, 0.0), 1.0)
            norm_w = min(max(w / img_w, 0.0), 1.0)
            norm_h = min(max(h / img_h, 0.0), 1.0)
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Skip frames with no rare class at all -- only add genuinely
        # useful new examples, don't just dump car-only test images in.
        if not (classes_present & RARE_CLASSES):
            continue

        out_name = f"testsplit_{sequence_name}_{img_filename}"
        shutil.copy(img_path, out_images / out_name)
        (out_labels / (Path(out_name).stem + ".txt")).write_text("\n".join(lines))
        added += 1

    return added


def main():
    parser = argparse.ArgumentParser(description="Add real rare-class test-split images into the training set")
    parser.add_argument("--xml-dir", required=True, help="folder containing test-split *.xml annotation files")
    parser.add_argument("--img-dir", required=True, help="folder containing per-sequence test-split image subfolders")
    parser.add_argument("--train-dir", required=True, help="existing train/ folder to add images into (must contain images/ and labels/)")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    train_dir = Path(args.train_dir)
    out_images = train_dir / "images"
    out_labels = train_dir / "labels"
    if not out_images.exists() or not out_labels.exists():
        raise SystemExit(f"--train-dir must contain existing images/ and labels/ subfolders: {train_dir}")

    before_counts = count_class_boxes(sorted(out_labels.glob("*.txt")))

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No .xml files found in {xml_dir}")

    total_added = 0
    for xml_path in xml_files:
        print(f"Scanning {xml_path.name} for rare-class frames...")
        n = add_rare_frames_from_sequence(xml_path, img_dir, out_images, out_labels)
        total_added += n

    after_counts = count_class_boxes(sorted(out_labels.glob("*.txt")))

    print(f"\nAdded {total_added} real (non-duplicate) rare-class images from the test split.\n")
    print(f"{'Class':<10}{'Before':>10}{'After':>10}{'Change':>10}")
    for class_id, name in enumerate(CLASS_NAMES):
        b, a = before_counts[class_id], after_counts[class_id]
        print(f"{name:<10}{b:>10}{a:>10}{a - b:>+10}")

    print(
        "\nREMINDER: these images came from UA-DETRAC's official TEST split. "
        "Disclose this as a custom (non-benchmark-standard) split in any writeup."
    )


if __name__ == "__main__":
    main()

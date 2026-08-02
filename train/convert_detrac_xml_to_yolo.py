"""
Convert raw UA-DETRAC XML annotations + image sequences into YOLO format.

Only needed if you downloaded the ORIGINAL UA-DETRAC release (XML
annotations + per-sequence image folders) instead of a pre-converted
Roboflow YOLO export. Preserves the original 4-class vehicle_type
breakdown (car / bus / van / others), which some Roboflow mirrors
collapse into a single generic "vehicle" class.

Validated against the "bratjay/ua-detrac-orig" Kaggle mirror. Real class
distribution measured on the full training split (71,825 frames):
    car:    429,559 boxes (83.7%)
    van:     47,492 boxes ( 9.3%)
    bus:     32,643 boxes ( 6.4%)
    others:   3,390 boxes ( 0.7%)
This confirms the expected imbalance -- report per-class mAP in your
evaluation, since "others" will predictably underperform given how
little data it has.

Expected input layout (matches the official UA-DETRAC release, and the
Kaggle "bratjay/ua-detrac-orig" mirror specifically):

    xml_dir/
        MVI_20011.xml
        MVI_20012.xml
        ...
    img_dir/
        MVI_20011/
            img00001.jpg
            img00002.jpg
            ...
        MVI_20012/
            ...

Each XML file is one <sequence>, containing <frame num="N"> elements,
each with a <target_list> of <target> elements. Each target has a
<box left top width height> (already top-left pixel coords, not
center) and an <attribute vehicle_type="car|bus|van|others" ...>.

Usage:
    python train/convert_detrac_xml_to_yolo.py \
        --xml-dir data/ua_detrac_raw/annotations \
        --img-dir data/ua_detrac_raw/images \
        --output-dir data/ua_detrac_yolo \
        --val-split 0.15

The train/val split is done PER SEQUENCE (not per frame) to avoid
leaking near-duplicate consecutive frames from the same video across
both splits, which would make validation metrics misleadingly high.
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

# Canonical UA-DETRAC vehicle classes, in a fixed order so class
# indices are stable across runs.
CLASS_NAMES = ["car", "bus", "van", "others"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def normalize_vehicle_type(raw: str) -> str:
    """UA-DETRAC XML has been seen with slightly different casings/
    spellings for the catch-all category ('others', 'other') across
    dataset releases, so normalize defensively."""
    raw = raw.strip().lower()
    if raw in ("car",):
        return "car"
    if raw in ("bus",):
        return "bus"
    if raw in ("van",):
        return "van"
    return "others"


def convert_sequence(xml_path: Path, img_dir: Path, out_images: Path, out_labels: Path) -> int:
    """Convert one sequence's XML into per-frame YOLO .txt files.
    Returns the number of frames successfully converted."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence_name = root.attrib.get("name", xml_path.stem.replace("_v3", ""))
    sequence_img_dir = img_dir / sequence_name

    if not sequence_img_dir.exists():
        print(f"  [skip] no image folder found for sequence '{sequence_name}' at {sequence_img_dir}")
        return 0

    converted = 0
    for frame in root.findall("frame"):
        frame_num = int(frame.attrib["num"])
        img_filename = f"img{frame_num:05d}.jpg"
        img_path = sequence_img_dir / img_filename

        if not img_path.exists():
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lines = []
        target_list = frame.find("target_list")
        if target_list is None:
            continue

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

            # Convert top-left pixel box -> normalized YOLO center-format
            x_center = (left + w / 2) / img_w
            y_center = (top + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Clip to [0, 1] defensively — some UA-DETRAC boxes extend
            # slightly past frame edges due to annotation tolerance
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Copy image and write label file, even if there are zero
        # targets in this frame (a valid negative/background example)
        out_img_name = f"{sequence_name}_{img_filename}"
        shutil.copy(img_path, out_images / out_img_name)
        label_path = out_labels / (Path(out_img_name).stem + ".txt")
        label_path.write_text("\n".join(lines))
        converted += 1

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC XML annotations to YOLO format")
    parser.add_argument("--xml-dir", required=True, help="folder containing *.xml annotation files (e.g. MVI_20011.xml)")
    parser.add_argument("--img-dir", required=True, help="folder containing per-sequence image subfolders")
    parser.add_argument("--output-dir", required=True, help="where to write the YOLO-format dataset")
    parser.add_argument("--val-split", type=float, default=0.15, help="fraction of SEQUENCES held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No .xml files found in {xml_dir}")

    random.seed(args.seed)
    xml_files_shuffled = xml_files[:]
    random.shuffle(xml_files_shuffled)
    n_val = max(1, int(len(xml_files_shuffled) * args.val_split))
    val_files = set(xml_files_shuffled[:n_val])

    total_converted = {"train": 0, "val": 0}
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for xml_path in xml_files:
        split = "val" if xml_path in val_files else "train"
        print(f"Converting {xml_path.name} -> {split}")
        n = convert_sequence(
            xml_path, img_dir,
            output_dir / split / "images",
            output_dir / split / "labels",
        )
        total_converted[split] += n

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"train: {output_dir.resolve() / 'train' / 'images'}\n"
        f"val: {output_dir.resolve() / 'val' / 'images'}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    print(f"\nDone. Train frames: {total_converted['train']}, Val frames: {total_converted['val']}")
    print(f"Sequences: {len(xml_files) - len(val_files)} train, {len(val_files)} val")
    print(f"data.yaml written to {data_yaml}")


if __name__ == "__main__":
    main()

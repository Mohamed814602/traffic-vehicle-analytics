# Dataset: UA-DETRAC

This project fine-tunes on **UA-DETRAC**, a public benchmark of 100 real-world
traffic video sequences (140,000+ frames, 1.21M labeled vehicle bounding boxes,
4 vehicle classes: car, bus, van, others) captured across varied weather and
lighting conditions.

Paper / original source: https://arxiv.org/abs/1511.04136

## Option A — Kaggle original XML release (recommended — validated, 4-class)

This is the path actually validated for this project. It preserves the full
car/bus/van/others breakdown, unlike some pre-converted mirrors that collapse
everything into a single generic "vehicle" class.

1. Create a free Kaggle account at https://kaggle.com
2. Add the dataset **[bratjay/ua-detrac-orig](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig)**
   as an input to your Kaggle Notebook (or download it locally if you have
   enough disk space — it's several GB).
3. Confirmed folder structure (Kaggle mounts it under `/kaggle/input/datasets/bratjay/ua-detrac-orig/`):
   ```
   DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/   # 60 sequence .xml files
   DETRAC-Images/DETRAC-Images/                                  # per-sequence image folders
       MVI_20011/
           img00001.jpg
           img00002.jpg
           ...
   ```
4. Run the converter (see `train/convert_detrac_xml_to_yolo.py`):
   ```bash
   python train/convert_detrac_xml_to_yolo.py \
     --xml-dir /kaggle/input/datasets/bratjay/ua-detrac-orig/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML \
     --img-dir /kaggle/input/datasets/bratjay/ua-detrac-orig/DETRAC-Images/DETRAC-Images \
     --output-dir /kaggle/working/ua_detrac_yolo \
     --val-split 0.15
   ```
   This splits by whole sequence (not by frame) to avoid leaking near-duplicate
   consecutive frames across train/val.

**Confirmed real output** (51 train / 9 val sequences, 15% split):
- Train: 71,825 frames — Val: 10,260 frames (82,085 total from the 60-sequence
  official training set)
- Class distribution (train split):

  | Class | Boxes | % |
  |---|---|---|
  | car | 429,559 | 83.7% |
  | van | 47,492 | 9.3% |
  | bus | 32,643 | 6.4% |
  | others | 3,390 | 0.7% |

  This confirms the expected severe imbalance — report **per-class mAP** in
  your evaluation, since `others` (0.7% of boxes) will likely underperform
  `car` by a wide margin. That's expected, not a bug.

## Option B — Roboflow mirror (faster, but single-class only)

A pre-converted YOLO-format mirror exists on Roboflow Universe
(`cs474-ug2-vehicle-detection/ua-detrac-rvwkg`), but as of the version checked
for this project, it collapses all 4 vehicle types into one generic `vehicle`
class (`nc: 1`) — you lose the car/bus/van/others breakdown entirely. Only use
this if you don't need per-class results and want the fastest path to a
working baseline:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("cs474-ug2-vehicle-detection").project("ua-detrac-rvwkg")
dataset = project.version(2).download("yolov11", location="data/ua_detrac")
```

(Note: `"yolov11"` is confirmed as a valid Roboflow export format string for
object detection projects, in addition to the older `"yolov8"`.)

## Option C — dtrnngc/ua-detrac-dataset mirror (tried, not recommended)

Also investigated during this project: a pre-converted YOLO-format mirror
at `dtrnngc/ua-detrac-dataset` (83,791 frames, matching the official
training set size exactly). Already in YOLO format (no XML conversion
needed), and does have multiple classes (`1`, `2`, `3` seen in samples) —
but **ships with no `data.yaml` or classes/names file anywhere**, so
there's no reliable way to know which class ID maps to which vehicle
type. We abandoned this option for that reason rather than guess.
If you go looking at Kaggle mirrors yourself, this is worth knowing about
in advance rather than discovering after downloading it.

## Balancing the training set

Oversampling alone (see main README) moved car from 83.7%→79.7%. Combining
it with undersampling car-only images (`train/undersample_car.py`,
`--remove-fraction 0.5`) pushed it further, confirmed real result:

| Class | Original | Oversampled only | Oversampled + Undersampled |
|---|---|---|---|
| car | 83.7% | 79.7% | 78.7% |
| van | 9.3% | 9.7% | 10.1% |
| bus | 6.4% | 8.5% | 8.9% |
| others | 0.7% | 2.1% | 2.2% |

Undersampling's ceiling is limited by data structure, not the technique
itself: only 16% of images are pure car-only (25,896 of 161,224 after
oversampling) — the rest of car's presence comes from co-occurring with
other classes in the same image, which undersampling deliberately never
touches (removing those would also remove the rare-class instances this
whole effort is trying to preserve).

For a genuinely different lever (real data instead of more duplication/
removal), `train/add_test_split_rare_data.py` merges real rare-class
images from UA-DETRAC's official **test** split — see the main README's
"Fine-tune on UA-DETRAC" section for the honest caveat about benchmark
comparability this introduces.

## Verifying class distribution after conversion

```python
from collections import Counter
import glob

counts = Counter()
for filepath in glob.glob("<output_dir>/train/labels/*.txt"):
    with open(filepath) as f:
        for line in f:
            if line.strip():
                counts[int(line.split()[0])] += 1

names = ["car", "bus", "van", "others"]
total = sum(counts.values())
for class_id, count in sorted(counts.items()):
    print(f"{names[class_id]}: {count:,} boxes ({100*count/total:.1f}%)")
```

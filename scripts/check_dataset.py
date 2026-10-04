"""Validate the YOLO dataset in dataset/ and print a report.

Usage (from the project root, with .venv activated):
    python scripts/check_dataset.py
    python scripts/check_dataset.py --data dataset/data.yaml
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_LIST = 10


def main():
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"), help="path to data.yaml")
    args = parser.parse_args()

    errors, warnings = [], []
    data_yaml = Path(args.data).resolve()

    print("=" * 50)
    print(" PPE DATASET CHECK")
    print("=" * 50)

    # data.yaml
    if not data_yaml.parent.is_dir():
        print(f"[X] Dataset directory not found: {data_yaml.parent}")
        return 1
    print(f"[OK] Dataset directory: {data_yaml.parent}")
    if not data_yaml.is_file():
        print(f"[X] data.yaml not found: {data_yaml}")
        return 1
    print(f"[OK] data.yaml: {data_yaml}")

    try:
        import yaml
    except ImportError:
        print("[X] PyYAML is not installed. Activate .venv and run: pip install pyyaml")
        return 1
    try:
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print(f"[X] data.yaml is not valid YAML: {e}")
        return 1

    names = cfg.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not names:
        print("[X] data.yaml has no 'names' (class list).")
        return 1
    names = {int(k): str(v) for k, v in names.items()}

    root = Path(cfg["path"]) if cfg.get("path") else data_yaml.parent
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve() if (data_yaml.parent / root).exists() else root.resolve()
    print(f"[OK] Dataset root: {root}")

    print("\nClasses (from data.yaml):")
    for i in sorted(names):
        print(f"  {i:<3} {names[i]}")

    image_counts, label_counts = {}, {}
    class_counts = Counter()
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            if split in ("train", "val"):
                errors.append(f"data.yaml has no '{split}' entry.")
            continue
        img_dir = (root / rel).resolve()
        # YOLO convention: .../images/<split> -> .../labels/<split>
        parts = list(img_dir.parts)
        if "images" not in parts:
            errors.append(f"{split}: image folder path must contain 'images': {img_dir}")
            continue
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        lbl_dir = Path(*parts)

        if not img_dir.is_dir():
            (errors if split in ("train", "val") else warnings).append(f"{split}: image folder missing: {img_dir}")
            continue
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}
        image_counts[split], label_counts[split] = len(images), len(labels)

        if not images:
            (errors if split in ("train", "val") else warnings).append(f"{split}: no images in {img_dir}")
        if not lbl_dir.is_dir():
            errors.append(f"{split}: label folder missing: {lbl_dir}")
        elif not labels:
            errors.append(f"{split}: no YOLO .txt label files in {lbl_dir}")

        no_label = sorted(set(images) - set(labels))
        no_image = sorted(set(labels) - set(images))
        if no_label:
            warnings.append(f"{split}: {len(no_label)} image(s) without a label file (treated as background): "
                            + ", ".join(images[s].name for s in no_label[:MAX_LIST]) + (" ..." if len(no_label) > MAX_LIST else ""))
        if no_image:
            warnings.append(f"{split}: {len(no_image)} label file(s) without an image (ignored by YOLO): "
                            + ", ".join(labels[s].name for s in no_image[:MAX_LIST]) + (" ..." if len(no_image) > MAX_LIST else ""))

        bad_lines, bad_ids, empty = [], Counter(), 0
        for stem, lbl in labels.items():
            lines = [ln.split() for ln in lbl.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for n, vals in enumerate(lines, 1):
                try:
                    cls = int(vals[0])
                    coords = [float(v) for v in vals[1:]]
                except (ValueError, IndexError):
                    bad_lines.append(f"{lbl.name}:{n}")
                    continue
                if len(coords) != 4 and (len(coords) < 6 or len(coords) % 2):
                    bad_lines.append(f"{lbl.name}:{n}")
                    continue
                if any(c < 0 or c > 1.0001 for c in coords):
                    bad_lines.append(f"{lbl.name}:{n} (coords not normalized 0-1)")
                if cls not in names:
                    bad_ids[cls] += 1
                class_counts[cls] += 1
        if empty:
            warnings.append(f"{split}: {empty} empty label file(s) (images with no objects).")
        if bad_lines:
            errors.append(f"{split}: {len(bad_lines)} malformed label line(s): " + ", ".join(bad_lines[:MAX_LIST]))
        for cls, cnt in bad_ids.items():
            errors.append(f"{split}: class ID {cls} used {cnt} time(s) but not defined in data.yaml")

    print("\nImages:")
    for s, n in image_counts.items():
        print(f"  {s.capitalize():<11} {n}")
    print("\nLabels:")
    for s, n in label_counts.items():
        print(f"  {s.capitalize():<11} {n}")

    print("\nClass IDs used in labels (all splits):")
    if class_counts:
        for cls in sorted(class_counts):
            print(f"  {cls:<3} {names.get(cls, '??? NOT IN data.yaml'):<12} {class_counts[cls]} boxes")
        unused = sorted(set(names) - set(class_counts))
        if unused:
            warnings.append("classes defined but never used: " + ", ".join(f"{i} {names[i]}" for i in unused))
    else:
        print("  (none)")

    print("\nWarnings:" if warnings else "\nWarnings: none")
    for w in warnings:
        print(f"  [!] {w}")
    print("\nErrors:" if errors else "\nErrors: none")
    for e in errors:
        print(f"  [X] {e}")

    print("\n" + "=" * 50)
    if not image_counts:
        print(" RESULT: DATASET NOT FOUND - run: python scripts/download_dataset.py")
    elif errors:
        print(" RESULT: DATASET HAS ERRORS - fix them before training")
    else:
        print(" RESULT: DATASET VALID - ready for training")
    print("=" * 50)
    return 1 if errors or not image_counts else 0


if __name__ == "__main__":
    sys.exit(main())

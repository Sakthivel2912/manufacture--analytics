"""Download the official Ultralytics Construction-PPE dataset into dataset/.

Usage (from the project root, with .venv activated):
    python scripts/download_dataset.py

The project's own dataset/data.yaml is kept (the copy inside the zip is skipped).
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
ZIP_PATH = DATASET_DIR / "construction-ppe.zip"


def progress(block_num, block_size, total_size):
    if total_size > 0:
        done = min(block_num * block_size, total_size)
        pct = done * 100 / total_size
        sys.stdout.write(f"\r  Downloading... {pct:5.1f}% ({done / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
        sys.stdout.flush()


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if (DATASET_DIR / "images" / "train").is_dir() and any((DATASET_DIR / "images" / "train").iterdir()):
        print("Dataset already present in dataset/images/train. Delete dataset/images and dataset/labels to re-download.")
        return 0

    print(f"Source: {URL}")
    try:
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=progress)
        print()
    except Exception as e:
        print(f"\nERROR: download failed: {e}")
        print("Download the zip manually from the URL above, save it as dataset/construction-ppe.zip and re-run.")
        return 1

    print("Extracting...")
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            for member in zf.infolist():
                if Path(member.filename).name == "data.yaml":
                    continue  # keep our own dataset/data.yaml
                zf.extract(member, DATASET_DIR)
    except zipfile.BadZipFile:
        print("ERROR: the downloaded file is not a valid zip. Delete dataset/construction-ppe.zip and try again.")
        return 1

    ZIP_PATH.unlink(missing_ok=True)
    print("Done. Now run: python scripts/check_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

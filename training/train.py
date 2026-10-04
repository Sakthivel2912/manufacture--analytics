"""Train a lightweight YOLO PPE detection model.

Usage (from the project root, with .venv activated):
    python training/train.py                       # uses the settings below
    python training/train.py --epochs 50 --batch 4 # override settings

Results are saved to runs/detect/ppe_training/ (best model: weights/best.pt).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ===================== EASY-TO-EDIT SETTINGS =====================
MODEL = "yolo26n.pt"  # YOLO26 nano (smallest/fastest). Alternative: "yolo11n.pt"
DATA = "dataset/data.yaml"
EPOCHS = 30
IMG_SIZE = 640  # lower to 480 or 416 if training is too slow / out of memory
BATCH = 8  # reduce to 4 or 2 if you get "out of memory" errors
WORKERS = 2  # data-loading processes; set to 0 if you get errors on Windows
DEVICE = None  # None = auto (NVIDIA GPU if available, else CPU). Or "cpu", or 0 for first GPU
PATIENCE = 10  # stop early if validation doesn't improve for this many epochs
RUN_NAME = "ppe_training"
# =================================================================


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO PPE detector.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of training images to use (e.g. 0.05 for a quick smoke test)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        print(f"ERROR: dataset config not found: {data_path}")
        print("Download the dataset first: python scripts/download_dataset.py")
        return 1
    if not (data_path.parent / "images" / "train").is_dir():
        print("ERROR: training images not found in dataset/images/train")
        print("Download the dataset first: python scripts/download_dataset.py")
        return 1

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: {e}. Did you activate .venv and install requirements?")
        return 1

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model} | Device: {'GPU (CUDA)' if device != 'cpu' else 'CPU'} | "
          f"Epochs: {args.epochs} | Image size: {args.imgsz} | Batch: {args.batch}")

    # Pretrained weights are downloaded automatically into models/ on first use.
    model_path = Path(args.model)
    if not model_path.is_absolute() and model_path.parent == Path("."):
        (ROOT / "models").mkdir(exist_ok=True)
        model_path = ROOT / "models" / model_path.name

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        patience=PATIENCE,
        fraction=args.fraction,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print("\nTraining finished.")
    print(f"Best model: {best}" if best.exists() else "WARNING: best.pt was not created.")
    print("Test it with: python scripts/test_model.py --image path/to/image.jpg")
    return 0


if __name__ == "__main__":  # required on Windows (multiprocessing data loaders)
    sys.exit(main())

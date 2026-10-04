"""Run the trained PPE model on one image.

Usage (from the project root, with .venv activated):
    python scripts/test_model.py --image test.jpg
    python scripts/test_model.py --image test.jpg --conf 0.4 --show
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "runs" / "detect" / "ppe_training" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Test the trained PPE detection model on an image.")
    parser.add_argument("--image", required=True, help="path to an image file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="path to trained weights (best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (0-1)")
    parser.add_argument("--show", action="store_true", help="also open a window with the result")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print("Trained model not found. Please train the model first.")
        print(f"(Expected: {model_path})")
        return 1

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 1

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: {e}. Did you activate .venv and install requirements?")
        return 1

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read the image (unsupported or corrupted file): {image_path}")
        return 1

    try:
        model = YOLO(str(model_path))
        result = model.predict(image, conf=args.conf, verbose=False)[0]
    except Exception as e:
        print(f"Inference failed: {e}")
        return 1

    print(f"\nDetections in {image_path.name} (conf >= {args.conf}):")
    if len(result.boxes) == 0:
        print("  No objects detected.")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        print(f"  {result.names[cls_id]:<12} {float(box.conf[0]):.2f}")

    out_dir = ROOT / "runs" / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_result.jpg"
    annotated = result.plot()
    cv2.imwrite(str(out_path), annotated)
    print(f"\nSaved result image: {out_path}")

    if args.show:
        try:
            cv2.imshow("PPE detection (press any key to close)", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Could not open a display window; open the saved image instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

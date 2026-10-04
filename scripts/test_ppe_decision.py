
import argparse
from pathlib import Path
import sys
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ppe_decision import evaluate_person_ppe, CONFIDENCE_THRESHOLD
from src.ppe_annotate import draw_decisions

DEFAULT_MODEL = ROOT / "runs" / "detect" / "ppe_training" / "weights" / "best.pt"

def run(image_path, model_path=DEFAULT_MODEL, conf=CONFIDENCE_THRESHOLD):
    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), conf=conf, verbose=False)

    result = results[0]
    names = result.names
    detections = []
    person_boxes = []

    if result.boxes is not None:
        for b in result.boxes:
            cls_id = int(b.cls[0].item())
            score = float(b.conf[0].item())
            box = [float(x) for x in b.xyxy[0].tolist()]
            name = names[cls_id]
            detections.append({"name": name, "confidence": score, "box": box})
            if name == "Person":
                person_boxes.append(box)

    decisions = []
    for i, pbox in enumerate(person_boxes, 1):
        pdet = next((d for d in detections if d["name"]=="Person" and d["box"]==pbox), None)
        d = evaluate_person_ppe(pbox, detections, pdet["confidence"] if pdet else 0.0)
        decisions.append(d)

        print(f"\nPERSON {i} (confidence {d.person_confidence:.2f})")
        for item in ("helmet","vest","boots"):
            x = d.items[item]
            mark = "DETECTED" if x.detected else "MISSING"
            extra = f" ({x.confidence:.2f})" if x.detected else ""
            print(f"  {item.title():<7} : {mark}{extra}")
        print("RESULT:", d.message)
        if d.missing:
            print("Missing:", ", ".join(x.title() for x in d.missing))

    if not person_boxes:
        print("\nNO PERSON DETECTED - ACCESS DENIED")

    annotated = draw_decisions(result.orig_img.copy(), person_boxes, decisions)
    out_dir = ROOT / "runs" / "ppe_decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(image_path).stem + "_ppe.jpg")
    cv2.imwrite(str(out), annotated)
    print("\nSaved:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=False)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    args = ap.parse_args()

    if not args.image:
        ap.error("--image is required for this test. Example: --image .\\dataset\\images\\train\\image100.jpg")
    image = Path(args.image)
    if not image.exists():
        ap.error(f"Image not found: {image}")
    model = Path(args.model)
    if not model.exists():
        ap.error(f"Model not found: {model}")
    run(image, model, args.conf)


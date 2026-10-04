# Creates the PPE decision-engine files in the current AI-PPE-Safety-System project.
# Refuses to overwrite existing files.

$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
if (-not (Test-Path (Join-Path $root "dataset")) -or
    -not (Test-Path (Join-Path $root "requirements.txt"))) {
    Write-Error "Run this script from the AI-PPE-Safety-System project root."
    exit 1
}

$files = @{}
$files["src/__init__.py"] = @'
"""PPE safety decision package."""
'@
$files["src/ppe_decision.py"] = @'

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# Only these three PPE items determine access.
REQUIRED = ("helmet", "vest", "boots")
CONFIDENCE_THRESHOLD = 0.25

# Vertical position bands relative to the person's bounding box.
BANDS = {
    "helmet": (-0.10, 0.35),
    "vest": (0.12, 0.75),
    "boots": (0.60, 1.10),
}

@dataclass
class PPEItem:
    name: str
    detected: bool
    confidence: float = 0.0

@dataclass
class PersonDecision:
    person_confidence: float
    allowed: bool
    items: Dict[str, PPEItem]
    missing: List[str] = field(default_factory=list)

    @property
    def message(self):
        return "ACCESS ALLOWED" if self.allowed else "ACCESS DENIED"

def _xyxy(box):
    return tuple(float(x) for x in box)

def _center(box):
    x1, y1, x2, y2 = _xyxy(box)
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def _area(box):
    x1, y1, x2, y2 = _xyxy(box)
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def _intersection(a, b):
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    x1, y1 = max(ax1,bx1), max(ay1,by1)
    x2, y2 = min(ax2,bx2), min(ay2,by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2-x1)*(y2-y1)

def _fits(person_box, item_box, item):
    px1, py1, px2, py2 = _xyxy(person_box)
    cx, cy = _center(item_box)
    pw = max(1.0, px2-px1)
    ph = max(1.0, py2-py1)

    # Horizontal centre must be inside the person with 10% slack.
    if cx < px1 - 0.10*pw or cx > px2 + 0.10*pw:
        return False

    rel_y = (cy-py1)/ph
    lo, hi = BANDS[item]
    if rel_y < lo or rel_y > hi:
        return False

    # At least 50% of the PPE box must overlap the person.
    overlap = _intersection(person_box, item_box)
    if overlap / max(_area(item_box), 1.0) < 0.50:
        return False
    return True

def evaluate_person_ppe(person_box, detections, person_confidence=0.0):
    """
    detections: iterable of dicts with keys class/name, confidence/conf, box.
    Only helmet, vest and boots affect the decision.
    """
    best = {k: None for k in REQUIRED}
    for d in detections:
        name = d.get("name", d.get("class"))
        if name not in REQUIRED:
            continue
        conf = float(d.get("confidence", d.get("conf", 0.0)))
        if conf < CONFIDENCE_THRESHOLD:
            continue
        box = d["box"]
        if not _fits(person_box, box, name):
            continue
        if best[name] is None or conf > best[name]["confidence"]:
            best[name] = {"confidence": conf, "box": box}

    items = {}
    missing = []
    for name in REQUIRED:
        hit = best[name]
        items[name] = PPEItem(name, hit is not None, hit["confidence"] if hit else 0.0)
        if hit is None:
            missing.append(name)

    return PersonDecision(
        person_confidence=float(person_confidence),
        allowed=len(missing) == 0,
        items=items,
        missing=missing,
    )

def evaluate_detections(detections):
    """Evaluate every Person detection in a frame."""
    persons = [d for d in detections
               if d.get("name", d.get("class")) == "Person"
               and float(d.get("confidence", d.get("conf", 0.0))) >= CONFIDENCE_THRESHOLD]
    results = []
    used = set()

    for p in persons:
        decision = evaluate_person_ppe(
            p["box"], detections,
            float(p.get("confidence", p.get("conf", 0.0)))
        )
        results.append(decision)
    return results

'@
$files["src/ppe_annotate.py"] = @'

import cv2

def draw_decisions(frame, person_boxes, decisions):
    """Draw person/PPE decision boxes and a simple status panel."""
    for i, (box, decision) in enumerate(zip(person_boxes, decisions), 1):
        x1,y1,x2,y2 = [int(v) for v in box]
        ok = decision.allowed
        color = (0, 180, 0) if ok else (0, 0, 220)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

        label = f"PERSON {i}: {'ALLOWED' if ok else 'DENIED'}"
        cv2.putText(frame, label, (x1, max(25,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y = y1 + 25
        for item in ("helmet","vest","boots"):
            status = "OK" if decision.items[item].detected else "MISSING"
            txt = f"{item.upper()}: {status}"
            cv2.putText(frame, txt, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 22
    return frame

'@
$files["scripts/test_ppe_decision.py"] = @'

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

'@
$files["tests/test_ppe_decision.py"] = @'

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ppe_decision import evaluate_person_ppe

def det(name, box, confidence=0.9):
    return {"name": name, "box": box, "confidence": confidence}

class TestPPEDecision(unittest.TestCase):
    person = [0,0,100,200]

    def test_all_required_allowed(self):
        ds = [
            det("helmet",[30,0,70,35]),
            det("vest",[20,45,80,130]),
            det("boots",[20,150,50,200]),
        ]
        r = evaluate_person_ppe(self.person, ds)
        self.assertTrue(r.allowed)
        self.assertEqual(r.missing, [])

    def test_missing_helmet_denied(self):
        ds = [det("vest",[20,45,80,130]), det("boots",[20,150,50,200])]
        r = evaluate_person_ppe(self.person, ds)
        self.assertFalse(r.allowed)
        self.assertIn("helmet", r.missing)

    def test_missing_vest_denied(self):
        ds = [det("helmet",[30,0,70,35]), det("boots",[20,150,50,200])]
        r = evaluate_person_ppe(self.person, ds)
        self.assertFalse(r.allowed)
        self.assertIn("vest", r.missing)

    def test_missing_boots_denied(self):
        ds = [det("helmet",[30,0,70,35]), det("vest",[20,45,80,130])]
        r = evaluate_person_ppe(self.person, ds)
        self.assertFalse(r.allowed)
        self.assertIn("boots", r.missing)

    def test_gloves_do_not_affect_decision(self):
        ds = [
            det("helmet",[30,0,70,35]),
            det("vest",[20,45,80,130]),
            det("boots",[20,150,50,200]),
        ]
        r = evaluate_person_ppe(self.person, ds)
        self.assertTrue(r.allowed)

    def test_low_confidence_counts_missing(self):
        ds = [
            det("helmet",[30,0,70,35],0.10),
            det("vest",[20,45,80,130]),
            det("boots",[20,150,50,200]),
        ]
        r = evaluate_person_ppe(self.person, ds)
        self.assertFalse(r.allowed)
        self.assertIn("helmet", r.missing)

if __name__ == "__main__":
    unittest.main()

'@

foreach ($rel in $files.Keys) {
    $target = Join-Path $root $rel
    if (Test-Path $target) {
        Write-Host "[SKIP] Already exists: $rel"
        continue
    }
    $dir = Split-Path $target -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Set-Content -Path $target -Value $files[$rel] -Encoding UTF8
    Write-Host "[CREATED] $rel"
}

Write-Host ""
Write-Host "Done. Existing files were not overwritten."
Write-Host "Next:"
Write-Host "  python -m unittest tests.test_ppe_decision -v"
Write-Host '  python scripts\test_ppe_decision.py --image ".\dataset\images\train\image100.jpg"'

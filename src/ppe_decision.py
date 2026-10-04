
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


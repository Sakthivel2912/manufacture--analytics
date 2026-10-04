
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


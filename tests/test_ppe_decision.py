
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


"""The shipped finger replays (held-out trials of the S01 finger models) must be consistent with their own metadata."""
import json
import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CASES = {("finger_2", "ME"): ["thumb", "pinky"], ("finger_2", "MI"): ["thumb", "pinky"],
         ("finger_3", "ME"): ["thumb", "index", "pinky"], ("finger_3", "MI"): ["thumb", "index", "pinky"]}


def test_finger_replays_are_consistent_and_served(client=None):
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        tasks = c.get("/api/tasks").json()
        for (task, mode), labels in CASES.items():
            path = os.path.join(DATA, f"replay_{task}_{mode}.json")
            if not os.path.exists(path):
                pytest.skip("finger replays not shipped")
            d = json.load(open(path))
            assert d["label_order"] == labels and task in tasks
            recs = d["records"]
            assert recs and all(set(r["probs"]) == set(labels) for r in recs[:200])
            assert abs(np.mean([r["true_class"] == r["predicted_class"] for r in recs]) - d["meta"]["per_window_accuracy"]) < 1e-9
            assert all(abs(sum(r["probs"].values()) - 1) < 1e-4 for r in recs[:200])
            assert "S01" in d["meta"]["training_note"] and "not the full training" in d["meta"]["training_note"]
            served = c.get(f"/api/replay/log/{task}?mode={mode}").json()
            assert served["label_order"] == labels and len(served["records"]) == len(recs)

"""The shipped finger replays (held-out trials of the S01 finger models) must be consistent with their own metadata."""
import gzip
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
            if not os.path.exists(path) and not os.path.exists(path + ".gz"):
                pytest.skip("finger replays not shipped")
            d = json.load(gzip.open(path + ".gz", "rt", encoding="utf-8") if os.path.exists(path + ".gz") else open(path))
            assert d["label_order"] == labels and task in tasks
            recs = d["records"]
            assert recs and all(set(r["probs"]) == set(labels) for r in recs[:200])
            assert abs(np.mean([r["true_class"] == r["predicted_class"] for r in recs]) - d["meta"]["per_window_accuracy"]) < 1e-9
            assert all(abs(sum(r["probs"].values()) - 1) < 1e-4 for r in recs[:200])
            assert "S01" in d["meta"]["training_note"] and "not the full training" in d["meta"]["training_note"]
            served = c.get(f"/api/replay/log/{task}?mode={mode}").json()
            assert served["label_order"] == labels and len(served["records"]) == len(recs)


def test_finger_replays_carry_real_signal_and_band_power():
    """The signal-export notebook adds full-rate traces and band power; they must be present, finite and aligned in shape."""
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        for (task, mode) in CASES:
            j = c.get(f"/api/replay/log/{task}?mode={mode}").json()
            sig = j["meta"]["signal"]
            assert sig["samples_per_window"] == 100 and sig["sampling_rate_hz"] == 100
            assert j["channel_names"] == ["C3", "CZ", "C4"] and "BioSemi" in str(sig["channel_mapping"])
            r = j["records"][len(j["records"]) // 2]
            assert len(r["waveform"]) == 3 and len(r["waveform"][0]) == 100
            assert set(r["bands"]) == {"theta", "mu", "beta", "gamma_low"} and len(r["bands"]["mu"]) == 3
            total = sum(r["bands"][b][0] for b in r["bands"])
            assert 0.99 < total < 1.01   # relative power of the 4-40 Hz band adds up to one
            assert "matched by ONNX probabilities" in sig["alignment"] or "ONNX reproduces" in sig["alignment"]

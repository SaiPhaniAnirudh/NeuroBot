"""Checks on the real exported flagship ONNX models shipped in checkpoints/ (skipped if absent)."""
import json
import os

import numpy as np
import pytest

import model_registry

REAL = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "manifest.json")
pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL) or not json.load(open(REAL)).get("models"), reason="no real models exported")

EXPECTED = {("left_right", "ME"): 0.8333, ("left_right", "MI"): 1.0,
            ("fists_feet", "ME"): 1.0, ("fists_feet", "MI"): 0.8889}


@pytest.fixture(scope="module")
def real():
    model_registry.load(REAL)
    yield
    model_registry.load()


def test_all_four_real_models_load(real):
    assert model_registry.load_errors() == [] and model_registry.n_models() == 4


@pytest.mark.parametrize("key", list(EXPECTED))
def test_manifest_matches_reported_accuracy(real, key):
    ev = model_registry.public_info(*key)["evaluation"]
    assert ev["n_heldout_trials"] == 18 and abs(ev["majority_vote_accuracy"] - EXPECTED[key]) < 1e-3
    assert ev["onnx_vs_torch_max_abs_diff"] < 1e-5


@pytest.mark.parametrize("key", list(EXPECTED))
def test_real_model_outputs_valid_probabilities(real, key):
    x = np.random.default_rng(0).standard_normal((5, 64, 160)).astype(np.float32)
    p = model_registry.predict(*key, x)
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1.0) and np.all(p >= 0)


def test_replay_logs_agree_with_manifest(real):
    here = os.path.join(os.path.dirname(REAL), "..", "data")
    for (task, mode), acc in EXPECTED.items():
        d = json.load(open(os.path.join(here, f"replay_{task}_{mode}.json")))
        assert d["label_order"] == model_registry.label_order(task, mode)
        assert abs(d["meta"]["majority_vote_accuracy"] - acc) < 1e-3
        recs = d["records"]
        w = np.mean([r["predicted_class"] == r["true_class"] for r in recs])
        assert abs(w - d["meta"]["per_window_accuracy"]) < 1e-6

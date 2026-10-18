import json
import os
import shutil

import numpy as np

import model_registry
from conftest import LABELS, build_manifest, make_tiny_onnx


def test_loads_all_models(tmp_root):
    path = build_manifest(os.path.join(tmp_root, "reg_ok"))
    model_registry.load(path)
    assert model_registry.n_models() == 4 and not model_registry.load_errors()
    assert model_registry.label_order("fists_feet", "ME") == LABELS["fists_feet"]
    p = model_registry.predict("left_right", "ME", np.random.default_rng(0).standard_normal((300, 64, 160)).astype(np.float32))
    assert p.shape == (300, 3) and np.allclose(p.sum(axis=1), 1.0)  # crosses the batch boundary (128)


def test_bad_entries_are_reported_and_good_ones_still_load(tmp_root):
    d = os.path.join(tmp_root, "reg_bad")
    path = build_manifest(d, combos=[("left_right", "ME")])
    manifest = json.load(open(path))
    good = manifest["models"][0]
    make_tiny_onnx(os.path.join(d, "wrong_k.onnx"), 2, 9)
    make_tiny_onnx(os.path.join(d, "dup.onnx"), 3, 9)
    manifest["models"] += [
        {**good, "id": "missing-file", "mode": "MI", "onnx_path": "nope.onnx"},
        {**good, "id": "bad-labels", "mode": "MI", "label_order": ["a", "b", "c"]},
        {**good, "id": "wrong-k", "mode": "MI", "onnx_path": "wrong_k.onnx"},
        {**good, "id": "traversal", "mode": "MI", "onnx_path": "../manifest.json"},
        {**good, "id": "bad-mode", "mode": "XX"},
        {**good, "id": "duplicate", "onnx_path": "dup.onnx"},
        {"id": "incomplete"},
    ]
    json.dump(manifest, open(path, "w"))
    model_registry.load(path)
    assert model_registry.n_models() == 1 and model_registry.has_model("left_right", "ME")
    errs = " | ".join(model_registry.load_errors())
    for name in ("missing-file", "bad-labels", "wrong-k", "traversal", "bad-mode", "duplicate", "incomplete"):
        assert name in errs
    model_registry.load(os.environ["NEUROBOT_MANIFEST"])


def test_missing_manifest_is_reported(tmp_root):
    model_registry.load(os.path.join(tmp_root, "does_not_exist", "manifest.json"))
    assert model_registry.n_models() == 0 and "manifest not found" in model_registry.load_errors()[0]
    model_registry.load(os.environ["NEUROBOT_MANIFEST"])


def test_public_info_hides_file_paths():
    model_registry.load(os.environ["NEUROBOT_MANIFEST"])
    assert "onnx_path" not in model_registry.public_info("left_right", "ME")

"""Shared fixtures: tiny ONNX models (built with the onnx helper API, no PyTorch needed) and a manifest."""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

LABELS = {
    "left_right": ["left", "rest", "right"],
    "fists_feet": ["both_feet", "both_fists", "rest"],  # deliberately not the 'natural' order
}


def make_tiny_onnx(path: str, n_classes: int, seed: int) -> None:
    """window (N, 64, 160) -> mean over time -> (N, 64) -> linear layer -> logits (N, K)."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((64, n_classes)) * 3.0).astype(np.float32)
    b = rng.standard_normal(n_classes).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["window"], ["m"], axes=[2], keepdims=0),
        helper.make_node("MatMul", ["m", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("window", TensorProto.FLOAT, ["batch", 64, 160])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_classes])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def build_manifest(directory: str, combos=None) -> str:
    os.makedirs(directory, exist_ok=True)
    combos = combos or [(t, m) for t in LABELS for m in ("ME", "MI")]
    entries = []
    for i, (task, mode) in enumerate(combos):
        mid = f"tiny-{task}-{mode.lower()}"
        make_tiny_onnx(os.path.join(directory, mid + ".onnx"), len(LABELS[task]), seed=i)
        entries.append({"id": mid, "task": task, "mode": mode, "arch": "tiny", "title": mid,
                        "label_order": LABELS[task], "onnx_path": mid + ".onnx", "output": "logits",
                        "evaluation": {"majority_vote_accuracy": 0.5, "n_heldout_trials": 18}})
    path = os.path.join(directory, "manifest.json")
    with open(path, "w") as fh:
        json.dump({"models": entries}, fh)
    return path


_TMP = tempfile.mkdtemp(prefix="neurobot_test_")
os.environ["NEUROBOT_MANIFEST"] = build_manifest(os.path.join(_TMP, "full"))


@pytest.fixture(scope="session")
def tmp_root():
    return _TMP


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import main
    import model_registry
    model_registry.load(os.environ["NEUROBOT_MANIFEST"])
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_root):
    """Server with no models loaded."""
    from fastapi.testclient import TestClient
    import main
    import model_registry
    path = os.path.join(tmp_root, "empty", "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"models": []}, fh)
    old = model_registry.MANIFEST_PATH
    model_registry.MANIFEST_PATH = path
    try:
        with TestClient(main.app) as c:
            yield c
    finally:
        model_registry.MANIFEST_PATH = old
        model_registry.load(os.environ["NEUROBOT_MANIFEST"])


def window_payload(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((64, 160))
    w = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
    return w.tolist()

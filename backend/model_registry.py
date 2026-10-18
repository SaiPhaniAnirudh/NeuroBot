"""
Model registry for the NeuroBot backend.

Models are ONNX files described by checkpoints/manifest.json (written by the export
notebook, neurobot_export_flagship_onnx.ipynb). One specialist model per
(task, mode): task in {left_right, fists_feet}, mode in {ME, MI}.

The server never fakes output. A task/mode without a valid model is reported as
unavailable, and every problem found while loading is kept in `load_errors` and shown
by /api/health.
"""

import json
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

import preprocessing

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:  # pragma: no cover
    ort = None
    HAS_ORT = False

TASK_LABELS = {
    "left_right": {"left", "rest", "right"},
    "fists_feet": {"both_fists", "rest", "both_feet"},
}
MODES = ("ME", "MI")
MAX_BATCH = 128

MANIFEST_PATH = os.environ.get(
    "NEUROBOT_MANIFEST",
    os.path.join(os.path.dirname(__file__), "checkpoints", "manifest.json"),
)


class ModelUnavailable(Exception):
    pass


class _Loaded:
    def __init__(self, entry: dict, session, input_name: str, output_name: str):
        self.entry = entry
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.lock = threading.Lock()


_models: Dict[Tuple[str, str], _Loaded] = {}
_errors: List[str] = []
_loaded_from: Optional[str] = None


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_one(entry: dict, base_dir: str) -> _Loaded:
    for key in ("id", "task", "mode", "label_order", "onnx_path"):
        if key not in entry:
            raise ValueError(f"missing field '{key}'")
    task, mode, labels = entry["task"], entry["mode"], list(entry["label_order"])
    if task not in TASK_LABELS:
        raise ValueError(f"unknown task '{task}'")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'")
    if set(labels) != TASK_LABELS[task] or len(labels) != len(set(labels)):
        raise ValueError(f"label_order {labels} does not match task '{task}' ({sorted(TASK_LABELS[task])})")
    path = os.path.realpath(os.path.join(base_dir, entry["onnx_path"]))
    if os.path.commonpath([path, os.path.realpath(base_dir)]) != os.path.realpath(base_dir):
        raise ValueError("onnx_path points outside the checkpoints directory")
    if not os.path.isfile(path):
        raise ValueError(f"file not found: {entry['onnx_path']}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("NEUROBOT_ORT_THREADS", "2"))
    session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out_name = entry.get("output", session.get_outputs()[0].name)
    # smoke test: right shapes in, right number of classes out, finite values
    probe = np.zeros((2, preprocessing.N_CHANNELS, preprocessing.WINDOW), dtype=np.float32)
    logits = session.run([out_name], {inp.name: probe})[0]
    if logits.shape != (2, len(labels)):
        raise ValueError(f"model output has shape {logits.shape}, expected (2, {len(labels)})")
    if not np.all(np.isfinite(logits)):
        raise ValueError("model output is not finite on a zero input")
    return _Loaded(entry, session, inp.name, out_name)


def load(manifest_path: Optional[str] = None) -> None:
    """(Re)load every model in the manifest. Safe to call more than once."""
    global _loaded_from
    _models.clear()
    _errors.clear()
    path = manifest_path or MANIFEST_PATH
    _loaded_from = path
    if not HAS_ORT:
        _errors.append("onnxruntime is not installed")
        return
    if not os.path.exists(path):
        _errors.append(f"manifest not found: {os.path.basename(path)}")
        return
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        entries = manifest["models"]
    except Exception as exc:
        _errors.append(f"manifest unreadable: {exc}")
        return
    base_dir = os.path.dirname(os.path.abspath(path))
    for entry in entries:
        name = entry.get("id", "<no id>") if isinstance(entry, dict) else "<invalid entry>"
        try:
            loaded = _load_one(entry, base_dir)
            key = (entry["task"], entry["mode"])
            if key in _models:
                raise ValueError(f"duplicate model for {key[0]}/{key[1]} (already loaded: {_models[key].entry['id']})")
            _models[key] = loaded
        except Exception as exc:
            _errors.append(f"{name}: {exc}")


def load_errors() -> List[str]:
    return list(_errors)


def has_model(task: str, mode: str) -> bool:
    return (task, mode) in _models


def n_models() -> int:
    return len(_models)


def public_info(task: str, mode: str) -> dict:
    """Manifest entry without filesystem paths."""
    entry = dict(_get(task, mode).entry)
    entry.pop("onnx_path", None)
    return entry


def list_models() -> List[dict]:
    return [public_info(t, m) for (t, m) in sorted(_models)]


def label_order(task: str, mode: str) -> List[str]:
    return list(_get(task, mode).entry["label_order"])


def _get(task: str, mode: str) -> _Loaded:
    try:
        return _models[(task, mode)]
    except KeyError:
        raise ModelUnavailable(f"No model is loaded for task '{task}', mode '{mode}'.") from None


def predict(task: str, mode: str, windows: np.ndarray) -> np.ndarray:
    """windows: (n, 64, 160) preprocessed + z-scored. Returns softmax probabilities (n, K)."""
    m = _get(task, mode)
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (preprocessing.N_CHANNELS, preprocessing.WINDOW):
        raise ValueError(f"Expected windows of shape (n, 64, 160), got {tuple(windows.shape)}.")
    out = []
    for i in range(0, len(windows), MAX_BATCH):
        chunk = windows[i:i + MAX_BATCH]
        with m.lock:
            logits = m.session.run([m.output_name], {m.input_name: chunk})[0]
        out.append(_softmax(logits.astype(np.float64)))
    return np.concatenate(out, axis=0) if out else np.zeros((0, len(m.entry["label_order"])))

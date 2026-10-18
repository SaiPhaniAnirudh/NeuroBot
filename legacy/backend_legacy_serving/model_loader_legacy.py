"""
Loads the real trained EEGNet checkpoint (from the Kaggle run) once, at
import time. If the checkpoint is ever missing, is_model_loaded() returns
False and the API reports that clearly rather than silently faking output.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "models"))

from eegnet import EEGNet

CHECKPOINT_PATH = os.environ.get(
    "NEUROBOT_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "checkpoints", "eegnet.pt"),
)
LABEL_ORDER = ["left", "rest", "right"]  # must match the order used during training/export

_model = None
_meta = {}


def _load():
    global _model, _meta
    if _model is not None:
        return _model
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = EEGNet(
        n_channels=checkpoint["n_channels"],
        n_samples=checkpoint["n_samples"],
        n_classes=len(LABEL_ORDER),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _model = model
    _meta = {"n_channels": checkpoint["n_channels"], "n_samples": checkpoint["n_samples"]}
    return _model


def is_model_loaded() -> bool:
    return _load() is not None


def model_meta() -> dict:
    _load()
    return dict(_meta)


@torch.no_grad()
def predict(window) -> dict:
    """
    window: (n_channels, n_samples) array-like, already preprocessed and
    z-scored the same way as during training (see pipeline/data/preprocess.py).
    Returns {class_name: probability}.
    """
    model = _load()
    if model is None:
        raise RuntimeError(f"No checkpoint found at {CHECKPOINT_PATH}")
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1).numpy()[0]
    return {c: float(p) for c, p in zip(LABEL_ORDER, probs)}

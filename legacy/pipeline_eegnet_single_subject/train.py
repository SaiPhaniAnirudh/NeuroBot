"""
Training core for NeuroBot's EEGNet decoder.

Provides:
  - encode_labels / make_loader     : data plumbing
  - train_loop / evaluate           : basic train+eval with confusion matrix
  - run_subject_cv                  : 5-fold cross-validation per subject (matches base paper's methodology)
  - run_finetune_protocol           : base-model-then-fine-tune, matches base paper's Eq./protocol
  - save_checkpoint                 : writes the .pt file the (later) backend/UI will load
  - export_inference_log            : runs a trained model over ordered windows and dumps a
                                       JSON replay log — real predictions on real held-out EEG,
                                       consumed by ui/index.html (no backend needed for that yet)
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models.eegnet import EEGNet

LABEL_ORDER = ["left", "rest", "right"]


def encode_labels(y: np.ndarray) -> np.ndarray:
    return np.array([LABEL_ORDER.index(label) for label in y])


def make_loader(X, y, batch_size=64, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_loop(model, loader, optimizer, criterion, epochs, device, verbose=True):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch+1}/{epochs}  loss={total_loss/len(loader.dataset):.4f}")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())
    acc = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average=None, labels=list(range(len(LABEL_ORDER))), zero_division=0
    )
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(LABEL_ORDER))))
    return {"accuracy": acc, "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "confusion_matrix": cm.tolist()}


def run_subject_cv(X, y_raw, epochs=30, device="cpu", n_folds=5):
    """5-fold cross-validation on one subject's windows — matches the base
    paper's offline evaluation methodology."""
    y = encode_labels(y_raw)
    n_channels, n_samples = X.shape[1], X.shape[2]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/{n_folds}")
        model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=len(LABEL_ORDER)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_loader = make_loader(X[train_idx], y[train_idx])
        test_loader = make_loader(X[test_idx], y[test_idx], shuffle=False)

        train_loop(model, train_loader, optimizer, criterion, epochs, device, verbose=False)
        metrics = evaluate(model, test_loader, device)
        print(f"    fold accuracy: {metrics['accuracy']:.3f}")
        fold_results.append(metrics)

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    print(f"  Mean CV accuracy across {n_folds} folds: {mean_acc:.3f}")
    return fold_results, mean_acc


def run_finetune_protocol(X, y_raw, trial_ids, epochs=30, device="cpu"):
    """Base-model-then-fine-tune, matching the base paper: train on the
    first 70% of a session chronologically, fine-tune (block1 frozen) on
    part of the remainder, evaluate on held-out same-session data."""
    y = encode_labels(y_raw)
    y_raw = np.asarray(y_raw)
    trial_ids = np.asarray(trial_ids)
    n_channels, n_samples = X.shape[1], X.shape[2]

    split = int(len(X) * 0.7)
    X_base, y_base = X[:split], y[:split]
    X_late, y_late, y_late_raw, trial_late = X[split:], y[split:], y_raw[split:], trial_ids[split:]
    X_ft, X_eval, y_ft, y_eval, _, y_eval_raw, _, trial_eval = train_test_split(
        X_late, y_late, y_late_raw, trial_late, test_size=0.5, stratify=y_late, random_state=0
    )

    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=len(LABEL_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss()

    print("  Training base model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loop(model, make_loader(X_base, y_base), optimizer, criterion, epochs, device)
    base_metrics = evaluate(model, make_loader(X_eval, y_eval, shuffle=False), device)
    print(f"  Base model eval accuracy (per-window): {base_metrics['accuracy']:.3f}")

    print("  Fine-tuning (block1 frozen)...")
    trainable = model.frozen_finetune_parameters()
    ft_optimizer = torch.optim.Adam(trainable, lr=5e-4)
    train_loop(model, make_loader(X_ft, y_ft), ft_optimizer, criterion, max(5, epochs // 3), device)
    ft_metrics = evaluate(model, make_loader(X_eval, y_eval, shuffle=False), device)
    print(f"  Fine-tuned model eval accuracy (per-window): {ft_metrics['accuracy']:.3f}")

    mv_metrics = evaluate_majority_vote(model, X_eval, y_eval_raw, trial_eval, device)
    print(f"  Fine-tuned model eval accuracy (trial-level majority vote, {mv_metrics['n_trials']} trials): {mv_metrics['accuracy']:.3f}")

    return model, base_metrics, ft_metrics, mv_metrics, (X_eval, y_eval_raw, trial_eval)


@torch.no_grad()
def evaluate_majority_vote(model, X, y_raw, trial_ids, device) -> dict:
    """
    Trial-level majority-vote accuracy — the base paper's actual evaluation
    metric (its Methods section, citing Kuncheva 2014), not raw per-window
    accuracy. Each trial's final prediction is whichever class its
    constituent windows voted for most; a model can be wrong on individual
    noisy windows and still get the trial right.
    """
    model.eval()
    loader = make_loader(X, encode_labels(y_raw), shuffle=False)
    window_preds = []
    for xb, _ in loader:
        window_preds.extend(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    window_preds = np.array(window_preds)

    unique_trials = np.unique(trial_ids)
    trial_true, trial_pred = [], []
    for t in unique_trials:
        mask = trial_ids == t
        true_label = encode_labels(y_raw)[mask][0]  # all windows in a trial share the true label
        votes = window_preds[mask]
        winner = np.bincount(votes, minlength=len(LABEL_ORDER)).argmax()
        trial_true.append(true_label)
        trial_pred.append(winner)

    acc = accuracy_score(trial_true, trial_pred)
    cm = confusion_matrix(trial_true, trial_pred, labels=list(range(len(LABEL_ORDER))))
    return {"accuracy": acc, "n_trials": len(unique_trials), "confusion_matrix": cm.tolist()}


def save_checkpoint(model, n_channels, n_samples, path="./checkpoints/eegnet.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_channels": n_channels, "n_samples": n_samples}, path)
    print(f"Checkpoint saved to {path}")


@torch.no_grad()
def export_inference_log(model, X_eval, y_eval_raw, device="cpu",
                          waveform_channels=(8, 24, 40, 56), path="./inference_log.json"):
    """
    Runs the trained model over held-out windows *in order* and writes a
    JSON replay log: real predictions on real EEG, with a small waveform
    slice per window for the UI to plot. No synthetic data involved —
    this is the actual model's actual output.
    """
    model.eval()
    y_eval = encode_labels(y_eval_raw)
    records = []
    for i in range(len(X_eval)):
        window = X_eval[i]  # (n_channels, n_samples)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        records.append({
            "t": i * 0.125,
            "true_class": LABEL_ORDER[y_eval[i]],
            "predicted_class": LABEL_ORDER[pred_idx],
            "probs": {LABEL_ORDER[c]: float(probs[c]) for c in range(len(LABEL_ORDER))},
            "waveform": [window[ch, ::4].round(3).tolist() for ch in waveform_channels],  # downsampled for size
        })
    with open(path, "w") as f:
        json.dump({"label_order": LABEL_ORDER, "channels_shown": list(waveform_channels), "records": records}, f)
    print(f"Inference log written to {path} ({len(records)} windows)")

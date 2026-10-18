import sys, os
import numpy as np
import mne
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from data.preprocess import preprocess_raw, epoch_run, sliding_windows
from train import (run_subject_cv, run_finetune_protocol, evaluate_majority_vote,
                    save_checkpoint, export_inference_log, LABEL_ORDER)

SFREQ = 160.0
N_CHANNELS = 64
TRIAL_SEC = 4.0
N_TRIAL_PAIRS = 20


def make_synthetic_raw(seed=0):
    rng = np.random.default_rng(seed)
    ch_names = [f"EEG{i:03d}" for i in range(N_CHANNELS)]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    total_sec = N_TRIAL_PAIRS * 2 * TRIAL_SEC
    n_samples = int(total_sec * SFREQ)
    data = rng.normal(0, 1e-5, size=(N_CHANNELS, n_samples))

    onsets, durations, descriptions = [], [], []
    t = 0.0
    for i in range(N_TRIAL_PAIRS):
        onsets.append(t); durations.append(TRIAL_SEC); descriptions.append("T0")
        t += TRIAL_SEC
        label = "T1" if i % 2 == 0 else "T2"
        onsets.append(t); durations.append(TRIAL_SEC); descriptions.append(label)
        start = int(t * SFREQ); end = start + int(TRIAL_SEC * SFREQ)
        bump = slice(0, N_CHANNELS // 2) if label == "T1" else slice(N_CHANNELS // 2, N_CHANNELS)
        osc = np.sin(2 * np.pi * 10 * np.arange(end - start) / SFREQ) * 3e-5
        data[bump, start:end] += osc
        t += TRIAL_SEC

    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))
    return raw


def main():
    print("=== Building synthetic recording + preprocessing ===")
    raw = preprocess_raw(make_synthetic_raw())
    epochs = epoch_run(raw, run_number=3)
    X, y, trial_ids = sliding_windows(epochs, sfreq=raw.info["sfreq"])
    print(f"X: {X.shape}, classes: {np.unique(y, return_counts=True)}, unique trials: {len(np.unique(trial_ids))}")
    assert len(X) == len(y) == len(trial_ids)

    print("\n=== 5-fold cross-validation ===")
    fold_results, mean_acc = run_subject_cv(X, y, epochs=5, n_folds=3)
    assert 0.0 <= mean_acc <= 1.0

    print("\n=== Fine-tuning protocol + majority-vote accuracy ===")
    model, base_m, ft_m, mv_m, (X_eval, y_eval, trial_eval) = run_finetune_protocol(X, y, trial_ids, epochs=5)
    assert "confusion_matrix" in base_m and "confusion_matrix" in ft_m
    assert "accuracy" in mv_m and mv_m["n_trials"] > 0
    assert mv_m["n_trials"] <= len(np.unique(trial_eval))
    print(f"Per-window accuracy: {ft_m['accuracy']:.3f}  |  Majority-vote (trial-level) accuracy: {mv_m['accuracy']:.3f}")

    print("\n=== Checkpoint export ===")
    save_checkpoint(model, X.shape[1], X.shape[2], path="/tmp/neurobot_test_checkpoint.pt")
    assert os.path.exists("/tmp/neurobot_test_checkpoint.pt")

    print("\n=== Inference log export (for the UI) ===")
    export_inference_log(model, X_eval, y_eval, path="/tmp/neurobot_test_inference_log.json")
    assert os.path.exists("/tmp/neurobot_test_inference_log.json")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

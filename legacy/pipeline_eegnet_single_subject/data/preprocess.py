"""
Preprocessing & epoching for eegmmidb.

Common-average reference + broadband filter (4-40Hz, covering mu+beta),
then sliding-window segmentation matching the base paper's approach
(1s window, 125ms step) so a trained model can be queried every 125ms
during live decoding later.
"""

import numpy as np
import mne

from task_map import run_family, labels_for_run

WINDOW_SEC = 1.0
STEP_SEC = 0.125
TRIAL_TMIN = 0.0
TRIAL_TMAX = 4.0  # eegmmidb cue length


def load_raw(edf_path: str) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels(lambda ch: ch.strip("."))  # eegmmidb channel names have trailing dots
    return raw


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy()
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(l_freq=4.0, h_freq=40.0, fir_design="firwin", verbose=False)
    return raw


def epoch_run(raw: mne.io.BaseRaw, run_number: int) -> mne.Epochs:
    label_map = labels_for_run(run_number)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    class_event_id = {label_map[code]: eid for code, eid in event_id.items() if code in label_map}
    epochs = mne.Epochs(
        raw, events, event_id=class_event_id,
        tmin=TRIAL_TMIN, tmax=TRIAL_TMAX,
        baseline=None, preload=True, verbose=False,
    )
    return epochs


def sliding_windows(epochs: mne.Epochs, sfreq: float):
    """Returns X: (n_windows, n_channels, n_samples), y: (n_windows,) string labels,
    trial_ids: (n_windows,) int — which source trial each window came from, so
    predictions can later be grouped back into per-trial majority votes
    (the base paper's actual evaluation metric, not raw per-window accuracy)."""
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    id_to_label = {v: k for k, v in epochs.event_id.items()}

    win_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)

    X, y, trial_ids = [], [], []
    for trial_idx in range(data.shape[0]):
        trial = data[trial_idx]
        label = id_to_label[labels[trial_idx]]
        n_times = trial.shape[1]
        start = 0
        while start + win_samples <= n_times:
            window = trial[:, start:start + win_samples]
            window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
            X.append(window)
            y.append(label)
            trial_ids.append(trial_idx)
            start += step_samples

    return np.stack(X), np.array(y), np.array(trial_ids)


def process_run(edf_path: str, run_number: int):
    if run_family(run_number) == "baseline":
        raise ValueError(f"Run {run_number} is a baseline run; skip it in the caller")
    raw = load_raw(edf_path)
    raw = preprocess_raw(raw)
    epochs = epoch_run(raw, run_number)
    X, y, trial_ids = sliding_windows(epochs, sfreq=raw.info["sfreq"])
    return X, y, trial_ids

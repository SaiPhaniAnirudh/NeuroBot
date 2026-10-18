"""
Serving-time preprocessing for NeuroBot.

This must match the training pipeline exactly, otherwise the deployed models see
a different input distribution than the one they were trained and evaluated on:

    1. common average reference over the 64 EEG channels
    2. 4-40 Hz zero-phase FIR band-pass (the same filter MNE's raw.filter(4, 40,
       fir_design="firwin") builds at 160 Hz: 265 taps, Hamming window, -6 dB points
       at 3 Hz and 45 Hz; the taps are stored in fir_bandpass_160hz.npy so the server
       does not need MNE for this step)
    3. 1 s windows (160 samples) every 125 ms (20 samples)
    4. per-window, per-channel z-score  (paper eq. 3, epsilon = 1e-8)

tests/test_preprocessing.py checks the stored taps against MNE whenever MNE is installed.
"""

import os
from fractions import Fraction

import numpy as np
import scipy.signal

SFREQ = 160.0
N_CHANNELS = 64
WINDOW = 160
STRIDE = 20
EPS = 1e-8

_TAPS = np.load(os.path.join(os.path.dirname(__file__), "fir_bandpass_160hz.npy")).astype(np.float64)
_PAD = len(_TAPS) // 2

# PhysioNet eegmmidb channel order (the order the models were trained on)
STANDARD_64_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8",
    "T7", "T8", "T9", "T10",
    "TP7", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "IZ",
]
assert len(STANDARD_64_CHANNELS) == N_CHANNELS and len(set(STANDARD_64_CHANNELS)) == N_CHANNELS

# Channels drawn on the waveform display. Indices and names are derived from the
# standard order so the labels cannot drift away from the data.
DISPLAY_CHANNEL_NAMES = ["C3", "CZ", "C4", "FC3", "FC4", "CPZ"]
DISPLAY_CHANNELS = [STANDARD_64_CHANNELS.index(n) for n in DISPLAY_CHANNEL_NAMES]


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """x: (channels, samples)."""
    return x - x.mean(axis=0, keepdims=True)


def bandpass_4_40(x: np.ndarray) -> np.ndarray:
    """Zero-phase 4-40 Hz FIR band-pass at 160 Hz, edge handling as in MNE (odd reflection)."""
    if x.shape[1] <= _PAD:
        raise ValueError(f"Recording too short to filter ({x.shape[1]} samples).")
    padded = np.pad(x.astype(np.float64), ((0, 0), (_PAD, _PAD)), mode="reflect", reflect_type="odd")
    y = scipy.signal.oaconvolve(padded, _TAPS[None, :], mode="same", axes=1)
    return y[:, _PAD:-_PAD]


def resample_to_160(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Polyphase resampling to 160 Hz (only used for uploads recorded at another rate)."""
    if abs(sfreq - SFREQ) < 1e-6:
        return x
    ratio = Fraction(SFREQ / sfreq).limit_denominator(1000)
    return scipy.signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=1)


def preprocess_recording(x: np.ndarray, sfreq: float) -> np.ndarray:
    """(64, samples) raw recording -> CAR + 4-40 Hz at 160 Hz, float32."""
    if x.ndim != 2 or x.shape[0] != N_CHANNELS:
        raise ValueError(f"Expected (64, samples), got {tuple(x.shape)}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Recording contains NaN or infinite values.")
    x = resample_to_160(x.astype(np.float64), sfreq)
    return bandpass_4_40(common_average_reference(x)).astype(np.float32)


def zscore_windows(w: np.ndarray) -> np.ndarray:
    """w: (n, channels, samples); per-window, per-channel z-score."""
    mean = w.mean(axis=2, keepdims=True)
    std = w.std(axis=2, keepdims=True)
    return ((w - mean) / (std + EPS)).astype(np.float32)


def make_windows(x: np.ndarray):
    """(64, samples) preprocessed recording -> (starts, windows (n, 64, 160)), z-scored."""
    n = x.shape[1]
    if n < WINDOW:
        raise ValueError(f"Recording is too short ({n} samples at 160 Hz); at least {WINDOW} samples (1 s) are needed.")
    starts = np.arange(0, n - WINDOW + 1, STRIDE)
    idx = starts[:, None] + np.arange(WINDOW)[None, :]
    windows = x[:, idx].transpose(1, 0, 2)  # (n, channels, WINDOW)
    return starts, zscore_windows(windows)

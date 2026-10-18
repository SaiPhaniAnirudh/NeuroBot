"""
NeuroBot Preprocessing
======================
EEG signal preprocessing pipeline: Common Average Reference,
bandpass filtering, normalization, and artifact rejection.

Follows the Ding et al. (2025) protocol adapted for 64-channel
160 Hz EEGMMIDB data (vs. their 128-channel 1024 Hz BioSemi setup).
"""

import numpy as np
from scipy.signal import butter, filtfilt
import mne


def apply_car(X: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to EEG epochs.
    
    Subtracts the mean across channels at each time point,
    removing global noise sources shared across all electrodes.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n_epochs, n_channels, n_timepoints).
    
    Returns
    -------
    np.ndarray
        CAR-referenced data, same shape.
    """
    # Mean across channels (axis=1), subtract from each channel
    mean_signal = X.mean(axis=1, keepdims=True)
    return X - mean_signal


def bandpass_filter(X: np.ndarray, sfreq: float = 160.0,
                    low: float = 8.0, high: float = 30.0,
                    order: int = 5) -> np.ndarray:
    """
    Apply bandpass Butterworth filter to EEG epochs.
    
    Default 8–30 Hz captures mu (8–13 Hz) and beta (13–30 Hz) bands,
    which carry the strongest motor imagery/execution information.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n_epochs, n_channels, n_timepoints).
    sfreq : float
        Sampling frequency in Hz.
    low, high : float
        Bandpass cutoff frequencies in Hz.
    order : int
        Butterworth filter order.
    
    Returns
    -------
    np.ndarray
        Filtered data, same shape.
    """
    nyquist = sfreq / 2.0
    low_n = low / nyquist
    high_n = high / nyquist
    
    # Clamp to valid range
    low_n = max(low_n, 0.001)
    high_n = min(high_n, 0.999)
    
    b, a = butter(order, [low_n, high_n], btype='band')
    
    # Apply zero-phase filter along time axis
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_filtered[i, j] = filtfilt(b, a, X[i, j])
    
    return X_filtered


def bandpass_filter_broadband(X: np.ndarray, sfreq: float = 160.0,
                               low: float = 4.0, high: float = 40.0,
                               order: int = 5) -> np.ndarray:
    """
    Broadband 4–40 Hz filter (paper found broadband > single-band).
    Same interface as bandpass_filter().
    """
    return bandpass_filter(X, sfreq=sfreq, low=low, high=high, order=order)


def normalize_epochs(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize EEG epochs.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n_epochs, n_channels, n_timepoints).
    method : str
        'zscore' — per-channel, per-epoch z-score normalization
        'minmax' — per-channel, per-epoch min-max to [0, 1]
        'global_zscore' — z-score using global (all-epoch) statistics
    
    Returns
    -------
    np.ndarray
        Normalized data, same shape.
    """
    X_norm = np.zeros_like(X)
    
    if method == 'zscore':
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                mean = X[i, j].mean()
                std = X[i, j].std()
                if std < 1e-8:
                    X_norm[i, j] = 0.0
                else:
                    X_norm[i, j] = (X[i, j] - mean) / std
    
    elif method == 'minmax':
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                vmin = X[i, j].min()
                vmax = X[i, j].max()
                rng = vmax - vmin
                if rng < 1e-8:
                    X_norm[i, j] = 0.0
                else:
                    X_norm[i, j] = (X[i, j] - vmin) / rng
    
    elif method == 'global_zscore':
        # Compute statistics across ALL epochs for each channel
        for j in range(X.shape[1]):
            all_vals = X[:, j, :].flatten()
            mean = all_vals.mean()
            std = all_vals.std()
            if std < 1e-8:
                X_norm[:, j, :] = 0.0
            else:
                X_norm[:, j, :] = (X[:, j, :] - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm


def reject_artifacts(X: np.ndarray, y: np.ndarray, 
                     metadata: list = None,
                     threshold_uv: float = 100.0) -> tuple:
    """
    Reject epochs containing amplitude artifacts.
    
    Removes epochs where any channel exceeds ±threshold_uv.
    The paper used a 20 µV std threshold; we use peak amplitude
    since eegmmidb data is already relatively clean.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n_epochs, n_channels, n_timepoints).
    y : np.ndarray
        Shape (n_epochs,).
    metadata : list of dict, optional
        Per-epoch metadata (filtered in parallel).
    threshold_uv : float
        Maximum allowed peak-to-peak amplitude per channel (µV).
    
    Returns
    -------
    tuple
        Filtered (X, y, metadata, rejection_mask)
    """
    # Peak-to-peak amplitude per channel per epoch
    ptp = X.max(axis=2) - X.min(axis=2)  # (n_epochs, n_channels)
    
    # Reject if ANY channel exceeds threshold
    keep_mask = (ptp < threshold_uv).all(axis=1)
    
    n_rejected = (~keep_mask).sum()
    n_total = len(keep_mask)
    
    print(f"  Artifact rejection: kept {keep_mask.sum()}/{n_total} "
          f"epochs ({n_rejected} rejected, threshold={threshold_uv}µV)")
    
    X_clean = X[keep_mask]
    y_clean = y[keep_mask]
    
    meta_clean = None
    if metadata is not None:
        meta_clean = [m for m, keep in zip(metadata, keep_mask) if keep]
    
    return X_clean, y_clean, meta_clean, keep_mask


def preprocess_pipeline(X: np.ndarray, sfreq: float = 160.0,
                        filter_low: float = 8.0, filter_high: float = 30.0,
                        normalize: str = 'zscore',
                        apply_car_ref: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline in one call.
    
    Applies: CAR → Bandpass → Normalization (in order).
    
    Parameters
    ----------
    X : np.ndarray
        Raw EEG epochs (n_epochs, n_channels, n_timepoints).
    sfreq : float
        Sampling frequency.
    filter_low, filter_high : float
        Bandpass cutoff frequencies.
    normalize : str
        Normalization method ('zscore', 'minmax', 'global_zscore', or None).
    apply_car_ref : bool
        Whether to apply Common Average Reference.
    
    Returns
    -------
    np.ndarray
        Preprocessed data, same shape.
    """
    X_proc = X.copy()
    
    # Step 1: Common Average Reference
    if apply_car_ref:
        X_proc = apply_car(X_proc)
    
    # Step 2: Bandpass filter
    X_proc = bandpass_filter(X_proc, sfreq=sfreq, 
                             low=filter_low, high=filter_high)
    
    # Step 3: Normalization
    if normalize is not None:
        X_proc = normalize_epochs(X_proc, method=normalize)
    
    return X_proc


def compute_erd(X: np.ndarray, y: np.ndarray, sfreq: float = 160.0,
                freq_band: tuple = (8, 13)) -> dict:
    """
    Compute Event-Related Desynchronization (ERD) per channel per class.
    
    ERD% = (A - R) / R × 100, where A = active power, R = reference power.
    Negative ERD = desynchronization (power decrease during movement).
    
    Parameters
    ----------
    X : np.ndarray
        Preprocessed epochs (n_epochs, n_channels, n_timepoints).
    y : np.ndarray
        Labels.
    sfreq : float
        Sampling frequency.
    freq_band : tuple
        Frequency band for power computation (Hz).
    
    Returns
    -------
    dict
        {class_label: erd_per_channel array}
    """
    from scipy.signal import welch
    
    classes = np.unique(y)
    erd_results = {}
    
    # Use all-class mean power as reference
    all_power = np.zeros(X.shape[1])
    for ch in range(X.shape[1]):
        freqs, psd = welch(X[:, ch, :], fs=sfreq, axis=1)
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        all_power[ch] = psd[:, band_mask].mean()
    
    for cls in classes:
        cls_mask = y == cls
        cls_power = np.zeros(X.shape[1])
        
        for ch in range(X.shape[1]):
            freqs, psd = welch(X[cls_mask, ch, :], fs=sfreq, axis=1)
            band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
            cls_power[ch] = psd[:, band_mask].mean()
        
        # ERD% formula
        erd = (cls_power - all_power) / (all_power + 1e-10) * 100.0
        erd_results[int(cls)] = erd
    
    return erd_results

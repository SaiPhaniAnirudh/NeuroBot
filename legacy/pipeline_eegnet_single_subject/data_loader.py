"""
NeuroBot Data Loader
====================
Loads PhysioNet EEGMMIDB EDF+ files, extracts annotated epochs,
and provides trial-aware train/val/test splits.

Dataset: https://www.physionet.org/content/eegmmidb/1.0.0/
Format: 109 subjects × 14 runs, 64-channel EEG @ 160 Hz, EDF+
"""

import os
import glob
import numpy as np
import mne

# ── Run → task mapping (PhysioNet EEGMMIDB protocol) ──────────────────────
# Runs 3, 7, 11  → Motor Execution: Left Fist (T1) vs Right Fist (T2)
# Runs 4, 8, 12  → Motor Imagery:   Left Fist (T1) vs Right Fist (T2)
# Runs 5, 9, 13  → Motor Execution: Both Fists (T1) vs Both Feet (T2)
# Runs 6, 10, 14 → Motor Imagery:   Both Fists (T1) vs Both Feet (T2)

MI_LR_RUNS = [4, 8, 12]    # Motor Imagery — Left vs Right fist
ME_LR_RUNS = [3, 7, 11]    # Motor Execution — Left vs Right fist
MI_BF_RUNS = [6, 10, 14]   # Motor Imagery — Both Fists vs Both Feet
ME_BF_RUNS = [5, 9, 13]    # Motor Execution — Both Fists vs Both Feet

# Standard 10-20 channel names for the 64-channel BCI2000 montage
CHANNEL_NAMES_64 = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
    'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
    'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
    'Ft7.', 'Ft8.',
    'T7..', 'T8..', 'T9..', 'T10.',
    'Tp7.', 'Tp8.',
    'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
    'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
    'O1..', 'Oz..', 'O2..',
    'Iz..'
]

# Clean channel name mapping (strip dots, standardize case)
CLEAN_CHANNEL_MAP = {raw: raw.rstrip('.').upper() for raw in CHANNEL_NAMES_64}


def find_edf_files(data_root: str) -> dict:
    """
    Scan a directory tree for EEGMMIDB EDF files.
    
    Parameters
    ----------
    data_root : str
        Root directory containing subject folders (S001/, S002/, etc.)
        Works with both PhysioNet and Kaggle directory layouts.
    
    Returns
    -------
    dict
        Mapping of (subject_id, run_id) → absolute file path.
        Example: {(1, 4): '/data/S001/S001R04.edf', ...}
    """
    file_index = {}
    
    # Search for all EDF files recursively
    patterns = [
        os.path.join(data_root, '**', 'S*R*.edf'),
        os.path.join(data_root, '**', 's*r*.edf'),
    ]
    
    edf_files = []
    for pattern in patterns:
        edf_files.extend(glob.glob(pattern, recursive=True))
    
    for fpath in edf_files:
        basename = os.path.basename(fpath).upper()
        # Parse S{subject_id}R{run_id}.edf
        try:
            s_idx = basename.index('S') + 1
            r_idx = basename.index('R')
            subject_id = int(basename[s_idx:r_idx])
            run_id = int(basename[r_idx + 1:basename.index('.')])
            file_index[(subject_id, run_id)] = fpath
        except (ValueError, IndexError):
            continue
    
    return file_index


def load_raw_edf(filepath: str, verbose: bool = False) -> mne.io.Raw:
    """
    Load a single EDF+ file and standardize channel names.
    
    Parameters
    ----------
    filepath : str
        Path to the EDF+ file.
    verbose : bool
        MNE verbosity flag.
    
    Returns
    -------
    mne.io.Raw
        Raw EEG data with standardized channel names.
    """
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=verbose)
    
    # Standardize channel names (strip dots, uppercase)
    rename_map = {}
    for ch in raw.ch_names:
        clean = ch.rstrip('.').upper()
        if clean != ch:
            rename_map[ch] = clean
    
    if rename_map:
        raw.rename_channels(rename_map)
    
    # Set channel types to EEG
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names 
                           if ch not in ['STI 014', 'Status']})
    
    return raw


def extract_events(raw: mne.io.Raw) -> np.ndarray:
    """
    Extract events from EDF+ annotations.
    
    Returns
    -------
    np.ndarray
        MNE-style events array (n_events, 3): [sample, 0, event_id]
        Event IDs: T0=0 (rest), T1=1, T2=2
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return events, event_id


def load_subject_runs(data_root: str, subject_id: int, 
                      runs: list = None, verbose: bool = False) -> list:
    """
    Load all specified runs for a single subject.
    
    Parameters
    ----------
    data_root : str
        Root directory of the dataset.
    subject_id : int
        Subject number (1–109).
    runs : list of int
        Run numbers to load. Defaults to MI left/right (runs 4, 8, 12).
    verbose : bool
        MNE verbosity.
    
    Returns
    -------
    list of dict
        Each dict contains: {'raw': mne.io.Raw, 'events': ndarray, 
                              'event_id': dict, 'subject': int, 'run': int}
    """
    if runs is None:
        runs = MI_LR_RUNS
    
    file_index = find_edf_files(data_root)
    results = []
    
    for run_id in runs:
        key = (subject_id, run_id)
        if key not in file_index:
            print(f"  ⚠ File not found for S{subject_id:03d}R{run_id:02d}")
            continue
        
        raw = load_raw_edf(file_index[key], verbose=verbose)
        events, event_id = extract_events(raw)
        
        results.append({
            'raw': raw,
            'events': events,
            'event_id': event_id,
            'subject': subject_id,
            'run': run_id,
        })
    
    return results


def build_epoch_dataset(data_root: str, subject_id: int, 
                        runs: list = None, tmin: float = 0.0, 
                        tmax: float = 1.0, baseline: tuple = None,
                        verbose: bool = False) -> tuple:
    """
    Build a complete epoch dataset for a subject.
    
    Parameters
    ----------
    data_root : str
        Dataset root directory.
    subject_id : int
        Subject number (1–109).
    runs : list
        Run numbers (default: MI left/right).
    tmin, tmax : float
        Epoch window relative to event onset (seconds).
    baseline : tuple
        Baseline correction window. None = no baseline.
    verbose : bool
        MNE verbosity.
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list)
        X: (n_epochs, n_channels, n_timepoints) — EEG data
        y: (n_epochs,) — labels (0 = T1/left fist, 1 = T2/right fist)
        metadata: list of dicts with trial info
    """
    if runs is None:
        runs = MI_LR_RUNS
    
    all_X = []
    all_y = []
    all_meta = []
    
    run_data = load_subject_runs(data_root, subject_id, runs, verbose)
    
    for rd in run_data:
        raw = rd['raw']
        events = rd['events']
        event_id = rd['event_id']
        
        # Map annotation strings to our labels
        # PhysioNet uses 'T1' and 'T2' (or sometimes numeric strings)
        target_events = {}
        for key, val in event_id.items():
            if 'T1' in str(key) or key == '1' or key == 'T1':
                target_events[key] = val
            elif 'T2' in str(key) or key == '2' or key == 'T2':
                target_events[key] = val
        
        if not target_events:
            continue
        
        # Create MNE Epochs
        try:
            epochs = mne.Epochs(
                raw, events, event_id=target_events,
                tmin=tmin, tmax=tmax, baseline=baseline,
                preload=True, verbose=verbose,
                proj=False, picks='eeg'
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ Epoch creation failed for run {rd['run']}: {e}")
            continue
        
        if len(epochs) == 0:
            continue
        
        X = epochs.get_data()  # (n_epochs, n_channels, n_timepoints)
        
        # Build labels: map event IDs to binary 0/1
        # T1 (left fist) → 0, T2 (right fist) → 1
        t1_ids = [v for k, v in target_events.items() 
                  if 'T1' in str(k) or k == '1' or k == 'T1']
        t2_ids = [v for k, v in target_events.items() 
                  if 'T2' in str(k) or k == '2' or k == 'T2']
        
        y = np.array([0 if ev in t1_ids else 1 for ev in epochs.events[:, 2]])
        
        # Metadata for each trial
        for i in range(len(epochs)):
            all_meta.append({
                'subject': rd['subject'],
                'run': rd['run'],
                'trial_idx': i,
                'event_sample': int(epochs.events[i, 0]),
                'label': int(y[i]),
                'label_name': 'left_fist' if y[i] == 0 else 'right_fist',
            })
        
        all_X.append(X)
        all_y.append(y)
    
    if not all_X:
        return np.array([]), np.array([]), []
    
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    
    return X, y, all_meta


def get_trial_splits(metadata: list, n_folds: int = 5, 
                     fold_idx: int = 0, val_ratio: float = 0.15,
                     seed: int = 42) -> tuple:
    """
    Trial-aware train/val/test split to avoid data leakage.
    
    Splits are done at the trial level (not sample level), so overlapping
    sliding windows from the same trial never end up in different sets.
    
    Parameters
    ----------
    metadata : list of dict
        Trial metadata from build_epoch_dataset().
    n_folds : int
        Number of CV folds.
    fold_idx : int
        Which fold to use as test set (0-indexed).
    val_ratio : float
        Fraction of training set to use for validation.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        train_indices, val_indices, test_indices into the epoch array.
    """
    rng = np.random.RandomState(seed)
    
    # Group by (run, trial_idx) to get unique trials
    trial_keys = list(set((m['run'], m['trial_idx']) for m in metadata))
    trial_keys.sort()
    rng.shuffle(trial_keys)
    
    # K-fold split on trials
    fold_size = len(trial_keys) // n_folds
    test_start = fold_idx * fold_size
    test_end = test_start + fold_size if fold_idx < n_folds - 1 else len(trial_keys)
    
    test_trials = set(trial_keys[test_start:test_end])
    train_val_trials = [t for t in trial_keys if t not in test_trials]
    
    # Split train_val into train and val
    n_val = max(1, int(len(train_val_trials) * val_ratio))
    val_trials = set(train_val_trials[:n_val])
    train_trials = set(train_val_trials[n_val:])
    
    # Map back to indices
    train_idx, val_idx, test_idx = [], [], []
    for i, m in enumerate(metadata):
        key = (m['run'], m['trial_idx'])
        if key in test_trials:
            test_idx.append(i)
        elif key in val_trials:
            val_idx.append(i)
        else:
            train_idx.append(i)
    
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def get_multi_subject_dataset(data_root: str, subject_ids: list,
                              runs: list = None, tmin: float = 0.0,
                              tmax: float = 1.0, verbose: bool = False) -> tuple:
    """
    Load and concatenate epoch data across multiple subjects.
    Useful for cross-subject training and fine-tuning experiments.
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list)
        X, y, metadata — same format as build_epoch_dataset()
    """
    all_X, all_y, all_meta = [], [], []
    
    for sid in subject_ids:
        X, y, meta = build_epoch_dataset(
            data_root, sid, runs=runs, tmin=tmin, tmax=tmax, verbose=verbose
        )
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_meta.extend(meta)
    
    if not all_X:
        return np.array([]), np.array([]), []
    
    return (np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
            all_meta)

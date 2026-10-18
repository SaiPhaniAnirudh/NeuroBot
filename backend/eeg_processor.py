"""
NeuroBot EEG processing: turns uploaded or generated recordings into decoded replay sessions.

Honesty rules enforced here (they replace behaviour of the first version):
  * `true_class` is only filled when ground truth really exists (PhysioNet T0/T1/T2
    annotations in an uploaded EDF, or the synthetic generator's own block design).
    Otherwise it is null; it is never copied from the prediction.
  * The model is chosen by (task, mode) from the registry. Probabilities are never
    relabelled from one task to another.
  * Recordings that do not contain the 64 channels the models were trained on are rejected
    instead of being padded or tiled.
"""

import io
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import model_registry
import preprocessing as pp

try:  # MNE is only needed for EDF/BDF parsing; import it lazily to keep start-up light
    import importlib.util
    HAS_MNE = importlib.util.find_spec("mne") is not None
except Exception:  # pragma: no cover
    HAS_MNE = False

STANDARD_64_CHANNELS = pp.STANDARD_64_CHANNELS
MAX_SECONDS = float(os.environ.get("NEUROBOT_MAX_SECONDS", "900"))
WAVE_POINTS = 40  # 160 samples / 4

ANNOTATION_LABELS = {
    "left_right": {"T0": "rest", "T1": "left", "T2": "right"},
    "fists_feet": {"T0": "rest", "T1": "both_fists", "T2": "both_feet"},
}
RUN_INFO = {  # eegmmidb run number -> (task, mode)
    **{r: ("left_right", "ME") for r in (3, 7, 11)}, **{r: ("left_right", "MI") for r in (4, 8, 12)},
    **{r: ("fists_feet", "ME") for r in (5, 9, 13)}, **{r: ("fists_feet", "MI") for r in (6, 10, 14)},
}
PERSONAL_MODEL_WARNING = (
    "The deployed models are personal models of one subject each. Predictions are only meaningful for that "
    "subject's own recordings; on anyone else's data they are not expected to be better than chance."
)


# ── helpers ────────────────────────────────────────────────────────────────

def infer_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """PhysioNet names look like S001R03.edf (run 3). Returns (task, mode) or None."""
    m = re.search(r"R(\d{2})", os.path.basename(filename).upper())
    if m and int(m.group(1)) in RUN_INFO:
        return RUN_INFO[int(m.group(1))]
    return None


def resolve_mode(mode: str, task: str, filename: str, warnings: List[str]) -> str:
    inferred = infer_from_filename(filename)
    if mode == "auto":
        if inferred:
            if inferred[0] != task:
                warnings.append(f"File name suggests task '{inferred[0]}' but task '{task}' was requested.")
            return inferred[1]
        warnings.append("Mode (executed/imagined) was not given and could not be read from the file name; assumed ME.")
        return "ME"
    if inferred and (inferred[0] != task or inferred[1] != mode):
        warnings.append(f"File name suggests {inferred[0]}/{inferred[1]} but {task}/{mode} was requested.")
    return mode


def window_labels(starts: np.ndarray, segments: List[Tuple[int, int, str]]) -> List[Optional[str]]:
    """Label of each window = label of the annotated segment that contains the whole window."""
    out: List[Optional[str]] = [None] * len(starts)
    for i, s in enumerate(starts):
        e = s + pp.WINDOW
        for a, b, lab in segments:
            if a <= s and e <= b:
                out[i] = lab
                break
    return out


def decode_recording(preprocessed: np.ndarray, task: str, mode: str,
                     truth: Optional[List[Tuple[int, int, str]]] = None,
                     extra: Optional[Dict[str, Any]] = None,
                     warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    """preprocessed: (64, samples) at 160 Hz after CAR + band-pass."""
    if not model_registry.has_model(task, mode):
        raise model_registry.ModelUnavailable(f"No model is loaded for task '{task}', mode '{mode}'.")
    n_samples = preprocessed.shape[1]
    if n_samples / pp.SFREQ > MAX_SECONDS:
        raise ValueError(f"Recording is {n_samples / pp.SFREQ:.0f} s long; the limit is {MAX_SECONDS:.0f} s.")
    starts, windows = pp.make_windows(preprocessed)
    order = model_registry.label_order(task, mode)
    probs = model_registry.predict(task, mode, windows)
    truths = window_labels(starts, truth) if truth else [None] * len(starts)
    truths = [t if t in order else None for t in truths]

    wave = windows[:, pp.DISPLAY_CHANNELS, ::4][:, :, :WAVE_POINTS]
    records = []
    n_lab = n_ok = 0
    for i, s in enumerate(starts):
        p = probs[i]
        pred = order[int(np.argmax(p))]
        if truths[i] is not None:
            n_lab += 1
            n_ok += int(pred == truths[i])
        records.append({
            "t": round(float(s) / pp.SFREQ, 3),
            "true_class": truths[i],
            "predicted_class": pred,
            "probs": {c: float(p[k]) for k, c in enumerate(order)},
            "waveform": [[round(float(v), 3) for v in ch] for ch in wave[i]],
        })
    info = model_registry.public_info(task, mode)
    result: Dict[str, Any] = {
        "task": task,
        "mode": mode,
        "label_order": order,
        "channels_shown": pp.DISPLAY_CHANNELS,
        "channel_names": pp.DISPLAY_CHANNEL_NAMES,
        "total_records": len(records),
        "duration_seconds": round(n_samples / pp.SFREQ, 2),
        "sampling_rate": pp.SFREQ,
        "has_ground_truth": n_lab > 0,
        "n_labelled_windows": n_lab,
        "window_accuracy": (n_ok / n_lab) if n_lab else None,
        "model": {k: info.get(k) for k in ("id", "title", "arch", "mode", "trained_on", "protocol", "evaluation", "warning")},
        "warnings": list(warnings or []) + [PERSONAL_MODEL_WARNING],
        "records": records,
    }
    if extra:
        result.update(extra)
    return result


# ── EDF / BDF ──────────────────────────────────────────────────────────────

def process_edf(file_bytes: bytes, filename: str, task: str = "left_right", mode: str = "auto") -> Dict[str, Any]:
    if not HAS_MNE:
        raise RuntimeError("MNE-Python is not installed on this server, so EDF files cannot be read.")
    import mne  # lazy
    warnings: List[str] = []
    mode = resolve_mode(mode, task, filename, warnings)
    if not model_registry.has_model(task, mode):
        raise model_registry.ModelUnavailable(f"No model is loaded for task '{task}', mode '{mode}'.")

    suffix = ".bdf" if filename.lower().endswith(".bdf") else ".edf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        reader = mne.io.read_raw_bdf if suffix == ".bdf" else mne.io.read_raw_edf
        try:
            raw = reader(tmp_path, preload=True, verbose=False)
        except Exception as exc:
            raise ValueError(f"The file could not be read as {suffix[1:].upper()}: {str(exc)[:200]}") from None
        names = [c.strip().strip(".").upper() for c in raw.ch_names]
        index = {n: i for i, n in enumerate(names)}
        missing = [c for c in STANDARD_64_CHANNELS if c not in index]
        if missing:
            raise ValueError(
                f"The file has {pp.N_CHANNELS - len(missing)} of the 64 channels the models were trained on "
                f"(missing: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}). "
                "Only recordings in the PhysioNet 64-channel montage can be decoded.")
        sfreq = float(raw.info["sfreq"])
        if raw.n_times / sfreq > MAX_SECONDS:
            raise ValueError(f"Recording is {raw.n_times / sfreq:.0f} s long; the limit is {MAX_SECONDS:.0f} s.")
        data = raw.get_data(picks=[index[c] for c in STANDARD_64_CHANNELS]) * 1e6  # volts -> microvolts

        # ground truth from PhysioNet T0/T1/T2 annotations, expressed in 160 Hz samples
        labmap = ANNOTATION_LABELS[task]
        segments: List[Tuple[int, int, str]] = []
        for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            lab = labmap.get(str(desc).strip().upper())
            if lab:
                segments.append((int(round(onset * pp.SFREQ)), int(round((onset + dur) * pp.SFREQ)), lab))
        if not segments:
            warnings.append("The file has no T0/T1/T2 annotations, so there is no ground truth to compare against.")
        if abs(sfreq - pp.SFREQ) > 1e-6:
            warnings.append(f"Recording was resampled from {sfreq:g} Hz to 160 Hz.")

        processed = pp.preprocess_recording(data, sfreq)
        return decode_recording(processed, task, mode, truth=segments or None, warnings=warnings,
                                extra={"source_filename": filename, "matched_channels": pp.N_CHANNELS,
                                       "is_synthetic": False})
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── CSV / TSV / NPY ────────────────────────────────────────────────────────

def _parse_table(file_bytes: bytes) -> Tuple[np.ndarray, Optional[List[str]]]:
    text = file_bytes.decode("utf-8", errors="strict")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("The file is empty.")
    delim = "\t" if "\t" in lines[0] else ","
    first = [t.strip() for t in lines[0].split(delim)]
    names: Optional[List[str]] = None
    try:
        float(first[0])
        body = lines
    except ValueError:
        names = [t.strip('"').strip("'").strip(".").upper() for t in first]
        body = lines[1:]
    try:
        arr = np.loadtxt(io.StringIO("\n".join(body)), delimiter=delim, dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"The file contains non-numeric values: {exc}") from None
    return arr, names


def process_csv(file_bytes: bytes, filename: str, task: str = "left_right", mode: str = "auto",
                sfreq: float = 160.0) -> Dict[str, Any]:
    warnings: List[str] = []
    mode = resolve_mode(mode, task, filename, warnings)
    if not model_registry.has_model(task, mode):
        raise model_registry.ModelUnavailable(f"No model is loaded for task '{task}', mode '{mode}'.")
    if not (10.0 <= sfreq <= 5000.0):
        raise ValueError("sfreq must be between 10 and 5000 Hz.")

    names: Optional[List[str]] = None
    if filename.lower().endswith(".npy"):
        arr = np.load(io.BytesIO(file_bytes), allow_pickle=False).astype(np.float64)
        if arr.ndim != 2:
            raise ValueError("The .npy array must be 2-D.")
    else:
        arr, names = _parse_table(file_bytes)

    # orientation: header names, or the dimension that equals 64
    if names is not None and len(names) == arr.shape[1]:
        data = arr.T
    elif arr.shape[0] == pp.N_CHANNELS and arr.shape[1] != pp.N_CHANNELS:
        data = arr
    elif arr.shape[1] == pp.N_CHANNELS:
        data = arr.T
    else:
        raise ValueError(f"Expected 64 EEG channels (found a {arr.shape[0]} x {arr.shape[1]} table). "
                         "Only recordings with the 64 PhysioNet channels can be decoded.")
    if data.shape[0] != pp.N_CHANNELS:
        raise ValueError(f"Expected 64 EEG channels, found {data.shape[0]}.")

    if names is not None and set(STANDARD_64_CHANNELS) <= set(names):
        idx = {n: i for i, n in enumerate(names)}
        data = data[[idx[c] for c in STANDARD_64_CHANNELS]]
    else:
        warnings.append("Channel names were not found in the header; the 64 channels were assumed to be in "
                        "PhysioNet order (FC5, FC3, FC1, ... IZ).")
    if abs(sfreq - pp.SFREQ) > 1e-6:
        warnings.append(f"Recording was resampled from {sfreq:g} Hz to 160 Hz.")
    warnings.append("CSV uploads carry no annotations, so there is no ground truth to compare against.")

    processed = pp.preprocess_recording(data, sfreq)
    return decode_recording(processed, task, mode, truth=None, warnings=warnings,
                            extra={"source_filename": filename, "is_synthetic": False})


# ── synthetic demo session ─────────────────────────────────────────────────

def generate_synthetic_motor_eeg(task: str = "left_right", mode: str = "ME", duration_sec: float = 12.0,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Synthetic 64-channel session with 10 Hz mu activity and event-related desynchronisation
    on the motor channels. Ground truth is the block design. It demonstrates the pipeline;
    it says nothing about decoding accuracy because the models were trained on real EEG.
    """
    rng = np.random.default_rng(seed)
    sf = pp.SFREQ
    n = int(duration_sec * sf)
    t = np.arange(n) / sf
    x = rng.standard_normal((pp.N_CHANNELS, n)) * 4.0
    mu = 14.0 * np.sin(2 * np.pi * 10.0 * t)
    for ch in range(pp.N_CHANNELS):
        x[ch] += mu * (0.8 + 0.4 * rng.random())

    ci = {n_: STANDARD_64_CHANNELS.index(n_) for n_ in ("C3", "CZ", "C4")}
    classes = ["rest", "left", "rest", "right"] if task == "left_right" else ["rest", "both_fists", "rest", "both_feet"]
    block = int(3.0 * sf)
    segments: List[Tuple[int, int, str]] = []
    for b in range(int(duration_sec // 3.0)):
        cls = classes[b % len(classes)]
        a, e = b * block, min(n, (b + 1) * block)
        segments.append((a, e, cls))
        if cls == "left":      # contralateral ERD on C4
            x[ci["C4"], a:e] *= 0.25
        elif cls == "right":   # contralateral ERD on C3
            x[ci["C3"], a:e] *= 0.22
        elif cls == "both_fists":
            x[ci["C3"], a:e] *= 0.25
            x[ci["C4"], a:e] *= 0.25
        elif cls == "both_feet":  # foot area lies at the vertex
            x[ci["CZ"], a:e] *= 0.22
    processed = pp.preprocess_recording(x, sf)
    warnings = ["SYNTHETIC session: ground truth is the generator's block design and the models were trained on real "
                "EEG, so the accuracy shown here is not a performance measure."]
    return decode_recording(processed, task, mode, truth=segments, warnings=warnings,
                            extra={"source_filename": f"synthetic_{task}_{mode}_{duration_sec:.0f}s", "is_synthetic": True})


def sample_csv_text(seconds: float = 3.0, seed: int = 0) -> str:
    """Template CSV: header with the 64 channel names, one row per sample (synthetic values)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * pp.SFREQ)
    t = np.arange(n) / pp.SFREQ
    rows = [",".join(STANDARD_64_CHANNELS)]
    base = 12.0 * np.sin(2 * np.pi * 10.0 * t)[:, None] + rng.standard_normal((n, pp.N_CHANNELS)) * 3.0
    for r in base:
        rows.append(",".join(f"{v:.3f}" for v in r))
    return "\n".join(rows)

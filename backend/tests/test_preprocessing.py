import numpy as np
import pytest

import preprocessing as pp


def test_display_channels_match_their_names():
    assert [pp.STANDARD_64_CHANNELS[i] for i in pp.DISPLAY_CHANNELS] == pp.DISPLAY_CHANNEL_NAMES
    assert pp.DISPLAY_CHANNEL_NAMES[:3] == ["C3", "CZ", "C4"]


def test_car_removes_channel_mean():
    x = np.random.default_rng(0).standard_normal((64, 500))
    assert np.abs(pp.common_average_reference(x).mean(axis=0)).max() < 1e-12


def test_bandpass_passes_mu_and_rejects_drift_and_mains():
    t = np.arange(4000) / 160.0
    def gain(f):
        x = np.tile(np.sin(2 * np.pi * f * t), (64, 1))
        y = pp.bandpass_4_40(x)
        return np.abs(y[:, 800:-800]).max() / np.abs(x).max()
    assert gain(10.0) > 0.95
    assert gain(1.0) < 0.05
    assert gain(70.0) < 0.05


def test_bandpass_keeps_length_and_is_zero_phase():
    t = np.arange(3200) / 160.0
    x = np.tile(np.sin(2 * np.pi * 10 * t), (2, 1))
    y = pp.bandpass_4_40(x)
    assert y.shape == x.shape
    mid = slice(600, -600)
    assert np.corrcoef(x[0, mid], y[0, mid])[0, 1] > 0.999  # no phase shift


def test_stored_fir_matches_mne():
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal((64, 2400)), axis=1) * 0.5 + rng.standard_normal((64, 2400)) * 3
    raw = mne.io.RawArray(x * 1e-6, mne.create_info(64, 160.0, "eeg"))
    raw.set_eeg_reference("average", projection=False)
    car = raw.get_data() * 1e6
    ref = raw.copy().filter(4.0, 40.0, fir_design="firwin").get_data() * 1e6
    ours = pp.bandpass_4_40(car)
    assert np.abs(ours - ref).max() < 1e-8


def test_windows_count_zscore_and_content():
    x = np.random.default_rng(2).standard_normal((64, 800)).astype(np.float32)
    starts, w = pp.make_windows(x)
    assert w.shape == ((800 - 160) // 20 + 1, 64, 160) and starts[1] - starts[0] == 20
    assert np.abs(w.mean(axis=2)).max() < 1e-4 and np.abs(w.std(axis=2) - 1).max() < 1e-3
    raw0 = x[:, 20:180]
    assert np.allclose(w[1], (raw0 - raw0.mean(1, keepdims=True)) / (raw0.std(1, keepdims=True) + 1e-8), atol=1e-4)


def test_too_short_recording_is_rejected():
    with pytest.raises(ValueError):
        pp.make_windows(np.zeros((64, 100), dtype=np.float32))


def test_resample_changes_length_by_ratio():
    x = np.random.default_rng(3).standard_normal((64, 2000))
    y = pp.resample_to_160(x, 250.0)
    assert abs(y.shape[1] - round(2000 * 160 / 250)) <= 1

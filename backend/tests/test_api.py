import io
import json

import numpy as np
import pytest

import eeg_processor
import preprocessing as pp
from conftest import LABELS, window_payload


# ── status ─────────────────────────────────────────────────────────────────

def test_health_reports_models_and_no_errors(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["models_loaded"] == 4 and r["load_errors"] == []
    assert r["models"]["left_right"] == {"ME": True, "MI": True}
    assert set(r["supported_upload_formats"]) == {"edf", "bdf", "csv", "tsv", "npy"}


def test_models_endpoint_has_evaluation_and_no_paths(client):
    ms = client.get("/api/models").json()["models"]
    assert len(ms) == 4 and all("onnx_path" not in m and m["evaluation"] for m in ms)


def test_tasks_lists_labels_models_and_replays(client):
    t = client.get("/api/tasks").json()
    assert {"left_right", "fists_feet"} <= set(t) and set(t) <= {"left_right", "fists_feet", "finger_2", "finger_3"}
    assert all(t[k].get("replay_only") for k in t if k.startswith("finger"))   # finger tasks have replays but no live model
    assert t["left_right"]["available"] and t["left_right"]["models"] == {"ME": True, "MI": True}
    assert t["left_right"]["label_order"] == ["left", "rest", "right"]
    assert t["left_right"]["replays"][0]["file"] == "replay_left_right_ME.json"


def test_empty_server_is_honest(empty_client):
    h = empty_client.get("/api/health").json()
    assert h["models_loaded"] == 0 and h["model_loaded"] is False
    assert empty_client.post("/api/predict", json={"window": window_payload()}).status_code == 503
    assert empty_client.get("/api/sample/generate").status_code == 503
    csv = eeg_processor.sample_csv_text(3.0).encode()
    r = empty_client.post("/api/upload/csv", files={"file": ("a.csv", csv)})
    assert r.status_code == 503


# ── predict ────────────────────────────────────────────────────────────────

def test_predict_uses_manifest_label_order(client):
    r = client.post("/api/predict", json={"window": window_payload(), "task": "fists_feet", "mode": "MI"})
    assert r.status_code == 200
    j = r.json()
    assert j["label_order"] == LABELS["fists_feet"] and list(j["probs"]) == LABELS["fists_feet"]
    assert abs(sum(j["probs"].values()) - 1) < 1e-6 and j["predicted_class"] == max(j["probs"], key=j["probs"].get)
    assert j["model_id"] == "tiny-fists_feet-mi"


def test_predict_is_deterministic_and_modes_differ(client):
    a = client.post("/api/predict", json={"window": window_payload(1), "mode": "ME"}).json()["probs"]
    b = client.post("/api/predict", json={"window": window_payload(1), "mode": "ME"}).json()["probs"]
    c = client.post("/api/predict", json={"window": window_payload(1), "mode": "MI"}).json()["probs"]
    assert a == b and a != c


@pytest.mark.parametrize("body", [
    {"window": [[0.0] * 160] * 63},
    {"window": [[0.0] * 159] * 64},
    {"window": [[0.0, 1.0], [1.0]]},
    {"window": window_payload(), "task": "nonsense"},
    {"window": window_payload(), "mode": "XX"},
], ids=["63_channels", "159_samples", "ragged", "bad_task", "bad_mode"])
def test_predict_rejects_bad_input(client, body):
    assert client.post("/api/predict", json=body).status_code == 422


# ── replay ─────────────────────────────────────────────────────────────────

def test_replay_log_has_provenance(client):
    r = client.get("/api/replay/log/left_right")
    assert r.status_code == 200
    j = r.json()
    assert len(j["records"]) == 450 and j["provenance"]["subject"] == "S007"
    assert "not training data" in j["provenance"]["source"]
    assert client.get("/api/replay/log/left_right?mode=MI").json()["provenance"]["mode"] == "MI"


def test_replay_unknown_task_is_404(client):
    assert client.get("/api/replay/log/nope").status_code == 404


def test_websocket_streams_meta_then_records(client):
    with client.websocket_connect("/ws/replay/fists_feet") as ws:
        meta = ws.receive_json()
        assert meta["type"] == "meta" and meta["total_records"] == 450 and meta["provenance"]
        rec = ws.receive_json()
        assert rec["type"] == "record" and set(rec["probs"]) == set(meta["label_order"])


def test_websocket_unknown_task_reports_error(client):
    with client.websocket_connect("/ws/replay/nope") as ws:
        assert "error" in ws.receive_json()


# ── uploads: CSV ───────────────────────────────────────────────────────────

def _csv(seconds=4.0, header=True, channels=64, shuffle=False, rate=160):
    rng = np.random.default_rng(0)
    n = int(seconds * rate)
    data = rng.standard_normal((n, channels)) * 5
    names = pp.STANDARD_64_CHANNELS[:channels]
    if shuffle:
        order = rng.permutation(channels)
        data, names = data[:, order], [names[i] for i in order]
    lines = ([",".join(names)] if header else []) + [",".join(f"{v:.3f}" for v in row) for row in data]
    return "\n".join(lines).encode()


def test_csv_upload_never_invents_ground_truth(client):
    r = client.post("/api/upload/csv?task=left_right&mode=ME", files={"file": ("x.csv", _csv())})
    assert r.status_code == 200
    j = r.json()
    assert j["has_ground_truth"] is False and j["window_accuracy"] is None
    assert all(rec["true_class"] is None for rec in j["records"])
    assert j["total_records"] == (640 - 160) // 20 + 1 and j["channel_names"] == pp.DISPLAY_CHANNEL_NAMES
    assert any("personal models" in w for w in j["warnings"])


def test_csv_header_order_is_respected(client):
    a = client.post("/api/upload/csv", files={"file": ("a.csv", _csv())}).json()
    b = client.post("/api/upload/csv", files={"file": ("b.csv", _csv(shuffle=True))}).json()
    # same data, columns shuffled but named: decoded output must be identical
    assert [r["predicted_class"] for r in a["records"]] == [r["predicted_class"] for r in b["records"]]


@pytest.mark.parametrize("payload,code", [
    (_csv(channels=32), 422),         # not 64 channels: rejected, not tiled
    (b"", 422),
    (_csv(seconds=0.5), 422),         # shorter than one window
    (b"a,b\n1,x\n2,3\n", 422),        # non-numeric
], ids=["32_channels", "empty_bytes", "short_window", "non_numeric"])
def test_csv_rejects_bad_files(client, payload, code):
    assert client.post("/api/upload/csv", files={"file": ("x.csv", payload)}).status_code == code


def test_csv_wrong_extension_and_size_limit(client, monkeypatch):
    assert client.post("/api/upload/csv", files={"file": ("x.pdf", b"1,2")}).status_code == 400
    import main
    monkeypatch.setattr(main, "MAX_CSV_BYTES", 1000)
    assert client.post("/api/upload/csv", files={"file": ("x.csv", _csv())}).status_code == 413


def test_csv_other_sampling_rate_is_resampled(client):
    r = client.post("/api/upload/csv?sfreq=250", files={"file": ("x.csv", _csv(seconds=8.0, rate=250))})
    assert r.status_code == 200 and any("resampled" in w for w in r.json()["warnings"])
    assert abs(r.json()["duration_seconds"] - 8.0) < 0.1


def test_npy_upload(client):
    buf = io.BytesIO()
    np.save(buf, np.random.default_rng(0).standard_normal((64, 800)))
    r = client.post("/api/upload/csv", files={"file": ("x.npy", buf.getvalue())})
    assert r.status_code == 200 and r.json()["total_records"] == 33


# ── uploads: EDF ───────────────────────────────────────────────────────────

def _edf_bytes(tmp_path, names=None, seconds=20, annotations=True):
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    mne.set_log_level("ERROR")
    names = names or [c.title() + "." for c in pp.STANDARD_64_CHANNELS]  # PhysioNet style: 'Fc5.'
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((len(names), seconds * 160)) * 20e-6, mne.create_info(names, 160.0, "eeg"))
    if annotations:
        raw.set_annotations(mne.Annotations([0.0, 4.2, 8.3, 12.4, 16.5], [4.2, 4.1, 4.2, 4.1, 3.5],
                                            ["T0", "T1", "T0", "T2", "T0"]))
    out = str(tmp_path / "S001R03.edf")
    mne.export.export_raw(out, raw, fmt="edf", overwrite=True, verbose=False)
    return open(out, "rb").read()


def test_edf_upload_uses_real_annotations_as_ground_truth(client, tmp_path):
    r = client.post("/api/upload/edf?task=left_right", files={"file": ("S001R03.edf", _edf_bytes(tmp_path))})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mode"] == "ME"  # inferred from run 03
    assert j["has_ground_truth"] and 0 < j["n_labelled_windows"] < j["total_records"]
    labelled = {rec["true_class"] for rec in j["records"] if rec["true_class"]}
    assert labelled <= {"rest", "left", "right"} and "left" in labelled and "right" in labelled
    assert all(rec["true_class"] is None for rec in j["records"] if rec["t"] < 0 or (3.3 < rec["t"] < 4.2))  # windows straddling a boundary


def test_edf_fists_feet_labels_follow_task(client, tmp_path):
    j = client.post("/api/upload/edf?task=fists_feet&mode=MI", files={"file": ("S001R03.edf", _edf_bytes(tmp_path))}).json()
    assert j["mode"] == "MI" and {r["true_class"] for r in j["records"] if r["true_class"]} <= {"rest", "both_fists", "both_feet"}
    assert any("suggests" in w for w in j["warnings"])  # run 03 file name vs requested fists_feet/MI


def test_edf_without_annotations_has_no_ground_truth(client, tmp_path):
    j = client.post("/api/upload/edf", files={"file": ("rec.edf", _edf_bytes(tmp_path, annotations=False))}).json()
    assert j["has_ground_truth"] is False and all(r["true_class"] is None for r in j["records"])


def test_edf_with_missing_channels_is_rejected(client, tmp_path):
    names = [c.title() + "." for c in pp.STANDARD_64_CHANNELS[:40]]
    r = client.post("/api/upload/edf", files={"file": ("x.edf", _edf_bytes(tmp_path, names=names))})
    assert r.status_code == 422 and "64 channels" in r.json()["detail"]


def test_edf_wrong_extension_and_garbage(client):
    assert client.post("/api/upload/edf", files={"file": ("x.txt", b"hi")}).status_code == 400
    assert client.post("/api/upload/edf", files={"file": ("x.edf", b"not an edf")}).status_code == 422


# ── synthetic session and template ─────────────────────────────────────────

def test_synthetic_session_is_flagged_and_has_designed_labels(client):
    j = client.get("/api/sample/generate?task=fists_feet&mode=MI&duration=12").json()
    assert j["is_synthetic"] is True and j["has_ground_truth"] is True
    assert any("SYNTHETIC" in w for w in j["warnings"])
    assert {r["true_class"] for r in j["records"] if r["true_class"]} == {"rest", "both_fists", "both_feet"}


def test_synthetic_is_reproducible_with_seed():
    import model_registry, os
    model_registry.load(os.environ["NEUROBOT_MANIFEST"])
    a = eeg_processor.generate_synthetic_motor_eeg("left_right", "ME", 9.0, seed=5)
    b = eeg_processor.generate_synthetic_motor_eeg("left_right", "ME", 9.0, seed=5)
    assert a["records"] == b["records"]


def test_sample_template_round_trips_through_the_csv_endpoint(client):
    text = client.get("/api/sample/download/csv").text
    assert text.splitlines()[0].split(",") == pp.STANDARD_64_CHANNELS
    r = client.post("/api/upload/csv", files={"file": ("template.csv", text.encode())})
    assert r.status_code == 200 and r.json()["total_records"] == (480 - 160) // 20 + 1


def test_project_zip_download_redirects_or_serves(client):
    r = client.get("/api/download/project-zip", follow_redirects=False)
    assert r.status_code in (200, 307)


# ── replay logs written by the export notebook ─────────────────────────────

def test_notebook_replay_log_is_preferred_and_describes_itself(client, tmp_path, monkeypatch):
    import main, shutil, os
    shutil.copytree(main.DATA_DIR, tmp_path / "data")
    doc = {"label_order": ["left", "rest", "right"], "channels_shown": pp.DISPLAY_CHANNELS,
           "channel_names": pp.DISPLAY_CHANNEL_NAMES,
           "meta": {"source": "Held-out trials of the exported model x", "subject": "S007", "mode": "ME"},
           "records": [{"t": 0.0, "trial": 1, "true_class": "left", "predicted_class": "left",
                        "probs": {"left": .8, "rest": .1, "right": .1}, "waveform": [[0.0] * 40] * 6}]}
    (tmp_path / "data" / "replay_left_right_ME.json").write_text(json.dumps(doc))
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "data"))
    j = client.get("/api/replay/log/left_right?mode=ME").json()
    assert j["replay_file"] == "replay_left_right_ME.json" and j["provenance"]["subject"] == "S007"
    t = client.get("/api/tasks").json()["left_right"]
    assert t["replays"][0]["file"] == "replay_left_right_ME.json" and t["replays"][0]["mode"] == "ME"
    # no mode given: the model-based replay wins over the earlier window-level log
    assert client.get("/api/replay/log/left_right").json()["replay_file"] == "replay_left_right_ME.json"

"""Finger replays are served only when a real replay file exists; nothing is invented."""
import json

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path))
    with TestClient(main.app) as c:
        yield c, tmp_path


def test_finger_task_absent_without_replay_file(client):
    c, _ = client
    assert "finger_2" not in c.get("/api/tasks").json()
    assert c.get("/api/replay/log/finger_2?mode=ME").status_code == 404


def test_finger_replay_served_when_file_exists(client):
    c, d = client
    doc = {"label_order": ["thumb", "pinky"], "channels_shown": [], "channel_names": [],
           "meta": {"model_id": "tcformer-finger2-s01-me"},
           "records": [{"t": 0.0, "trial": 0, "true_class": "thumb", "predicted_class": "thumb", "probs": {"thumb": 0.7, "pinky": 0.3}}]}
    (d / "replay_finger_2_ME.json").write_text(json.dumps(doc))
    t = c.get("/api/tasks").json()["finger_2"]
    assert t["replay_only"] and t["label_order"] == ["thumb", "pinky"] and not any(t["models"].values())
    r = c.get("/api/replay/log/finger_2?mode=ME").json()
    assert r["label_order"] == ["thumb", "pinky"] and r["provenance"]["model_id"] == "tcformer-finger2-s01-me"


def test_unknown_task_still_rejected(client):
    c, _ = client
    assert c.get("/api/replay/log/finger_9").status_code == 404

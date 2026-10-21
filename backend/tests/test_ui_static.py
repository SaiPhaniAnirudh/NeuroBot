"""The web UI must read the exported models' replays from the backend and must not present simulated data as a result."""
import os

UI = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "index.html")


def _html():
    with open(UI, encoding="utf-8") as fh:
        return fh.read()


def test_ui_prefers_backend_replay_over_bundled_sample():
    h = _html()
    assert "/api/replay/log/${taskKey}?mode=${PRESET_MODE}" in h


def test_finger_demo_is_flagged_as_simulated():
    h = _html()
    assert "simulated: true" in h and "SIMULATED DEMO" in h and "n/a (simulated demo)" in h


def test_ui_has_no_hardcoded_localhost_port_label():
    assert "BACKEND ONLINE (:8000)" not in _html()

# NeuroBot

EEG-based movement-intention decoding (PhysioNet eegmmidb, 64 channels) with a software-only
three-tier application: a FastAPI backend that serves ONNX models, and a browser Decode Console
with a 3D bionic-limb model. No external API is called anywhere.

The paper (IEEE-style report) describes the method and results. **The Kaggle notebooks are the source of
truth for every number in it.**

## Layout

| Path | What it is |
|---|---|
| `backend/` | FastAPI server, ONNX model registry, preprocessing, 45 automated tests |
| `backend/checkpoints/manifest.json` | list of deployed models (empty until you add the exported ones) |
| `backend/data/` | stored replay logs and `replay_index.json` (what each log is) |
| `ui/index.html` | Decode Console (single file; Three.js is loaded from a CDN) |
| `notebooks/` | `neurobot_v6_baseline_both_tasks` (baseline vs cross-subject pretraining), `neurobot_finger_v2_1_fixed` (finger pilot), `neurobot_export_flagship_onnx_v3` (model export) |
| `results/` | `v6_results.json`, `finger_results_v2.json` |
| `scripts/recompute_paper_stats.py` | regenerates paper Tables VIII and IX from `results/` |
| `legacy/` | the first single-subject EEGNet code and notebook, kept for reference only (window-level split, kernel length 64: superseded) |

## Run locally

```bash
pip install -r backend/requirements-dev.txt
cd backend && uvicorn main:app --port 8000      # UI and API on http://localhost:8000
pytest tests                                    # 45 tests
python ../scripts/recompute_paper_stats.py      # paper Tables VIII and IX
```

## Adding trained models

1. Run `notebooks/neurobot_export_flagship_onnx_v3.ipynb` on Kaggle (GPU and Internet on). The default
   `EXPORT_VARIANT = "baseline"` takes about 30-60 minutes; `"cross"` adds cross-subject pretraining
   and takes about 5-6 hours per task (see the notebook header).
2. Download `neurobot_models.zip` from the Output tab and unzip it.
3. Copy the `.onnx` files into `backend/checkpoints/` and append the entries of `manifest_entries.json`
   to the `models` list of `backend/checkpoints/manifest.json`. Copy the `replay_*.json` files into `backend/data/`.
4. `pytest backend/tests`, start the server, and check `/api/health`: the models must be listed as loaded
   and `load_errors` must be empty.

Each model is a **personal model of one subject** (the flagship subjects were picked because they decode
well, so their scores are best-case). The API says so in every response.

## API

`GET /api/health`, `GET /api/models`, `GET /api/tasks`, `POST /api/predict`, `GET /api/replay/log/{task}`,
`WS /ws/replay/{task}`, `POST /api/upload/edf`, `POST /api/upload/csv`, `GET /api/sample/generate`,
`GET /api/sample/download/csv`. Tasks: `left_right`, `fists_feet`. Modes: `ME` (executed), `MI` (imagined).

Rules the server follows:

* If no model is loaded for a task and mode, it answers HTTP 503. It never substitutes made-up output.
* `true_class` is only filled when ground truth really exists: the T0/T1/T2 annotations of a PhysioNet EDF,
  or the block design of the synthetic demo session (flagged `is_synthetic`). CSV uploads have none.
* Recordings must contain the 64 PhysioNet channels (EDF by channel name; CSV by header or exactly 64 columns).
* Preprocessing matches training: common average reference, 4-40 Hz zero-phase FIR (fixed taps equal to
  MNE's filter; checked against MNE in the tests), 1 s windows every 125 ms, per-window z-score.

## Deploy on Render

`render.yaml` is included (free web service, health check `/api/health`). Free instances sleep when idle
and have limited memory; keep `NEUROBOT_MAX_SECONDS` (default 900) and the upload limits in mind.

## Known limits

* The UI's Signal & 10-20 Cap tab shows band-power and amplitude values that are not computed from the
  signal yet; do not quote them.
* The robot hand has five digits, but the finger dataset decodes thumb, index, middle and pinky, and the
  deployed models decode whole-limb commands, so decoded commands curl the whole hand.
* The stored replay logs come from an earlier window-level-split run (see `backend/data/replay_index.json`).

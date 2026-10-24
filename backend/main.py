"""
NeuroBot backend (v0.4.0) - serves the trained ONNX models and the replay logs.

    GET  /api/health                  status, loaded models, load errors, limits
    GET  /api/models                  metadata and held-out accuracy of every loaded model
    GET  /api/tasks                   tasks, their labels, which models and replays exist
    POST /api/predict                 one preprocessed (64, 160) window -> class probabilities
    GET  /api/replay/log/{task}       stored replay log (optional ?mode=ME|MI)
    WS   /ws/replay/{task}            the same log streamed at 125 ms per record
    POST /api/upload/edf              EDF/BDF in the PhysioNet montage -> decoded session
    POST /api/upload/csv              CSV/TSV/NPY with 64 channels -> decoded session
    GET  /api/sample/generate         synthetic demo session (clearly flagged as synthetic)
    GET  /api/sample/download/csv     example CSV in the accepted format

Nothing here calls an external service. If no model is loaded for a task/mode the API says so
(HTTP 503); it never substitutes made-up output.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import eeg_processor
import model_registry
import preprocessing as pp

log = logging.getLogger("neurobot")
VERSION = "0.4.0"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")

TaskName = Literal["left_right", "fists_feet"]
ModeName = Literal["ME", "MI"]

MAX_EDF_BYTES = int(os.environ.get("NEUROBOT_MAX_EDF_MB", "40")) * 1024 * 1024
MAX_CSV_BYTES = int(os.environ.get("NEUROBOT_MAX_CSV_MB", "25")) * 1024 * 1024

# Replay logs written by the export notebook are named replay_{task}_{mode}.json.
# The two logs below are the earlier ones (window-level split); data/replay_index.json says what they are.
LEGACY_LOGS = {"left_right": "inference_log_left_right.json", "fists_feet": "inference_log_fists_feet.json"}
TASK_LABEL_ORDER = {"left_right": ["left", "rest", "right"], "fists_feet": ["both_fists", "rest", "both_feet"]}
# Finger tasks are replay-only: they appear once a replay_finger_{k}_{mode}.json exists in data/ (written by the finger export notebook).
FINGER_LABEL_ORDER = {"finger_2": ["thumb", "pinky"], "finger_3": ["thumb", "index", "pinky"]}
ALL_TASKS = {**TASK_LABEL_ORDER, **FINGER_LABEL_ORDER}


def _ping_external(url: str):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroBot-InternalKeepAlive/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            log.info("Keep-alive self ping -> HTTP %d", resp.getcode())
    except Exception as exc:
        log.warning("Keep-alive self ping notice: %s", exc)


async def _keep_alive_task():
    # Only active if running on Render or explicitly enabled
    if os.environ.get("RENDER") != "true" and os.environ.get("NEUROBOT_KEEP_ALIVE") != "1":
        return
    url = os.environ.get("NEUROBOT_PUBLIC_URL", "https://neurobot-nrnn.onrender.com/api/health")
    log.info("Started 24/7 internal keep-alive loop targeting %s", url)
    # Wait 60s before initial ping so deployment finishes completely
    await asyncio.sleep(60)
    while True:
        try:
            await asyncio.to_thread(_ping_external, url)
        except Exception as e:
            log.debug("Keep-alive task error: %s", e)
        # Ping every 10 minutes (600 seconds) - safely before Render's 15 min sleep window
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_registry.load()
    log.info("loaded %d model(s); problems: %s", model_registry.n_models(), model_registry.load_errors() or "none")
    task = asyncio.create_task(_keep_alive_task())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="NeuroBot API", version=VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


# ── replay helpers ─────────────────────────────────────────────────────────

def _replay_index() -> dict:
    path = os.path.join(DATA_DIR, "replay_index.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _replay_candidates(task: str, mode: Optional[str]):
    modes = [mode] if mode else list(model_registry.MODES)
    files = [(f"replay_{task}_{m}.json", m) for m in modes]
    if task in LEGACY_LOGS:
        files.append((LEGACY_LOGS[task], None))
    return files


def _find_replay(task: str, mode: Optional[str] = None):
    for fname, m in _replay_candidates(task, mode):
        if os.path.exists(os.path.join(DATA_DIR, fname)):
            return fname, m
    return None


def _load_replay(task: str, mode: Optional[str]):
    found = _find_replay(task, mode)
    if not found:
        raise HTTPException(status_code=404, detail=f"No replay log exists for task '{task}'" + (f", mode '{mode}'." if mode else "."))
    fname, m = found
    with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as fh:
        data = json.load(fh)
    prov = _replay_index().get(fname) or data.get("meta")  # logs from the export notebook describe themselves
    if prov:
        data["provenance"] = prov
    data["replay_file"] = fname
    return data


# ── status ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "models_loaded": model_registry.n_models(),
        "model_loaded": model_registry.n_models() > 0,
        "models": {t: {m: model_registry.has_model(t, m) for m in model_registry.MODES} for t in TASK_LABEL_ORDER},
        "load_errors": model_registry.load_errors(),
        "onnxruntime_available": model_registry.HAS_ORT,
        "mne_available": eeg_processor.HAS_MNE,
        "supported_upload_formats": ["edf", "bdf", "csv", "tsv", "npy"],
        "limits": {"edf_mb": MAX_EDF_BYTES // 2**20, "csv_mb": MAX_CSV_BYTES // 2**20,
                   "max_seconds": eeg_processor.MAX_SECONDS},
    }


@app.get("/api/models")
def list_models():
    return {"models": model_registry.list_models(), "load_errors": model_registry.load_errors()}


@app.get("/api/tasks")
def list_tasks():
    out = {}
    index = _replay_index()
    for task, order in TASK_LABEL_ORDER.items():
        replays = []
        for fname, m in _replay_candidates(task, None):
            if os.path.exists(os.path.join(DATA_DIR, fname)):
                replays.append({"file": fname, "mode": m, "source": (index.get(fname) or {}).get("source")})
        out[task] = {
            "available": bool(replays),
            "file": replays[0]["file"] if replays else LEGACY_LOGS[task],
            "label_order": order,
            "models": {m: model_registry.has_model(task, m) for m in model_registry.MODES},
            "replays": replays,
        }
    for task, order in FINGER_LABEL_ORDER.items():
        replays = [{"file": f, "mode": m, "source": None} for f, m in _replay_candidates(task, None)
                   if os.path.exists(os.path.join(DATA_DIR, f))]
        if replays:
            out[task] = {"available": True, "file": replays[0]["file"], "label_order": order,
                         "models": {m: False for m in model_registry.MODES}, "replays": replays, "replay_only": True}
    return out


# ── live inference ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    window: list[list[float]]
    task: TaskName = "left_right"
    mode: ModeName = "ME"


@app.post("/api/predict")
def predict(req: PredictRequest):
    """One window that is already preprocessed (CAR, 4-40 Hz) and z-scored per channel: shape (64, 160)."""
    try:
        arr = np.asarray(req.window, dtype=np.float32)
    except ValueError:
        raise HTTPException(status_code=422, detail="window must be a rectangular array of numbers.")
    if arr.shape != (pp.N_CHANNELS, pp.WINDOW):
        raise HTTPException(status_code=422, detail=f"window must have shape (64, 160), got {tuple(arr.shape)}.")
    if not np.all(np.isfinite(arr)):
        raise HTTPException(status_code=422, detail="window contains NaN or infinite values.")
    try:
        p = model_registry.predict(req.task, req.mode, arr[None])[0]
    except model_registry.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    order = model_registry.label_order(req.task, req.mode)
    probs = {c: float(p[i]) for i, c in enumerate(order)}
    return {"task": req.task, "mode": req.mode, "model_id": model_registry.public_info(req.task, req.mode)["id"],
            "label_order": order, "probs": probs, "predicted_class": max(probs, key=probs.get)}


# ── replay ─────────────────────────────────────────────────────────────────

@app.get("/api/replay/log/{task}")
def get_replay_log(task: str, mode: Optional[ModeName] = None):
    if task not in ALL_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task}'. Valid: {list(ALL_TASKS)}")
    return _load_replay(task, mode)


@app.websocket("/ws/replay/{task}")
async def ws_replay(websocket: WebSocket, task: str, mode: Optional[str] = None):
    await websocket.accept()
    try:
        if task not in ALL_TASKS or mode not in (None, "ME", "MI"):
            await websocket.send_json({"error": f"Unknown task or mode. Valid tasks: {list(ALL_TASKS)}"})
            await websocket.close()
            return
        try:
            data = _load_replay(task, mode)
        except HTTPException as exc:
            await websocket.send_json({"error": exc.detail})
            await websocket.close()
            return
        await websocket.send_json({
            "type": "meta", "label_order": data["label_order"], "channels_shown": data["channels_shown"],
            "channel_names": data.get("channel_names"), "total_records": len(data["records"]),
            "provenance": data.get("provenance"), "replay_file": data["replay_file"],
        })
        for record in data["records"]:
            await websocket.send_json({"type": "record", **record})
            await asyncio.sleep(0.125)
        await websocket.send_json({"type": "done"})
    except (WebSocketDisconnect, RuntimeError):
        pass


# ── uploads ────────────────────────────────────────────────────────────────

async def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks, size = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=f"File is larger than the {limit // 2**20} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except model_registry.ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception:
        log.exception("processing failed")
        raise HTTPException(status_code=500, detail="Processing failed. The file may be corrupt or in an unsupported format.")


@app.post("/api/upload/edf")
async def upload_edf(file: UploadFile = File(...), task: TaskName = "left_right",
                     mode: Literal["ME", "MI", "auto"] = "auto"):
    """EDF/BDF in the PhysioNet 64-channel montage. T0/T1/T2 annotations, if present, give real ground truth."""
    if not (file.filename or "").lower().endswith((".edf", ".bdf")):
        raise HTTPException(status_code=400, detail="Expected an .edf or .bdf file.")
    content = await _read_limited(file, MAX_EDF_BYTES)
    return await asyncio.to_thread(_run, eeg_processor.process_edf, content, file.filename, task, mode)


@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...), task: TaskName = "left_right",
                     mode: Literal["ME", "MI", "auto"] = "auto",
                     sfreq: float = Query(160.0, description="Sampling rate in Hz")):
    """CSV/TSV (header with channel names, one row per sample) or NPY, 64 channels. No ground truth."""
    if not (file.filename or "").lower().endswith((".csv", ".tsv", ".txt", ".npy")):
        raise HTTPException(status_code=400, detail="Expected a .csv, .tsv or .npy file.")
    content = await _read_limited(file, MAX_CSV_BYTES)
    return await asyncio.to_thread(_run, eeg_processor.process_csv, content, file.filename, task, mode, sfreq)


@app.get("/api/sample/generate")
def generate_sample(task: TaskName = "left_right", mode: ModeName = "ME",
                    duration: float = Query(12.0, ge=3.0, le=60.0)):
    """Synthetic demonstration session. Flagged is_synthetic; its accuracy is not a performance figure."""
    return _run(eeg_processor.generate_synthetic_motor_eeg, task, mode, duration)


@app.get("/api/sample/download/csv")
def download_sample_csv():
    return PlainTextResponse(
        content=eeg_processor.sample_csv_text(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=neurobot_sample_64ch_160hz.csv"},
    )


@app.get("/api/download/project-zip")
def download_project_zip():
    """Serves the project archive if one is placed next to the backend folder."""
    zip_path = os.path.join(os.path.dirname(__file__), "..", "NeuroBot_Complete_Project.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename="NeuroBot_Complete_Project.zip", media_type="application/zip")
    raise HTTPException(status_code=404, detail="Project archive not found on this server.")


# UI last, so that it never shadows an API route
if os.path.isdir(UI_DIR):
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")

#!/usr/bin/env python3
"""Check a deployed NeuroBot (standard library only).

    python scripts/check_live.py                                  # checks the public deployment
    python scripts/check_live.py --api http://127.0.0.1:8765 --ui http://127.0.0.1:8765

It waits for a sleeping Space to wake up (up to about 2 minutes), then checks the API, every replay log
(limb and finger), the signal fields added by the signal-export notebook, and the page served by the front end.
Exit code 0 means every check passed.
"""
import argparse, gzip, json, sys, time, urllib.request

LIMB = [("left_right", 160, 6), ("fists_feet", 160, 6)]
FINGER = [("finger_2", 100, 3), ("finger_3", 100, 3)]
ok_all = True


def get(url, timeout=90):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "neurobot-check"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return r.status, raw


def check(name, cond, detail=""):
    global ok_all
    print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else ""))
    ok_all &= bool(cond)


def wait_for_backend(api, minutes=2):
    end = time.time() + 60 * minutes
    while time.time() < end:
        try:
            s, b = get(api + "/api/health", 30)
            if s == 200:
                return json.loads(b)
        except Exception as e:
            print("  waiting for the backend to wake up...", type(e).__name__)
        time.sleep(8)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://energyvenom-neurobot.hf.space")
    ap.add_argument("--ui", default="https://neurobot-ui.vercel.app")
    a = ap.parse_args()
    api, ui = a.api.rstrip("/"), a.ui.rstrip("/")

    print(f"Backend: {api}\nFront end: {ui}\n")
    h = wait_for_backend(api)
    check("backend answers /api/health", h is not None)
    if h:
        check("four limb models loaded", h.get("models_loaded") == 4, str(h.get("models_loaded")))
        check("no model load errors", not h.get("load_errors"), str(h.get("load_errors")))
    try:
        tasks = json.loads(get(api + "/api/tasks")[1])
        for t, _, _ in LIMB + FINGER:
            check(f"/api/tasks lists {t} with a replay", t in tasks and tasks[t].get("available"))
    except Exception as e:
        check("/api/tasks", False, str(e))

    for task, n, nch in LIMB + FINGER:
        for mode in ("ME", "MI"):
            tag = f"{task} {mode}"
            try:
                t0 = time.time()
                s, b = get(f"{api}/api/replay/log/{task}?mode={mode}", 120)
                secs = time.time() - t0
                d = json.loads(b)
                check(f"{tag}: answered in under 30 s", secs < 30, f"{secs:.1f} s, {len(b) / 1e6:.1f} MB")
                recs = d["records"]; sig = d.get("meta", {}).get("signal", {})
                check(f"{tag}: replay served", s == 200 and len(recs) > 100, f"{len(recs)} windows")
                r = recs[len(recs) // 2]
                check(f"{tag}: real trace of {nch} channels x {n} samples",
                      len(r.get("waveform", [])) == nch and len(r["waveform"][0]) == n and sig.get("samples_per_window") == n)
                bands = r.get("bands", {})
                check(f"{tag}: band power adds up to 1", set(bands) == {"theta", "mu", "beta", "gamma_low"} and
                      0.99 < sum(bands[k][0] for k in bands) < 1.01)
                check(f"{tag}: alignment note present", "ONNX" in sig.get("alignment", ""))
                check(f"{tag}: not marked simulated", not d.get("meta", {}).get("simulated"))
            except Exception as e:
                check(f"{tag}: replay", False, f"{type(e).__name__}: {e}")

    try:
        s, b = get(ui + "/")
        html = b.decode("utf-8", "replace")
        check("front end page loads", s == 200 and "NEUROBOT" in html.upper())
        check("front end has the retrying fetch (new version)", "fetchBackend" in html)
        check("front end has no synthetic-EEG code", "getSynthesizedFingerWaveform" not in html and "CALIBRATED" not in html)
        check("front end shows band power from the log", "bandHeaderText" in html and "gamma_low" in html)
        check("no Render address in the page", "onrender.com" not in html)
        s, b = get(ui + "/api/health")
        check("front end forwards /api/* to the backend", s == 200)
    except Exception as e:
        check("front end", False, f"{type(e).__name__}: {e}")

    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

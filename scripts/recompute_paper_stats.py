"""
Recomputes the numbers of paper Tables VIII and IX from the committed result files.

    python scripts/recompute_paper_stats.py

Table VIII: TCFormer without vs with cross-subject pretraining (results/v6_results.json).
Table IX:   finger-level pilot, subject S01 (results/finger_results_v2.json).
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def table_viii():
    r = json.load(open(os.path.join(ROOT, "v6_results.json")))
    subs = range(1, 11)
    print("Table VIII  (mean +- population std over 10 subjects, seed-averaged majority vote, %)")
    print(f"{'condition':12s} {'no pretrain':>14s} {'pretrained':>14s} {'delta':>7s} {'b/t/w':>8s} {'t-test p':>9s} {'Wilcoxon p':>11s}")
    for task in "AB":
        for mode in ("ME", "MI"):
            base = np.array([r[f"{task}|tcformer|baseline|{s}"][mode]["ensemble"] for s in subs]) * 100
            cross = np.array([r[f"{task}|tcformer|cross|{s}"][mode]["ensemble"] for s in subs]) * 100
            d = cross - base
            p_t = stats.ttest_rel(cross, base).pvalue
            p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"Task {task} {mode:2s}   {base.mean():6.1f} +- {base.std():4.1f} {cross.mean():6.1f} +- {cross.std():4.1f} "
                  f"{d.mean():+7.1f} {int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum()):<3d} {p_t:9.3f} {p_w:11.3f}")


def table_ix():
    f = json.load(open(os.path.join(ROOT, "finger_results_v2.json")))
    print("\nTable IX  (finger pilot, subject S01, majority-vote accuracy %, three seeds averaged)")
    print(f"{'config / mode':16s} {'trials':>6s} {'chance':>7s} {'TCFormer':>9s} {'EEGNet':>8s} {'ensemble':>9s}")
    for cls, label in (("2class", "2-finger"), ("3class", "3-finger")):
        for mode in ("ME", "MI"):
            row = {m: f[f"S01|{cls}|{m}|baseline"][mode] for m in ("tcformer", "eegnet", "arch_ensemble")}
            t = row["tcformer"]
            print(f"{label + ' ' + mode:16s} {t['n_eval_trials']:6d} {t['chance'] * 100:7.1f} "
                  f"{row['tcformer']['ensemble_mv'] * 100:9.1f} {row['eegnet']['ensemble_mv'] * 100:8.1f} "
                  f"{row['arch_ensemble']['ensemble_mv'] * 100:9.1f}")


if __name__ == "__main__":
    table_viii()
    table_ix()

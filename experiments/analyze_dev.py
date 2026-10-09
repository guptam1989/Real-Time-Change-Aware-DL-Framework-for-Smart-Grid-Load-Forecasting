"""Development check: overall and post-alarm accuracy of the combination variants
on the development datasets (validation and test spans)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "results_v3", "agg")
RES3 = os.path.join(HERE, "results_v3")
VARS = ["Best on validation", "Mean", "EWA", "FS", "FS (baselines only)", "FS+reset", "FS+spawn", "FS+reset+spawn",
        "RCAF", "Trigger: periodic", "Trigger: CUSUM", "Trigger: BOCPD", "RCAF (own experts only)"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


for ds in sys.argv[1:] or ["RWD1", "RWD2"]:
    meta = json.load(open(os.path.join(RES3, f"alarms_{ds}.json")))
    n_tr, n_va = meta["n_tr"], meta["n_va"]
    res = {v: dict(val=[], test=[], post_val=[], post_test=[]) for v in VARS}
    for s in range(3):
        p = os.path.join(AGG, f"{ds}_s{s}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        y, vp, tp = d["y"], d["val_pos"], d["test_pos"]
        post = np.zeros(len(y), bool)
        for a in meta["alarms_ae"]:
            if a >= n_tr:
                post[a - n_tr:a - n_tr + 14 * 24] = True
        pv = vp[post[vp]]
        pt = tp[post[tp]]
        for v in VARS:
            P = d[f"pred::{v}"]
            res[v]["val"].append(rmse(P[vp], y[vp]))
            res[v]["test"].append(rmse(P[tp], y[tp]))
            if len(pv):
                res[v]["post_val"].append(rmse(P[pv], y[pv]))
            if len(pt):
                res[v]["post_test"].append(rmse(P[pt], y[pt]))
    print(f"\n=== {ds}  (alarms after training: {sum(a >= n_tr for a in meta['alarms_ae'])})")
    print(f"{'variant':26s} {'val':>8s} {'test':>8s} {'post-val':>9s} {'post-test':>9s}")
    for v in VARS:
        r = res[v]
        f = lambda k: f"{np.mean(r[k]):8.4f}" if r[k] else "     n/a"
        print(f"{v:26s} {f('val')} {f('test')} {f('post_val'):>9s} {f('post_test'):>9s}")
    info = json.load(open(os.path.join(AGG, f"{ds}_s0.json")))
    print("selected:", {k: info[k]["params"] for k in ("FS", "FS+reset", "FS+spawn", "FS+reset+spawn")}, "RCAF ->", info["RCAF"].get("selected_variant"))
    print("interval:", info["RCAF"].get("interval"))

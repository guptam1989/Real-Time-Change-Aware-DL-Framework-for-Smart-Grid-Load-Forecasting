"""Weight diagnostics for the case studies (post-hoc analysis, not part of the method):
weight entropy and the weights of selected experts for RCAF and plain fixed share
(FS) around a documented event, and the mean weight of the change-aware and
adapted experts over the test span.

Usage: python analyze_weights.py IT 2020-03-10 [seed]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aggregate as ag  # noqa: E402


def entropy(w):
    w = np.clip(w, 1e-12, 1)
    return float(-(w * np.log(w)).sum(-1).mean())


def main(ds, start, seed=0):
    F = ag.load_feats(ds)
    meta = json.load(open(os.path.join(ag.RES3, f"alarms_{ds}.json")))
    n, n_tr, n_va = meta["n"], meta["n_tr"], meta["n_va"]
    vt = np.arange(n_tr, n - ag.H)
    N = len(vt)
    y = F["y"][vt[:, None] + np.arange(1, ag.H + 1)]
    sigma = float(np.std(F["y"][:n_tr]))
    info = json.load(open(os.path.join(ag.AGG, f"{ds}_s{seed}.json")))
    d = np.load(os.path.join(ag.AGG, f"{ds}_s{seed}.npz"), allow_pickle=True)
    names = ag.BASELINE_POOL + ag.RCAF_EXPERTS
    P = np.stack([ag.load_pred(ds, m, seed) for m in names], -1)
    aw = np.ones((N, len(names)), bool)
    pf = info["FS"]["params"]
    _, _, _, path_fs = ag.combine(P, aw, y, sigma, np.zeros(N, bool), [pf["eta"]], [pf["alpha"]], [0.0], [1],
                                  None, record=[0, ag.H - 1])
    path_rc = d["weight_path"]
    labels = list(d["weight_labels"])
    tt = pd.to_datetime(F["time_index"])[vt]
    t0 = pd.Timestamp(start)
    i0 = int(np.searchsorted(tt, t0))
    win = (tt >= t0) & (tt < t0 + pd.Timedelta(days=56))
    tp = np.arange(n_va - n_tr, N)
    print(ds, "seed", seed, "RCAF variant", info["RCAF"]["selected_variant"], info["RCAF"]["params"], "FS", pf)
    for k, hname in ((1, "h=24"), (0, "h=1")):
        wf, wr = path_fs[:, k, :], path_rc[:, k, :]
        print(f"[{hname}] entropy at event start: FS {entropy(wf[i0]):.3f}  RCAF {entropy(wr[i0]):.3f}"
              f"  (uniform over 11: {np.log(11):.3f})")
        print(f"[{hname}] weights at event start  FS:", {m: round(float(wf[i0, j]), 3) for j, m in enumerate(names)})
        print(f"[{hname}] weights at event start RCAF:", {m: round(float(wr[i0, j]), 3) for j, m in enumerate(labels)})
        print(f"[{hname}] mean weights in event window  FS:", {m: round(float(wf[win, j].mean()), 3) for j, m in enumerate(names)})
        print(f"[{hname}] mean weights in event window RCAF:", {m: round(float(wr[win, j].mean()), 3) for j, m in enumerate(labels)})
        ca = [j for j, m in enumerate(labels) if m.startswith("RCAF") and not m.endswith("*")]
        sp = [j for j, m in enumerate(labels) if m.endswith("*")]
        print(f"[{hname}] RCAF test-span mean weight: change-aware {wr[tp][:, ca].sum(1).mean():.3f}, "
              f"adapted {wr[tp][:, sp].sum(1).mean() if sp else 0:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)

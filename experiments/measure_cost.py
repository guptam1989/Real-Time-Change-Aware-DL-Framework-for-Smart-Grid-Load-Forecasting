"""Streaming cost on a single CPU thread (edge-device proxy).

For each dataset: per-sample latency of every RCAF component (RSSA update,
GARCH update, time-domain features, AE descriptor, CPD score, TFT forecast),
per-forecast latency of the baselines (batch size 1), model sizes, and the
wall-clock time of one CPD-triggered adaptation on the CPU.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import time_domain  # noqa: E402
from rssa import RSSA  # noqa: E402
from run_forecast import (CACHE, DEEP, H, L, RES, Windows, finetune, input_spec,  # noqa: E402
                          load_merged, make_model)

torch.set_num_threads(1)
N_STEPS = 500


def timeit(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return 1e3 * (time.perf_counter() - t0) / n


def main():
    best = load_merged("best_cfg")
    sel = load_merged("adapt_selection")
    rows, comp_rows = [], []
    for ds in ("RWD1", "RWD2"):
        F = pickle.load(open(os.path.join(CACHE, f"{ds}_rssa.pkl"), "rb"))
        n_va = F["n_va"]
        load = F["y"]
        # ---- RCAF per-sample components
        z = (load - load[:F["n_tr"]].mean()) / load[:F["n_tr"]].std()
        r = RSSA(w=24)
        r.init(z[n_va - 336:n_va])
        k = iter(range(n_va, n_va + 10 ** 6))
        t_rssa = timeit(lambda: r.update(z[next(k)]), N_STEPS)
        gp = {"omega": 1.0, "alpha": 0.1, "beta": 0.8}
        h = [1.0]
        t_garch = timeit(lambda: h.append(gp["omega"] + gp["alpha"] * 0.01 + gp["beta"] * h[-1]), N_STEPS)
        den = F["den_level"]
        t_time = timeit(lambda: time_domain(den[n_va - 192:n_va])[-1], 200)
        from cpd import train_ae
        ae = train_ae(np.random.randn(256, 24), m=8, epochs=1)
        w = torch.as_tensor(F["den"][n_va - 24:n_va][None], dtype=torch.float32)
        t_ae = timeit(lambda: ae(w), N_STEPS)
        D = F["latent"][n_va - 336:n_va]

        def cpd_score():
            past, rec = D[:168], D[168:]
            mp, mr = np.median(past, 0), np.median(rec, 0)
            sp = 1.4826 * np.median(np.abs(past - mp), 0)
            sr = 1.4826 * np.median(np.abs(rec - mr), 0)
            return np.mean(((mr - mp) / np.sqrt((sp ** 2 + sr ** 2) / 168 + 1e-6)) ** 2)
        t_cpd = timeit(cpd_score, N_STEPS)

        # ---- forecasters (batch size 1)
        for m in DEEP:
            groups = input_spec(m, F)
            win = Windows(groups, F, "cpu")
            model = make_model(m, win.X.shape[1], groups, best[f"{ds}|{m}"]["cfg"]).eval()
            pt = os.path.join(RES, "preds", f"{ds}_{m}_s0.pt")
            if os.path.exists(pt):
                model.load_state_dict(torch.load(pt, map_location="cpu"))
            x, f, _ = win.batch(np.array([n_va]))
            with torch.no_grad():
                t_fc = timeit(lambda: model(x, f), 200)
            params = int(sum(p.numel() for p in model.parameters()))
            row = dict(dataset=ds, model=m, forecast_ms=t_fc, params=params, size_KB=params * 4 / 1024)
            if m == "RCAF":
                ft = sel[ds]["ft"]
                idx = np.arange(n_va - H - ft["window"], n_va - H + 1)
                t0 = time.perf_counter()
                finetune(model, win, idx, None, 0, ft["epochs"], ft["lr"], True)
                row["adapt_cpu_s"] = time.perf_counter() - t0
                pipe = t_rssa + t_garch + t_time + t_ae + t_cpd
                row["pipeline_ms"] = pipe
                row["total_ms"] = pipe + t_fc
                comp_rows += [dict(dataset=ds, component=c, ms=v) for c, v in
                              [("RSSA update", t_rssa), ("GARCH update", t_garch), ("Time-domain features", t_time),
                               ("AE descriptor", t_ae), ("AE-CPD score", t_cpd), ("TFT forecast (24 h)", t_fc)]]
            else:
                row["total_ms"] = t_fc
            row["throughput_hz"] = 1e3 / row["total_ms"]
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "cost_forecasters.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(os.path.join(RES, "cost_components.csv"), index=False)
    print(pd.DataFrame(comp_rows).round(3).to_string())


if __name__ == "__main__":
    main()

"""Metrics, significance tests and result tables from saved test-span forecasts.

All metrics are computed on the raw load in original units (kW for RWD1,
MW for RWD2) over every hourly forecast origin of the test span.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PRED = os.path.join(RES, "preds")
CACHE = os.path.join(HERE, "cache")
H = 24
HORIZONS = [1, 6, 12, 24]
MAIN = ["SeasonalNaive", "SARIMA-RT", "XGBoost", "LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT", "RCAF"]


def truth(ds):
    with open(os.path.join(CACHE, f"{ds}_rssa.pkl"), "rb") as f:
        F = pickle.load(f)
    n, n_va = len(F["y"]), F["n_va"]
    te = np.arange(n_va, n - H)
    return F["y"][te[:, None] + np.arange(1, H + 1)], F, te


def metrics(y, p):
    e = p - y
    out = {}
    for h in HORIZONS:
        yh, eh = y[:, h - 1], e[:, h - 1]
        out[f"RMSE@{h}"] = np.sqrt(np.mean(eh ** 2))
        out[f"MAE@{h}"] = np.mean(np.abs(eh))
        out[f"MAPE@{h}"] = 100 * np.mean(np.abs(eh / yh))
        out[f"R2@{h}"] = 1 - np.sum(eh ** 2) / np.sum((yh - yh.mean()) ** 2)
    out["RMSE"] = np.sqrt(np.mean(e ** 2))
    out["MAE"] = np.mean(np.abs(e))
    out["MAPE"] = 100 * np.mean(np.abs(e / y))
    out["R2"] = float(np.mean([1 - np.sum(e[:, h] ** 2) / np.sum((y[:, h] - y[:, h].mean()) ** 2) for h in range(H)]))
    return out


def load_runs(ds, name, key="static"):
    runs = []
    for f in sorted(glob.glob(os.path.join(PRED, f"{ds}_{name}_s*.npz"))):
        d = np.load(f)
        if key in d:
            runs.append(d[key])
    return runs


def dm_test(e1, e2, lag=H):
    """Diebold-Mariano test on squared-error loss averaged over horizons per origin,
    with a Newey-West (Bartlett) long-run variance (overlapping multi-step forecasts)."""
    d = np.mean(e1 ** 2, 1) - np.mean(e2 ** 2, 1)
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def pinball(yq, y, qs=(0.1, 0.5, 0.9)):
    return float(np.mean([np.mean(np.maximum(q * (y - yq[..., i]), (q - 1) * (y - yq[..., i]))) for i, q in enumerate(qs)]))


def summarise(rows):
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("dataset", "model", "seed")]
    g = df.groupby(["dataset", "model"], sort=False)[num]
    return g.mean().add_suffix(""), g.std(ddof=0).fillna(0).add_suffix("_std")


def main():
    out_rows, ens, abl_rows, prob_rows, dm_rows, adapt_rows = [], {}, [], [], [], []
    datasets = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(PRED, "*.npz"))})
    for ds in datasets:
        y, F, te = truth(ds)
        # ---- main comparison: baselines static; RCAF with change-aware adaptation
        for m in MAIN:
            key = "cpd" if m == "RCAF" else "static"
            runs = load_runs(ds, m, key)
            for s, p in enumerate(runs):
                out_rows.append(dict(dataset=ds, model=m, seed=s, **metrics(y, p)))
            if runs:
                ens[(ds, m)] = np.mean(runs, 0)
        # ---- ablations
        variants = [("RCAF (full: AE-CPD-triggered adaptation)", "RCAF", "cpd"),
                    ("RCAF without adaptation (static)", "RCAF", "static"),
                    ("RCAF with periodic adaptation (same budget)", "RCAF", "periodic"),
                    ("RCAF with CUSUM-triggered adaptation", "RCAF", "cusum"),
                    ("RCAF with BOCPD-triggered adaptation", "RCAF", "bocpd"),
                    ("RCAF without autoregressive highway", "RCAF-noHighway", "cpd"),
                    ("RCAF without RSSA denoising", "RCAF-noRSSA", "cpd"),
                    ("RCAF without GARCH feature", "RCAF-noGARCH", "cpd"),
                    ("RCAF, time-domain features only (alpha = 0)", "RCAF-alpha0", "cpd"),
                    ("RCAF, AE latent features only (alpha = 1)", "RCAF-alpha1", "cpd"),
                    ("RCAF without fused features", "RCAF-noFusion", "cpd"),
                    ("TFT (static)", "TFT", "static"),
                    ("TFT with AE-CPD-triggered adaptation", "TFT", "cpd"),
                    ("LSTM (static)", "LSTM", "static"),
                    ("LSTM with AE-CPD-triggered adaptation", "LSTM", "cpd")]
        for label, name, key in variants:
            for s, p in enumerate(load_runs(ds, name, key)):
                abl_rows.append(dict(dataset=ds, model=label, seed=s, **metrics(y, p)))
        # ---- probabilistic forecasts (quantiles 0.1/0.5/0.9)
        for label, name, key in [("RCAF (adaptive)", "RCAF", "cpd_q"), ("RCAF (static)", "RCAF", "static_q"),
                                 ("TFT (static)", "TFT", "static_q")]:
            for s, q in enumerate(load_runs(ds, name, key)):
                cover = np.mean((y >= q[..., 0]) & (y <= q[..., 2]))
                prob_rows.append(dict(dataset=ds, model=label, seed=s, pinball=pinball(q, y),
                                      PICP80=100 * cover, width=float(np.mean(q[..., 2] - q[..., 0]))))
        # ---- Diebold-Mariano: RCAF (seed-averaged) vs each baseline (seed-averaged)
        if (ds, "RCAF") in ens:
            er = ens[(ds, "RCAF")] - y
            for m in MAIN[:-1]:
                if (ds, m) in ens:
                    stat, p = dm_test(ens[(ds, m)] - y, er)
                    dm_rows.append(dict(dataset=ds, baseline=m, DM=stat, p=p))
        # ---- adaptation statistics from run metadata
        for f in sorted(glob.glob(os.path.join(PRED, f"{ds}_RCAF_s*.json"))):
            r = json.load(open(f))
            adapt_rows.append(dict(dataset=ds, seed=r["spec"]["seed"], **{k: v for k, v in r.items() if k.startswith("adapt_")},
                                   train_time=r["train_time"], params=r["params"], epochs=r.get("epochs")))

    os.makedirs(RES, exist_ok=True)
    for rows, name in [(out_rows, "forecast_main"), (abl_rows, "forecast_ablation"), (prob_rows, "forecast_prob")]:
        if not rows:
            continue
        mean, std = summarise(rows)
        tab = pd.concat([mean, std], axis=1)
        tab.to_csv(os.path.join(RES, f"{name}.csv"))
        cols = [c for c in ["RMSE", "MAE", "MAPE", "R2", "RMSE@1", "RMSE@24", "pinball", "PICP80", "width"] if c in mean.columns]
        print(f"\n==== {name}")
        print(mean[cols].round(4).to_string())
    if dm_rows:
        dm = pd.DataFrame(dm_rows)
        dm.to_csv(os.path.join(RES, "forecast_dm.csv"), index=False)
        print("\n==== Diebold-Mariano (positive: RCAF more accurate)")
        print(dm.round(4).to_string())
    if adapt_rows:
        a = pd.DataFrame(adapt_rows)
        a.to_csv(os.path.join(RES, "adaptation_stats.csv"), index=False)
        print(a.round(3).to_string())


if __name__ == "__main__":
    main()

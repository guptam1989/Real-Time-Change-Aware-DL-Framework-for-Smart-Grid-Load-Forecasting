"""Evaluation of the v3 experiments: per-dataset metrics, cross-dataset ranks
with Friedman/Nemenyi tests, Holm-corrected Diebold-Mariano tests, accuracy in
the windows after documented structural events, and interval quality."""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES3 = os.path.join(HERE, "results_v3")
PRED3 = os.path.join(RES3, "preds")
AGG = os.path.join(RES3, "agg")
CACHE = os.path.join(HERE, "cache")
H = 24
DEV = ["RWD1", "RWD2"]
HELDOUT = ["DE", "FR", "IT", "ES", "GB", "ERCOT", "CAISO", "SPP"]
SINGLE = ["SeasonalNaive", "SARIMA-RT", "XGBoost", "LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT"]
EXPERTS = ["RCAF", "RCAF-TRF", "RCAF-GBM"]
EXPERT_LABEL = {"RCAF": "RCAF-TFT expert", "RCAF-TRF": "RCAF-TRF expert", "RCAF-GBM": "RCAF-GBM expert"}
COMBOS = ["Mean", "Best on validation", "EWA", "FS", "FS (baselines only)"]
ABLATIONS = ["RCAF", "FS", "FS+reset", "FS+spawn", "FS+reset+spawn", "Trigger: periodic", "Trigger: CUSUM",
             "Trigger: BOCPD", "RCAF (own experts only)", "FS (baselines only)"]
# documented structural events (start dates, local); windows of 8 weeks (COVID) or 4 weeks (weather)
EVENTS = {
    "RWD2": ("2017-12-26", 28, "cold wave Dec 2017 - Jan 2018"),
    "DE": ("2020-03-22", 56, "COVID-19 contact ban"),
    "FR": ("2020-03-17", 56, "COVID-19 lockdown"),
    "IT": ("2020-03-10", 56, "COVID-19 lockdown"),
    "ES": ("2020-03-15", 56, "COVID-19 state of alarm"),
    "GB": ("2020-03-24", 56, "COVID-19 lockdown"),
    "ERCOT": ("2021-02-13", 28, "Winter Storm Uri"),
    "CAISO": ("2020-08-14", 28, "August 2020 heat wave"),
    "SPP": ("2021-02-13", 28, "Winter Storm Uri"),
}


def feats(ds):
    with open(os.path.join(CACHE, f"{ds}_rssa.pkl"), "rb") as f:
        return pickle.load(f)


def metrics(y, p, scale):
    e = p - y
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(100 * np.mean(np.abs(e / y))), MASE=float(np.mean(np.abs(e)) / scale),
                R2=float(np.mean([1 - np.sum(e[:, h] ** 2) / np.sum((y[:, h] - y[:, h].mean()) ** 2) for h in range(H)])),
                **{f"RMSE@{h}": float(np.sqrt(np.mean(e[:, h - 1] ** 2))) for h in (1, 24)})


def dm(e1, e2, lag=H):
    d = np.mean(e1 ** 2, 1) - np.mean(e2 ** 2, 1)
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if not lrv > 0:   # identical forecasts (e.g. RCAF without any reaction equals FS)
        return 0.0, 1.0
    s = d.mean() / np.sqrt(lrv / n)
    return float(s), float(2 * (1 - stats.norm.cdf(abs(s))))


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load_all(ds):
    """Test-span forecasts of every method for every seed: {method: [S, Ntest, H]}."""
    F = feats(ds)
    meta = json.load(open(os.path.join(RES3, f"alarms_{ds}.json")))
    n_tr, n_va, n = meta["n_tr"], meta["n_va"], meta["n"]
    vt = np.arange(n_tr, n - H)
    tp = np.arange(n_va - n_tr, len(vt))
    y = F["y"][vt[tp][:, None] + np.arange(1, H + 1)]
    out, q = {}, {}
    for m in SINGLE + EXPERTS:
        runs = []
        for f in sorted(glob.glob(os.path.join(PRED3, f"{ds}_{m}_s*.npz"))):
            d = np.load(f)
            runs.append(d["static"][tp])
            if "static_q" in d:
                q.setdefault(m, []).append(d["static_q"][tp])
        if runs:
            out[EXPERT_LABEL.get(m, m)] = np.stack(runs)
    hw = {}
    for f in sorted(glob.glob(os.path.join(AGG, f"{ds}_s*.npz"))):
        d = np.load(f)
        for k in d.files:
            if k.startswith("pred::"):
                out.setdefault(k[6:], []).append(d[k][tp])
            if k.startswith("lo::"):
                hw.setdefault(k[4:], [[], []])[0].append(d[k][tp])
            if k.startswith("up::"):
                hw.setdefault(k[4:], [[], []])[1].append(d[k][tp])
    out = {k: np.stack(v) if isinstance(v, list) else v for k, v in out.items()}
    hw = {k: (np.stack(v[0]), np.stack(v[1])) for k, v in hw.items()}
    q = {k: np.stack(v) for k, v in q.items()}
    scale = float(np.mean(np.abs(F["y"][24:n_tr] - F["y"][:n_tr - 24])))
    times = pd.to_datetime(F["time_index"][vt[tp]])
    return y, out, hw, q, scale, times, meta, F


def friedman_nemenyi(ranks):
    """ranks: DataFrame datasets x methods (1 = best). Returns chi2, p, CD (alpha 0.05)."""
    N, k = ranks.shape
    R = ranks.mean(0).values
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    p = 1 - stats.chi2.cdf(chi2, k - 1)
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return float(chi2), float(p), float(cd)


def main():
    rows, dm_rows, ev_rows, prob_rows, abl_rows, det_rows, post_rows = [], [], [], [], [], [], []
    datasets = [d for d in DEV + HELDOUT if glob.glob(os.path.join(AGG, f"{d}_s*.npz"))]
    for ds in datasets:
        y, out, hw, q, scale, times, meta, F = load_all(ds)
        group = "development" if ds in DEV else "held-out"
        for m, P in out.items():
            for s in range(P.shape[0]):
                rows.append(dict(dataset=ds, group=group, model=m, seed=s, **metrics(y, P[s], scale)))
        # Diebold-Mariano: RCAF vs single models and combination baselines (seed-averaged forecasts)
        ref = out["RCAF"].mean(0) - y
        comp = [m for m in SINGLE + list(EXPERT_LABEL.values()) + COMBOS[:-1] + ["FS (baselines only)"] if m in out]
        res = [dm(out[m].mean(0) - y, ref) for m in comp]
        adj = holm([r[1] for r in res])
        for m, (st, p), pa in zip(comp, res, adj):
            dm_rows.append(dict(dataset=ds, group=group, baseline=m, DM=st, p=p, p_holm=pa))
        # accuracy in the 14 days after every AE-CPD alarm in the test span
        post = np.zeros(len(times), bool)
        vt0 = meta["n_va"]
        for a in meta["alarms_ae"]:
            if a >= vt0:
                post[a - vt0:a - vt0 + 14 * 24] = True
        if post.any():
            for m, P in out.items():
                post_rows.append(dict(dataset=ds, model=m, n_alarms=int(sum(a >= vt0 for a in meta["alarms_ae"])),
                                      hours=int(post.sum()),
                                      MASE=float(np.mean(np.abs(P[:, post].mean(0) - y[post])) / scale),
                                      RMSE=float(np.sqrt(np.mean((P[:, post].mean(0) - y[post]) ** 2)))))
        # documented event window
        if ds in EVENTS:
            start, days, label = EVENTS[ds]
            t0 = pd.Timestamp(start)
            win = (times >= t0) & (times < t0 + pd.Timedelta(days=days))
            if win.sum() > 0:
                for m, P in out.items():
                    ev_rows.append(dict(dataset=ds, event=label, model=m,
                                        RMSE=float(np.sqrt(np.mean((P[:, win].mean(0) - y[win]) ** 2))),
                                        MASE=float(np.mean(np.abs(P[:, win].mean(0) - y[win])) / scale)))
                # detection delay of each trigger for the documented event
                tt = pd.to_datetime(F["time_index"])
                for trig in ("alarms_ae", "alarms_cusum", "alarms_bocpd"):
                    al = [a for a in meta[trig] if tt[a] >= t0 - pd.Timedelta(days=3)]
                    al = [a for a in al if tt[a] <= t0 + pd.Timedelta(days=days)]
                    det_rows.append(dict(dataset=ds, event=label, trigger=trig.split("_")[1],
                                         first_alarm=str(tt[al[0]]) if al else "",
                                         delay_h=float((tt[al[0]] - t0) / pd.Timedelta(hours=1)) if al else np.nan,
                                         alarms_test=int(np.sum(np.array(meta[trig]) >= meta["n_va"]))))
        # probabilistic: conformal intervals around the RCAF forecast vs. TFT quantiles
        pred = out["RCAF"]
        for k, (WL, WU) in hw.items():
            lo, hi = pred - WL, pred + WU
            cov = np.mean((y[None] >= lo) & (y[None] <= hi))
            isc = np.mean((hi - lo) + 10 * (lo - y) * (y < lo) + 10 * (y - hi) * (y > hi))
            pin = np.mean([np.mean(np.maximum(a * (y - b), (a - 1) * (y - b))) for a, b in ((0.1, lo), (0.5, pred), (0.9, hi))])
            prob_rows.append(dict(dataset=ds, method=k, PICP80=100 * cov, width=float(np.mean(hi - lo)),
                                  interval_score=float(isc), pinball=float(pin)))
        for m, Q in q.items():
            lo, md, hi = Q[..., 0], Q[..., 1], Q[..., 2]
            cov = np.mean((y[None] >= lo) & (y[None] <= hi))
            isc = np.mean((hi - lo) + 10 * (lo - y) * (y < lo) + 10 * (y - hi) * (y > hi))
            pin = np.mean([np.mean(np.maximum(a * (y - b), (a - 1) * (y - b))) for a, b in ((0.1, lo), (0.5, md), (0.9, hi))])
            prob_rows.append(dict(dataset=ds, method=f"{EXPERT_LABEL.get(m, m)} (quantiles)", PICP80=100 * cov,
                                  width=float(np.mean(hi - lo)), interval_score=float(isc), pinball=float(pin)))
    # what the validation span selected on every dataset (seed-wise)
    sel_rows = []
    for ds in datasets:
        for f in sorted(glob.glob(os.path.join(AGG, f"{ds}_s*.json"))):
            info = json.load(open(f))
            r = info["RCAF"]
            sel_rows.append(dict(dataset=ds, seed=int(f[-6]), variant=r.get("selected_variant"),
                                 **{f"agg_{k}": v for k, v in r["params"].items()},
                                 **{f"aci_{k}": v for k, v in r.get("interval", {}).items()},
                                 best_single_on_val=info.get("best_single_on_val")))
    pd.DataFrame(sel_rows).to_csv(os.path.join(RES3, "selection.csv"), index=False)
    df = pd.DataFrame(rows)
    os.makedirs(RES3, exist_ok=True)
    df.to_csv(os.path.join(RES3, "metrics_all_seeds.csv"), index=False)
    mean = df.groupby(["dataset", "group", "model"], sort=False).mean(numeric_only=True).reset_index()
    std = df.groupby(["dataset", "group", "model"], sort=False).std(ddof=0, numeric_only=True).reset_index()
    mean.to_csv(os.path.join(RES3, "metrics_mean.csv"), index=False)
    std.to_csv(os.path.join(RES3, "metrics_std.csv"), index=False)
    pd.DataFrame(dm_rows).to_csv(os.path.join(RES3, "dm_tests.csv"), index=False)
    pd.DataFrame(ev_rows).to_csv(os.path.join(RES3, "event_windows.csv"), index=False)
    pd.DataFrame(det_rows).to_csv(os.path.join(RES3, "event_detection.csv"), index=False)
    pd.DataFrame(post_rows).to_csv(os.path.join(RES3, "post_alarm.csv"), index=False)
    pd.DataFrame(prob_rows).to_csv(os.path.join(RES3, "intervals.csv"), index=False)
    # cross-dataset ranks (MASE) for the main comparison and the ablation
    for name, methods in (("main", SINGLE + ["RCAF"]), ("combination", COMBOS + ["RCAF"]),
                          ("ablation", ABLATIONS), ("experts", list(EXPERT_LABEL.values()) + ["RCAF"])):
        for grp, dsets in (("all", datasets), ("held-out", [d for d in datasets if d in HELDOUT])):
            sub = mean[mean.dataset.isin(dsets) & mean.model.isin(methods)]
            if sub.empty or sub.dataset.nunique() < 2:
                continue
            piv = sub.pivot(index="dataset", columns="model", values="MASE")[methods].dropna()
            ranks = piv.rank(axis=1)
            chi2, p, cd = friedman_nemenyi(ranks)
            r = ranks.mean(0).sort_values()
            pd.DataFrame(dict(mean_rank=r)).assign(chi2=chi2, p=p, CD=cd, N=len(piv)).to_csv(
                os.path.join(RES3, f"ranks_{name}_{grp}.csv"))
            print(f"\n== ranks {name} ({grp}, N={len(piv)}): Friedman chi2={chi2:.1f} p={p:.2g} CD={cd:.2f}")
            print(r.round(2).to_string())
    with pd.option_context("display.width", 200):
        print(mean.pivot(index="model", columns="dataset", values="MASE").round(3).to_string())


if __name__ == "__main__":
    main()

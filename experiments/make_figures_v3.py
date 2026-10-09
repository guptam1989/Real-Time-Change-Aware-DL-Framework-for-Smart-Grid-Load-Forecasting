"""Figures for the v3 experiments (multi-dataset evaluation, events, intervals)."""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES3 = os.path.join(HERE, "results_v3")
AGG = os.path.join(RES3, "agg")
PRED3 = os.path.join(RES3, "preds")
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "..", "figures")
H = 24
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300})
UNITS = {"RWD1": "kW"}
COL = {"RCAF": "#c0392b", "FS": "#2c7fb8", "Mean": "#7b3294", "TFT": "#1a9641", "XGBoost": "#4d4d4d",
       "Transformer": "#e66101", "best": "#4d4d4d"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def feats(ds):
    with open(os.path.join(CACHE, f"{ds}_rssa.pkl"), "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ critical difference diagram
def fig_cd(name="main", grp="all"):
    r = pd.read_csv(os.path.join(RES3, f"ranks_{name}_{grp}.csv"), index_col=0)
    ranks, cd, N = r["mean_rank"], float(r["CD"].iloc[0]), int(r["N"].iloc[0])
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * ((k + 1) // 2) + 1.0))
    ax.set_xlim(1, k)
    ax.axis("off")
    ax.plot([1, k], [0.85, 0.85], color="k", lw=0.8)
    for i in range(1, k + 1):
        ax.plot([i, i], [0.85, 0.88], color="k", lw=0.8)
        ax.text(i, 0.91, str(i), ha="center", fontsize=7.5)
    ax.plot([1, 1 + cd], [0.98, 0.98], color="k", lw=1.6)
    ax.text(1 + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5)
    display = {"FS (baselines only)": "FS (standard forecasters only)", "SeasonalNaive": "Seasonal naive"}
    names, vals = [display.get(n, n) for n in ranks.index], list(ranks.values)
    half = (k + 1) // 2
    for j, (nm, v) in enumerate(zip(names, vals)):
        left = j < half
        yy = 0.75 - 0.075 * (j if left else (j - half))
        xx = 1 - 0.05 if left else k + 0.05
        ax.plot([v, v], [0.85, yy], color="#555", lw=0.7)
        ax.plot([v, xx], [yy, yy], color="#555", lw=0.7)
        lab = f"{nm} ({v:.2f})"
        ax.text(xx - 0.05 if left else xx + 0.05, yy, lab, ha="right" if left else "left", va="center",
                fontsize=7.5, fontweight="bold" if nm == "RCAF" else "normal",
                color=COL["RCAF"] if nm == "RCAF" else "k")
    # cliques: groups of methods whose rank difference is below CD
    order = np.argsort(vals)
    sv = np.array(vals)[order]
    cliques, ybar = [], 0.80
    for a in range(k):
        b = a
        while b + 1 < k and sv[b + 1] - sv[a] < cd:
            b += 1
        if b > a and not any(a >= c0 and b <= c1 for c0, c1 in cliques):
            cliques.append((a, b))
    for c0, c1 in cliques:
        ax.plot([sv[c0] - 0.03, sv[c1] + 0.03], [ybar, ybar], color="#c0392b", lw=2.2, alpha=0.6)
        ybar -= 0.02
    ylow = 0.75 - 0.075 * (half - 1)
    ax.set_ylim(ylow - 0.12, 1.05)
    ax.text((1 + k) / 2, ylow - 0.1, f"mean rank over {N} datasets (MASE; 1 = best)", ha="center", fontsize=7.5)
    save(fig, f"fig_cd_{name}_{grp}.png")


# ------------------------------------------------------------------ gains per dataset
def fig_gain():
    m = pd.read_csv(os.path.join(RES3, "metrics_mean.csv"))
    single = ["SeasonalNaive", "SARIMA-RT", "XGBoost", "LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT"]
    rows = []
    for ds, d in m.groupby("dataset", sort=False):
        d = d.set_index("model")
        best_single = d.loc[[s for s in single if s in d.index], "MASE"].min()
        rows.append(dict(dataset=ds, vs_best=100 * (d.loc["RCAF", "MASE"] / best_single - 1),
                         vs_fs=100 * (d.loc["RCAF", "MASE"] / d.loc["FS", "MASE"] - 1),
                         vs_mean=100 * (d.loc["RCAF", "MASE"] / d.loc["Mean", "MASE"] - 1)))
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(r))
    ax.bar(x - 0.27, r.vs_best, 0.27, color="#c0392b", label="vs. best single model (in hindsight)")
    ax.bar(x, r.vs_fs, 0.27, color="#2c7fb8", label="vs. fixed-share combination")
    ax.bar(x + 0.27, r.vs_mean, 0.27, color="#7b3294", label="vs. simple mean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, r.dataset)
    ax.set_ylabel("MASE change of RCAF (%)")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_gain.png")


# ------------------------------------------------------------------ event case study
def fig_event(ds="IT", start="2020-03-10", before=21, after=42):
    F = feats(ds)
    meta = json.load(open(os.path.join(RES3, f"alarms_{ds}.json")))
    n_tr, n = meta["n_tr"], meta["n"]
    vt = np.arange(n_tr, n - H)
    tt = pd.to_datetime(F["time_index"])
    t0 = pd.Timestamp(start)
    sel = (tt[vt] >= t0 - pd.Timedelta(days=before)) & (tt[vt] < t0 + pd.Timedelta(days=after))
    pos = np.where(sel)[0]
    d = np.load(os.path.join(AGG, f"{ds}_s0.npz"), allow_pickle=True)
    info = json.load(open(os.path.join(AGG, f"{ds}_s0.json")))
    y = d["y"]
    best = info["best_single_on_val"]
    preds = {"RCAF": d["pred::RCAF"], "FS (no CPD)": d["pred::FS"],
             f"best single on val. ({best})": np.load(os.path.join(PRED3, f"{ds}_{best}_s0.npz"))["static"]}
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 0.8, 1, 1.2]))
    times = tt[vt[pos]]
    ax[0].plot(tt[vt[pos] + 24], y[pos, -1], color="k", lw=0.7, label="observed load")
    ax[0].plot(tt[vt[pos] + 24], preds["RCAF"][pos, -1], color=COL["RCAF"], lw=0.8, label="RCAF (24 h ahead)")
    ax[0].set_ylabel("MW")
    ax[0].legend(fontsize=7, frameon=False, ncol=2)
    ax[1].plot(times, F["score"][vt[pos]], color="#7b3294", lw=0.8)
    ax[1].axhline(meta["threshold"], color="k", ls="--", lw=0.7)
    ax[1].set_ylabel("AE-CPD score")
    for nm, P in preds.items():
        e = pd.Series(np.mean((P[pos] - y[pos]) ** 2, 1)).rolling(168, min_periods=24).mean() ** 0.5
        ax[2].plot(times, e, lw=1.1 if nm == "RCAF" else 0.8, label=nm,
                   color=COL["RCAF"] if nm == "RCAF" else (COL["FS"] if nm.startswith("FS") else "#4d4d4d"))
    ax[2].set_ylabel("7-day RMSE (MW)")
    ax[2].legend(fontsize=7, frameon=True, framealpha=0.85, loc="upper left")
    W = d["weight_path"][:, 1, :]                      # horizon 24
    labs = list(d["weight_labels"])
    groups = {"change-aware experts": [i for i, l in enumerate(labs) if l.startswith("RCAF") and not l.endswith("*")],
              "spawned (adapted) experts": [i for i, l in enumerate(labs) if l.endswith("*")],
              "standard forecasters": [i for i, l in enumerate(labs) if not l.startswith("RCAF")]}
    stack = [W[pos][:, g].sum(1) for g in groups.values()]
    ax[3].stackplot(times, *stack, labels=list(groups), colors=["#c0392b", "#f4a582", "#92c5de"], alpha=0.85)
    ax[3].set_ylabel("weight (h = 24)")
    ax[3].set_ylim(0, 1)
    ax[3].legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    for a in meta["alarms_ae"]:
        if times.min() <= tt[a] <= times.max():
            for k in range(4):
                ax[k].axvline(tt[a], color="#7b3294", ls=":", lw=0.9)
    for k in range(4):
        ax[k].axvline(t0, color="k", lw=0.9)
    fig.autofmt_xdate()
    save(fig, f"fig_event_{ds}.png")


# ------------------------------------------------------------------ rolling coverage
def fig_coverage(cases=(("IT", "2020-03-10"), ("ERCOT", "2021-02-13"))):
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.2, 2.4 * len(cases)))
    for ax, (ds, start) in zip(np.atleast_1d(axes), cases):
        F = feats(ds)
        meta = json.load(open(os.path.join(RES3, f"alarms_{ds}.json")))
        n_tr, n, n_va = meta["n_tr"], meta["n"], meta["n_va"]
        vt = np.arange(n_tr, n - H)
        tt = pd.to_datetime(F["time_index"])
        d = np.load(os.path.join(AGG, f"{ds}_s0.npz"), allow_pickle=True)
        y, pred = d["y"], d["pred::RCAF"]
        tp = vt >= n_va
        for key, lab, col in (("RCAF + ACI", "RCAF + change-aware ACI", "#c0392b"),
                              ("RCAF + rolling conformal", "RCAF + rolling conformal", "#2c7fb8")):
            lo, up = d[f"lo::{key}"], d[f"up::{key}"]
            cov = ((y >= pred - lo) & (y <= pred + up)).mean(1)
            ax.plot(tt[vt[tp]], pd.Series(cov[tp]).rolling(168, min_periods=24).mean() * 100, color=col, lw=0.9, label=lab)
        q = np.load(os.path.join(PRED3, f"{ds}_TFT_s0.npz"))["static_q"]
        cov = ((y >= q[..., 0]) & (y <= q[..., 2])).mean(1)
        ax.plot(tt[vt[tp]], pd.Series(cov[tp]).rolling(168, min_periods=24).mean() * 100, color="#1a9641", lw=0.9,
                label="TFT quantiles")
        ax.axhline(80, color="k", ls="--", lw=0.7)
        ax.axvline(pd.Timestamp(start), color="k", lw=0.9)
        ax.set_ylabel("7-day coverage (%)")
        ax.set_title(f"{ds}: empirical coverage of the nominal 80% interval (test span)", fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "fig_coverage.png")


# ------------------------------------------------------------------ semi-synthetic CPD
def fig_semisynth():
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RES3, "cpd_semisynth_*.csv")))
                    if "cost" not in p])
    meths = ["CUSUM", "Bayesian", "uLSIF", "RuLSIF", "KLIEP", "RCAF (AE-CPD)"]
    cols = ["#999999", "#2c7fb8", "#fdb863", "#e66101", "#1a9641", "#c0392b"]
    sets = ["LoadL", "LoadA", "LoadLA"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.13
    for i, m in enumerate(meths):
        d = df[df.method == m].set_index("dataset").reindex(sets)
        ax.bar(np.arange(3) + (i - 2.5) * w, d.F1, w, yerr=d.F1_std, color=cols[i],
               label="BOCPD" if m == "Bayesian" else m, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(range(3), ["LoadL (level)", "LoadA (daily amplitude)", "LoadLA (both)"])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig_semisynth.png")


def fig_attention(ds="DE", n_orig=2000):
    import torch
    from run_forecast import Windows, input_spec, make_model, origins
    best = json.load(open(os.path.join(RES3, f"best_cfg_{ds}.json")))
    F = feats(ds)
    groups = input_spec("RCAF", F)
    win = Windows(groups, F, "cpu")
    model = make_model("RCAF", win.X.shape[1], groups, best["RCAF"]["cfg"])
    model.load_state_dict(torch.load(os.path.join(PRED3, f"{ds}_RCAF_s0.pt"), map_location="cpu"))
    model.eval()
    te = origins(F, "test")[:n_orig]
    att, wts = [], []
    with torch.no_grad():
        for i in range(0, len(te), 250):
            x, f, _ = win.batch(te[i:i + 250])
            model(x, f)
            att.append(model.last_attention.numpy())
            wts.append(model.last_weights.numpy())
    A = np.concatenate(att).mean(0)
    Wv = np.concatenate(wts).mean((0, 1))
    names = [g for g, _ in groups if g not in ("latent", "time")] + ["fused (AE + time)"]
    names = [{"raw": "load", "den": "denoised log load", "garch": "GARCH $h_t$", "weather": "weather",
              "cal": "calendar"}.get(n, n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
    im = axes[0].imshow(A[:, :168], aspect="auto", cmap="viridis", extent=[-167, 0, 24, 1])
    axes[0].set_xlabel("encoder time step (hours before origin)")
    axes[0].set_ylabel("forecast horizon h")
    axes[0].set_title("mean attention of decoder steps to history", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    order = np.argsort(Wv)
    axes[1].barh(np.array(names)[order], Wv[order], color="#2c7fb8")
    axes[1].set_xlabel("mean selection weight")
    axes[1].set_title("encoder variable selection", fontsize=9)
    fig.tight_layout()
    save(fig, f"fig_attention_{ds}.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["cd", "gain", "event", "coverage", "semisynth"]
    for w in which:
        globals()[f"fig_{w}"]()

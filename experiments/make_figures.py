"""Result figures for the paper, generated from saved experiment outputs."""
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
RES = os.path.join(HERE, "results")
PRED = os.path.join(RES, "preds")
CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "..", "figures")
H = 24
UNITS = {"RWD1": "kW", "RWD2": "MW"}
NAMES = {"RWD1": "RWD1 (UCI household)", "RWD2": "RWD2 (PJM East)"}
PAL = {"RCAF": "#c0392b", "TFT": "#2c7fb8", "PatchTST": "#7b3294", "DLinear": "#1a9641", "LSTM": "#e66101",
       "GRU": "#fdb863", "Transformer": "#5e3c99", "XGBoost": "#4d4d4d", "SARIMA-RT": "#a6611a",
       "SeasonalNaive": "#999999"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "savefig.dpi": 300})


def feats(ds):
    with open(os.path.join(CACHE, f"{ds}_rssa.pkl"), "rb") as f:
        return pickle.load(f)


def runs(ds, name, key):
    out = []
    for f in sorted(glob.glob(os.path.join(PRED, f"{ds}_{name}_s*.npz"))):
        d = np.load(f)
        if key in d:
            out.append(d[key])
    return out


def merged(name):
    out = {}
    for p in glob.glob(os.path.join(RES, f"{name}_*.json")):
        out.update(json.load(open(p)))
    return out


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ pipeline stages
def fig_pipeline(ds="RWD2", days=42):
    F = feats(ds)
    alarms = np.asarray(merged("alarms")[ds]["alarms_ae"])
    sel = merged("adapt_selection")[ds]
    n_tr, n_va = F["n_tr"], F["n_va"]
    s0 = n_va + 24 * 60
    al_in = alarms[(alarms >= n_va)]
    if len(al_in):
        s0 = max(n_va, int(al_in[0]) - 24 * days // 2)
    sl = slice(s0, s0 + 24 * days)
    t = pd.to_datetime(F["time_index"][sl])
    thr = np.quantile(F["score"][2 * 168 + 24:n_tr], sel["q"])
    fig, ax = plt.subplots(5, 1, figsize=(7.2, 8.2), sharex=True)
    ax[0].plot(t, F["y"][sl], lw=0.6, color="#555")
    ax[0].set_ylabel(f"load ({UNITS[ds]})")
    ax[1].plot(t, F["den_level"][sl], lw=0.8, color="#2c7fb8")
    ax[1].set_ylabel("RSSA output")
    ax[2].plot(t, F["den"][sl], lw=0.8, color="#1a9641")
    ax[2].set_ylabel("log, z-scored")
    ax[3].plot(t, F["garch"][sl], lw=0.8, color="#e66101")
    ax[3].set_ylabel("GARCH log $h_t$ (z)")
    ax[4].plot(t, F["score"][sl], lw=0.8, color="#7b3294")
    ax[4].axhline(thr, color="k", ls="--", lw=0.8, label="threshold")
    for a in alarms[(alarms >= s0) & (alarms < s0 + 24 * days)]:
        for k in range(5):
            ax[k].axvline(pd.to_datetime(F["time_index"][a]), color="#c0392b", lw=1.0, alpha=0.8)
    ax[4].set_ylabel("AE-CPD score")
    ax[4].legend(loc="upper right", frameon=False)
    for k, lab in enumerate("ABCDE"):
        ax[k].text(0.005, 0.85, lab, transform=ax[k].transAxes, fontweight="bold")
    fig.autofmt_xdate()
    save(fig, "fig_pipeline_steps.png")


# ------------------------------------------------------------------ forecasts
def fig_predictions(days=14):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4))
    for ax, ds in zip(axes, ["RWD1", "RWD2"]):
        F = feats(ds)
        n_va, n = F["n_va"], len(F["y"])
        te = np.arange(n_va, n - H)
        q = np.mean(runs(ds, "RCAF", "cpd_q"), 0)
        tft = np.mean(runs(ds, "TFT", "static"), 0)
        alarms = np.asarray(merged("alarms")[ds]["alarms_ae"])
        al = alarms[alarms >= n_va + 24 * 7]
        c = int(al[0]) if len(al) else n_va + 24 * 30
        i0 = max(0, c - n_va - 24 * days // 2)
        idx = np.arange(i0, min(i0 + 24 * days, len(te)))
        h = 24
        tt = pd.to_datetime(F["time_index"][te[idx] + h])
        ax.plot(tt, F["y"][te[idx] + h], color="k", lw=0.8, label="observed")
        ax.plot(tt, tft[idx, h - 1], color=PAL["TFT"], lw=0.8, label="TFT (static)")
        ax.plot(tt, q[idx, h - 1, 1], color=PAL["RCAF"], lw=0.9, label="RCAF median")
        ax.fill_between(tt, q[idx, h - 1, 0], q[idx, h - 1, 2], color=PAL["RCAF"], alpha=0.18, lw=0,
                        label="RCAF 80% interval")
        for a in alarms:
            if te[idx[0]] <= a <= te[idx[-1]]:
                ax.axvline(pd.to_datetime(F["time_index"][a]), color="#7b3294", ls="--", lw=0.9)
        ax.set_title(f"{NAMES[ds]}: 24-h-ahead forecasts", fontsize=9)
        ax.set_ylabel(UNITS[ds])
    axes[0].legend(ncol=4, frameon=False, fontsize=7.5, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "fig_predictions.png")


# ------------------------------------------------------------------ CPD figures
LABEL = {"Bayesian": "BOCPD"}   # display names


def fig_cpd():
    from cpd import BOCPD, AECPD, CUSUM, DensityRatio, make_synthetic, roc_auc, run_detector
    df = pd.read_csv(os.path.join(RES, "cpd_synthetic.csv"))
    meths = ["CUSUM", "Bayesian", "uLSIF", "RuLSIF", "KLIEP", "RCAF (AE-CPD)"]
    cols = ["#999999", "#2c7fb8", "#fdb863", "#e66101", "#1a9641", "#c0392b"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.13
    for i, m in enumerate(meths):
        d = df[df.method == m].set_index("dataset").loc[["SynD1", "SynD2", "SynD3"]]
        ax.bar(np.arange(3) + (i - 2.5) * w, d.F1, w, yerr=d.F1_std, color=cols[i], label=LABEL.get(m, m),
               capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(range(3), ["SynD1 (mean)", "SynD2 (variance)", "SynD3 (mean + variance)"])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "fig_cpd_f1.png")

    # pooled ROC curves on the test series (threshold-free scores)
    cache = os.path.join(RES, "cpd_test_scores.pkl")
    if os.path.exists(cache):
        S = pickle.load(open(cache, "rb"))
    else:
        ae_pack = pickle.load(open(os.path.join(RES, "aecpd_synthetic.pkl"), "rb"))
        S = {}
        for kind in ("mean", "var", "both"):
            for seed in range(101, 111):
                x, cps, _ = make_synthetic(kind, seed=seed)
                lab = np.zeros(len(x), int)
                for c in cps:
                    lab[c:c + 51] = 1
                for m in meths:
                    det = {"CUSUM": CUSUM(), "Bayesian": BOCPD(), "uLSIF": DensityRatio("ulsif"),
                           "RuLSIF": DensityRatio("rulsif"), "KLIEP": DensityRatio("kliep"),
                           "RCAF (AE-CPD)": AECPD(ae_pack[0], scale=ae_pack[1])}[m]
                    S[(kind, seed, m)] = (run_detector(det, x)[0], lab)
        pickle.dump(S, open(cache, "wb"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, kind, title in zip(axes, ("mean", "var", "both"), ("SynD1", "SynD2", "SynD3")):
        for m, c in zip(meths, cols):
            sc = np.concatenate([S[(kind, s, m)][0] for s in range(101, 111)])
            lab = np.concatenate([S[(kind, s, m)][1] for s in range(101, 111)])
            order = np.argsort(-sc)
            tp = np.cumsum(lab[order]) / lab.sum()
            fp = np.cumsum(1 - lab[order]) / (1 - lab).sum()
            ax.plot(fp, tp, color=c, lw=1.0, label=f"{LABEL.get(m, m)} ({roc_auc(lab, sc):.2f})")
        ax.plot([0, 1], [0, 1], color="k", lw=0.5, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("false positive rate")
        ax.legend(fontsize=5.8, frameon=False, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    fig.tight_layout()
    save(fig, "fig_cpd_roc.png")


# ------------------------------------------------------------------ horizon-wise RMSE
def fig_horizon():
    models = ["SeasonalNaive", "SARIMA-RT", "XGBoost", "LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT", "RCAF"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ds in zip(axes, ["RWD1", "RWD2"]):
        F = feats(ds)
        n_va, n = F["n_va"], len(F["y"])
        te = np.arange(n_va, n - H)
        y = F["y"][te[:, None] + np.arange(1, H + 1)]
        for m in models:
            r = runs(ds, m, "cpd" if m == "RCAF" else "static")
            if not r:
                continue
            rm = np.array([np.sqrt(np.mean((p - y) ** 2, 0)) for p in r])
            ax.plot(range(1, H + 1), rm.mean(0), color=PAL[m], lw=1.6 if m == "RCAF" else 0.9,
                    label="RCAF (adaptive)" if m == "RCAF" else m)
        ax.set_title(NAMES[ds], fontsize=9)
        ax.set_xlabel("horizon h (hours)")
        ax.set_ylabel(f"RMSE ({UNITS[ds]})")
    axes[1].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_rmse_horizon.png")


# ------------------------------------------------------------------ training curves
def fig_loss():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, ds in zip(axes, ["RWD1", "RWD2"]):
        for m in ["LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT", "RCAF"]:
            p = os.path.join(PRED, f"{ds}_{m}_s0.json")
            if not os.path.exists(p):
                continue
            hist = json.load(open(p)).get("history")
            if not hist:
                continue
            v = [hh[1] for hh in hist]
            ax.plot(range(1, len(v) + 1), v, color=PAL[m], lw=1.4 if m == "RCAF" else 0.9, label=m)
        ax.set_title(NAMES[ds], fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation MSE (standardised)")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig_loss.png")


# ------------------------------------------------------------------ adaptation over time
def fig_adaptation(win=24 * 14):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8))
    for ax, ds in zip(axes, ["RWD1", "RWD2"]):
        F = feats(ds)
        n_va, n = F["n_va"], len(F["y"])
        te = np.arange(n_va, n - H)
        y = F["y"][te[:, None] + np.arange(1, H + 1)]
        tt = pd.to_datetime(F["time_index"][te])
        meta = merged("alarms")[ds]
        for key, lab, col in [("static", "RCAF static", "#2c7fb8"), ("periodic", "RCAF periodic", "#1a9641"),
                              ("cpd", "RCAF CPD-triggered", "#c0392b")]:
            r = runs(ds, "RCAF", key)
            if not r:
                continue
            se = np.mean((np.mean(r, 0) - y) ** 2, 1)
            roll = np.sqrt(pd.Series(se).rolling(win, min_periods=win // 2).mean())
            ax.plot(tt, roll, color=col, lw=0.9, label=lab)
        for a in meta["alarms_ae"]:
            if a >= n_va:
                ax.axvline(pd.to_datetime(F["time_index"][a]), color="#7b3294", ls=":", lw=0.8)
        ax.set_ylabel(f"rolling RMSE ({UNITS[ds]})")
        ax.set_title(f"{NAMES[ds]}: 14-day rolling RMSE over the test span (dotted: AE-CPD alarms)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "fig_adaptation.png")


# ------------------------------------------------------------------ interpretability
def fig_attention(ds="RWD2", n_orig=2000):
    import torch
    from run_forecast import Windows, input_spec, make_model, origins
    best = merged("best_cfg")
    F = feats(ds)
    groups = input_spec("RCAF", F)
    win = Windows(groups, F, "cpu")
    model = make_model("RCAF", win.X.shape[1], groups, best[f"{ds}|RCAF"]["cfg"])
    model.load_state_dict(torch.load(os.path.join(PRED, f"{ds}_RCAF_s0.pt"), map_location="cpu"))
    model.eval()
    te = origins(F, "test")[:n_orig]
    att, wts = [], []
    with torch.no_grad():
        for i in range(0, len(te), 250):
            x, f, _ = win.batch(te[i:i + 250])
            model(x, f)
            att.append(model.last_attention.numpy())
            wts.append(model.last_weights.numpy())
    A = np.concatenate(att).mean(0)            # [H, L+H]
    Wv = np.concatenate(wts).mean((0, 1))      # [V]
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
    save(fig, "fig_attention.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["pipeline", "predictions", "cpd", "horizon", "loss", "adaptation", "attention"]
    for w in which:
        globals()[f"fig_{w}"]()

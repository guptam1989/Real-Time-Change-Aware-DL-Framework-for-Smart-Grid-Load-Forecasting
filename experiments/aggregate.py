"""Phase 3 of RCAF v3: change-aware online aggregation and adaptive conformal intervals.

For every horizon h the combiner keeps a weight vector over the experts and
updates it with exponentially weighted averaging and fixed share (Herbster &
Warmuth, 1998) as soon as the target of an earlier forecast becomes known
(delayed feedback: the loss of origin t-h is observed at time t). Adapted
experts created by CPD-triggered updates are "sleeping" until their wake-up
time; while asleep their weight follows the mixture (sleeping-experts
formulation). A CPD alarm re-shares the weights towards the uniform vector.
Hyper-parameters (eta, alpha, rho) are selected on the validation span; all
combinations are evaluated in one vectorised pass.

Prediction intervals: adaptive conformal inference (Gibbs & Candes, 2021) on the
absolute residuals of the combined forecast, per horizon, with a step size that
is multiplied by a boost factor during the week after an alarm.
"""
import glob
import itertools
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES3 = os.path.join(HERE, "results_v3")
PRED3 = os.path.join(RES3, "preds")
AGG = os.path.join(RES3, "agg")
CACHE = os.path.join(HERE, "cache")
H = 24
BASELINE_POOL = ["SARIMA-RT", "XGBoost", "LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT"]
RCAF_EXPERTS = ["RCAF", "RCAF-TRF", "RCAF-GBM"]          # change-aware experts
ADAPTIVE = ["RCAF", "RCAF-TRF", "RCAF-GBM"]              # experts that receive CPD-triggered updates
ETA = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
ALPHA = [0.0, 0.002, 0.01, 0.05, 0.1]
RHO = [0.0, 0.5, 0.9]
ABOOST = [1, 5]       # share-rate multiplier during the week after an alarm
GAMMA = [0.0003, 0.001, 0.003, 0.01, 0.03]
BOOST = [1, 3, 10]
WINDOW = 720
LEVEL = 0.2          # 80 % intervals
BURN = 168           # validation origins ignored while weights and intervals warm up


def load_feats(ds):
    with open(os.path.join(CACHE, f"{ds}_rssa.pkl"), "rb") as f:
        return pickle.load(f)


def load_pred(ds, name, seed, key="static"):
    p = os.path.join(PRED3, f"{ds}_{name}_s{seed}.npz")
    if not os.path.exists(p):
        p = os.path.join(PRED3, f"{ds}_{name}_s0.npz")   # deterministic models
    d = np.load(p)
    return d[key] if key in d else None


def combine(P, awake, y, sigma, resets, etas, alphas, rhos, boosts=(1,), boost_mask=None, record=None):
    """Vectorised fixed-share EWA with sleeping experts and delayed feedback.

    P: [N, H, K] expert forecasts; awake: [N, K] bool; y: [N, H] targets;
    resets: bool [N] (alarm known at origin i); boost_mask: bool [N], True
    during the week after an alarm, when the share rate is multiplied by the
    boost factor. Returns forecasts [C, N, H] for the C parameter combinations,
    the combinations and the final weights. If `record` is a list of horizon
    indices, the weight path of the first combination is also returned."""
    N, _, K = P.shape
    combos = list(itertools.product(etas, alphas, rhos, boosts))
    C = len(combos)
    eta = np.array([c[0] for c in combos], np.float32)[:, None, None]
    alp = np.array([c[1] for c in combos], np.float32)[:, None, None]
    rho = np.array([c[2] for c in combos], np.float32)[:, None, None]
    alp_b = np.minimum(alp * np.array([c[3] for c in combos], np.float32)[:, None, None], 0.5)
    if boost_mask is None:
        boost_mask = np.zeros(N, bool)
    P = P.astype(np.float32)
    w = np.full((C, H, K), 1.0 / K, np.float32)
    out = np.zeros((C, N, H), np.float32)
    path = np.zeros((N, len(record), K), np.float32) if record else None
    hh = np.arange(H)
    for i in range(N):
        # 1. feedback: origin s = i - h resolves for horizon h (target observed now)
        s = i - (hh + 1)
        ok = s >= 0
        if ok.any():
            hs, ss = hh[ok], s[ok]
            Pe = P[ss, hs, :]                                   # [h', K]
            yt = y[ss, hs][:, None]                             # [h', 1]
            le = ((Pe - yt) / sigma) ** 2                       # expert losses
            lm = ((out[:, ss, hs] - yt[:, 0][None]) / sigma) ** 2   # mixture loss [C, h']
            aw = awake[ss][:, :]                                # [h', K]
            loss = np.where(aw[None], le[None], lm[..., None])  # sleeping experts follow the mixture
            wk = w[:, hs, :] * np.exp(-eta * (loss - loss.min(-1, keepdims=True)))
            wk /= wk.sum(-1, keepdims=True)
            a = alp_b if boost_mask[i] else alp
            w[:, hs, :] = (1 - a) * wk + a / K
        # 2. CPD alarm: re-share weights towards uniform
        if resets[i]:
            w = (1 - rho) * w + rho / K
        if path is not None:
            path[i] = w[0][record]
        # 3. forecast with the awake experts
        a = awake[i]
        wa = w[:, :, a]
        wa = wa / wa.sum(-1, keepdims=True)
        out[:, i, :] = np.einsum("chk,hk->ch", wa, P[i][:, a])
    if record:
        return out, combos, w, path
    return out, combos, w


def aci2(pred, y, alarms_mask, gammas, boosts, window=WINDOW, level=LEVEL):
    """Asymmetric adaptive conformal intervals: the upper and lower bounds are
    calibrated separately (miscoverage level/2 each) on the signed residuals.
    Returns (lower half-widths, upper half-widths), each [G, N, H]."""
    e = y - pred
    up, combos = aci(pred, y, alarms_mask, gammas, boosts, window, level / 2, signed=e)
    cap_up = aci.capped
    lo, _ = aci(pred, y, alarms_mask, gammas, boosts, window, level / 2, signed=-e)
    aci.capped = cap_up | aci.capped
    return lo, up, combos


def aci(pred, y, alarms_mask, gammas, boosts, window=WINDOW, level=LEVEL, signed=None):
    """Adaptive conformal intervals for every (gamma, boost) pair. Returns
    half-widths [G, N, H]. alarms_mask[i] marks origins within a week after an alarm.
    If `signed` is given, the scores are the signed residuals (one-sided bound)."""
    N = len(pred)
    combos = list(itertools.product(gammas, boosts))
    G = len(combos)
    gam = np.array([c[0] for c in combos])[:, None]
    bst = np.array([c[1] for c in combos])[:, None]
    a_t = np.full((G, H), level)
    hw = np.zeros((G, N, H))
    capped = np.zeros((G, N, H), bool)   # working level <= 0: idealised interval would be the real line
    buf = np.full((H, window), np.nan)
    cnt = np.zeros(H, int)
    hh = np.arange(H)
    resid = np.abs(y - pred) if signed is None else signed
    for i in range(N):
        s = i - (hh + 1)
        ok = s >= 0
        if ok.any():
            hs, ss = hh[ok], s[ok]
            r = resid[ss, hs]
            err = (r > hw[:, ss, hs]).astype(float)          # [G, h']
            g = np.where(alarms_mask[i], gam * bst, gam)     # [G, 1]
            a_t[:, hs] += g * (level - err)
            buf[hs, cnt[hs] % window] = r
            cnt[hs] += 1
        valid = np.minimum(cnt, window)
        if valid.min() == 0:
            hw[:, i, :] = np.nan
            continue
        srt = np.sort(buf, axis=1)                            # NaNs sort last
        q = np.clip(1 - a_t, 0.0, 1.0)                        # [G, H]
        pos = np.clip(np.ceil(q * (valid[None] + 1)) - 1, 0, valid[None] - 1).astype(int)
        hw[:, i, :] = srt[hh[None, :], pos]
        hw[:, i, :] = np.where(q >= 1.0, np.abs(srt[hh, valid - 1])[None] * 1.5, hw[:, i, :])
        hw[:, i, :] = np.where(q <= 0.0, 0.0, hw[:, i, :])
        capped[:, i, :] = q >= 1.0
    aci.capped = capped
    return hw, combos


def interval_score(lo, hi, y, a=LEVEL):
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def rmse(p, y):
    return float(np.sqrt(np.nanmean((p - y) ** 2)))


def run_dataset(ds, seeds=(0, 1, 2)):
    F = load_feats(ds)
    meta = json.load(open(os.path.join(RES3, f"alarms_{ds}.json")))
    n, n_tr, n_va = meta["n"], meta["n_tr"], meta["n_va"]
    vt = np.arange(n_tr, n - H)
    N = len(vt)
    y = F["y"][vt[:, None] + np.arange(1, H + 1)]
    sigma = float(np.std(F["y"][:n_tr]))
    val_pos = np.arange(BURN, (n_va - H) - n_tr)                # tuning positions (targets in val)
    test_pos = np.arange(n_va - n_tr, N)
    delay = meta["delay"]

    def mask_at(times):
        m = np.zeros(N, bool)
        for t in times:
            if n_tr <= t < n - H:
                m[t - n_tr] = True
        return m

    def week_after(times):
        m = np.zeros(N, bool)
        for t in times:
            if t >= n_tr - 168:
                m[max(0, t - n_tr):max(0, t - n_tr + 168)] = True
        return m

    def wake_mask(times):
        times = [t for t in times if t >= n_tr]
        m = np.zeros(N, bool)
        if times:
            m[max(0, times[0] - n_tr):] = True
        return m

    triggers = {"cpd": (meta["alarms_ae"], [a + delay for a in meta["alarms_ae"]]),
                "periodic": (meta["periodic"], meta["periodic"]),
                "cusum": (meta["alarms_cusum"], [a + delay for a in meta["alarms_cusum"]]),
                "bocpd": (meta["alarms_bocpd"], [a + delay for a in meta["alarms_bocpd"]])}
    os.makedirs(AGG, exist_ok=True)
    results = []
    for seed in seeds:
        stat = {m: load_pred(ds, m, seed) for m in BASELINE_POOL + RCAF_EXPERTS}

        def pool(names, trig=None, spawn=False):
            cols, aw = [stat[m] for m in names], [np.ones(N, bool)] * len(names)
            labels = list(names)
            if spawn and trig is not None:
                for m in ADAPTIVE:
                    if m in names:
                        cols.append(load_pred(ds, m, seed, trig))
                        aw.append(wake_mask(triggers[trig][1]))
                        labels.append(m + "*")
            return np.stack(cols, -1), np.stack(aw, -1), labels

        variants = {
            # name: (pool names, trigger for resets, spawn adapted experts, tune rho)
            "EWA": (BASELINE_POOL + RCAF_EXPERTS, None, False, False, [0.0]),
            "FS": (BASELINE_POOL + RCAF_EXPERTS, None, False, True, [0.0]),
            "FS+reset": (BASELINE_POOL + RCAF_EXPERTS, "cpd", False, True, RHO),
            "FS+spawn": (BASELINE_POOL + RCAF_EXPERTS, "cpd", True, True, [0.0]),
            "FS+reset+spawn": (BASELINE_POOL + RCAF_EXPERTS, "cpd", True, True, RHO),
            "Trigger: periodic": (BASELINE_POOL + RCAF_EXPERTS, "periodic", True, True, RHO),
            "Trigger: CUSUM": (BASELINE_POOL + RCAF_EXPERTS, "cusum", True, True, RHO),
            "Trigger: BOCPD": (BASELINE_POOL + RCAF_EXPERTS, "bocpd", True, True, RHO),
            "FS (baselines only)": (BASELINE_POOL, None, False, True, [0.0]),
            "RCAF (own experts only)": (RCAF_EXPERTS, "cpd", True, True, RHO),
        }
        seed_out, paths = {}, {}
        for vname, (names, trig, spawn, use_alpha, rhos) in variants.items():
            P, aw, labels = pool(names, trig, spawn)
            change_aware = bool(trig) and max(rhos) > 0
            resets = mask_at(triggers[trig][0]) if change_aware else np.zeros(N, bool)
            bmask = week_after(triggers[trig][0]) if change_aware else None
            alphas = ALPHA if use_alpha else [0.0]
            boosts = ABOOST if (change_aware and use_alpha) else [1]
            out, combos, wfinal = combine(P, aw, y, sigma, resets, ETA, alphas, rhos, boosts, bmask)
            scores = np.array([rmse(out[c, val_pos], y[val_pos]) for c in range(len(combos))])
            best = scores.min()
            ties = [c for c in range(len(combos)) if scores[c] <= best * (1 + 1e-9)]
            # when validation cannot distinguish (no alarm in the validation span): rho = 0.5, no boost
            c = min(ties, key=lambda j: (abs(combos[j][2] - 0.5), combos[j][3], combos[j][1]))
            pred = out[c].astype(np.float64)
            prm = dict(eta=combos[c][0], alpha=combos[c][1], rho=combos[c][2], boost=combos[c][3])
            seed_out[vname] = dict(pred=pred, params=prm, val_rmse=float(scores[c]), labels=labels,
                                   weights=wfinal[c].tolist())
            if vname in ("FS+reset", "FS+reset+spawn"):   # weight path (horizons 1 and 24)
                _, _, _, path = combine(P, aw, y, sigma, resets, [prm["eta"]], [prm["alpha"]], [prm["rho"]],
                                        [prm["boost"]], bmask, record=[0, H - 1])
                paths[vname] = path
            del out
        # RCAF: the strength of the reaction to alarms (re-sharing rho, share boost, expert spawning,
        # including no reaction) is selected on the validation span
        sel_v = min(("FS+reset", "FS+reset+spawn"), key=lambda v: seed_out[v]["val_rmse"])
        seed_out["RCAF"] = dict(seed_out[sel_v], selected_variant=sel_v)
        paths["RCAF"] = paths[sel_v]
        # simple combinations
        P, aw, _ = pool(BASELINE_POOL + RCAF_EXPERTS)
        seed_out["Mean"] = dict(pred=P.mean(-1), params={}, val_rmse=rmse(P.mean(-1)[val_pos], y[val_pos]))
        vr = {m: rmse(stat[m][val_pos], y[val_pos]) for m in BASELINE_POOL + RCAF_EXPERTS}
        bm = min(vr, key=vr.get)
        seed_out["Best on validation"] = dict(pred=stat[bm], params=dict(model=bm), val_rmse=vr[bm])

        # adaptive conformal intervals for the full RCAF forecast
        pred = seed_out["RCAF"]["pred"]
        am = np.zeros(N, bool)
        for a in triggers["cpd"][0]:
            if a >= n_tr:
                am[max(0, a - n_tr):max(0, a - n_tr) + 168] = True
        def score(lo, up):
            return float(np.nanmean(interval_score(pred[val_pos] - lo[val_pos], pred[val_pos] + up[val_pos],
                                                   y[val_pos])))
        lo, up, gcombos = aci2(pred, y, am, GAMMA, BOOST)          # asymmetric, change-aware
        cap_all = aci.capped
        isc = np.array([score(lo[g], up[g]) for g in range(len(gcombos))])
        g = int(np.nanargmin(isc))
        cap_frac = float(cap_all[g][test_pos].mean())
        noboost = [j for j, cmb in enumerate(gcombos) if cmb[1] == 1]
        g_nb = noboost[int(np.nanargmin(isc[noboost]))]
        lo0, up0, _ = aci2(pred, y, am, [0.0], [1])                # rolling conformal (no adaptation)
        hws, gs = aci(pred, y, am, GAMMA, [1])                     # symmetric ACI
        g_s = int(np.nanargmin([score(hws[j], hws[j]) for j in range(len(gs))]))
        seed_out["RCAF"]["interval"] = dict(gamma=gcombos[g][0], boost=gcombos[g][1], capped_fraction=cap_frac)
        intervals = {"RCAF + ACI": (lo[g], up[g]), "RCAF + ACI (no boost)": (lo[g_nb], up[g_nb]),
                     "RCAF + rolling conformal": (lo0[0], up0[0]), "RCAF + symmetric ACI": (hws[g_s], hws[g_s])}
        np.savez_compressed(os.path.join(AGG, f"{ds}_s{seed}.npz"), origins=vt, y=y,
                            test_pos=test_pos, val_pos=val_pos, weight_path=paths["RCAF"],
                            weight_labels=np.array(seed_out["RCAF"]["labels"]),
                            **{f"pred::{k}": v["pred"] for k, v in seed_out.items()},
                            **{f"lo::{k}": v[0] for k, v in intervals.items()},
                            **{f"up::{k}": v[1] for k, v in intervals.items()})
        info = {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in seed_out.items()}
        info["best_single_on_val"] = bm
        info["val_rmse_single"] = vr
        json.dump(info, open(os.path.join(AGG, f"{ds}_s{seed}.json"), "w"), indent=1, default=float)
        results.append(info)
        print(ds, seed, {k: round(rmse(v["pred"][test_pos], y[test_pos]), 3) for k, v in seed_out.items()}, flush=True)
    return results


if __name__ == "__main__":
    names = sys.argv[1:] or sorted({os.path.basename(p).split("_")[1].replace(".json", "")
                                    for p in glob.glob(os.path.join(RES3, "alarms_*.json"))})
    for ds in names:
        run_dataset(ds)

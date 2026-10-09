"""Gradient-boosting forecasters over the validation + test spans.

XGBoost (baseline): one model per horizon on 30 lagged loads, current and 1-h
lagged weather and calendar features.
RCAF-GBM (change-aware expert): the same inputs plus the RCAF features at the
forecast origin (denoised log load, log GARCH variance, seven time-domain
features, the nine-dimensional AE descriptor and the AE-CPD score). At the
adaptation times of a trigger it is updated by boosting continuation: 50 more
trees are fitted on the most recent 720 origins whose targets are observed plus
an equal number of replayed earlier origins (the same data recipe as the
fine-tuning of the neural experts). Updates are cumulative.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_forecast as rf  # noqa: E402

LAGS = list(range(24)) + [47, 71, 95, 119, 143, 167]
H = rf.H


def design(F, idx, change_aware):
    z, w, cal = F["raw"], F["weather"], F["cal"]
    cols = [z[idx - k] for k in LAGS]
    cols += [w[idx, j] for j in range(w.shape[1])] + [w[idx - 1, j] for j in range(w.shape[1])]
    cols += [cal[idx, j] for j in range(cal.shape[1])]
    if change_aware:
        cols += [F["den"][idx], F["garch"][idx], np.log1p(F["score"][idx])]
        cols += [F["time"][idx, j] for j in range(F["time"].shape[1])]
        cols += [F["latent"][idx, j] for j in range(F["latent"].shape[1])]
    return np.stack(cols, 1).astype(np.float32)


N_ADD, WINDOW = 50, 720


def _model(n_estimators, device, early=True):
    import xgboost as xgb
    kw = dict(n_estimators=n_estimators, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", device=device, random_state=0,
              n_jobs=int(os.environ.get("NCPUS", "4")))
    if early:
        kw["early_stopping_rounds"] = 50
    return xgb.XGBRegressor(**kw)


def run(ds, plans, out_dir, device="cpu"):
    F = rf.features(ds)
    n, n_tr, n_va = len(F["y"]), F["n_tr"], F["n_va"]
    tr = np.arange(max(LAGS), n_tr - H)
    va = rf.origins(F, "val")
    vt = rf.origins(F, "valtest")
    for name, ca in (("XGBoost", False), ("RCAF-GBM", True)):
        path = os.path.join(out_dir, f"{ds}_{name}_s0.npz")
        if os.path.exists(path):
            continue
        t0 = time.perf_counter()
        Xtr, Xva, Xvt = design(F, tr, ca), design(F, va, ca), design(F, vt, ca)
        static = np.zeros((len(vt), H))
        n_best, models = [], []
        for h in range(1, H + 1):
            m = _model(2000, device)
            m.fit(Xtr, F["raw"][tr + h], eval_set=[(Xva, F["raw"][va + h])], verbose=False)
            n_best.append(int(m.best_iteration) + 1)
            models.append(m.get_booster()[: n_best[-1]])    # keep the early-stopped ensemble
            static[:, h - 1] = m.predict(Xvt, iteration_range=(0, n_best[-1]))
        out = {"origins": vt, "static": static * F["sd_y"] + F["mu_y"]}
        info = dict(train_time=time.perf_counter() - t0, n_trees=n_best, device=device)
        if ca:
            # CPD-/trigger-driven updates by boosting continuation (cumulative)
            upd_times = []
            for tag, plan in plans.items():
                times = [t for t in plan["times"] if vt[0] < t <= vt[-1]]
                pred = static.copy()
                current = list(models)
                for i, tau in enumerate(times):
                    t1 = time.perf_counter()
                    nxt = times[i + 1] if i + 1 < len(times) else vt[-1] + 1
                    seg = (vt >= tau) & (vt < nxt)
                    recent = np.arange(max(max(LAGS), tau - H - WINDOW), tau - H + 1)
                    idx = recent
                    if recent[0] > max(LAGS):
                        rng = np.random.default_rng(i)
                        old = rng.choice(np.arange(max(LAGS), recent[0]),
                                         size=min(len(recent), recent[0] - max(LAGS)), replace=False)
                        idx = np.concatenate([recent, old])
                    Xu = design(F, idx, ca)
                    Xseg = design(F, vt[seg], ca)
                    for h in range(1, H + 1):
                        m = _model(N_ADD, device, early=False)
                        m.fit(Xu, F["raw"][idx + h], xgb_model=current[h - 1], verbose=False)
                        current[h - 1] = m.get_booster()
                        if seg.any():
                            pred[seg, h - 1] = m.predict(Xseg)
                    upd_times.append(time.perf_counter() - t1)
                out[tag] = pred * F["sd_y"] + F["mu_y"]
                info[f"adapt_{tag}_n"] = len(times)
            info["update_time_mean"] = float(np.mean(upd_times)) if upd_times else 0.0
        np.savez_compressed(path, **out)
        json.dump(info, open(path.replace(".npz", ".json"), "w"), indent=1)
        print(ds, name, f"{time.perf_counter() - t0:.0f}s", flush=True)

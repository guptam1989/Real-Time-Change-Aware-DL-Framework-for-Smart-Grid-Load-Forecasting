"""Classical baselines on the same test origins as the deep models.

- Seasonal naive:   y_hat(t+h) = y(t+h-24)            (h <= 24)
- SARIMA-RT:        SARIMA(2,0,1)(1,0,1)_24 on the standardised load; parameters
                    re-estimated every 168 h on the most recent 4 weeks, state
                    re-filtered at every origin (online, no future data).
- XGBoost:          one gradient-boosted model per horizon on lagged load,
                    weather and calendar features; early stopping on validation.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_forecast import DATASETS, H, L, RES, features, origins  # noqa: E402

NPROC = int(os.environ.get("NCPUS", "2"))  # XGBoost threads; SARIMA chunks run sequentially
LAGS = list(range(24)) + [47, 71, 95, 119, 143, 167]


def seasonal_naive(F, idx):
    y = F["y"]
    return np.stack([y[idx + h - 24] for h in range(1, H + 1)], 1)


def _sarima_chunk(args):
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings("ignore")
    z, start, stop, fit_win, filt_win = args
    hist = z[start - fit_win:start]
    mod = SARIMAX(hist, order=(2, 0, 1), seasonal_order=(1, 0, 1, 24), trend="c")
    t0 = time.perf_counter()
    res = mod.fit(disp=False, maxiter=200)
    fit_time = time.perf_counter() - t0
    out, lat = [], []
    for t in range(start, stop):
        t1 = time.perf_counter()
        r = SARIMAX(z[t + 1 - filt_win:t + 1], order=(2, 0, 1), seasonal_order=(1, 0, 1, 24),
                    trend="c").filter(res.params)
        out.append(r.forecast(H))
        lat.append(time.perf_counter() - t1)
    return np.asarray(out), fit_time, float(np.mean(lat))


def sarima_rt(F, idx, refit=168, fit_win=672, filt_win=336):
    z = F["raw"]
    chunks = [(z, s, min(s + refit, idx[-1] + 1), fit_win, filt_win) for s in range(idx[0], idx[-1] + 1, refit)]
    res = list(map(_sarima_chunk, chunks))  # the CPU queue does not allow multi-process jobs
    pred = np.concatenate([r[0] for r in res])
    return pred * F["sd_y"] + F["mu_y"], dict(fit_time=float(np.mean([r[1] for r in res])),
                                                lat_ms=1e3 * float(np.mean([r[2] for r in res])))


def lag_features(F, idx):
    z, w, cal = F["raw"], F["weather"], F["cal"]
    cols = [z[idx - k] for k in LAGS]
    cols += [w[idx, j] for j in range(w.shape[1])] + [w[idx - 1, j] for j in range(w.shape[1])]
    cols += [cal[idx, j] for j in range(cal.shape[1])]
    return np.stack(cols, 1)


def xgboost(F, tr, va, te):
    import xgboost as xgb
    Xtr, Xva, Xte = lag_features(F, tr), lag_features(F, va), lag_features(F, te)
    pred = np.zeros((len(te), H))
    t0 = time.perf_counter()
    for h in range(1, H + 1):
        m = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, early_stopping_rounds=30, n_jobs=NPROC, tree_method="hist")
        m.fit(Xtr, F["raw"][tr + h], eval_set=[(Xva, F["raw"][va + h])], verbose=False)
        pred[:, h - 1] = m.predict(Xte)
    train_time = time.perf_counter() - t0
    return pred * F["sd_y"] + F["mu_y"], dict(train_time=train_time)


def main():
    os.makedirs(os.path.join(RES, "preds"), exist_ok=True)
    for ds in DATASETS:
        F = features(ds)
        tr, va, te = origins(F, "train"), origins(F, "val"), origins(F, "test")
        tr = tr[tr >= max(LAGS)]
        np.savez_compressed(os.path.join(RES, "preds", f"{ds}_SeasonalNaive_s0.npz"), static=seasonal_naive(F, te))
        t0 = time.time()
        p, info = xgboost(F, tr, va, te)
        np.savez_compressed(os.path.join(RES, "preds", f"{ds}_XGBoost_s0.npz"), static=p)
        print(ds, "xgboost", f"{time.time() - t0:.0f}s", info, flush=True)
        t0 = time.time()
        p, info = sarima_rt(F, te)
        np.savez_compressed(os.path.join(RES, "preds", f"{ds}_SARIMA-RT_s0.npz"), static=p)
        with open(os.path.join(RES, f"sarima_info_{ds}.txt"), "w") as f:
            f.write(str(info))
        print(ds, "sarima", f"{time.time() - t0:.0f}s", info, flush=True)


if __name__ == "__main__":
    main()

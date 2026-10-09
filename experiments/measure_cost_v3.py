"""Single-thread CPU cost of the v3 components (edge-device proxy): online
aggregation and conformal intervals per sample, forecaster inference per origin
(batch size 1) and the time of one CPD-triggered update."""
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
import aggregate as agg  # noqa: E402
import run_forecast as rf  # noqa: E402
from run_gbm import LAGS, design  # noqa: E402

torch.set_num_threads(1)
RES3 = os.path.join(HERE, "results_v3")


def main(ds="RWD2", n_steps=2000):
    F = pickle.load(open(os.path.join(rf.CACHE, f"{ds}_rssa.pkl"), "rb"))
    meta = json.load(open(os.path.join(RES3, f"alarms_{ds}.json")))
    n_tr = meta["n_tr"]
    vt = rf.origins(F, "valtest")[:n_steps]
    y = F["y"][vt[:, None] + np.arange(1, 25)]
    K = 14
    P = y[:, :, None] + np.random.default_rng(0).normal(0, 100, (len(vt), 24, K))
    aw = np.ones((len(vt), K), bool)
    rows = []
    t0 = time.perf_counter()
    agg.combine(P, aw, y, 1000.0, np.zeros(len(vt), bool), [1.0], [0.01], [0.5], [5], np.zeros(len(vt), bool))
    rows.append(dict(component="Online aggregation (14 experts, 24 horizons)", ms=1e3 * (time.perf_counter() - t0) / len(vt)))
    pred = P.mean(-1)
    t0 = time.perf_counter()
    agg.aci2(pred, y, np.zeros(len(vt), bool), [0.003], [1])
    rows.append(dict(component="Asymmetric ACI (24 horizons, window 720)", ms=1e3 * (time.perf_counter() - t0) / len(vt)))
    best = json.load(open(os.path.join(RES3, f"best_cfg_{ds}.json")))
    for m in ["RCAF", "RCAF-TRF"]:
        groups = rf.input_spec(m, F)
        win = rf.Windows(groups, F, "cpu")
        model = rf.make_model(m, win.X.shape[1], groups, best[m]["cfg"]).eval()
        x, f, _ = win.batch(np.array([vt[0]]))
        with torch.no_grad():
            model(x, f)
            t0 = time.perf_counter()
            for _ in range(200):
                model(x, f)
        rows.append(dict(component=f"{'RCAF-TFT' if m == 'RCAF' else m} forecast (24 h, batch 1)",
                         ms=1e3 * (time.perf_counter() - t0) / 200))
        ft = dict(window=720, epochs=1, lr=1e-4, replay=1)
        tau = int(vt[-1])
        idx = np.arange(tau - 24 - 720, tau - 24 + 1)
        old = np.random.default_rng(0).choice(np.arange(167, idx[0]), len(idx), replace=False)
        t0 = time.perf_counter()
        rf.finetune(model, win, np.concatenate([idx, old]), None, 0, ft["epochs"], ft["lr"], m == "RCAF")
        rows.append(dict(component=f"{'RCAF-TFT' if m == 'RCAF' else m} CPD-triggered fine-tuning",
                         ms=1e3 * (time.perf_counter() - t0)))
    # gradient boosting: inference of the 24 horizon models and one continuation update
    import xgboost as xgb
    tr = np.arange(max(LAGS), n_tr - 24)
    X = design(F, tr, True)
    models = []
    for h in range(1, 25):
        m = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=0)
        m.fit(X, F["raw"][tr + h], verbose=False)
        models.append(m)
    x1 = design(F, vt[:1], True)
    t0 = time.perf_counter()
    for _ in range(50):
        for m in models:
            m.predict(x1)
    rows.append(dict(component="RCAF-GBM forecast (24 models, one origin)", ms=1e3 * (time.perf_counter() - t0) / 50))
    tau = int(vt[-1])
    idx = np.arange(tau - 24 - 720, tau - 24 + 1)
    old = np.random.default_rng(0).choice(np.arange(max(LAGS), idx[0]), len(idx), replace=False)
    idx = np.concatenate([idx, old])
    Xu = design(F, idx, True)
    t0 = time.perf_counter()
    for h, m in enumerate(models, 1):
        u = xgb.XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=0)
        u.fit(Xu, F["raw"][idx + h], xgb_model=m.get_booster(), verbose=False)
    rows.append(dict(component="RCAF-GBM CPD-triggered update (50 trees x 24 models)", ms=1e3 * (time.perf_counter() - t0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES3, "cost_v3.csv"), index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["RWD2"]))

"""v3 pipeline: forecasters (experts) for change-aware online aggregation.

Stages (python run_v3.py <stage>; datasets selected with RCAF_DS=name[,name]):
  features  causal features for each dataset (RSSA version only)
  search    validation search. Held-out datasets: every neural model on a small
            grid. Development datasets: only the new RCAF-TRF expert (the other
            configurations were selected earlier).
  final     static and adapted forecasts over the validation + test spans for
            every neural expert and baseline (three seeds)
  gbm       RCAF-GBM (change-aware features) and XGBoost baseline, static and
            refitted at the adaptation times of each trigger
  all       features -> search -> final -> gbm

The adaptation settings are fixed for all datasets (frozen after the
development experiments): AE-CPD threshold = 0.95 quantile of the training-span
score, adaptation 24 h after an alarm, one fine-tuning epoch at lr 1e-4 on the
last 720 origins plus an equal number of replayed origins.
"""
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_forecast as rf  # noqa: E402

RES3 = os.path.join(HERE, "results_v3")
PRED3 = os.path.join(RES3, "preds")
DEV = {"RWD1": "rwd1_uci_household.csv", "RWD2": "rwd2_pjm_east.csv"}
HELDOUT = {"DE": "ho_de.csv", "FR": "ho_fr.csv", "IT": "ho_it.csv", "ES": "ho_es.csv", "GB": "ho_gb.csv",
           "ERCOT": "ho_ercot.csv", "CAISO": "ho_caiso.csv", "SPP": "ho_spp.csv"}
ALL = {**DEV, **HELDOUT}
_sel = os.environ.get("RCAF_DS")
DATASETS = {k: v for k, v in ALL.items() if not _sel or k in _sel.replace("+", ",").split(",")}

Q = 0.95
FT = dict(window=720, epochs=1, lr=1e-4, replay=1)
DELAY = 24
SMALL_GRID = [dict(d=d, lr=lr, dropout=0.1) for d, lr in itertools.product((64, 128), (1e-3, 3e-4))]
BASE_NEURAL = ["LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT"]
EXPERTS_NEURAL = ["RCAF", "RCAF-TRF"]


def cfg_path(ds):
    return os.path.join(RES3, f"best_cfg_{ds}.json")


def alarm_path(ds):
    return os.path.join(RES3, f"alarms_{ds}.json")


# ------------------------------------------------------------------ features
def stage_features():
    from features import build_features
    os.makedirs(rf.CACHE, exist_ok=True)
    for ds, f in DATASETS.items():
        path = os.path.join(rf.CACHE, f"{ds}_rssa.pkl")
        if os.path.exists(path):
            continue
        df = pd.read_csv(os.path.join(HERE, "data", f), parse_dates=["time"] +
                         (["local_time"] if "local_time" in pd.read_csv(os.path.join(HERE, "data", f), nrows=1) else []))
        t0 = time.time()
        F = build_features(df, use_rssa=True)
        F["time_index"] = df["time"].astype(str).values
        with open(path, "wb") as fh:
            pickle.dump(F, fh)
        print(ds, f"features {time.time() - t0:.0f}s", F["garch_params"], flush=True)


# ------------------------------------------------------------------ search
def stage_search():
    os.makedirs(RES3, exist_ok=True)
    old = rf.load_merged("best_cfg")  # configurations selected for the development datasets
    for ds in DATASETS:
        if os.path.exists(cfg_path(ds)):
            continue
        jobs = []
        if ds in DEV:
            best = {m: old[f"{ds}|{m}"] for m in BASE_NEURAL + ["RCAF"]}
            models = ["RCAF-TRF"]
        else:
            best = {}
            models = BASE_NEURAL + EXPERTS_NEURAL
        for m in models:
            grid = SMALL_GRID[:1] if m == "DLinear" else SMALL_GRID
            for c in grid:
                c = {**c, "highway": 1} if m == "RCAF" else dict(c)
                jobs.append(dict(ds=ds, model=m, cfg=c, seed=0, mode="search"))
        res = rf.run_jobs(rf.job_train, jobs, f"v3search_{ds}")
        for r in res:
            m = r["spec"]["model"]
            if m not in best or r["val_mse"] < best[m]["val_mse"]:
                best[m] = dict(cfg=r["spec"]["cfg"], val_mse=r["val_mse"])
        if "RCAF" in models:
            best["RCAF"]["cfg"]["alpha"] = 0.5
            ajobs = [dict(ds=ds, model="RCAF", cfg={**best["RCAF"]["cfg"], "alpha": a}, seed=0, mode="search")
                     for a in (0.25, 0.75)]
            for r in rf.run_jobs(rf.job_train, ajobs, f"v3alpha_{ds}"):
                if r["val_mse"] < best["RCAF"]["val_mse"]:
                    best["RCAF"] = dict(cfg=r["spec"]["cfg"], val_mse=r["val_mse"])
        json.dump(best, open(cfg_path(ds), "w"), indent=1)
        pd.DataFrame([dict(model=r["spec"]["model"], **r["spec"]["cfg"], val_mse=r["val_mse"],
                           train_time=r["train_time"], params=r["params"]) for r in res]).to_csv(
            os.path.join(RES3, f"search_{ds}.csv"), index=False)
        print(ds, json.dumps(best), flush=True)


# ------------------------------------------------------------------ adaptation plans
def make_plans(ds):
    F = rf.features(ds)
    n, n_tr, n_va = len(F["y"]), F["n_tr"], F["n_va"]
    al_ae = rf.trigger_alarms(F, "ae", Q)
    n_cal = max(1, int(np.sum(al_ae < n_va)))
    al_cu = rf.trigger_alarms(F, "cusum", target_count=n_cal)
    al_bo = rf.trigger_alarms(F, "bocpd", target_count=n_cal)
    n_vt = int(np.sum(al_ae >= n_tr))
    k = max(1, n_vt)
    periodic = (n_tr + np.arange(1, k + 1) * (n - n_tr) // (k + 1)).astype(int)
    thr = float(np.quantile(F["score"][2 * 168 + 24:n_tr], Q))
    meta = dict(threshold=thr, q=Q, ft=FT, delay=DELAY, n=n, n_tr=n_tr, n_va=n_va,
                alarms_ae=al_ae.tolist(), alarms_cusum=al_cu.tolist(), alarms_bocpd=al_bo.tolist(),
                periodic=periodic.tolist())
    plans = {"cpd": dict(times=(al_ae + DELAY).tolist(), ft=FT),
             "periodic": dict(times=periodic.tolist(), ft=FT),
             "cusum": dict(times=(al_cu + DELAY).tolist(), ft=FT),
             "bocpd": dict(times=(al_bo + DELAY).tolist(), ft=FT)}
    return plans, meta


def stage_final():
    os.makedirs(PRED3, exist_ok=True)
    jobs = []
    for ds in DATASETS:
        best = json.load(open(cfg_path(ds)))
        plans, meta = make_plans(ds)
        json.dump(meta, open(alarm_path(ds), "w"))
        for seed in rf.SEEDS:
            for m in BASE_NEURAL + EXPERTS_NEURAL:
                spec = dict(ds=ds, model=m, cfg=best[m]["cfg"], seed=seed, mode="final", span="valtest",
                            out=os.path.join(PRED3, f"{ds}_{m}_s{seed}"))
                if m in EXPERTS_NEURAL:
                    spec["adapt"] = plans
                    spec["save_model"] = seed == 0
                jobs.append(spec)
    rf.run_jobs(rf.job_train, jobs, "v3final_" + "_".join(DATASETS))


def stage_gbm():
    if rf.GPUS and rf.GPUS[0] is not None:  # same device as the neural jobs (MIG UUID)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rf.GPUS[0])
    import run_gbm
    for ds in DATASETS:
        plans, meta = make_plans(ds)
        run_gbm.run(ds, plans, PRED3)


def stage_all():
    t0 = time.time()
    stage_features()
    print(f"features done {time.time() - t0:.0f}s", flush=True)
    stage_search()
    print(f"search done {time.time() - t0:.0f}s", flush=True)
    stage_final()
    print(f"final done {time.time() - t0:.0f}s", flush=True)
    stage_gbm()
    print(f"gbm done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"features": stage_features, "search": stage_search, "final": stage_final, "gbm": stage_gbm,
     "all": stage_all}[sys.argv[1]]()

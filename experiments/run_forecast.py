"""Real-world forecasting experiments (Sections 5.2-5.4).

Stages (python run_forecast.py <stage>):
  features  compute causal features for each dataset (with / without RSSA)
  search    hyper-parameter search on the validation span (seed 0)
  final     train every model with the selected configuration (3 seeds) and
            forecast the test span; RCAF variants with online adaptation
  evaluate  metrics, Diebold-Mariano tests, tables (CSV)

Protocol: chronological split 60/10/30 %; input window L = 168 h; direct
multi-horizon forecasts for h = 1..24 h; all errors computed on the raw load
in original units.
"""
import copy
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
RES = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
ALL_DATASETS = {"RWD1": "rwd1_uci_household.csv", "RWD2": "rwd2_pjm_east.csv"}
_sel = os.environ.get("RCAF_DS")
DATASETS = {k: v for k, v in ALL_DATASETS.items() if not _sel or k in _sel.replace("+", ",").split(",")}
TAG = "_".join(DATASETS)
L, H = 168, 24
SEEDS = [0, 1, 2]
# RCAF_GPUS=pbs: use the GPU assigned by the PBS scheduler (inherited CUDA_VISIBLE_DEVICES).
# RCAF_GPUS=auto: use the single GPU with the most free memory at start-up.
_g = os.environ.get("RCAF_GPUS", "4")


def _freest_gpu():
    """UUID of the GPU or MIG slice with the most free memory. UUIDs are
    unambiguous; numeric indices differ between CUDA and nvidia-smi when MIG is
    enabled, and a MIG-enabled parent GPU cannot be used directly, so every
    usable device is probed in a short subprocess."""
    import re
    import subprocess
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    devices, parent = [], None
    for line in listing.splitlines():
        m = re.search(r"UUID: ((?:GPU|MIG)-[0-9a-f-]+)", line)
        if not m:
            continue
        if m.group(1).startswith("GPU-"):
            parent = m.group(1)
            devices.append(parent)
        else:
            if parent in devices:
                devices.remove(parent)            # MIG-enabled parent is not directly usable
            devices.append(m.group(1))
    probe = "import torch; f, t = torch.cuda.mem_get_info(); print(f)"
    best, best_free = None, -1
    for u in devices:
        try:
            r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=u))
            free = int(r.stdout.strip().splitlines()[-1])
        except Exception:
            continue
        if free > best_free:
            best, best_free = u, free
    print(f"selected device {best} with {best_free / 2 ** 30:.1f} GiB free", flush=True)
    return best


GPUS = [None] if _g == "pbs" else ([_freest_gpu()] if _g == "auto" else _g.split(","))
WORKERS_PER_GPU = int(os.environ.get("RCAF_WPG", "6"))


def save_json(obj, name):
    with open(os.path.join(RES, f"{name}_{TAG}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_merged(name):
    import glob
    out = {}
    for p in glob.glob(os.path.join(RES, f"{name}_*.json")):
        out.update(json.load(open(p)))
    return out
DEEP = ["LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT", "RCAF"]


# ---------------------------------------------------------------- data access
def features(ds, use_rssa=True):
    path = os.path.join(CACHE, f"{ds}_{'rssa' if use_rssa else 'raw'}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def input_spec(model, F, drop=()):
    """Column groups (name, array [n, w]) for each model family."""
    base = [("raw", F["raw"][:, None]), ("weather", F["weather"])]
    if model in ("RCAF", "RCAF-TRF"):
        g = [("raw", F["raw"][:, None]), ("den", F["den"][:, None]), ("garch", F["garch"][:, None]),
             ("weather", F["weather"]), ("latent", F["latent"]), ("time", F["time"])]
        g = [x for x in g if x[0] not in drop]
    else:
        g = base
    return g + [("cal", F["cal"])]


def origins(F, part):
    n, n_tr, n_va = len(F["y"]), F["n_tr"], F["n_va"]
    if part == "train":
        return np.arange(L - 1, n_tr - H)
    if part == "val":
        return np.arange(n_tr, n_va - H)
    if part == "valtest":  # every origin after the training span (streaming evaluation)
        return np.arange(n_tr, n - H)
    return np.arange(n_va, n - H)


# ---------------------------------------------------------------- deep models
def make_model(name, n_in, groups, cfg):
    from models import TFT, RNN, TransformerForecaster, DLinear, PatchTST
    d, dr = cfg["d"], cfg["dropout"]
    if name in ("TFT", "RCAF"):
        return TFT([(g, a.shape[1]) for g, a in groups], n_known=6, horizon=H, d=d,
                   heads=4, lstm_layers=2, dropout=dr, alpha=cfg.get("alpha", 0.5),
                   highway=bool(cfg.get("highway", 0)), seq_len=L)
    if name in ("LSTM", "GRU"):
        return RNN(n_in, H, d, 2, dr, name.lower())
    if name == "Transformer":
        return TransformerForecaster(n_in, H, d, 4, 2, dr)
    if name == "RCAF-TRF":  # change-aware Transformer expert (RCAF inputs + autoregressive highway)
        return TransformerForecaster(n_in, H, d, 4, 2, dr, highway=True, seq_len=L)
    if name == "DLinear":
        return DLinear(L, H)
    if name == "PatchTST":
        return PatchTST(L, H, 24, 12, d, 4, 2, dr)
    raise ValueError(name)


class Windows:
    """GPU-resident sliding windows over the feature matrix."""

    def __init__(self, groups, F, device):
        import torch
        self.X = torch.as_tensor(np.concatenate([a for _, a in groups], 1), dtype=torch.float32, device=device)
        self.cal = torch.as_tensor(F["cal"], dtype=torch.float32, device=device)
        self.y = torch.as_tensor(F["raw"], dtype=torch.float32, device=device)
        self.offL = torch.arange(-L + 1, 1, device=device)
        self.offH = torch.arange(1, H + 1, device=device)
        self.device = device

    def batch(self, t):
        import torch
        t = torch.as_tensor(t, device=self.device)
        return self.X[t[:, None] + self.offL], self.cal[t[:, None] + self.offH], self.y[t[:, None] + self.offH]


def fit(model, win, tr, va, cfg, seed, quantile, max_epochs=60, patience=8, batch=256, verbose=False, hist=None):
    import torch
    from models import quantile_loss
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    lossf = (lambda p, y: quantile_loss(p, y)) if quantile else (lambda p, y: ((p - y) ** 2).mean())
    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        tl = []
        for i in range(0, len(perm), batch):
            x, f, y = win.batch(perm[i:i + batch])
            opt.zero_grad()
            loss = lossf(model(x, f), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss))
        v = evaluate_loss(model, win, va, quantile)
        if hist is not None:
            hist.append((float(np.mean(tl)), v))
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best


def evaluate_loss(model, win, idx, quantile):
    p = predict_z(model, win, idx, quantile)
    import torch
    y = win.y[torch.as_tensor(idx, device=win.device)[:, None] + win.offH].cpu().numpy()
    return float(np.mean((p - y) ** 2))


def predict_z(model, win, idx, quantile, all_q=False, batch=512):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            x, f, _ = win.batch(idx[i:i + batch])
            p = model(x, f)
            if quantile and not all_q:
                p = p[..., 1]
            out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def finetune(model, win, idx, cfg, seed, epochs, lr, quantile, batch=64):
    import torch
    from models import quantile_loss
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(idx)
        for i in range(0, len(perm), batch):
            x, f, y = win.batch(perm[i:i + batch])
            opt.zero_grad()
            p = model(x, f)
            loss = quantile_loss(p, y) if quantile else ((p - y) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def alarms_from(score, thr, start, end, refractory=168):
    al, last = [], -10 ** 9
    for t in range(start, end):
        if score[t] > thr and t - last > refractory:
            al.append(t)
            last = t
    return np.asarray(al, int)


def online_predict(model, win, F, eval_idx, adapt_times, ft, seed, quantile, all_q=False):
    """Forecast eval_idx sequentially; at each adaptation time tau the model is
    fine-tuned on the most recent `ft['window']` origins whose targets are
    observed by tau, and used for all origins >= tau."""
    import torch
    model = copy.deepcopy(model)
    preds, times = [], []
    bounds = [t for t in adapt_times if eval_idx[0] < t <= eval_idx[-1]]
    cuts = [eval_idx[0]] + bounds + [eval_idx[-1] + 1]
    for i in range(len(cuts) - 1):
        if i > 0:
            tau = cuts[i]
            idx = np.arange(max(L - 1, tau - H - ft["window"]), tau - H + 1)
            if ft.get("replay") and idx[0] > L - 1:
                # experience replay: an equal number of origins drawn from all earlier data
                rng = np.random.default_rng(seed + i)
                old = rng.choice(np.arange(L - 1, idx[0]), size=min(len(idx), idx[0] - L + 1), replace=False)
                idx = np.concatenate([idx, old])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            finetune(model, win, idx, None, seed + i, ft["epochs"], ft["lr"], quantile)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        seg = eval_idx[(eval_idx >= cuts[i]) & (eval_idx < cuts[i + 1])]
        if len(seg):
            preds.append(predict_z(model, win, seg, quantile, all_q))
    return np.concatenate(preds), times


def to_units(pz, F):
    return pz * F["sd_y"] + F["mu_y"]


# ---------------------------------------------------------------- job runner
def run_jobs(fn, jobs, tag="jobs"):
    """Run each job in its own subprocess (one log file per job), keeping
    len(GPUS) * WORKERS_PER_GPU jobs in flight. Failed jobs are retried once."""
    import subprocess
    jdir = os.path.join(HERE, "jobs", f"{tag}_{TAG}")
    os.makedirs(jdir, exist_ok=True)
    slots = [g for g in GPUS for _ in range(WORKERS_PER_GPU)]
    pending = list(enumerate(jobs))
    running, tries = {}, {}
    while pending or running:
        while pending and len(running) < len(slots):
            i, job = pending.pop(0)
            spec_path = os.path.join(jdir, f"{i}.spec.json")
            res_path = spec_path.replace(".spec.json", ".result.json")
            if os.path.exists(res_path) and os.path.exists(spec_path) and json.load(open(spec_path)) == json.loads(json.dumps(job)):
                continue  # identical job already completed (resume)
            free = [s for s in range(len(slots)) if s not in {v[1] for v in running.values()}][0]
            with open(spec_path, "w") as f:
                json.dump(job, f)
            env = dict(os.environ, OMP_NUM_THREADS="1")
            if slots[free] is not None:
                env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(slots[free]),
                           RCAF_GPUS=str(slots[free]))    # children must not probe the devices again
            log = open(os.path.join(jdir, f"{i}.log"), "w")
            log.write(f"gpu={slots[free]}\n")
            log.flush()
            p = subprocess.Popen([sys.executable, "-W", "ignore", os.path.abspath(__file__), "job", spec_path],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            running[i] = (p, free, job, log)
        time.sleep(2)
        for i in list(running):
            p, slot, job, log = running[i]
            if p.poll() is None:
                continue
            log.close()
            del running[i]
            if p.returncode != 0:
                tries[i] = tries.get(i, 0) + 1
                print(f"job {i} failed (rc={p.returncode}), see {jdir}/{i}.log", flush=True)
                if tries[i] < 2:
                    pending.append((i, job))
    out = []
    for i in range(len(jobs)):
        rp = os.path.join(jdir, f"{i}.result.json")
        out.append(json.load(open(rp)) if os.path.exists(rp) else None)
    return [o for o in out if o is not None]


def _job_entry(spec_path):
    import torch
    torch.set_num_threads(1)
    spec = json.load(open(spec_path))
    res = job_train(spec)
    with open(spec_path.replace(".spec.json", ".result.json"), "w") as f:
        json.dump(res, f, default=float)


def job_train(spec):
    """spec: dict(ds, model, cfg, seed, mode, drop, rssa, out)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = features(spec["ds"], spec.get("rssa", True))
    name = spec["model"]
    groups = input_spec(name, F, spec.get("drop", ()))
    win = Windows(groups, F, dev)
    quantile = name in ("TFT", "RCAF")
    model = make_model(name, win.X.shape[1], groups, spec["cfg"]).to(dev)
    tr, va = origins(F, "train"), origins(F, "val")
    t0 = time.perf_counter()
    hist = []
    vloss = fit(model, win, tr, va, spec["cfg"], spec["seed"], quantile, hist=hist)
    train_time = time.perf_counter() - t0
    res = dict(spec={k: v for k, v in spec.items() if k not in ("out", "adapt", "val_adapt")},
               val_mse=vloss, train_time=train_time, history=hist, epochs=len(hist),
               params=int(sum(p.numel() for p in model.parameters())))
    if spec["mode"] == "search":
        return res
    out = {}
    te = origins(F, spec.get("span", "test"))
    out["origins"] = te
    out["static"] = to_units(predict_z(model, win, te, quantile), F)
    if quantile:
        out["static_q"] = to_units(predict_z(model, win, te, True, all_q=True), F)
    for tag, plan in spec.get("adapt", {}).items():
        pz, times = online_predict(model, win, F, te, plan["times"], plan["ft"], spec["seed"], quantile)
        out[tag] = to_units(pz, F)
        res[f"adapt_{tag}_n"] = len(times)
        res[f"adapt_{tag}_time"] = float(np.mean(times)) if times else 0.0
        if quantile and tag == "cpd":
            pq, _ = online_predict(model, win, F, te, plan["times"], plan["ft"], spec["seed"], quantile, all_q=True)
            out["cpd_q"] = to_units(pq, F)
    if spec.get("val_adapt"):
        # validation-span online runs used to select the adaptation settings
        vres = {}
        for tag, plan in spec["val_adapt"].items():
            pz, _ = online_predict(model, win, F, va, plan["times"], plan["ft"], spec["seed"], quantile)
            yv = win.y[torch.as_tensor(va, device=dev)[:, None] + win.offH].cpu().numpy()
            vres[tag] = float(np.mean((pz - yv) ** 2))
        res["val_adapt"] = vres
    if spec.get("save_model"):
        torch.save(model.state_dict(), spec["out"] + ".pt")
    np.savez_compressed(spec["out"] + ".npz", **out)
    with open(spec["out"] + ".json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


# ---------------------------------------------------------------- stages
GRID = [dict(d=d, lr=lr, dropout=dr) for d, lr, dr in itertools.product([32, 64, 128], [1e-3, 3e-4], [0.1, 0.2])]


def stage_features():
    from features import build_features
    os.makedirs(CACHE, exist_ok=True)
    for ds, f in DATASETS.items():
        df = pd.read_csv(os.path.join(HERE, "data", f), parse_dates=["time"])
        for use in (True, False):
            t0 = time.time()
            F = build_features(df, use_rssa=use)
            F["time_index"] = df["time"].astype(str).values
            with open(os.path.join(CACHE, f"{ds}_{'rssa' if use else 'raw'}.pkl"), "wb") as fh:
                pickle.dump(F, fh)
            print(ds, "rssa" if use else "raw", f"{time.time() - t0:.0f}s", F["garch_params"], flush=True)


def stage_search():
    os.makedirs(RES, exist_ok=True)
    jobs = [dict(ds=ds, model=m, cfg=c, seed=0, mode="search") for ds in DATASETS for m in DEEP
            for c in (GRID[:1] if m == "DLinear" else GRID)]
    res = run_jobs(job_train, jobs, "search")
    best = {}
    for r in res:
        k = f"{r['spec']['ds']}|{r['spec']['model']}"
        if k not in best or r["val_mse"] < best[k]["val_mse"]:
            best[k] = dict(cfg=r["spec"]["cfg"], val_mse=r["val_mse"])
    # alpha for RCAF
    ajobs = [dict(ds=ds, model="RCAF", cfg={**best[f"{ds}|RCAF"]["cfg"], "alpha": a}, seed=0, mode="search")
             for ds in DATASETS for a in (0.25, 0.75)]
    for k in best:
        best[k]["cfg"].setdefault("alpha", 0.5)
    ares = run_jobs(job_train, ajobs, "alpha")
    for r in ares:  # alpha = 0.5 is already in the first grid; keep the best of all
        k = f"{r['spec']['ds']}|RCAF"
        if r["val_mse"] < best[k]["val_mse"]:
            best[k] = dict(cfg=r["spec"]["cfg"], val_mse=r["val_mse"])
    pd.DataFrame([dict(job=f"{r['spec']['ds']}|{r['spec']['model']}", **r["spec"]["cfg"], val_mse=r["val_mse"],
                       train_time=r["train_time"], params=r["params"]) for r in res + ares]).to_csv(
        os.path.join(RES, f"search_all_{TAG}.csv"), index=False)
    save_json(best, "best_cfg")
    print(json.dumps(best, indent=1), flush=True)


def deseason(F):
    d = F["den"]
    return np.concatenate([np.zeros(168), d[168:] - d[:-168]])


def trigger_alarms(F, kind, q=None, target_count=None):
    """Alarm indices over the whole series for a trigger type.
    AE-CPD threshold: q-quantile of the score on the training span. Other
    triggers are calibrated to give the same number of alarms on the
    training + validation span as AE-CPD."""
    n, n_tr, n_va = len(F["y"]), F["n_tr"], F["n_va"]
    if kind == "ae":
        s = F["score"]
        thr = np.quantile(s[2 * 168 + 24:n_tr], q)
        return alarms_from(s, thr, 0, n)
    from cpd import CUSUM, BOCPD, run_detector
    x = deseason(F)  # week-over-week difference removes daily/weekly cycles
    if kind == "cusum":
        cands = np.geomspace(5, 2000, 60)
        best = None
        for h in cands:
            _, al = run_detector(CUSUM(n_ref=168), x[:n_va], thr=h, refractory=168)
            if best is None or abs(len(al) - target_count) < abs(best[1] - target_count):
                best = (h, len(al))
        _, al = run_detector(CUSUM(n_ref=168), x, thr=best[0], refractory=168)
        return al
    sc, _ = run_detector(BOCPD(hazard=1 / 720, lag=24), x)
    cands = np.unique(np.quantile(sc[:n_va], np.linspace(0.5, 0.9999, 400)))
    best = min(cands, key=lambda t: abs(len(alarms_from(sc, t, 0, n_va)) - target_count))
    return alarms_from(sc, best, 0, n)


FT_GRID = [dict(window=720, epochs=e, lr=lr, replay=r) for e in (1, 3) for lr in (1e-4, 3e-4) for r in (0, 1)]
Q_GRID = [0.90, 0.95, 0.98, 0.99]
DELAY = 24  # hours of post-change data collected before adapting


def stage_select_adapt():
    """Select the AE-CPD threshold quantile and fine-tuning settings on the validation span."""
    best = load_merged("best_cfg")
    jobs = []
    for ds in DATASETS:
        F = features(ds)
        plans = {}
        for q in Q_GRID:
            al = trigger_alarms(F, "ae", q)
            for j, ft in enumerate(FT_GRID):
                plans[f"q{q}_ft{j}"] = dict(times=(al + DELAY).tolist(), ft=ft)
        jobs.append(dict(ds=ds, model="RCAF", cfg=best[f"{ds}|RCAF"]["cfg"], seed=0, mode="final",
                         val_adapt=plans, out=os.path.join(RES, f"select_{ds}")))
    res = run_jobs(job_train, jobs, "select")
    sel = {}
    for r in res:
        ds = r["spec"]["ds"]
        va = r["val_adapt"]
        k = min(va, key=va.get)
        q = float(k.split("_")[0][1:])
        ft = FT_GRID[int(k.split("ft")[1])]
        sel[ds] = dict(q=q, ft=ft, val_mse=va[k], static_val_mse=r["val_mse"], all=va)
    save_json(sel, "adapt_selection")
    print(json.dumps(sel, indent=1), flush=True)


def stage_final():
    best = load_merged("best_cfg")
    sel = load_merged("adapt_selection")
    os.makedirs(os.path.join(RES, "preds"), exist_ok=True)
    jobs, meta = [], {}
    for ds in DATASETS:
        F = features(ds)
        n_va, n = F["n_va"], len(F["y"])
        al_ae = trigger_alarms(F, "ae", sel[ds]["q"])
        n_cal = int(np.sum(al_ae < n_va))
        al_cu = trigger_alarms(F, "cusum", target_count=n_cal)
        al_bo = trigger_alarms(F, "bocpd", target_count=n_cal)
        test_al = al_ae[al_ae >= n_va]
        K = max(24, int((n - n_va) / max(len(test_al), 1)))
        periodic = np.arange(n_va + K, n, K)
        ft = sel[ds]["ft"]
        plans_full = {"cpd": dict(times=(al_ae + DELAY).tolist(), ft=ft),
                      "periodic": dict(times=periodic.tolist(), ft=ft),
                      "cusum": dict(times=(al_cu + DELAY).tolist(), ft=ft),
                      "bocpd": dict(times=(al_bo + DELAY).tolist(), ft=ft)}
        meta[ds] = dict(alarms_ae=al_ae.tolist(), alarms_cusum=al_cu.tolist(), alarms_bocpd=al_bo.tolist(),
                        periodic=periodic.tolist(), K=K, n_test_alarms=int(len(test_al)))
        for seed in SEEDS:
            for m in DEEP:
                spec = dict(ds=ds, model=m, cfg=best[f"{ds}|{m}"]["cfg"], seed=seed, mode="final",
                            out=os.path.join(RES, "preds", f"{ds}_{m}_s{seed}"))
                if m == "RCAF":
                    spec["adapt"] = plans_full
                    spec["save_model"] = seed == 0
                if m in ("LSTM", "TFT"):
                    spec["adapt"] = {"cpd": plans_full["cpd"]}
                jobs.append(spec)
            rc = best[f"{ds}|RCAF"]["cfg"]
            al_raw = trigger_alarms(features(ds, False), "ae", sel[ds]["q"])
            for tag, drop, rssa, cfg in [("noHighway", (), True, {**rc, "highway": 0}),
                                         ("noRSSA", (), False, rc), ("noGARCH", ("garch",), True, rc),
                                         ("alpha0", (), True, {**rc, "alpha": 0.0}),
                                         ("alpha1", (), True, {**rc, "alpha": 1.0}),
                                         ("noFusion", ("latent", "time"), True, rc)]:
                plan = (dict(times=(al_raw + DELAY).tolist(), ft=ft) if not rssa else plans_full["cpd"])
                jobs.append(dict(ds=ds, model="RCAF", cfg=cfg, seed=seed, mode="final", drop=drop, rssa=rssa,
                                 adapt={"cpd": plan},
                                 out=os.path.join(RES, "preds", f"{ds}_RCAF-{tag}_s{seed}")))
    save_json(meta, "alarms")
    run_jobs(job_train, jobs, "final")


def stage_rcaf_search():
    """Re-run the validation search for RCAF with the autoregressive highway
    (added after the validation diagnostic in val_diagnostic.py)."""
    best = load_merged("best_cfg")
    jobs = [dict(ds=ds, model="RCAF", cfg={**c, "highway": 1}, seed=0, mode="search") for ds in DATASETS for c in GRID]
    res = run_jobs(job_train, jobs, "search_hw")
    for ds in DATASETS:
        r = min([r for r in res if r["spec"]["ds"] == ds], key=lambda r: r["val_mse"])
        best[f"{ds}|RCAF_v1"] = best[f"{ds}|RCAF"]
        best[f"{ds}|RCAF"] = dict(cfg={**r["spec"]["cfg"], "alpha": 0.5}, val_mse=r["val_mse"])
    ajobs = [dict(ds=ds, model="RCAF", cfg={**best[f"{ds}|RCAF"]["cfg"], "alpha": a}, seed=0, mode="search")
             for ds in DATASETS for a in (0.25, 0.75)]
    ares = run_jobs(job_train, ajobs, "alpha_hw")
    for r in ares:
        k = f"{r['spec']['ds']}|RCAF"
        if r["val_mse"] < best[k]["val_mse"]:
            best[k] = dict(cfg=r["spec"]["cfg"], val_mse=r["val_mse"])
    pd.DataFrame([dict(job=f"{r['spec']['ds']}|RCAF+highway", **r["spec"]["cfg"], val_mse=r["val_mse"],
                       train_time=r["train_time"], params=r["params"]) for r in res + ares]).to_csv(
        os.path.join(RES, f"search_hw_{TAG}.csv"), index=False)
    save_json({k: v for k, v in best.items() if k.split("|")[0] in DATASETS}, "best_cfg")
    print(json.dumps(best, indent=1), flush=True)


def stage_v2():
    stage_rcaf_search()
    stage_select_adapt()
    stage_final()
    print("v2 done", flush=True)


def stage_all():
    for ds in DATASETS:
        if not os.path.exists(os.path.join(CACHE, f"{ds}_raw.pkl")):
            stage_features()
            break
    t0 = time.time()
    if not os.path.exists(os.path.join(RES, f"best_cfg_{TAG}.json")):
        stage_search()
    print(f"search done {time.time() - t0:.0f}s", flush=True)
    stage_select_adapt()
    print(f"select done {time.time() - t0:.0f}s", flush=True)
    stage_final()
    print(f"final done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        _job_entry(sys.argv[2])
        sys.exit(0)
    {"features": stage_features, "search": stage_search, "select": stage_select_adapt,
     "final": stage_final, "all": stage_all, "v2": stage_v2}[sys.argv[1]]()

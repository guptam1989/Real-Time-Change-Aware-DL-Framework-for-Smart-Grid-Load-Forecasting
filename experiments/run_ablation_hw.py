"""Post-hoc ablation, run after the held-out evaluation (not part of the frozen method).

RCAF-TFT differs from the TFT baseline in two ways: the change-aware inputs and
the autoregressive highway; RCAF-TRF differs from the Transformer baseline in the
same two ways. This script trains the standard TFT and Transformer with the
highway but with the standard inputs (TFT-HW, Transformer-HW), so that
TFT -> TFT-HW -> RCAF-TFT (and Transformer -> Transformer-HW -> RCAF-TRF)
separates the effect of the highway from that of the change-aware inputs.
Each variant receives the same validation search as the held-out experts
(run_v3.SMALL_GRID) and is then trained with three seeds.

Usage: RCAF_DS=DE+FR python run_ablation_hw.py search|final|eval|all
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_forecast as rf  # noqa: E402
import run_v3 as v3  # noqa: E402

VARIANTS = {"TFT-HW": "TFT", "Transformer-HW": "Transformer"}   # name -> base model (standard inputs)
_make_model = rf.make_model


def make_model(name, n_in, groups, cfg):
    if name == "Transformer" and cfg.get("highway"):
        from models import TransformerForecaster
        return TransformerForecaster(n_in, rf.H, cfg["d"], 4, 2, cfg["dropout"], highway=True, seq_len=rf.L)
    return _make_model(name, n_in, groups, cfg)   # TFT: highway taken from cfg


rf.make_model = make_model


def cfg_path(ds):
    return os.path.join(v3.RES3, f"ablation_hw_cfg_{ds}.json")


def run_jobs(jobs, tag):
    """rf.run_jobs with this script as the job entry point (so that the patched
    model factory is used); completed identical jobs are skipped."""
    jdir = os.path.join(HERE, "jobs", f"hw_{tag}")
    os.makedirs(jdir, exist_ok=True)
    slots = [g for g in rf.GPUS for _ in range(rf.WORKERS_PER_GPU)]
    pending, running, tries = list(enumerate(jobs)), {}, {}
    while pending or running:
        while pending and len(running) < len(slots):
            i, job = pending.pop(0)
            spec_path = os.path.join(jdir, f"{i}.spec.json")
            res_path = spec_path.replace(".spec.json", ".result.json")
            if os.path.exists(res_path) and os.path.exists(spec_path) and \
                    json.load(open(spec_path)) == json.loads(json.dumps(job)):
                continue
            free = [s for s in range(len(slots)) if s not in {v[1] for v in running.values()}][0]
            with open(spec_path, "w") as f:
                json.dump(job, f)
            env = dict(os.environ, OMP_NUM_THREADS="1")
            if slots[free] is not None:
                env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(slots[free]),
                           RCAF_GPUS=str(slots[free]))
            log = open(os.path.join(jdir, f"{i}.log"), "w")
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
        if os.path.exists(rp):
            out.append(json.load(open(rp)))
    return out


def stage_search():
    for ds in v3.DATASETS:
        if os.path.exists(cfg_path(ds)):
            continue
        jobs = [dict(ds=ds, model=base, cfg={**c, "highway": 1}, seed=0, mode="search", variant=name)
                for name, base in VARIANTS.items() for c in v3.SMALL_GRID]
        best, res = {}, run_jobs(jobs, f"search_{ds}")
        if len(res) < len(jobs):   # killed jobs: fail so that the job script retries (completed jobs are kept)
            raise RuntimeError(f"{ds}: {len(jobs) - len(res)} search jobs missing")
        for r in res:
            name = r["spec"]["variant"]
            if name not in best or r["val_mse"] < best[name]["val_mse"]:
                best[name] = dict(cfg=r["spec"]["cfg"], val_mse=r["val_mse"])
        json.dump(best, open(cfg_path(ds), "w"), indent=1)
        print(ds, json.dumps(best), flush=True)


def stage_final():
    jobs = []
    for ds in v3.DATASETS:
        best = json.load(open(cfg_path(ds)))
        for name, base in VARIANTS.items():
            for seed in rf.SEEDS:
                jobs.append(dict(ds=ds, model=base, cfg=best[name]["cfg"], seed=seed, mode="final", span="valtest",
                                 variant=name, out=os.path.join(v3.PRED3, f"{ds}_{name}_s{seed}")))
    res = run_jobs(jobs, "final_" + "_".join(v3.DATASETS))
    if len(res) < len(jobs):
        raise RuntimeError(f"{len(jobs) - len(res)} final jobs missing")


def stage_eval():
    """Test MASE (as in evaluate_v3) of the base models, the highway variants and
    the change-aware experts; DM test (seed-averaged forecasts) of each
    change-aware expert against its highway variant."""
    import evaluate_v3 as ev
    pairs = [("TFT", "TFT-HW", "RCAF"), ("Transformer", "Transformer-HW", "RCAF-TRF"), ("XGBoost", None, "RCAF-GBM")]
    rows = []
    for ds in ev.DEV + ev.HELDOUT:
        F = ev.feats(ds)
        meta = json.load(open(os.path.join(v3.RES3, f"alarms_{ds}.json")))
        n_tr, n_va, n = meta["n_tr"], meta["n_va"], meta["n"]
        vt = np.arange(n_tr, n - rf.H)
        tp = np.arange(n_va - n_tr, len(vt))
        y = F["y"][vt[tp][:, None] + np.arange(1, rf.H + 1)]
        scale = float(np.mean(np.abs(F["y"][24:n_tr] - F["y"][:n_tr - 24])))

        def load(m):
            runs = [np.load(os.path.join(v3.PRED3, f"{ds}_{m}_s{s}.npz"))["static"][tp] for s in rf.SEEDS
                    if os.path.exists(os.path.join(v3.PRED3, f"{ds}_{m}_s{s}.npz"))]
            return np.stack(runs) if runs else None

        for base, hw, expert in pairs:
            P = {k: load(k) for k in (base, hw, expert) if k is not None}
            if any(v is None for v in P.values()):
                continue
            ref = hw if hw is not None else base     # the change-aware inputs are added to this model
            r = dict(dataset=ds, base=base)
            for k, lab in ((base, "base"), (hw, "highway"), (expert, "change_aware")):
                if k is None:
                    continue
                r[f"MASE_{lab}"] = float(np.mean([np.mean(np.abs(p - y)) / scale for p in P[k]]))
                r[f"MASE_{lab}_sd"] = float(np.std([np.mean(np.abs(p - y)) / scale for p in P[k]]))
            # DM > 0: the second model is more accurate than the first (squared and absolute error loss;
            # the absolute-error version is consistent with the MASE)
            r["DM_inputs"], r["p_inputs"] = ev.dm(P[ref].mean(0) - y, P[expert].mean(0) - y)
            r["DM_inputs_abs"], r["p_inputs_abs"] = ev.dm(np.sqrt(np.abs(P[ref].mean(0) - y)),
                                                          np.sqrt(np.abs(P[expert].mean(0) - y)))
            if hw is not None:
                r["DM_highway"], r["p_highway"] = ev.dm(P[base].mean(0) - y, P[hw].mean(0) - y)
                r["DM_highway_abs"], r["p_highway_abs"] = ev.dm(np.sqrt(np.abs(P[base].mean(0) - y)),
                                                                np.sqrt(np.abs(P[hw].mean(0) - y)))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(v3.RES3, "ablation_hw.csv"), index=False)
    print(df.round(4).to_string(), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        import torch
        torch.set_num_threads(1)
        spec = json.load(open(sys.argv[2]))
        res = rf.job_train(spec)
        with open(sys.argv[2].replace(".spec.json", ".result.json"), "w") as f:
            json.dump(res, f, default=float)
    elif sys.argv[1] == "all":
        stage_search()
        stage_final()
    else:
        {"search": stage_search, "final": stage_final, "eval": stage_eval}[sys.argv[1]]()

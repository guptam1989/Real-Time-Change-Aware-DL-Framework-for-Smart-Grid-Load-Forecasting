"""Synthetic change point detection benchmark (Section 5.1).

Protocol
- SynD1 (mean), SynD2 (variance), SynD3 (mean + variance): 3 validation
  series (seeds 1-3) and 10 test series (seeds 101-110) per type, n = 5000.
- The AE of AE-CPD is trained on the pooled validation series only.
- For every detector one alarm threshold is selected on the pooled
  validation series (all three types) by maximising F1; it is then fixed for
  all test series. Refractory period 50 steps, detection margin 50 steps.
- Streaming cost (latency, throughput, state memory) is measured on a single
  CPU thread.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpd import (AECPD, BOCPD, CUSUM, DensityRatio, NumpyAE, evaluate, make_synthetic,
                 offline_pelt, run_detector)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KINDS = {"mean": "SynD1", "var": "SynD2", "both": "SynD3"}
VAL_SEEDS = [1, 2, 3]
TEST_SEEDS = list(range(101, 111))
METHODS = ["CUSUM", "Bayesian", "uLSIF", "RuLSIF", "KLIEP", "RCAF (AE-CPD)"]
REFRACTORY, MARGIN = 50, 50
NPROC = int(os.environ.get("RCAF_CPD_PROCS", "1"))  # the cluster CPU queue kills multi-process jobs


def pmap(fn, jobs):
    if NPROC == 1:
        return list(map(fn, jobs))
    with get_context("spawn").Pool(min(NPROC, len(jobs))) as p:
        return p.map(fn, jobs)


def series(kind, seed):
    return make_synthetic(kind, seed=seed)


def build(name, ae_pack=None):
    if name == "CUSUM":
        return CUSUM()
    if name == "Bayesian":
        return BOCPD()
    if name in ("uLSIF", "RuLSIF", "KLIEP"):
        return DensityRatio(name.lower())
    ae, scale = ae_pack
    return AECPD(ae, scale=scale)


def alarms_from_scores(sc, thr):
    al, last = [], -10 ** 9
    for t, v in enumerate(sc):
        if v > thr and t - last > REFRACTORY:
            al.append(t)
            last = t
    return np.asarray(al, int)


def _scores_job(args):
    name, kind, seed, ae_pack = args
    x, cps, outl = series(kind, seed)
    det = build(name, ae_pack)
    t0 = time.perf_counter()
    sc, _ = run_detector(det, x)
    return (name, kind, seed), sc, time.perf_counter() - t0


def _cusum_job(args):
    kind, seed, h = args
    x, cps, _ = series(kind, seed)
    sc, al = run_detector(CUSUM(), x, thr=h, refractory=REFRACTORY)
    return (kind, seed, h), sc, al


def state_bytes(det):
    tot = 0
    for v in vars(det).values():
        if isinstance(v, np.ndarray):
            tot += v.nbytes
        elif isinstance(v, list):
            tot += sum(np.asarray(e).nbytes for e in v)
        elif isinstance(v, NumpyAE):
            tot += v.n_params() * 8
    return tot


def main():
    os.makedirs(OUT, exist_ok=True)
    val = {(k, s): series(k, s) for k in KINDS for s in VAL_SEEDS}
    test = {(k, s): series(k, s) for k in KINDS for s in TEST_SEEDS}

    ae_pack = AECPD.fit([v[0] for v in val.values()], seed=0)
    print("AE trained", flush=True)
    with open(os.path.join(OUT, "aecpd_synthetic.pkl"), "wb") as f:
        pickle.dump(ae_pack, f)

    # ---------------- scores for non-resetting detectors
    jobs = [(m, k, s, ae_pack if m == "RCAF (AE-CPD)" else None)
            for m in METHODS if m != "CUSUM" for (k, s) in list(val) + list(test)]
    res = pmap(_scores_job, jobs)
    scores = {key: sc for key, sc, _ in res}
    print("scores done", flush=True)

    # CUSUM resets after alarms: evaluate a grid of decision intervals h.
    hs = np.round(np.geomspace(2, 200, 40), 3)
    cjobs = [(k, s, h) for (k, s) in list(val) + list(test) for h in hs]
    cres = pmap(_cusum_job, cjobs)
    cusum = {key: (sc, al) for key, sc, al in cres}
    print("cusum grid done", flush=True)

    # ---------------- threshold selection on validation (pooled over types)
    def pooled_f1(get_alarms, keys):
        tp = fp = fn = 0
        for k, s in keys:
            _, cps, _ = (val if (k, s) in val else test)[(k, s)]
            r = evaluate(cps, get_alarms(k, s), 5000, MARGIN)
            tp, fp, fn = tp + r["TP"], fp + r["FP"], fn + r["FN"]
        pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return 2 * pre * rec / max(pre + rec, 1e-12)

    thr = {}
    for m in METHODS:
        if m == "CUSUM":
            best = max(hs, key=lambda h: pooled_f1(lambda k, s: cusum[(k, s, h)][1], list(val)))
        else:
            pos = np.concatenate([scores[(m, k, s)] for (k, s) in val])
            cands = np.unique(np.quantile(pos[pos > 0], np.linspace(0.5, 0.9995, 200)))
            best = max(cands, key=lambda t: pooled_f1(
                lambda k, s: alarms_from_scores(scores[(m, k, s)], t), list(val)))
        thr[m] = float(best)
        print("threshold", m, thr[m], flush=True)

    pens = np.geomspace(1, 200, 30)
    def pelt_al(k, s, pen, cache={}):
        key = (k, s, pen)
        if key not in cache:
            x = (val if (k, s) in val else test)[(k, s)][0]
            cache[key] = offline_pelt(x, pen=pen * np.log(len(x)))
        return cache[key]
    best_pen = max(pens, key=lambda pen: pooled_f1(lambda k, s: pelt_al(k, s, pen), list(val)))

    # ---------------- test evaluation
    rows = []
    for kind, dname in KINDS.items():
        for m in METHODS + ["PELT (offline)"]:
            per, tp, fp, fn, delays = [], 0, 0, 0, []
            for s in TEST_SEEDS:
                x, cps, _ = test[(kind, s)]
                if m == "CUSUM":
                    h = thr[m]
                    sc, al = cusum[(kind, s, min(hs, key=lambda v: abs(v - h)))]
                elif m == "PELT (offline)":
                    sc, al = None, pelt_al(kind, s, best_pen)
                else:
                    sc = scores[(m, kind, s)]
                    al = alarms_from_scores(sc, thr[m])
                if m == "CUSUM":
                    # threshold-free AUC uses the CUSUM statistic without resets
                    sc = run_detector(CUSUM(), x)[0]
                r = evaluate(cps, al, len(x), MARGIN, scores=sc)
                per.append(r)
                tp, fp, fn = tp + r["TP"], fp + r["FP"], fn + r["FN"]
                if not np.isnan(r["delay"]):
                    delays.append(r["delay"])
            pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            f1s = [r["F1"] for r in per]
            rows.append(dict(dataset=dname, method=m, REC=rec, PRE=pre,
                             F1=2 * pre * rec / max(pre + rec, 1e-12),
                             F1_std=float(np.std(f1s)),
                             FPR=float(np.mean([r["FPR"] for r in per])),
                             AUC=float(np.mean([r["AUC"] for r in per])) if "AUC" in per[0] else np.nan,
                             AUC_std=float(np.std([r["AUC"] for r in per])) if "AUC" in per[0] else np.nan,
                             delay=float(np.mean(delays)) if delays else np.nan,
                             n_cps=int(sum(len(test[(kind, s)][1]) for s in TEST_SEEDS))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cpd_synthetic.csv"), index=False)
    print(df.round(3).to_string())

    # ---------------- streaming cost on one CPU thread (SynD1 test series 101)
    x, cps, _ = test[("mean", 101)]
    cost = []
    for m in METHODS:
        det = build(m, ae_pack if m == "RCAF (AE-CPD)" else None)
        det.reset()
        for v in x[:300]:
            det.step(float(v))                    # warm-up (fills buffers)
        t0 = time.perf_counter()
        for v in x[300:2300]:
            det.step(float(v))
        lat = (time.perf_counter() - t0) / 2000
        cost.append(dict(method=m, LAT_ms=1e3 * lat, THR_hz=1 / lat, state_KB=state_bytes(det) / 1024))
    cdf = pd.DataFrame(cost)
    cdf.to_csv(os.path.join(OUT, "cpd_streaming_cost.csv"), index=False)
    print(cdf.round(3).to_string())

    # dataset summary for Table 1
    summ = []
    for kind, dname in KINDS.items():
        summ.append(dict(dataset=dname, n_series_test=len(TEST_SEEDS), length=5000,
                         cps_test=int(sum(len(test[(kind, s)][1]) for s in TEST_SEEDS)),
                         outliers_per_series=15))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "synthetic_summary.csv"), index=False)
    with open(os.path.join(OUT, "cpd_thresholds.json"), "w") as f:
        json.dump(dict(thresholds=thr, pelt_pen=float(best_pen)), f, indent=1)

    # example scores for the figure (SynD3, seed 101)
    x, cps, _ = test[("both", 101)]
    ex = {"x": x.tolist(), "cps": cps.tolist(),
          **{m: scores[(m, "both", 101)].tolist() for m in METHODS if m != "CUSUM"}}
    with open(os.path.join(OUT, "cpd_example.json"), "w") as f:
        json.dump(ex, f)


if __name__ == "__main__":
    main()

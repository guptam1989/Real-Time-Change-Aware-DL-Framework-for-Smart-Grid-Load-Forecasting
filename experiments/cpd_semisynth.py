"""Semi-synthetic change point benchmark on real hourly load.

Base series: 6,000-h segments of the ENTSO-E country loads (training spans only,
2015-2018). Changes are injected at known times after a 1,500-h warm-up:
  LoadL  level shifts: the load is multiplied by a factor in [0.85, 1.15]
         (at least 5 % change between consecutive segments);
  LoadA  daily-profile amplitude changes: x' = m + k (x - m), m = centred 24-h
         mean, k log-uniform in [0.6, 1.6] (ratio >= 1.3);
  LoadLA both.
Detectors run as they would inside RCAF on hourly data: AE-CPD on the
RSSA-denoised log load (24-h windows, weekly two-sample test), CUSUM and BOCPD
on the week-over-week difference, density-ratio detectors on 24-h subsequences.
One threshold per detector is selected on three validation series per type;
a change counts as detected if an alarm occurs within 168 h after it.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cpd import BOCPD, CUSUM, DensityRatio, NumpyAE, evaluate, run_detector, train_ae  # noqa: E402
from rssa import rssa_filter  # noqa: E402

OUT = os.path.join(HERE, "results_v3")
COUNTRIES = ["DE", "FR", "IT", "ES", "GB"]
SEG, WARM, MARGIN, REFR = 6000, 1500, 168, 168
METHODS = ["CUSUM", "Bayesian", "uLSIF", "RuLSIF", "KLIEP", "RCAF (AE-CPD)"]


def base_segments():
    segs = {}
    for c in COUNTRIES:
        x = pd.read_csv(os.path.join(HERE, "data", f"ho_{c.lower()}.csv"))["load"].values
        n_tr = int(0.6 * len(x))
        for j in range(n_tr // SEG):
            segs[(c, j)] = x[j * SEG:(j + 1) * SEG].astype(float)
    return segs


def inject(x, kind, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    cps, t = [], WARM + int(rng.integers(200, 700))
    while t < n - 300:
        cps.append(t)
        t += int(rng.integers(500, 1200))
    m = pd.Series(x).rolling(24, center=True, min_periods=1).mean().values
    out = x.copy()
    f, k = 1.0, 1.0
    bounds = [0] + cps + [n]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if i > 0:
            if kind in ("L", "LA"):
                while True:
                    nf = float(np.exp(rng.uniform(np.log(0.85), np.log(1.15))))
                    if abs(np.log(nf / f)) >= np.log(1.05):
                        f = nf
                        break
            if kind in ("A", "LA"):
                while True:
                    nk = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6))))
                    if max(nk / k, k / nk) >= 1.3:
                        k = nk
                        break
        out[a:b] = f * (m[a:b] + k * (x[a:b] - m[a:b]))
    return out, np.array(cps)


def prep(x):
    """Standardised RSSA-denoised log load and its week-over-week difference."""
    mu, sd = x[:WARM].mean(), x[:WARM].std()
    den = rssa_filter((x - mu) / sd) * sd + mu
    ld = np.log(np.maximum(den, 0.05 * np.median(x[:WARM])))
    dz = (ld - ld[:WARM].mean()) / ld[:WARM].std()
    wk = np.concatenate([np.zeros(168), dz[168:] - dz[:-168]])
    return dz, wk


class AECPDHourly:
    """RCAF's AE-CPD as used on hourly load: 24-h windows, 8-d latent code,
    weekly robust two-sample statistic (Eq. 8)."""
    resets = False
    name = "RCAF (AE-CPD)"

    def __init__(self, dz, w=24, m=8, n=168, seed=0):
        W = np.lib.stride_tricks.sliding_window_view(dz[:WARM], w)
        self.ae = NumpyAE(train_ae(W, m=m, epochs=600, seed=seed, lr=5e-3))
        self.w, self.n = w, n

    def scores(self, dz):
        Wall = np.lib.stride_tricks.sliding_window_view(dz, self.w)
        rec, z = self.ae(Wall)
        e = np.mean((rec - Wall) ** 2, 1)
        D = np.concatenate([z, np.log(e + 1e-6)[:, None]], 1)
        Dfull = np.zeros((len(dz), D.shape[1]))
        Dfull[self.w - 1:] = D
        s = np.zeros(len(dz))
        n = self.n
        for t in range(2 * n + self.w, len(dz)):
            past, rec_ = Dfull[t - 2 * n + 1:t - n + 1], Dfull[t - n + 1:t + 1]
            mp, mr = np.median(past, 0), np.median(rec_, 0)
            sp = 1.4826 * np.median(np.abs(past - mp), 0)
            sr = 1.4826 * np.median(np.abs(rec_ - mr), 0)
            s[t] = np.mean(((mr - mp) / np.sqrt((sp ** 2 + sr ** 2) / n + 1e-6)) ** 2)
        return s


def alarms(sc, thr, start=WARM + 336):
    al, last = [], -10 ** 9
    for t in range(start, len(sc)):
        if sc[t] > thr and t - last > REFR:
            al.append(t)
            last = t
    return np.asarray(al, int)


def score_series(method, x):
    dz, wk = prep(x)
    t0 = time.perf_counter()
    if method == "RCAF (AE-CPD)":
        sc = AECPDHourly(dz).scores(dz)
    elif method == "CUSUM":
        sc = run_detector(CUSUM(n_ref=168), wk)[0]
    elif method == "Bayesian":
        sc = run_detector(BOCPD(hazard=1 / 720, lag=24), wk)[0]
    else:  # ~0.1 s per evaluation on 24-h subsequences: evaluated once per day, held in between
        sc = run_detector(DensityRatio(method.lower(), k=24, n=168, stride=24), dz)[0]
    sc[:WARM + 336] = 0.0
    return sc, (time.perf_counter() - t0) / len(x), dz, wk


def main(kinds=("L", "A", "LA")):
    from cpd import roc_auc
    os.makedirs(OUT, exist_ok=True)
    segs = base_segments()
    test_keys = [(c, j) for c in COUNTRIES for j in (0, 1)]
    val_keys = [(c, 2) for c in COUNTRIES[:3]]
    rows, cost = [], {}
    names = {"L": "LoadL", "A": "LoadA", "LA": "LoadLA"}
    base_seed = {"L": 1000, "A": 2000, "LA": 3000}
    for kind in kinds:
        dname = names[kind]
        series = {}
        for i, key in enumerate(val_keys + test_keys):
            series[key] = inject(segs[key], kind, seed=base_seed[kind] + i)
        for m in METHODS:
            sc, lat, dz, wk = {}, [], {}, {}
            for key, (x, cps) in series.items():
                s, l, d1, d2 = score_series(m, x)
                sc[key], dz[key], wk[key] = s, d1, d2
                lat.append(l)
            cost[m] = float(np.mean(lat)) * 1e3
            # threshold on the validation series (pooled F1)
            if m == "CUSUM":
                cands = np.geomspace(2, 2000, 60)

                def al_fn(key, h):
                    return run_detector(CUSUM(n_ref=168), wk[key], thr=h, refractory=REFR)[1]
            else:
                pos = np.concatenate([sc[k][sc[k] > 0] for k in val_keys])
                cands = np.unique(np.quantile(pos, np.linspace(0.5, 0.9995, 200)))

                def al_fn(key, h):
                    return alarms(sc[key], h)

            def pooled(keys, h):
                tp = fp = fn = 0
                for k in keys:
                    a = al_fn(k, h)
                    a = a[a >= WARM + 336]
                    r = evaluate(series[k][1], a, SEG, MARGIN)
                    tp, fp, fn = tp + r["TP"], fp + r["FP"], fn + r["FN"]
                pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                return 2 * pre * rec / max(pre + rec, 1e-12), pre, rec
            best = max(cands, key=lambda h: pooled(val_keys, h)[0])
            per = []
            st = WARM + 336
            for k in test_keys:
                a = al_fn(k, best)
                a = a[a >= st]
                r = evaluate(series[k][1], a, SEG, MARGIN)
                lab = np.zeros(SEG, int)
                for c in series[k][1]:
                    lab[c:c + MARGIN + 1] = 1
                r["AUC"] = roc_auc(lab[st:], sc[k][st:])       # after the warm-up only
                per.append(r)
            tp = sum(r["TP"] for r in per)
            fp = sum(r["FP"] for r in per)
            fn = sum(r["FN"] for r in per)
            pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            delays = [r["delay"] for r in per if not np.isnan(r["delay"])]
            rows.append(dict(dataset=dname, method=m, REC=rec, PRE=pre, F1=2 * pre * rec / max(pre + rec, 1e-12),
                             F1_std=float(np.std([r["F1"] for r in per])),
                             AUC=float(np.mean([r["AUC"] for r in per])), AUC_std=float(np.std([r["AUC"] for r in per])),
                             delay=float(np.mean(delays)) if delays else np.nan,
                             n_cps=int(sum(len(series[k][1]) for k in test_keys))))
            print(rows[-1], flush=True)
    tag = "_".join(kinds)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"cpd_semisynth_{tag}.csv"), index=False)
    pd.DataFrame([dict(method=k, LAT_ms=v) for k, v in cost.items()]).to_csv(
        os.path.join(OUT, f"cpd_semisynth_cost_{tag}.csv"), index=False)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("L", "A", "LA"))

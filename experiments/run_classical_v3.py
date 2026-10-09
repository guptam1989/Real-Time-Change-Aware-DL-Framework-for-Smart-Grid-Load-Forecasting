"""Seasonal naive and SARIMA-RT over the validation + test spans (single process;
the cluster's CPU queue does not allow multi-process jobs)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_forecast as rf  # noqa: E402
from run_classical import _sarima_chunk  # noqa: E402
from run_v3 import DATASETS, PRED3  # noqa: E402

H = rf.H


def main():
    os.makedirs(PRED3, exist_ok=True)
    for ds in DATASETS:
        F = rf.features(ds)
        vt = rf.origins(F, "valtest")
        p = os.path.join(PRED3, f"{ds}_SeasonalNaive_s0.npz")
        if not os.path.exists(p):
            sn = np.stack([F["y"][vt + h - 24] for h in range(1, H + 1)], 1)
            np.savez_compressed(p, origins=vt, static=sn)
        p = os.path.join(PRED3, f"{ds}_SARIMA-RT_s0.npz")
        if os.path.exists(p):
            continue
        t0 = time.time()
        z = F["raw"]
        chunks = [(z, s, min(s + 168, vt[-1] + 1), 672, 336) for s in range(vt[0], vt[-1] + 1, 168)]
        res = [_sarima_chunk(c) for c in chunks]
        pred = np.concatenate([r[0] for r in res]) * F["sd_y"] + F["mu_y"]
        np.savez_compressed(p, origins=vt, static=pred)
        json.dump(dict(fit_time=float(np.mean([r[1] for r in res])), lat_ms=1e3 * float(np.mean([r[2] for r in res])),
                       wall=time.time() - t0), open(p.replace(".npz", ".json"), "w"))
        print(ds, "SARIMA-RT", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

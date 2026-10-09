"""Validation-span diagnostic: RMSE per horizon for the selected configurations
(seed 0). Uses only the training and validation spans."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_forecast import (H, RES, Windows, features, fit, input_spec, load_merged,  # noqa: E402
                          make_model, origins, predict_z)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    best = load_merged("best_cfg")
    out = {}
    for ds in ("RWD1", "RWD2"):
        F = features(ds)
        tr, va = origins(F, "train"), origins(F, "val")
        for m in ("DLinear", "LSTM", "TFT", "RCAF"):
            groups = input_spec(m, F)
            win = Windows(groups, F, dev)
            q = m in ("TFT", "RCAF")
            model = make_model(m, win.X.shape[1], groups, best[f"{ds}|{m}"]["cfg"]).to(dev)
            fit(model, win, tr, va, best[f"{ds}|{m}"]["cfg"], 0, q)
            p = predict_z(model, win, va, q)
            y = F["raw"][va[:, None] + np.arange(1, H + 1)]
            r = np.sqrt(np.mean((p - y) ** 2, 0)) * F["sd_y"]
            out[f"{ds}|{m}"] = r.tolist()
            print(ds, m, "val RMSE h=1,6,12,24:", np.round(r[[0, 5, 11, 23]], 3), flush=True)
    json.dump(out, open(os.path.join(RES, "val_diagnostic.json"), "w"))


if __name__ == "__main__":
    main()

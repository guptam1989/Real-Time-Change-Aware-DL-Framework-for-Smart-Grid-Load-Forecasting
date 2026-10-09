"""Phase 1 and Phase 2 of RCAF: adaptive preprocessing and change-aware features.

All features at time t depend only on observations up to t. Every statistic
used for scaling, the GARCH parameters and the autoencoder are estimated on
the training span only.
"""
import numpy as np
import pandas as pd
import torch
from arch import arch_model

from cpd import AECPD, train_ae
from rssa import rssa_filter

WEATHER = ["temp", "hum", "wind"]


def split_points(n, train=0.6, val=0.1):
    return int(n * train), int(n * (train + val))


def calendar(time):
    t = pd.to_datetime(time)
    h, dw, dy = t.dt.hour.values, t.dt.dayofweek.values, t.dt.dayofyear.values
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
                     np.sin(2 * np.pi * dy / 365.25), np.cos(2 * np.pi * dy / 365.25)], 1)


def zfit(a, n_train):
    mu, sd = np.nanmean(a[:n_train], 0), np.nanstd(a[:n_train], 0) + 1e-8
    return (a - mu) / sd, (mu, sd)


def time_domain(xd, n=24):
    """Table 3 features on the denoised load (causal trailing windows)."""
    s = pd.Series(xd)
    y = np.log(s)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    feats = pd.DataFrame({
        "std": y.rolling(n).std(),
        "skew": y.rolling(n).skew(),
        "kurt": y.rolling(n).kurt(),
        "ema": y.ewm(alpha=0.1, adjust=False).mean(),
        "roc": (s - s.shift(n)) / s.shift(n) * 100,
        "rsi": 100 - 100 / (1 + up / (dn + 1e-12)),
        "acf24": y.rolling(168).corr(y.shift(24)),
    })
    return feats.bfill().fillna(0).values


def build_features(df, use_rssa=True, ae_w=24, ae_m=8, seed=0, device="cpu"):
    """Returns dict of aligned arrays and the scaling of the target."""
    n = len(df)
    n_tr, n_va = split_points(n)
    load = df["load"].values.astype(float)
    floor = 0.05 * np.median(load[:n_tr])

    # ---- Phase 1: RSSA denoising (causal), log transform, GARCH variability
    if use_rssa:
        z, (mu_l, sd_l) = zfit(load, n_tr)
        den = rssa_filter(z) * sd_l + mu_l
    else:
        den = load.copy()
    den = np.maximum(den, floor)
    logden = np.log(den)
    # One-step residual: the denoised log level at t-1 is the conditional mean of log x_t.
    eta = np.zeros(n)
    eta[1:] = np.log(np.maximum(load[1:], floor)) - logden[:-1]
    am = arch_model(100 * eta[:n_tr], mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
    fit = am.fit(disp="off")
    om, a1, b1 = fit.params["omega"], fit.params["alpha[1]"], fit.params["beta[1]"]
    h = np.empty(n)
    h[0] = om / max(1 - a1 - b1, 1e-3)
    e2 = (100 * eta) ** 2
    for t in range(1, n):                                   # recursive GARCH(1,1)
        h[t] = om + a1 * e2[t - 1] + b1 * h[t - 1]
    garch = np.log(h)

    # ---- Phase 2: time-domain features and AE latent features
    tdf = time_domain(den)
    den_z, _ = zfit(logden, n_tr)
    W_tr = np.lib.stride_tricks.sliding_window_view(den_z[:n_tr], ae_w)
    rng = np.random.default_rng(seed)
    W_fit = W_tr[rng.choice(len(W_tr), min(20000, len(W_tr)), replace=False)]
    ae = train_ae(W_fit, m=ae_m, epochs=600, seed=seed, lr=5e-3)
    Wall = np.lib.stride_tricks.sliding_window_view(den_z, ae_w)
    with torch.no_grad():
        X = torch.as_tensor(Wall, dtype=torch.float32)
        rec, zl = ae(X)
        err = ((rec - X) ** 2).mean(1).numpy()
    lat = np.zeros((n, ae_m + 1))
    lat[ae_w - 1:, :ae_m] = zl.numpy()
    lat[ae_w - 1:, ae_m] = np.log(err + 1e-6)

    # ---- AE-CPD score: weekly two-sample test on the AE descriptors
    cpd_n = 168
    D = lat
    score = np.zeros(n)
    for t in range(2 * cpd_n + ae_w, n):
        past, rec_ = D[t - 2 * cpd_n + 1: t - cpd_n + 1], D[t - cpd_n + 1: t + 1]
        mp, mr = np.median(past, 0), np.median(rec_, 0)
        sp = 1.4826 * np.median(np.abs(past - mp), 0)
        sr = 1.4826 * np.median(np.abs(rec_ - mr), 0)
        zz = (mr - mp) / np.sqrt((sp ** 2 + sr ** 2) / cpd_n + 1e-6)
        score[t] = np.mean(zz ** 2)

    # ---- assemble and scale (train statistics)
    raw_z, (mu_y, sd_y) = zfit(load, n_tr)
    wz, _ = zfit(df[WEATHER].values.astype(float), n_tr)
    tz, _ = zfit(tdf, n_tr)
    lz, _ = zfit(lat, n_tr)
    gz, _ = zfit(garch[:, None], n_tr)
    cal = calendar(df["local_time"] if "local_time" in df else df["time"])  # load follows local time
    return dict(raw=raw_z, den=den_z, weather=wz, time=tz, latent=lz, garch=gz[:, 0],
                cal=cal, score=score, y=load, mu_y=mu_y, sd_y=sd_y, n_tr=n_tr, n_va=n_va,
                garch_params=dict(omega=om, alpha=a1, beta=b1), den_level=den)

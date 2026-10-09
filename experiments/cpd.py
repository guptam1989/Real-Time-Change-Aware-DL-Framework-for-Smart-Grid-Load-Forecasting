"""Change point detection: synthetic data, AE-CPD and baseline detectors, evaluation.

Every detector produces a per-step score s_t computed only from x_1..x_t.
Alarms are raised when s_t crosses a threshold (selected on separate
validation series), followed by a refractory period. Detectors that model a
reference regime (CUSUM, AE-CPD) are re-initialised after each alarm.
"""
import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------- synthetic data

def make_synthetic(kind, n=5000, seed=0, seg=(150, 350), n_out=15):
    """SynD1 (mean), SynD2 (variance), SynD3 (mean + variance) piecewise Gaussian series."""
    rng = np.random.default_rng(seed)
    cps, t = [], 0
    while True:
        t += int(rng.integers(*seg))
        if t >= n - seg[0] // 2:
            break
        cps.append(t)
    bounds = [0] + cps + [n]
    mu, sd = 0.0, 1.0
    x = np.empty(n)
    for i in range(len(bounds) - 1):
        if i > 0:
            if kind in ("mean", "both"):
                while True:
                    m = rng.uniform(-3, 3)
                    if abs(m - mu) >= 1.0:
                        mu = m
                        break
            if kind in ("var", "both"):
                while True:
                    s = np.exp(rng.uniform(np.log(0.3), np.log(3)))
                    if max(s / sd, sd / s) >= 1.5:
                        sd = s
                        break
        a, b = bounds[i], bounds[i + 1]
        x[a:b] = mu + sd * rng.standard_normal(b - a)
    out_idx = rng.choice(np.arange(20, n), n_out, replace=False)
    x[out_idx] += rng.choice([-1, 1], n_out) * 5 * np.std(x)
    return x, np.array(cps), np.sort(out_idx)


# ---------------------------------------------------------------- detectors

class CUSUM:
    """Two-sided CUSUM for mean shifts with a re-estimated reference after each alarm."""
    name = "CUSUM"

    def __init__(self, n_ref=50, drift=0.5):
        self.n_ref, self.k = n_ref, drift

    def reset(self):
        self.ref, self.sp, self.sn = [], 0.0, 0.0

    def step(self, x):
        if len(self.ref) < self.n_ref:
            self.ref.append(x)
            if len(self.ref) == self.n_ref:
                r = np.asarray(self.ref)
                med = np.median(r)
                self.mu, self.sd = med, 1.4826 * np.median(np.abs(r - med)) + 1e-8
            return 0.0
        z = (x - self.mu) / self.sd
        self.sp = max(0.0, self.sp + z - self.k)
        self.sn = max(0.0, self.sn - z - self.k)
        return max(self.sp, self.sn)

    resets = True


class BOCPD:
    """Bayesian online change point detection (Adams & MacKay, 2007), Normal-Gamma model.
    Score: posterior probability that a change occurred within the last `lag` steps."""
    name = "Bayesian"
    resets = False

    def __init__(self, hazard=1 / 250, rmax=400, lag=10, mu0=0.0, kappa0=0.1, alpha0=1.0, beta0=1.0):
        self.h, self.rmax, self.lag = hazard, rmax, lag
        self.prior = (mu0, kappa0, alpha0, beta0)

    def reset(self):
        mu0, k0, a0, b0 = self.prior
        self.R = np.array([1.0])
        self.mu, self.k = np.array([mu0]), np.array([k0])
        self.a, self.b = np.array([a0]), np.array([b0])

    def step(self, x):
        mu, k, a, b = self.mu, self.k, self.a, self.b
        scale2 = b * (k + 1) / (a * k)
        df = 2 * a
        z = (x - mu) ** 2 / scale2
        logp = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * scale2)
                - (df + 1) / 2 * np.log1p(z / df))
        p = np.exp(logp - logp.max())
        growth = self.R * p * (1 - self.h)
        cp = (self.R * p * self.h).sum()
        R = np.concatenate([[cp], growth])
        R /= R.sum()
        mu0, k0, a0, b0 = self.prior
        nmu = (k * mu + x) / (k + 1)
        nb = b + k * (x - mu) ** 2 / (2 * (k + 1))
        self.mu = np.concatenate([[mu0], nmu])
        self.k = np.concatenate([[k0], k + 1])
        self.a = np.concatenate([[a0], a + 0.5])
        self.b = np.concatenate([[b0], nb])
        if len(R) > self.rmax:
            R, self.mu, self.k, self.a, self.b = (v[: self.rmax] for v in (R, self.mu, self.k, self.a, self.b))
            R /= R.sum()
        self.R = R
        return float(R[: self.lag].sum())


def _subseq(x, k):
    return np.lib.stride_tricks.sliding_window_view(x, k)


class DensityRatio:
    """Sliding-window density-ratio CPD (Kawahara & Sugiyama 2012; Liu et al. 2013).
    method in {'ulsif', 'rulsif', 'kliep'}; score = symmetric divergence between
    the reference window and the test window of subsequences."""
    resets = False

    def __init__(self, method="rulsif", k=10, n=50, alpha=0.1, lam=0.01, stride=1):
        self.method, self.k, self.n, self.lam = method, k, n, lam
        self.stride = stride  # evaluate the divergence every `stride` steps and hold it in between
        self.alpha = 0.0 if method == "ulsif" else alpha
        self.name = {"ulsif": "uLSIF", "rulsif": "RuLSIF", "kliep": "KLIEP"}[method]

    def reset(self):
        self.buf, self.count, self.last = [], 0, 0.0

    @staticmethod
    def _gauss(A, C, sigma):
        d = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2 * sigma ** 2))

    def _pe(self, Xr, Xt):
        # Estimate r(x) = p_t(x) / (alpha p_t + (1-alpha) p_r) with kernel centres at test samples.
        allx = np.vstack([Xr, Xt])
        d = np.sqrt(((allx[:, None] - allx[None]) ** 2).sum(-1))
        sigma = np.median(d[d > 0]) + 1e-8
        Kt, Kr = self._gauss(Xt, Xt, sigma), self._gauss(Xr, Xt, sigma)
        nt, nr = len(Xt), len(Xr)
        if self.method == "kliep":
            theta = np.ones(nt) / nt
            b = Kr.mean(0)
            for _ in range(60):
                theta += 1e-3 * Kt.T @ (1.0 / (Kt @ theta + 1e-12))
                theta += (1 - b @ theta) * b / (b @ b)
                theta = np.maximum(theta, 0)
                theta /= b @ theta
            g = Kt @ theta
            return float(np.mean(np.log(g + 1e-12)))          # KL divergence estimate
        H = self.alpha * Kt.T @ Kt / nt + (1 - self.alpha) * Kr.T @ Kr / nr
        h = Kt.mean(0)
        theta = np.linalg.solve(H + self.lam * np.eye(nt), h)
        gt, gr = Kt @ theta, Kr @ theta
        return float(-self.alpha * (gt ** 2).mean() / 2 - (1 - self.alpha) * (gr ** 2).mean() / 2 + gt.mean() - 0.5)

    def step(self, x):
        self.buf.append(x)
        need = 2 * self.n + self.k - 1
        if len(self.buf) < need:
            return 0.0
        if len(self.buf) > need:
            self.buf.pop(0)
        self.count += 1
        if (self.count - 1) % self.stride:
            return self.last
        Y = _subseq(np.asarray(self.buf), self.k)
        Xr, Xt = Y[: self.n], Y[self.n:]
        mu, sd = Xr.mean(), Xr.std() + 1e-8                  # scale-free comparison
        Xr, Xt = (Xr - mu) / sd, (Xt - mu) / sd
        self.last = self._pe(Xr, Xt) + self._pe(Xt, Xr)
        return self.last


def train_ae(windows, m=4, epochs=200, noise=0.1, l2=1e-4, lr=1e-2, seed=0, model=None, hidden=32):
    """Denoising autoencoder with L2 weight penalty (Eq. loss). Returns a torch module
    whose forward gives (reconstruction, latent code)."""
    import torch
    import torch.nn as nn

    class _AE(nn.Module):
        def __init__(self, w, m):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(w, hidden), nn.Tanh(), nn.Linear(hidden, m))
            self.dec = nn.Sequential(nn.Linear(m, hidden), nn.Tanh(), nn.Linear(hidden, w))

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    torch.manual_seed(seed)
    X = torch.as_tensor(windows, dtype=torch.float32)
    ae = model if model is not None else _AE(X.shape[1], m)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        rec, _ = ae(X + noise * torch.randn_like(X))
        loss = ((rec - X) ** 2).mean() + l2 * sum((p ** 2).sum() for n_, p in ae.named_parameters() if "weight" in n_)
        loss.backward()
        opt.step()
    return ae


class NumpyAE:
    """NumPy copy of a trained AE for lightweight streaming inference."""

    def __init__(self, ae):
        sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in ae.state_dict().items()}
        self.W = [(sd[f"{p}.{i}.weight"], sd[f"{p}.{i}.bias"]) for p in ("enc", "dec") for i in (0, 2)]

    def __call__(self, x):
        (w1, b1), (w2, b2), (w3, b3), (w4, b4) = self.W
        z = np.tanh(x @ w1.T + b1) @ w2.T + b2
        return np.tanh(z @ w3.T + b3) @ w4.T + b4, z

    def n_params(self):
        return int(sum(w.size + b.size for w, b in self.W))


class AECPD:
    """Autoencoder-based online CPD used in RCAF (latent two-sample test).

    A denoising AE (Eq. autoencoder/loss) is trained offline on windows of
    historical data. Each incoming window of length w is described by its
    latent code z and its log reconstruction error. At time t the descriptors
    of the n most recent windows are compared with those of the n preceding
    windows using a robust (median/MAD) standardised difference; a large
    difference indicates a structural change. Only past data are used.
    """
    name = "RCAF (AE-CPD)"
    resets = False

    def __init__(self, ae, w=16, n=40, scale=None):
        self.ae, self.w, self.n = ae, w, n
        self.scale = scale  # (mu, sd) used to standardise the raw series

    @staticmethod
    def fit(series_list, w=16, m=4, epochs=400, seed=0, max_windows=20000):
        """Train the AE on windows drawn from historical/validation series."""
        allx = np.concatenate(series_list)
        mu, sd = np.median(allx), 1.4826 * np.median(np.abs(allx - np.median(allx))) + 1e-8
        W = np.vstack([_subseq((s - mu) / sd, w) for s in series_list])
        rng = np.random.default_rng(seed)
        if len(W) > max_windows:
            W = W[rng.choice(len(W), max_windows, replace=False)]
        ae = train_ae(W, m=m, epochs=epochs, seed=seed, lr=5e-3)
        return NumpyAE(ae), (mu, sd)

    def reset(self):
        self.buf, self.desc = [], []

    def descriptor(self, win):
        rec, z = self.ae(win)
        e = float(np.mean((rec - win) ** 2))
        return np.concatenate([z, [np.log(e + 1e-6)]])

    def step(self, x):
        mu, sd = self.scale
        self.buf.append((x - mu) / sd)
        if len(self.buf) > self.w:
            self.buf.pop(0)
        if len(self.buf) < self.w:
            return 0.0
        self.desc.append(self.descriptor(np.asarray(self.buf)))
        if len(self.desc) > 2 * self.n:
            self.desc.pop(0)
        if len(self.desc) < 2 * self.n:
            return 0.0
        D = np.asarray(self.desc)
        past, rec = D[: self.n], D[self.n:]
        mp, mr = np.median(past, 0), np.median(rec, 0)
        sp = 1.4826 * np.median(np.abs(past - mp), 0)
        sr = 1.4826 * np.median(np.abs(rec - mr), 0)
        z = (mr - mp) / np.sqrt((sp ** 2 + sr ** 2) / self.n + 1e-6)
        return float(np.mean(z ** 2))

    def latent(self):
        return None if not self.desc else self.desc[-1]


def run_detector(det, x, thr=None, refractory=50):
    """Run a detector over a series. Returns per-step scores and alarm indices."""
    det.reset()
    scores, alarms, last = np.zeros(len(x)), [], -10 ** 9
    for t, v in enumerate(x):
        s = det.step(float(v))
        scores[t] = s
        if thr is not None and s > thr and t - last > refractory:
            alarms.append(t)
            last = t
            if det.resets:
                det.reset()
    return scores, np.asarray(alarms, int)


def offline_pelt(x, pen=None):
    import ruptures as rpt
    pen = pen if pen is not None else 3 * np.log(len(x)) * np.var(x)
    algo = rpt.Pelt(model="normal", min_size=20, jump=5).fit(x[:, None])
    return np.asarray(algo.predict(pen=pen)[:-1], int)


# ---------------------------------------------------------------- evaluation

def evaluate(cps, alarms, n, margin=50, scores=None):
    """A CP is detected if an alarm falls in [c, c + margin]; each CP and alarm match once."""
    used, tp, delays = set(), 0, []
    for c in cps:
        cand = [a for a in alarms if c <= a <= c + margin and a not in used]
        if cand:
            used.add(cand[0])
            tp += 1
            delays.append(cand[0] - c)
    fp = len(alarms) - len(used)
    fn = len(cps) - tp
    rec = tp / max(len(cps), 1)
    pre = tp / max(len(alarms), 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-12)
    fpr = fp / max(n - len(cps), 1)
    out = dict(TP=tp, FP=fp, FN=fn, REC=rec, PRE=pre, F1=f1, FPR=fpr,
               delay=float(np.mean(delays)) if delays else np.nan)
    if scores is not None:
        lab = np.zeros(n, int)
        for c in cps:
            lab[c: min(c + margin + 1, n)] = 1
        out["AUC"] = roc_auc(lab, np.nan_to_num(scores))
    return out


def roc_auc(labels, scores):
    """Mann-Whitney estimate of the ROC AUC (ties receive average ranks)."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    r = rankdata(scores)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

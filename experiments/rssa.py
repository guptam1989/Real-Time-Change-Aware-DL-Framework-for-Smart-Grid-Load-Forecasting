"""Recursive singular spectrum analysis (RSSA) for causal, streaming denoising.

The lag-covariance matrix Pi_tau is updated with each new lagged vector
(Eq. recursive covariance). Its eigenvectors are tracked by a first-order
perturbation rotation Q <- Q (I + P_V) (Erdogmus et al., 2004), followed by
re-orthonormalisation; eigenvalues are the Rayleigh quotients q_i^T Pi q_i.
Perturbation entries are clipped where eigenvalues are nearly degenerate,
and an exact eigendecomposition refresh is performed every `refresh` steps
to bound the accumulated approximation error.

The denoised value at time t is the last coordinate of the newest lagged
vector projected onto the signal subspace, so the filter is causal.
"""
import numpy as np


class RSSA:
    def __init__(self, w=24, energy=0.95, refresh=168, clip=0.5):
        self.w, self.energy, self.refresh, self.clip = w, energy, refresh, clip

    def _signal_dim(self):
        lam = np.clip(self.lam, 0, None)
        cum = np.cumsum(lam) / max(lam.sum(), 1e-12)
        return int(min(np.searchsorted(cum, self.energy) + 1, self.w))

    def _exact(self):
        lam, Q = np.linalg.eigh(self.Pi)
        idx = np.argsort(lam)[::-1]
        self.lam, self.Q = lam[idx], Q[:, idx]

    def init(self, x0):
        """Batch initialisation from the first L0 observations."""
        x0 = np.asarray(x0, float)
        T = np.lib.stride_tricks.sliding_window_view(x0, self.w).T  # w x k0
        self.tau = T.shape[1]
        self.Pi = T @ T.T / self.tau
        self._exact()
        self.buf = list(x0[-self.w:])
        self.steps = 0
        l = self._signal_dim()
        Qs = self.Q[:, :l]
        rec = (Qs @ (Qs.T @ T))[-1]
        return np.concatenate([x0[: self.w - 1], rec])

    def update(self, x):
        self.buf.append(float(x))
        self.buf.pop(0)
        chi = np.asarray(self.buf)
        self.tau += 1
        tau = self.tau
        self.Pi = (tau - 1) / tau * self.Pi + np.outer(chi, chi) / tau
        self.steps += 1
        if self.steps % self.refresh == 0:
            self._exact()
        else:
            a = self.Q.T @ chi                                  # alpha_tau
            d = (tau - 1) * self.lam + a * a                    # d_i + alpha_i^2
            den = d[None, :] - d[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                PV = np.where(np.abs(den) > 1e-9, np.outer(a, a) / den, 0.0)
            np.fill_diagonal(PV, 0.0)
            PV = np.clip(PV, -self.clip, self.clip)
            Q, _ = np.linalg.qr(self.Q @ (np.eye(self.w) + PV))
            lam = np.einsum("ij,ij->j", Q, self.Pi @ Q)        # Rayleigh quotients
            idx = np.argsort(lam)[::-1]
            self.lam, self.Q = lam[idx], Q[:, idx]
        l = self._signal_dim()
        Qs = self.Q[:, :l]
        return float(Qs[-1] @ (Qs.T @ chi))


def rssa_filter(x, w=24, init_len=336, energy=0.95, refresh=168):
    """Causally denoise a whole series."""
    x = np.asarray(x, float)
    m = RSSA(w=w, energy=energy, refresh=refresh)
    out = np.empty_like(x)
    out[:init_len] = m.init(x[:init_len])
    for t in range(init_len, len(x)):
        out[t] = m.update(x[t])
    return out


def exact_ssa_filter(x, w=24, init_len=336, energy=0.95):
    """Reference: exact eigendecomposition at every step (same causal estimator)."""
    return rssa_filter(x, w, init_len, energy, refresh=1)

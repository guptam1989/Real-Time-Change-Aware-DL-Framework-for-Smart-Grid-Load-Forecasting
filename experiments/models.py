"""Forecasting models: RCAF (modified TFT), vanilla TFT, LSTM, GRU, Transformer,
DLinear and PatchTST. All map an input window [B, L, F] (plus known future
calendar features [B, H, C] for the TFT variants) to [B, H, Q] quantiles or
[B, H] point forecasts of the standardised load."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.5, 0.9)


# ------------------------------------------------------------------ TFT blocks
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return torch.sigmoid(self.a(x)) * self.b(x)


class GRN(nn.Module):
    """Gated residual network (Lim et al., 2021), Eq. (GRN)."""

    def __init__(self, d_in, d, d_out=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d
        self.l2 = nn.Linear(d_in, d)
        self.l1 = nn.Linear(d, d_out)
        self.glu = GLU(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.l1(self.drop(F.elu(self.l2(x))))
        return self.norm(self.skip(x) + self.glu(self.drop(h)))


class VariableSelection(nn.Module):
    """Embeds each input variable and combines them with softmax weights."""

    def __init__(self, var_dims, d, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Linear(k, d) for k in var_dims])
        self.grns = nn.ModuleList([GRN(d, d, dropout=dropout) for _ in var_dims])
        self.sel = GRN(d * len(var_dims), d, len(var_dims), dropout)

    def forward(self, xs):
        e = [emb(x) for emb, x in zip(self.embs, xs)]              # each [B, T, d]
        w = torch.softmax(self.sel(torch.cat(e, -1)), -1)          # [B, T, V]
        h = torch.stack([g(v) for g, v in zip(self.grns, e)], -2)  # [B, T, V, d]
        return (w.unsqueeze(-1) * h).sum(-2), w


class TFT(nn.Module):
    """Temporal fusion transformer used as the RCAF forecaster.

    var_groups: list of (name, width) describing the observed input columns,
        in order. Groups named 'latent' and 'time' are fused with
        F = alpha * W_L latent + (1 - alpha) * W_T time (Eq. fused) before
        variable selection, when both are present.
    n_known: number of known-future (calendar) columns, which are also the
        last n_known observed columns.
    """

    def __init__(self, var_groups, n_known, horizon=24, d=64, heads=4, lstm_layers=2,
                 dropout=0.1, alpha=0.5, quantiles=QUANTILES, highway=False, seq_len=168):
        super().__init__()
        # Autoregressive highway (Lai et al., 2018): a linear map from the load history
        # to every horizon, added to all quantile outputs.
        self.highway = nn.Linear(seq_len, horizon) if highway else None
        self.groups = var_groups
        self.horizon, self.q = horizon, len(quantiles)
        names = [g for g, _ in var_groups]
        self.fuse = "latent" in names and "time" in names
        dims = []
        for g, w in var_groups:
            if self.fuse and g in ("latent", "time"):
                continue
            dims.append(w)
        if self.fuse:
            self.alpha = alpha
            wl = dict(var_groups)["latent"]
            wt = dict(var_groups)["time"]
            self.proj_l, self.proj_t = nn.Linear(wl, d), nn.Linear(wt, d)
            dims.append(d)
        self.enc_vs = VariableSelection(dims, d, dropout)
        self.dec_vs = VariableSelection([n_known], d, dropout)
        self.enc = nn.LSTM(d, d, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.dec = nn.LSTM(d, d, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.post_lstm = nn.Sequential(nn.Dropout(dropout))
        self.lstm_glu, self.lstm_norm = GLU(d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, dropout=dropout)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_glu, self.attn_norm = GLU(d), nn.LayerNorm(d)
        self.ff = GRN(d, d, dropout=dropout)
        self.out_glu, self.out_norm = GLU(d), nn.LayerNorm(d)
        self.head = nn.Linear(d, self.q)
        self.last_attention = None
        self.last_weights = None

    def _split(self, x):
        out, i = {}, 0
        for g, w in self.groups:
            out.setdefault(g, []).append(x[..., i:i + w])
            i += w
        return out

    def forward(self, x, fut):
        parts, xs = self._split(x), []
        for g, w in self.groups:
            if self.fuse and g in ("latent", "time"):
                continue
            xs.append(parts[g].pop(0))
        if self.fuse:
            fl = self.proj_l(self._split(x)["latent"][0])
            ft = self.proj_t(self._split(x)["time"][0])
            xs.append(self.alpha * fl + (1 - self.alpha) * ft)
        e, w_enc = self.enc_vs(xs)
        dvec, _ = self.dec_vs([fut])
        he, state = self.enc(e)
        hd, _ = self.dec(dvec, state)
        lstm_out = torch.cat([he, hd], 1)
        skip = torch.cat([e, dvec], 1)
        phi = self.lstm_norm(skip + self.lstm_glu(self.post_lstm(lstm_out)))
        theta = self.enrich(phi)
        T = theta.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        a, att = self.attn(theta, theta, theta, attn_mask=mask, need_weights=True, average_attn_weights=True)
        delta = self.attn_norm(theta + self.attn_glu(a))
        psi = self.ff(delta)
        out = self.out_norm(phi + self.out_glu(psi))[:, -self.horizon:]
        self.last_attention = att[:, -self.horizon:].detach()
        self.last_weights = w_enc.detach()
        y = self.head(out)                                      # [B, H, Q]
        if self.highway is not None:
            y = y + self.highway(x[..., 0]).unsqueeze(-1)       # column 0 is the load history
        return y


# ------------------------------------------------------------------ baselines
class RNN(nn.Module):
    def __init__(self, n_in, horizon=24, d=64, layers=2, dropout=0.1, cell="lstm"):
        super().__init__()
        R = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = R(n_in, d, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, horizon))

    def forward(self, x, fut=None):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


class PosEnc(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]]


class TransformerForecaster(nn.Module):
    def __init__(self, n_in, horizon=24, d=64, heads=4, layers=2, dropout=0.1, highway=False, seq_len=168):
        super().__init__()
        self.inp, self.pos = nn.Linear(n_in, d), PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, horizon)
        self.highway = nn.Linear(seq_len, horizon) if highway else None

    def forward(self, x, fut=None):
        y = self.head(self.enc(self.pos(self.inp(x)))[:, -1])
        if self.highway is not None:
            y = y + self.highway(x[..., 0])                     # column 0 is the load history
        return y


class DLinear(nn.Module):
    """Zeng et al. (2023): moving-average decomposition + linear maps (load channel)."""

    def __init__(self, seq_len, horizon=24, kernel=25):
        super().__init__()
        self.k = kernel
        self.trend, self.season = nn.Linear(seq_len, horizon), nn.Linear(seq_len, horizon)

    def forward(self, x, fut=None):
        y = x[..., 0]
        pad = (self.k - 1) // 2
        yp = torch.cat([y[:, :1].repeat(1, pad), y, y[:, -1:].repeat(1, pad)], 1)
        trend = F.avg_pool1d(yp.unsqueeze(1), self.k, 1).squeeze(1)
        return self.trend(trend) + self.season(y - trend)


class PatchTST(nn.Module):
    """Nie et al. (2023): patching + Transformer encoder with instance normalisation (load channel)."""

    def __init__(self, seq_len, horizon=24, patch=24, stride=12, d=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        n = (seq_len - patch) // stride + 2
        self.emb, self.pos = nn.Linear(patch, d), nn.Parameter(torch.randn(n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(n * d, horizon))

    def forward(self, x, fut=None):
        y = x[..., 0]
        mu, sd = y.mean(1, keepdim=True), y.std(1, keepdim=True) + 1e-5
        y = (y - mu) / sd
        y = torch.cat([y, y[:, -1:].repeat(1, self.stride)], 1)
        p = y.unfold(1, self.patch, self.stride)
        out = self.head(self.enc(self.emb(p) + self.pos))
        return out * sd + mu


def quantile_loss(pred, y, quantiles=QUANTILES):
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()

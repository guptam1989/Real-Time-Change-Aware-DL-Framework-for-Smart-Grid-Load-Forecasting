"""Architecture diagrams for the paper (framework overview and RCAF forecaster)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
C1, C2, C3, CIN, COUT = "#dbe9f6", "#fde9d9", "#e2f0d9", "#fff2cc", "#d9f2e6"
EDGE = "#2f3b4c"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=EDGE, lw=1.1, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, fs=7.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=EDGE))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fs, style="italic")


def framework():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11.2)
    ax.axis("off")
    # input
    box(ax, 0.2, 4.2, 3.0, 2.8, "IoT smart-meter /\ngrid stream\n\nload $x_t$\ntemperature, humidity,\nwind, calendar", CIN, 8.5)
    # phase panels
    for x, w, title in [(3.8, 6.0, "Phase 1\nadaptive preprocessing"),
                        (10.3, 5.6, "Phase 2\nchange-aware features"),
                        (16.4, 5.4, "Phase 3\nforecasting")]:
        ax.add_patch(FancyBboxPatch((x, 0.4), w, 10.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white", ec="#7f8c9d", lw=1.0, ls="--"))
        ax.text(x + w / 2, 9.95, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
    # phase 1
    box(ax, 4.2, 7.2, 5.2, 1.9, "Recursive SSA (causal)\nrank-one covariance update,\nperturbation eigen-tracking", C1, 8.5)
    box(ax, 4.2, 4.6, 5.2, 1.9, "Log transform\n$y_t=\\log\\tilde{x}_t$\nZ-score (training statistics)", C1, 8.5)
    box(ax, 4.2, 1.0, 5.2, 2.6, "Recursive GARCH(1,1)\n$\\eta_t=\\log x_t-\\log\\tilde{x}_{t-1}$\n$h_t=\\gamma_0+\\gamma_1\\eta_{t-1}^2+\\delta_1 h_{t-1}$", C1, 8.5)
    arrow(ax, 6.8, 7.2, 6.8, 6.5, "$\\tilde{x}_t$")
    arrow(ax, 6.8, 4.6, 6.8, 3.6)
    arrow(ax, 3.2, 5.6, 4.2, 8.1)
    # phase 2
    box(ax, 10.7, 7.2, 4.8, 1.9, "Time-domain features\nstd, skewness, kurtosis,\nEMA, ROC, RSI, ACF(24)", C2, 8.5)
    box(ax, 10.7, 4.6, 4.8, 1.9, "Denoising autoencoder\nlatent code $z_t$ +\nlog reconstruction error", C2, 8.5)
    box(ax, 10.7, 1.0, 4.8, 2.6, "AE-CPD\nweekly two-sample test on\nAE descriptors $\\rightarrow$ alarm\nwhen score > threshold", C2, 8.5)
    arrow(ax, 9.4, 5.5, 10.7, 8.1)
    arrow(ax, 9.4, 5.5, 10.7, 5.5)
    arrow(ax, 13.1, 4.6, 13.1, 3.6)
    # phase 3
    box(ax, 16.8, 7.2, 4.6, 1.9, "Fusion  $F=\\alpha W_L z+(1-\\alpha)W_T f$\n+ GARCH $h_t$, load, weather,\ncalendar inputs", C3, 8.5)
    box(ax, 16.8, 4.6, 4.6, 1.9, "Modified TFT\nvariable selection, GRN, LSTM,\nmasked attention", C3, 8.5)
    box(ax, 16.8, 1.0, 4.6, 2.6, "Online adaptation\non alarm: wait 24 h, fine-tune\non last 30 days\n$\\rightarrow$ updated forecaster", C3, 8.5)
    arrow(ax, 15.5, 8.1, 16.8, 8.1)
    arrow(ax, 15.5, 5.5, 16.8, 7.8)
    # GARCH variability h_t routed below the Phase 2 boxes to the fusion input
    ax.plot([6.8, 6.8, 16.15, 16.15], [1.0, 0.65, 0.65, 7.5], color=EDGE, lw=1.1)
    arrow(ax, 16.15, 7.5, 16.8, 7.5)
    ax.text(12.9, 0.72, "$h_t$", ha="center", va="bottom", fontsize=8, style="italic")
    arrow(ax, 19.1, 7.2, 19.1, 6.5)
    arrow(ax, 15.5, 2.3, 16.8, 2.3, "alarm")
    arrow(ax, 19.6, 3.6, 19.6, 4.6)
    ax.text(19.1, 0.55, "output: 24-step quantile forecasts", ha="center", fontsize=8.5, style="italic")
    fig.savefig(os.path.join(OUT, "fig_framework.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def framework_v3():
    """Framework with the change-aware aggregation phase."""
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12)
    ax.axis("off")
    box(ax, 0.2, 4.6, 2.9, 2.8, "IoT smart-meter /\ngrid stream\n\nload $x_t$\nweather $\\mathbf{u}_t$\ncalendar $\\mathbf{c}_t$", CIN, 8.5)
    for x, w, title in [(3.6, 5.2, "Phase 1\nadaptive preprocessing"),
                        (9.2, 5.0, "Phase 2\nfeatures and CPD"),
                        (14.6, 9.2, "Phase 3\nchange-aware forecasting")]:
        ax.add_patch(FancyBboxPatch((x, 0.4), w, 11.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="white", ec="#7f8c9d", lw=1.0, ls="--"))
        ax.text(x + w / 2, 10.75, title, ha="center", va="center", fontsize=9.5, fontweight="bold")
    # phase 1
    box(ax, 3.9, 7.6, 4.6, 2.2, "Recursive SSA\n(causal denoising,\nperturbation eigen-tracking)", C1, 8.3)
    box(ax, 3.9, 4.6, 4.6, 2.2, "Log transform and\nstandardisation\n(training statistics)", C1, 8.3)
    box(ax, 3.9, 1.0, 4.6, 2.8, "Recursive GARCH(1,1)\nconditional variance $h_t$\nof the one-step residual", C1, 8.3)
    arrow(ax, 6.2, 7.6, 6.2, 6.8)
    arrow(ax, 6.2, 4.6, 6.2, 3.8)
    arrow(ax, 3.1, 6.0, 3.9, 8.7)
    # phase 2
    box(ax, 9.5, 7.6, 4.4, 2.2, "Time-domain features\n(std, skewness, kurtosis,\nEMA, ROC, RSI, ACF)", C2, 8.3)
    box(ax, 9.5, 4.6, 4.4, 2.2, "Denoising autoencoder\nlatent code + log\nreconstruction error", C2, 8.3)
    box(ax, 9.5, 1.0, 4.4, 2.8, "AE-CPD\nweekly robust two-sample\ntest on AE descriptors\n$\\rightarrow$ alarms", C2, 8.3)
    arrow(ax, 8.5, 5.7, 9.5, 8.7)
    arrow(ax, 8.5, 5.7, 9.5, 5.7)
    arrow(ax, 11.7, 4.6, 11.7, 3.8)
    # phase 3
    box(ax, 14.9, 7.3, 4.2, 2.8, "Change-aware experts\nRCAF-TFT, RCAF-TRF,\nRCAF-GBM\n(fused features, $h_t$)", C3, 8.3)
    box(ax, 19.4, 7.3, 4.1, 2.8, "Standard forecasters\nSARIMA, XGBoost, LSTM,\nGRU, Transformer, DLinear,\nPatchTST, TFT", C3, 8.3)
    box(ax, 14.9, 3.9, 4.2, 2.6, "Expert spawning\non alarm: adapted copies\n(fine-tuned / refitted)\nwake up after 24 h", C3, 8.3)
    box(ax, 19.4, 3.9, 4.1, 2.6, "Online aggregation\n(fixed-share EWA\nper horizon); on alarm:\nre-share $\\rho$, boost share", COUT, 8.3)
    box(ax, 17.0, 0.9, 4.6, 2.3, "Adaptive conformal intervals\n(boosted step after alarms)\n$\\rightarrow$ point + 80% interval", COUT, 8.3)
    arrow(ax, 13.9, 8.7, 14.9, 8.7)
    arrow(ax, 13.9, 5.7, 14.9, 8.3)
    arrow(ax, 13.9, 2.4, 14.9, 5.2, "alarm")
    arrow(ax, 17.0, 7.3, 17.0, 6.5)
    arrow(ax, 21.45, 7.3, 21.45, 6.5)
    arrow(ax, 19.1, 5.2, 19.4, 5.2)
    arrow(ax, 21.45, 3.9, 20.6, 3.2)
    ax.plot([8.5, 8.9, 8.9], [2.4, 2.4, 0.5], color=EDGE, lw=1.1)
    ax.plot([8.9, 14.45, 14.45], [0.5, 0.5, 7.0], color=EDGE, lw=1.1)
    arrow(ax, 14.45, 7.0, 14.9, 7.6)
    ax.text(11.7, 0.55, "$h_t$", ha="center", va="bottom", fontsize=8, style="italic")
    fig.savefig(os.path.join(OUT, "fig_framework_v3.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def forecaster():
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15)
    ax.axis("off")
    box(ax, 0.3, 13.0, 4.1, 1.4, "Load, denoised log load,\nweather, GARCH $h_t$", CIN, 8)
    box(ax, 4.9, 13.0, 4.1, 1.4, "Fused features\n$F=\\alpha W_L z+(1-\\alpha)W_T f$", C2, 8)
    box(ax, 9.6, 13.0, 4.1, 1.4, "Known future\ncalendar (24 h)", CIN, 8)
    box(ax, 0.8, 10.6, 7.7, 1.4, "Encoder variable selection  $\\tilde{\\gamma}_\\tau=\\sum_v \\xi_{\\tau,v}\\,\\mathrm{GRN}(\\gamma_{\\tau,v})$", C1, 8)
    box(ax, 9.3, 10.6, 4.4, 1.4, "Decoder variable\nselection", C1, 8)
    arrow(ax, 2.35, 13.0, 3.5, 12.0)
    arrow(ax, 6.95, 13.0, 5.8, 12.0)
    arrow(ax, 11.65, 13.0, 11.5, 12.0)
    box(ax, 0.8, 8.2, 7.7, 1.4, "LSTM encoder (168 h history)", C1, 8.5)
    box(ax, 9.3, 8.2, 4.4, 1.4, "LSTM decoder\n(24 h)", C1, 8.5)
    arrow(ax, 4.65, 10.6, 4.65, 9.6)
    arrow(ax, 11.5, 10.6, 11.5, 9.6)
    arrow(ax, 8.5, 8.9, 9.3, 8.9, "state")
    box(ax, 0.8, 5.8, 12.9, 1.4, "Gated skip + LayerNorm, enrichment GRN", C2, 8.5)
    arrow(ax, 4.65, 8.2, 4.65, 7.2)
    arrow(ax, 11.5, 8.2, 11.5, 7.2)
    box(ax, 0.8, 3.6, 12.9, 1.4, "Masked multi-head self-attention (4 heads) + gated residual + position-wise GRN", C2, 8.5)
    arrow(ax, 7.25, 5.8, 7.25, 5.0)
    box(ax, 1.6, 1.2, 11.3, 1.6, "Quantile head: $\\hat{y}_{t+h,q}$,  $q\\in\\{0.1,0.5,0.9\\}$,  $h=1,\\dots,24$\ntrained with the pinball loss", COUT, 8.5)
    arrow(ax, 7.25, 3.6, 7.25, 2.8)
    fig.savefig(os.path.join(OUT, "fig_rcaf_tft.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    framework()
    forecaster()
    framework_v3()

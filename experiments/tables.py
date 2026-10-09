"""Generate the LaTeX results tables from the result files and inline them into
sn-article.tex between %%BEGIN-TABLE:<name> and %%END-TABLE:<name> markers."""
import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TEX = os.path.join(HERE, "..", "sn-article.tex")
NAMES = {"RWD1": "RWD1 (household, kW)", "RWD2": "RWD2 (PJM East, MW)"}


def merged(name):
    out = {}
    for p in glob.glob(os.path.join(RES, f"{name}_*.json")):
        out.update(json.load(open(p)))
    return out


def fmt(v, nd):
    return f"{v:.{nd}f}"


def bold_best(vals, lower=True):
    arr = np.array([np.nan if v is None else v for v in vals], float)
    if np.all(np.isnan(arr)):
        return [False] * len(vals)
    b = np.nanmin(arr) if lower else np.nanmax(arr)
    return [abs(a - b) < 1e-12 if not np.isnan(a) else False for a in arr]


# ------------------------------------------------------------------ datasets
def t_datasets():
    rows = []
    for ds, f, src in [("RWD1", "rwd1_uci_household.csv", "UCI household smart meter \\cite{hebrail2006individual} (Sceaux, France), hourly mean active power"),
                       ("RWD2", "rwd2_pjm_east.csv", "PJM East (PJME) hourly grid load \\cite{mulla2018hourly}")]:
        d = pd.read_csv(os.path.join(HERE, "data", f), parse_dates=["time"])
        n = len(d)
        n_tr, n_va = int(n * 0.6), int(n * 0.7)
        al = merged("alarms").get(ds, {}).get("alarms_ae", [])
        n_al = sum(1 for a in al if a >= n_va)
        rows.append(f"{ds} & {src} & {d.time.min():%Y-%m-%d} to {d.time.max():%Y-%m-%d} & {n:,} & "
                    f"{n_tr:,} / {n_va - n_tr:,} / {n - n_va:,} h & {d.load.mean():.2f} ({d.load.std():.2f}) & {n_al} \\\\")
    s = pd.read_csv(os.path.join(RES, "synthetic_summary.csv"))
    kinds = {"SynD1": "mean shifts", "SynD2": "variance shifts", "SynD3": "mean and variance shifts"}
    for _, r in s.iterrows():
        rows.append(f"{r.dataset} & Piecewise Gaussian, {kinds[r.dataset]}, 15 outliers per series & synthetic & "
                    f"13 $\\times$ 5,000 & 3 / 10 series (val. / test) & -- & {int(r.cps_test)} (true) \\\\")
    return rows


def table_datasets():
    rows = t_datasets()
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Datasets. Real-world series are hourly and are joined with ERA5 temperature, relative humidity and wind speed \\cite{hersbach2020era5,openmeteo}; splits are chronological (train / validation / test). The last column gives the number of AE-CPD alarms in the test span (real data) or the number of true change points in the test series (synthetic data).}",
        "\\label{tab_dataset_all}",
        "\\footnotesize",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}p{0.065\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}>{\\raggedright\\arraybackslash}p{0.08\\textwidth}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}>{\\raggedright\\arraybackslash}p{0.08\\textwidth}>{\\raggedright\\arraybackslash}p{0.06\\textwidth}@{}}",
        "\\toprule",
        "ID & Source & Period & Length & Train / val. / test & Mean (SD) & CPs \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


# ------------------------------------------------------------------ hyper-parameters
def table_hyper():
    best = merged("best_cfg")
    sel = merged("adapt_selection")
    models = ["LSTM", "GRU", "Transformer", "PatchTST", "TFT", "RCAF"]
    rows = []
    for m in models:
        cells = []
        for ds in ("RWD1", "RWD2"):
            c = best[f"{ds}|{m}"]["cfg"]
            a = f", $\\alpha$={c['alpha']}" if m == "RCAF" else ""
            cells.append(f"$d$={c['d']}, lr={c['lr']:g}, dropout={c['dropout']}{a}")
        rows.append(f"{m} & {cells[0]} & {cells[1]} \\\\")
    ad = []
    for ds in ("RWD1", "RWD2"):
        s = sel[ds]
        ft = s["ft"]
        ad.append(f"$q$={s['q']}, {ft['epochs']} epoch(s), lr={ft['lr']:g}, replay={'yes' if ft.get('replay') else 'no'}")
    rows.append(f"RCAF adaptation & {ad[0]} & {ad[1]} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hyper-parameters selected on the validation span (seed 0). Search space for the neural models: hidden size $d\\in\\{32,64,128\\}$, learning rate $\\in\\{10^{-3},3\\times10^{-4}\\}$, dropout $\\in\\{0.1,0.2\\}$; for RCAF additionally $\\alpha\\in\\{0.25,0.5,0.75\\}$, the AE-CPD threshold quantile $q\\in\\{0.90,0.95,0.98,0.99\\}$, fine-tuning epochs $\\in\\{1,3\\}$, fine-tuning learning rate $\\in\\{10^{-4},3\\times10^{-4}\\}$ and replay $\\in\\{$no, yes$\\}$. Fixed settings: input window 168~h, horizon 24~h, two LSTM layers, four attention heads, batch size 256, AdamW, at most 60 epochs with early stopping (patience 8).}",
        "\\label{tab_hyperparameter_SS1}",
        "\\footnotesize",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}p{0.14\\linewidth}>{\\raggedright\\arraybackslash}p{0.36\\linewidth}>{\\raggedright\\arraybackslash}p{0.36\\linewidth}@{}}",
        "\\toprule",
        "Model & RWD1 & RWD2 \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}"])


# ------------------------------------------------------------------ CPD
def table_cpd():
    df = pd.read_csv(os.path.join(RES, "cpd_synthetic.csv"))
    lines = []
    for ds in ["SynD1", "SynD2", "SynD3"]:
        d = df[df.dataset == ds].reset_index(drop=True)
        online = ~d.method.str.contains("offline")
        bests = {c: [b and o for b, o in zip(bold_best([v if o else None for v, o in zip(d[c], online)], lower=c in ("FPR", "delay")), online)]
                 for c in ["REC", "PRE", "F1", "FPR", "AUC"]}
        for i, r in d.iterrows():
            def cell(c, nd=3, std=None):
                v = r[c]
                if pd.isna(v):
                    return "--"
                s = fmt(v, nd) + (f" $\\pm$ {r[std]:.3f}" if std else "")
                return f"\\textbf{{{s}}}" if bests[c][i] else s
            first = f"\\multirow{{{len(d)}}}{{*}}{{{ds}}}" if i == 0 else ""
            name = r.method.replace("RCAF (AE-CPD)", "\\textbf{RCAF (AE-CPD)}")
            delay = "--" if pd.isna(r.delay) else f"{r.delay:.1f}"
            lines.append(f"{first} & {name} & {cell('REC')} & {cell('PRE')} & {cell('F1', std='F1_std')} & "
                         f"{cell('FPR', 4)} & {cell('AUC', std='AUC_std')} & {delay} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Change point detection on the synthetic test series (10 series per dataset, 196 change points each). A change point counts as detected if an alarm occurs within 50 steps after it. REC, PRE and F1 are pooled over series (F1 $\\pm$ standard deviation across series); FPR and AUC are averaged over series; delay is the mean detection delay in steps. Thresholds were selected once per method on separate validation series. PELT is an offline reference that sees the whole series. Best online value in bold.}",
        "\\label{table_2}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llcccccc@{}}",
        "\\toprule",
        "Dataset & Method & REC & PRE & F1 & FPR & AUC & Delay \\\\",
        "\\midrule",
        *lines,
        "\\end{tabular*}",
        "\\end{table}"])


def table_cpd_cost():
    df = pd.read_csv(os.path.join(RES, "cpd_streaming_cost.csv"))
    rows = [f"{r.method} & {r.LAT_ms:.3f} & {r.THR_hz:,.0f} & {r.state_KB:.1f} \\\\" for r in df.itertuples()]
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Streaming cost of the change point detectors on one CPU thread (AMD EPYC 9354 host, SynD1 test series, 2,000 consecutive samples after warm-up). LAT: mean processing time per sample; THR = 1/LAT; State: memory held by the detector (buffers and model parameters).}",
        "\\label{tab:streaming_perf}",
        "\\footnotesize",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & LAT (ms) & THR (samples/s) & State (KB) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"])


# ------------------------------------------------------------------ forecasting
LABELS = {"SeasonalNaive": "Seasonal naive", "SARIMA-RT": "SARIMA-RT", "XGBoost": "XGBoost", "LSTM": "LSTM",
          "GRU": "GRU", "Transformer": "Transformer", "DLinear": "DLinear", "PatchTST": "PatchTST",
          "TFT": "TFT", "RCAF": "\\textbf{RCAF}"}


def table_forecast():
    m = pd.read_csv(os.path.join(RES, "forecast_main.csv"))
    lines = []
    cols = [("RMSE", 3), ("MAE", 3), ("MAPE", 2), ("R2", 3), ("RMSE@1", 3), ("RMSE@24", 3)]
    for ds in ["RWD1", "RWD2"]:
        d = m[m.dataset == ds].reset_index(drop=True)
        scale = 1.0
        nd_adj = 0 if ds == "RWD2" else None
        bests = {c: bold_best(list(d[c]), lower=c != "R2") for c, _ in cols}
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{NAMES[ds]}}}}} \\\\")
        for i, r in d.iterrows():
            cells = []
            for c, nd in cols:
                ndd = nd if (nd_adj is None or c in ("MAPE", "R2")) else nd_adj
                s = f"{r[c]:.{ndd}f}"
                if r.get(c + "_std", 0) > 0:
                    s += f"$\\pm${r[c + '_std']:.{ndd}f}"
                cells.append(f"\\textbf{{{s}}}" if bests[c][i] else s)
            lines.append(f"{LABELS[r.model]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Day-ahead forecasting accuracy on the test spans (all hourly origins, horizons 1--24~h, original units). Neural models: mean $\\pm$ standard deviation over three seeds; Seasonal naive, SARIMA-RT and XGBoost are deterministic. RMSE, MAE, MAPE and $R^2$ are averaged over the 24 horizons; RMSE@1 and RMSE@24 refer to the 1-h and 24-h horizons. RCAF uses AE-CPD-triggered adaptation. Best value per column in bold.}",
        "\\label{Tab_11}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "Model & RMSE & MAE & MAPE (\\%) & $R^2$ & RMSE@1 & RMSE@24 \\\\",
        "\\midrule",
        *lines,
        "\\end{tabular*}",
        "\\end{table*}"])


def table_dm():
    d = pd.read_csv(os.path.join(RES, "forecast_dm.csv"))
    piv = []
    for b in d.baseline.unique():
        r1 = d[(d.baseline == b) & (d.dataset == "RWD1")]
        r2 = d[(d.baseline == b) & (d.dataset == "RWD2")]
        cell = lambda r: "--" if r.empty else f"{r.DM.iloc[0]:.2f} ({r.p.iloc[0]:.3f})"
        piv.append(f"{LABELS.get(b, b)} & {cell(r1)} & {cell(r2)} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Diebold--Mariano tests \\cite{diebold1995comparing} of RCAF against each baseline (seed-averaged forecasts, squared-error loss averaged over the 24 horizons, Newey--West variance with 24 lags). Positive statistics indicate that RCAF is more accurate; two-sided $p$-values in parentheses.}",
        "\\label{tab_dm}",
        "\\footnotesize",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Baseline & RWD1: DM ($p$) & RWD2: DM ($p$) \\\\",
        "\\midrule",
        *piv,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"])


def table_ablation():
    a = pd.read_csv(os.path.join(RES, "forecast_ablation.csv"))
    order = list(dict.fromkeys(a.model))
    lines = []
    for ds in ["RWD1", "RWD2"]:
        d = a[a.dataset == ds].set_index("model")
        nd = 3 if ds == "RWD1" else 0
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{NAMES[ds]}}}}} \\\\")
        full = d.loc[order[0], "RMSE"]
        for mname in order:
            if mname not in d.index:
                continue
            r = d.loc[mname]
            delta = 100 * (r.RMSE - full) / full
            label = mname.replace("(alpha = 0)", "($\\alpha=0$)").replace("(alpha = 1)", "($\\alpha=1$)")
            lines.append(f"{label} & {r.RMSE:.{nd}f}$\\pm${r.RMSE_std:.{nd}f} & {r.MAE:.{nd}f} & "
                         f"{'--' if mname == order[0] else f'{delta:+.1f}'} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study on the test spans (mean over three seeds). $\\Delta$RMSE is the relative change with respect to the full RCAF model (positive: worse). Adaptation variants use the same fine-tuning settings; CUSUM and BOCPD triggers are calibrated to raise the same number of alarms as AE-CPD on the training and validation spans, and periodic adaptation uses the same number of updates as AE-CPD in the test span.}",
        "\\label{tab:ablation_values}",
        "\\footnotesize",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}p{0.52\\linewidth}ccc@{}}",
        "\\toprule",
        "Variant & RMSE & MAE & $\\Delta$RMSE (\\%) \\\\",
        "\\midrule",
        *lines,
        "\\end{tabular*}",
        "\\end{table}"])


def table_prob():
    p = pd.read_csv(os.path.join(RES, "forecast_prob.csv"))
    lines = []
    for ds in ["RWD1", "RWD2"]:
        d = p[p.dataset == ds]
        nd = 3 if ds == "RWD1" else 0
        for _, r in d.iterrows():
            lines.append(f"{ds} & {r.model} & {r.pinball:.{nd + 1}f} & {r.PICP80:.1f} & {r.width:.{nd}f} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Probabilistic accuracy of the quantile forecasts (mean over three seeds): average pinball loss over $q\\in\\{0.1,0.5,0.9\\}$, empirical coverage of the nominal 80\\% interval [$q_{0.1}$, $q_{0.9}$] (PICP) and mean interval width.}",
        "\\label{tab_prob}",
        "\\footnotesize",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Model & Pinball & PICP (\\%) & Width \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"])


def table_cost():
    c = pd.read_csv(os.path.join(RES, "cost_forecasters.csv"))
    comp = pd.read_csv(os.path.join(RES, "cost_components.csv"))
    ad = pd.read_csv(os.path.join(RES, "adaptation_stats.csv"))
    lines = []
    for ds in ["RWD1", "RWD2"]:
        d = c[c.dataset == ds]
        for _, r in d.iterrows():
            extra = ""
            if r.model == "RCAF":
                g = ad[ad.dataset == ds].adapt_cpd_time.mean()
                extra = f"{r.adapt_cpu_s:.1f} / {g:.1f}"
            lines.append(f"{ds} & {r.model} & {int(r.params):,} & {r.forecast_ms:.2f} & {r.total_ms:.2f} & "
                         f"{r.throughput_hz:.0f} & {extra if extra else '--'} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    comp_lines = []
    for comp_name in comp.component.unique():
        v = comp[comp.component == comp_name].set_index("dataset").ms
        comp_lines.append(f"{comp_name} & {v.get('RWD1', np.nan):.3f} & {v.get('RWD2', np.nan):.3f} \\\\")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Inference cost on one CPU thread of an AMD EPYC 9354 (edge-device proxy). Top: per-sample latency of the RCAF pipeline components (ms). Bottom: parameters, latency of one 24-h forecast (batch size 1), total per-sample latency including the RCAF streaming pipeline, throughput, and wall-clock time of one CPD-triggered adaptation on the CPU / on the GPU used for training (s).}",
        "\\label{tab:online_capability}",
        "\\footnotesize",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "RCAF component & RWD1 & RWD2 \\\\",
        "\\midrule",
        *comp_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2mm}",
        "",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrr@{}}",
        "\\toprule",
        "Data & Model & Params & Forecast (ms) & Total (ms) & THR (Hz) & Adapt. CPU / GPU (s) \\\\",
        "\\midrule",
        *lines,
        "\\end{tabular*}",
        "\\end{table*}"])


TABLES = {"datasets": table_datasets, "hyper": table_hyper, "cpd": table_cpd, "cpdcost": table_cpd_cost,
          "forecast": table_forecast, "dm": table_dm, "ablation": table_ablation, "prob": table_prob,
          "cost": table_cost}


def inline():
    tex = open(TEX, encoding="latin-1").read()
    for name, fn in TABLES.items():
        pat = re.compile(rf"(%%BEGIN-TABLE:{name}\n).*?(%%END-TABLE:{name})", re.S)
        if not pat.search(tex):
            print("marker missing:", name)
            continue
        try:
            body = fn()
        except FileNotFoundError as e:
            print("skip", name, e)
            continue
        tex = pat.sub(lambda mm: mm.group(1) + body + "\n" + mm.group(2), tex)
        print("inlined", name)
    open(TEX, "w", encoding="latin-1").write(tex)


if __name__ == "__main__":
    inline()

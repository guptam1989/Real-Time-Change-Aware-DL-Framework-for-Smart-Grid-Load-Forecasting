"""LaTeX tables for the v3 manuscript, generated from the result files and
inlined between %%BEGIN-TABLE:<name> and %%END-TABLE:<name> markers."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES3 = os.path.join(HERE, "results_v3")
RES = os.path.join(HERE, "results")
TEX = os.path.join(HERE, "..", "sn-article.tex")
DEV = ["RWD1", "RWD2"]
HELDOUT = ["DE", "FR", "IT", "ES", "GB", "ERCOT", "CAISO", "SPP"]
SINGLE = ["SeasonalNaive", "SARIMA-RT", "XGBoost", "LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT"]
SHORT = {"SeasonalNaive": "SNaive", "SARIMA-RT": "SARIMA", "XGBoost": "XGB", "Transformer": "Trans.",
         "PatchTST": "PatchTST", "DLinear": "DLinear", "LSTM": "LSTM", "GRU": "GRU", "TFT": "TFT",
         "RCAF": "\\textbf{RCAF}"}


def datasets_present(m):
    return [d for d in DEV + HELDOUT if d in set(m.dataset)]


def bold_min(vals):
    vals = np.asarray(vals, float)
    b = np.nanmin(vals)
    return [abs(v - b) < 1e-12 for v in vals]


def table_datasets_v3():
    src = {"RWD1": "UCI household (FR)", "RWD2": "PJM East (US)",
           "DE": "ENTSO-E Germany", "FR": "ENTSO-E France", "IT": "ENTSO-E Italy",
           "ES": "ENTSO-E Spain", "GB": "ENTSO-E Great Britain",
           "ERCOT": "EIA-930 Texas", "CAISO": "EIA-930 California", "SPP": "EIA-930 Central US"}
    events = {"RWD2": "cold wave, Dec 2017--Jan 2018", "DE": "COVID-19 lockdown, Mar 2020",
              "FR": "COVID-19 lockdown, Mar 2020", "IT": "COVID-19 lockdown, Mar 2020",
              "ES": "COVID-19 lockdown, Mar 2020", "GB": "COVID-19 lockdown, Mar 2020",
              "ERCOT": "Winter Storm Uri, Feb 2021", "CAISO": "heat wave, Aug 2020", "SPP": "Winter Storm Uri, Feb 2021"}
    files = {"RWD1": "rwd1_uci_household.csv", "RWD2": "rwd2_pjm_east.csv", **{d: f"ho_{d.lower()}.csv" for d in HELDOUT}}
    rows = []
    for ds in DEV + HELDOUT:
        p = os.path.join(HERE, "data", files[ds])
        if not os.path.exists(p):
            continue
        if ds == DEV[0]:
            rows.append("\\multicolumn{8}{@{}l}{\\textit{Development datasets}} \\\\")
        if ds == HELDOUT[0]:
            rows.append("\\multicolumn{8}{@{}l}{\\textit{Held-out datasets}} \\\\")
        d = pd.read_csv(p, parse_dates=["time"])
        n = len(d)
        n_tr, n_va = int(0.6 * n), int(0.7 * n)
        test_start = d.time.iloc[n_va]
        meta_p = os.path.join(RES3, f"alarms_{ds}.json")
        n_al = "--"
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            n_al = str(sum(1 for a in meta["alarms_ae"] if a >= meta["n_va"]))
        mean = f"{d.load.mean():.2f} kW" if ds == "RWD1" else f"{d.load.mean():,.0f} MW"
        rows.append(f"{ds} & {src[ds]} & {d.time.min():%Y}--{d.time.max():%Y} & {n:,} & "
                    f"{test_start:%Y-%m-%d} & {mean} & {events.get(ds, '--')} & {n_al} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Real-world datasets (hourly). Splits are chronological (60\\% training, 10\\% validation, 30\\% test); the test span starts on the date shown. Development datasets were used while the method was designed; held-out datasets were prepared and evaluated once after the design was frozen. The last two columns give the documented structural event in the test span and the number of AE-CPD alarms in the test span.}",
        "\\label{tab_dataset_all}",
        "\\footnotesize",   # 7 pt in sn-jnl (\\scriptsize is 9 pt there)
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lllrlr>{\\raggedright\\arraybackslash}p{2.2cm}c@{}}",
        "\\toprule",
        "ID & Source & Period & Hours & Test from & Mean load & Event in test span & Alarms \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}"])


def table_main_mase():
    m = pd.read_csv(os.path.join(RES3, "metrics_mean.csv"))
    s = pd.read_csv(os.path.join(RES3, "metrics_std.csv"))
    cols = SINGLE + ["RCAF"]
    lines = []
    for ds in datasets_present(m):
        d = m[m.dataset == ds].set_index("model")
        sd = s[s.dataset == ds].set_index("model")
        vals = [d.loc[c, "MASE"] if c in d.index else np.nan for c in cols]
        b = bold_min(vals)
        cells = []
        for c, v, bb in zip(cols, vals, b):
            txt = f"{v:.3f}"
            cells.append(f"\\textbf{{{txt}}}" if bb else txt)
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
        if ds == DEV[-1] and any(x in set(m.dataset) for x in HELDOUT):
            lines.append("\\midrule")
    r = pd.read_csv(os.path.join(RES3, "ranks_main_all.csv"), index_col=0)
    rk = [r.loc[c, "mean_rank"] if c in r.index else np.nan for c in cols]
    b = bold_min(rk)
    lines.append("\\midrule")
    lines.append("Mean rank & " + " & ".join((f"\\textbf{{{v:.2f}}}" if bb else f"{v:.2f}") for v, bb in zip(rk, b)) + " \\\\")
    rh = os.path.join(RES3, "ranks_main_held-out.csv")
    if os.path.exists(rh):
        r2 = pd.read_csv(rh, index_col=0)
        rk2 = [r2.loc[c, "mean_rank"] if c in r2.index else np.nan for c in cols]
        b2 = bold_min(rk2)
        lines.append("Mean rank (held-out) & " + " & ".join((f"\\textbf{{{v:.2f}}}" if bb else f"{v:.2f}") for v, bb in zip(rk2, b2)) + " \\\\")
    chi = r["chi2"].iloc[0]
    p = r["p"].iloc[0]
    cd = r["CD"].iloc[0]
    ptxt = f"p<10^{{{int(np.ceil(np.log10(p)))}}}" if p < 1e-3 else f"p={p:.3f}"
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Day-ahead forecasting accuracy on the test spans: MASE averaged over horizons 1--24~h (scale: in-sample mean absolute error of the 24-h seasonal naive forecast; lower is better). Neural models are averaged over three seeds. Best value per dataset in bold. Mean ranks over all datasets: Friedman $\\chi^2={chi:.1f}$, ${ptxt}$; Nemenyi critical difference {cd:.2f}.}}",
        "\\label{Tab_11}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * len(cols) + "@{}}",
        "\\toprule",
        "Dataset & " + " & ".join(SHORT[c] for c in cols) + " \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


def table_combination():
    m = pd.read_csv(os.path.join(RES3, "metrics_mean.csv"))
    cols = ["Best on validation", "Mean", "EWA", "FS", "FS (baselines only)", "RCAF (own experts only)",
            "FS+reset", "FS+spawn", "FS+reset+spawn", "Trigger: periodic", "Trigger: CUSUM", "Trigger: BOCPD", "RCAF"]
    def stack(a, b):
        return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{a}\\\\{b}\\end{{tabular}}"
    head = [stack("Best", "val."), "Mean", "EWA", "FS", stack("FS", "base."), stack("RCAF", "own"), "+reset",
            "+spawn", "+both", stack("peri-", "odic"), "CUSUM", "BOCPD", "\\textbf{RCAF}"]
    lines = []
    for ds in datasets_present(m):
        d = m[m.dataset == ds].set_index("model")
        vals = [d.loc[c, "MASE"] if c in d.index else np.nan for c in cols]
        b = bold_min(vals)
        lines.append(f"{ds} & " + " & ".join((f"\\textbf{{{v:.3f}}}" if bb else f"{v:.3f}") for v, bb in zip(vals, b)) + " \\\\")
        if ds == DEV[-1]:
            lines.append("\\midrule")
    rp = os.path.join(RES3, "ranks_combination_all.csv")
    ra = os.path.join(RES3, "ranks_ablation_all.csv")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Combination baselines and ablation of the change-aware mechanisms (test MASE, mean over seeds). Best val.: single model with the lowest validation RMSE; Mean: equal weights; EWA: exponential weights; FS: fixed share; FS base.: fixed share over the standard forecasters only; RCAF own: RCAF with its own change-aware experts only; +reset: fixed share with re-sharing and boosted share rate after AE-CPD alarms; +spawn: fixed share with CPD-spawned adapted experts; +both: re-sharing and spawning; periodic, CUSUM, BOCPD: +both with the AE-CPD alarms replaced by evenly spaced updates of the same number or by CUSUM or BOCPD alarms; RCAF: +reset or +both, whichever is more accurate on the validation span. All parameters are selected on the validation span.}",
        "\\label{tab:ablation_values}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * len(cols) + "@{}}",
        "\\toprule",
        "Dataset & " + " & ".join(head) + " \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


def table_dm_summary():
    d = pd.read_csv(os.path.join(RES3, "dm_tests.csv"))
    rows = []
    for b, g in d.groupby("baseline", sort=False):
        w = int(((g.p_holm < 0.05) & (g.DM > 0)).sum())
        l_ = int(((g.p_holm < 0.05) & (g.DM < 0)).sum())
        t = len(g) - w - l_
        wh = g[g.dataset.isin(HELDOUT)]
        w2 = int(((wh.p_holm < 0.05) & (wh.DM > 0)).sum())
        l2 = int(((wh.p_holm < 0.05) & (wh.DM < 0)).sum())
        t2 = len(wh) - w2 - l2
        name = {"SeasonalNaive": "Seasonal naive", "FS (baselines only)": "FS (standard forecasters only)"}.get(b, b)
        rows.append(f"{name} & {w} / {t} / {l_} & {w2} / {t2} / {l2} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Diebold--Mariano tests of RCAF against each alternative (seed-averaged forecasts, squared-error loss averaged over horizons, Newey--West variance with 24 lags, Holm correction within each dataset): number of datasets on which RCAF is significantly more accurate / not significantly different / significantly less accurate ($\\alpha=0.05$).}",
        "\\label{tab_dm}",
        "\\footnotesize",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Alternative & All datasets & Held-out datasets \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"])


def table_events():
    e = pd.read_csv(os.path.join(RES3, "event_windows.csv"))
    det = pd.read_csv(os.path.join(RES3, "event_detection.csv"))
    meths = ["XGBoost", "PatchTST", "TFT", "Best on validation", "Mean", "FS", "FS+reset+spawn", "RCAF"]
    head = ["XGB", "PatchTST", "TFT", "Best val.", "Mean", "FS", "+both", "\\textbf{RCAF}"]
    lines = []
    for ds, g in e.groupby("dataset", sort=False):
        g = g.set_index("model")
        vals = [g.loc[m, "MASE"] if m in g.index else np.nan for m in meths]
        b = bold_min(vals)
        dd = det[(det.dataset == ds) & (det.trigger == "ae")]
        delay = "--" if dd.empty or pd.isna(dd.delay_h.iloc[0]) else f"${dd.delay_h.iloc[0] / 24:.1f}$"
        ev = {"cold wave Dec 2017 - Jan 2018": "cold wave", "COVID-19 contact ban": "lockdown",
              "COVID-19 lockdown": "lockdown", "COVID-19 state of alarm": "lockdown",
              "Winter Storm Uri": "storm Uri", "August 2020 heat wave": "heat wave"}.get(g["event"].iloc[0],
                                                                                    g["event"].iloc[0])
        lines.append(f"{ds} & {ev} & {delay} & " + " & ".join((f"\\textbf{{{v:.3f}}}" if bb else f"{v:.3f}")
                                                         for v, bb in zip(vals, b)) + " \\\\")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Accuracy after documented structural events: MASE over the forecast origins in the 8 weeks (COVID-19 lockdowns) or 4 weeks (weather events) after the event start, and the delay (days) of the first AE-CPD alarm relative to the event start (negative: alarm up to three days before the nominal start date; ``--'': no alarm within the window). Start dates: RWD2 2017-12-26; DE 2020-03-22; FR 2020-03-17; IT 2020-03-10; ES 2020-03-15; GB 2020-03-24; ERCOT and SPP 2021-02-13; CAISO 2020-08-14. +both: FS+reset+spawn (re-sharing and spawning always available).}",
        "\\label{tab_events}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llc" + "c" * len(meths) + "@{}}",
        "\\toprule",
        "Dataset & Event & Delay (d) & " + " & ".join(head) + " \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


def table_intervals():
    p = pd.read_csv(os.path.join(RES3, "intervals.csv"))
    order = ["RCAF + ACI", "RCAF + ACI (no boost)", "RCAF + symmetric ACI", "RCAF + rolling conformal",
             "RCAF-TFT expert (quantiles)", "TFT (quantiles)"]
    # relative interval score: normalise by the RCAF + ACI score on each dataset
    ref = p[p.method == "RCAF + ACI"].set_index("dataset").interval_score
    lines = []
    for mth in order:
        g = p[p.method == mth].set_index("dataset")
        if g.empty:
            continue
        rel = (g.interval_score / ref.reindex(g.index)).mean()
        gh = g[g.index.isin(HELDOUT)]
        lines.append(f"{mth} & {g.PICP80.mean():.1f} & {g.PICP80.min():.1f}--{g.PICP80.max():.1f} & "
                     f"{gh.PICP80.mean():.1f} & {rel:.3f} \\\\")
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Nominal 80\\% prediction intervals on the test spans: mean and range of the empirical coverage over all datasets, mean coverage on the held-out datasets, and the interval score relative to RCAF + ACI (averaged over datasets; lower is better).}",
        "\\label{tab_prob}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Cov. (\\%) & Range & Cov. held-out & Rel. IS \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"])


def table_semisynth():
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RES3, "cpd_semisynth_*.csv")))
                    if "cost" not in p])
    lines = []
    for ds in ["LoadL", "LoadA", "LoadLA"]:
        d = df[df.dataset == ds].reset_index(drop=True)
        bf = bold_min(-d.F1.values)
        ba = bold_min(-d.AUC.values)
        for i, r in d.iterrows():
            first = f"\\multirow{{{len(d)}}}{{*}}{{{ds}}}" if i == 0 else ""
            f1 = f"{r.F1:.3f} $\\pm$ {r.F1_std:.3f}"
            au = f"{r.AUC:.3f} $\\pm$ {r.AUC_std:.3f}"
            f1 = "\\textbf{" + f1 + "}" if bf[i] else f1
            au = "\\textbf{" + au + "}" if ba[i] else au
            delay = "--" if pd.isna(r.delay) else f"{r.delay:.0f}"
            meth = "BOCPD" if r.method == "Bayesian" else r.method
            lines.append(f"{first} & {meth} & {r.REC:.3f} & {r.PRE:.3f} & {f1} & {au} & {delay} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Change point detection on real hourly load with injected changes (10 test series of 6,000~h per type; detection margin 168~h). LoadL: level shifts of 5--15\\%; LoadA: changes in the amplitude of the daily profile; LoadLA: both. Thresholds were selected on three validation series per type.}",
        "\\label{tab_semisynth}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Data & Method & REC & PRE & F1 & AUC & Delay (h) \\\\",
        "\\midrule",
        *lines,
        "\\end{tabular}",
        "\\end{table}"])


def table_experts():
    """Expert-level ablation: effect of the autoregressive highway (HW) and of the
    change-aware inputs (CA), from the post-hoc ablation in run_ablation_hw.py."""
    p = os.path.join(RES3, "ablation_hw.csv")
    if not os.path.exists(p):
        return table_experts_simple()
    a = pd.read_csv(p)

    def pct(new, old, pval):
        ch = 100 * (new / old - 1)
        s = ("$-$" if ch < 0 else "+") + f"{abs(ch):.1f}"
        return s + ("$^{*}$" if pval < 0.05 else "\\phantom{$^{*}$}")

    lines, cnt = [], {}
    for ds in DEV + HELDOUT:
        g = a[a.dataset == ds].set_index("base")
        if g.empty:
            continue
        cells = []
        for base in ("TFT", "Transformer", "XGBoost"):
            if base not in g.index:
                cells += ["--"] * (3 if base != "XGBoost" else 2)
                continue
            r = g.loc[base]
            cells.append(f"{r.MASE_base:.3f}")
            if base != "XGBoost":
                cells.append(pct(r.MASE_highway, r.MASE_base, r.p_highway_abs))
                ref = r.MASE_highway
            else:
                ref = r.MASE_base
            cells.append(pct(r.MASE_change_aware, ref, r.p_inputs_abs))
            if ds in HELDOUT:
                for key, new, old, pv, dm in (("hw", r.get("MASE_highway", np.nan), r.MASE_base,
                                               r.get("p_highway_abs", 1.0), r.get("DM_highway_abs", 0.0)),
                                              ("ca", r.MASE_change_aware, ref, r.p_inputs_abs, r.DM_inputs_abs)):
                    if np.isnan(new):
                        continue
                    c = cnt.setdefault((base, key), [0, 0, 0])
                    c[0] += int(new < old)
                    c[1] += int(pv < 0.05 and dm > 0)     # DM > 0: the modified model is more accurate
                    c[2] += int(pv < 0.05 and dm < 0)
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
        if ds == DEV[-1]:
            lines.append("\\midrule")

    def summ(base, key):
        c = cnt.get((base, key))
        return "--" if c is None else f"{c[0]} ({c[1]}/{c[2]})"
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{@{}l}{Improved (held-out)} & " + summ("TFT", "hw") + " & " + summ("TFT", "ca") + " & & "
                 + summ("Transformer", "hw") + " & " + summ("Transformer", "ca") + " & & " + summ("XGBoost", "ca") + " \\\\")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Expert-level ablation (post hoc): test MASE of the standard TFT, Transformer and XGBoost, and the relative change (\\%) caused by adding the autoregressive highway to the standard model (HW: TFT-HW vs.\\ TFT, Transformer-HW vs.\\ Transformer) and by adding the change-aware Phase~1--2 inputs on top of it (CA: RCAF-TFT vs.\\ TFT-HW, RCAF-TRF vs.\\ Transformer-HW, RCAF-GBM vs.\\ XGBoost). Negative: more accurate. $^{*}$: Diebold--Mariano test on the absolute errors of the seed-averaged forecasts (consistent with the MASE) significant at the 5\\% level. The last row counts the held-out datasets with an improvement and, in parentheses, with a significant improvement / deterioration.}",
        "\\label{tab_experts}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcrrcrrcr@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{TFT} & \\multicolumn{3}{c}{Transformer} & \\multicolumn{2}{c}{Gradient boosting} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-9}",
        "Dataset & MASE & HW (\\%) & CA (\\%) & MASE & HW (\\%) & CA (\\%) & MASE & CA (\\%) \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


def table_experts_simple():
    m = pd.read_csv(os.path.join(RES3, "metrics_mean.csv"))
    pairs = [("TFT", "RCAF-TFT expert"), ("Transformer", "RCAF-TRF expert"), ("XGBoost", "RCAF-GBM expert")]
    lines, better = [], {p[1]: 0 for p in pairs}
    present = datasets_present(m)
    for ds in present:
        d = m[m.dataset == ds].set_index("model")
        cells = []
        for base, exp in pairs:
            if base in d.index and exp in d.index:
                b, e = d.loc[base, "MASE"], d.loc[exp, "MASE"]
                ch = 100 * (e / b - 1)
                better[exp] += int(e < b)
                txt = f"{b:.3f} & {e:.3f} ({ch:+.1f}\\%)"
                cells.append(txt)
            else:
                cells.append("-- & --")
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
        if ds == DEV[-1]:
            lines.append("\\midrule")
    lines.append("\\midrule")
    lines.append("Datasets improved & & " + " & & ".join(f"{better[p[1]]}/{len(present)}" for p in pairs) + " \\\\")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Effect of the change-aware inputs at the expert level: test MASE of each standard forecaster and of its change-aware counterpart, which additionally receives the Phase~1--2 features (and, for the neural experts, the autoregressive highway). Percentages are relative changes (negative: the change-aware expert is more accurate).}",
        "\\label{tab_experts}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "Dataset & TFT & RCAF-TFT & Transformer & RCAF-TRF & XGBoost & RCAF-GBM \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


def table_post_alarm():
    p = pd.read_csv(os.path.join(RES3, "post_alarm.csv"))
    meths = ["Best on validation", "Mean", "FS", "FS+reset", "FS+spawn", "FS+reset+spawn", "RCAF"]
    head = ["Best val.", "Mean", "FS", "+reset", "+spawn", "+both", "\\textbf{RCAF}"]
    lines = []
    for ds, g in p.groupby("dataset", sort=False):
        g = g.set_index("model")
        vals = [g.loc[m, "MASE"] if m in g.index else np.nan for m in meths]
        b = bold_min(vals)
        lines.append(f"{ds} & {int(g.n_alarms.iloc[0])} & " + " & ".join(
            (f"\\textbf{{{v:.3f}}}" if bb else f"{v:.3f}") for v, bb in zip(vals, b)) + " \\\\")
    return "\n".join([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Accuracy in the 14 days after each AE-CPD alarm in the test spans (MASE over the forecast origins inside these windows). The second column gives the number of alarms.}",
        "\\label{tab_post_alarm}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lc" + "c" * len(meths) + "@{}}",
        "\\toprule",
        "Dataset & Alarms & " + " & ".join(head) + " \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}"])


def table_cost_v3():
    c = pd.read_csv(os.path.join(RES3, "cost_v3.csv")).set_index("component").ms
    comp = pd.read_csv(os.path.join(RES, "cost_components.csv"))
    comp = comp[comp.dataset == "RWD2"].set_index("component").ms
    fc = pd.read_csv(os.path.join(RES, "cost_forecasters.csv"))
    fc = fc[fc.dataset == "RWD2"].set_index("model").forecast_ms
    sar = eval(open(os.path.join(RES, "sarima_info_RWD2.txt")).read())["lat_ms"]
    phase12 = comp.drop("TFT forecast (24 h)").sum()
    per_hour = [("Phase 1--2 (RSSA, GARCH, time-domain features, AE descriptor, AE-CPD score)", phase12)]
    per_hour += [("RCAF-TFT forecast", c["RCAF-TFT forecast (24 h, batch 1)"]),
                 ("RCAF-TRF forecast", c["RCAF-TRF forecast (24 h, batch 1)"]),
                 ("RCAF-GBM forecast (24 models)", c["RCAF-GBM forecast (24 models, one origin)"])]
    per_hour += [(f"{m} forecast", fc[m]) for m in ["LSTM", "GRU", "Transformer", "DLinear", "PatchTST", "TFT"]]
    per_hour += [("XGBoost forecast (24 models)", c["RCAF-GBM forecast (24 models, one origin)"]),
                 ("SARIMA-RT state update and forecast", sar),
                 ("Online aggregation (14 experts, 24 horizons)", c["Online aggregation (14 experts, 24 horizons)"]),
                 ("Asymmetric adaptive conformal intervals", c["Asymmetric ACI (24 horizons, window 720)"])]
    total = sum(v for _, v in per_hour)
    rows = [f"{k} & {v:.2f} \\\\" for k, v in per_hour]
    rows += ["\\midrule", f"Total per hour of data & {total:.1f} \\\\", "\\midrule",
             "\\multicolumn{2}{l}{\\textit{Per CPD-triggered update (s)}} \\\\",
             f"RCAF-TFT fine-tuning (1 epoch, 1,441 origins) & {c['RCAF-TFT CPD-triggered fine-tuning'] / 1e3:.1f} \\\\",
             f"RCAF-TRF fine-tuning & {c['RCAF-TRF CPD-triggered fine-tuning'] / 1e3:.1f} \\\\",
             f"RCAF-GBM boosting continuation (50 trees $\\times$ 24) & {c['RCAF-GBM CPD-triggered update (50 trees x 24 models)'] / 1e3:.1f} \\\\"]
    return "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cost of running RCAF on one CPU thread of an AMD EPYC 9354 (RWD2): time per new hour of data for each component (ms) and time of one CPD-triggered update (s). Updates run in the background and do not delay the forecasts of the other experts.}",
        "\\label{tab:online_capability}",
        "\\footnotesize",
        "\\begin{tabular}{p{0.72\\linewidth}r}",
        "\\toprule",
        "Component & Time (ms) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"])


def table_cpd_synthetic():
    import tables as t2   # synthetic CPD table of the earlier pipeline (unchanged results)
    return t2.table_cpd().replace("Bayesian", "BOCPD")


TABLES = {"datasets": table_datasets_v3, "experts": table_experts, "postalarm": table_post_alarm,
          "cpd": table_cpd_synthetic,
          "cost": table_cost_v3, "forecast": table_main_mase, "ablation": table_combination,
          "dm": table_dm_summary, "events": table_events, "prob": table_intervals, "semisynth": table_semisynth}


def inline(names=None):
    if not os.path.exists(TEX):   # manuscript not present (e.g. public code release): write one .tex file per table
        out = os.path.join(RES3, "tables")
        os.makedirs(out, exist_ok=True)
        for name, fn in TABLES.items():
            if names and name not in names:
                continue
            try:
                body = fn()
            except FileNotFoundError as e:
                print("skip", name, e)
                continue
            with open(os.path.join(out, f"{name}.tex"), "w", encoding="latin-1") as f:
                f.write(body + "\n")
            print("written", os.path.join(out, f"{name}.tex"))
        return
    tex = open(TEX, encoding="latin-1").read()
    for name, fn in TABLES.items():
        if names and name not in names:
            continue
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
    import sys
    inline(sys.argv[1:] or None)

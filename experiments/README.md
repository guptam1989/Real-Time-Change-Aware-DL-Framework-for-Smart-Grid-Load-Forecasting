# RCAF experiments: reproduction guide

This folder contains the code that produces every number, table and figure in the paper.
Run all commands from this folder. The scripts locate their inputs and outputs relative to it.

## Environment

Python 3.10 with the packages in `../requirements.txt`, which lists the versions used in the paper.
Neural experts were trained on NVIDIA H100 GPUs (MIG slices). Everything also runs on a CPU, only more
slowly. `RCAF_GPUS` selects a GPU or MIG device by UUID (`auto` picks the device with the most free
memory). `RCAF_WPG` sets the number of parallel training jobs per device. `RCAF_DS` selects the datasets,
for example `RCAF_DS=DE+FR`.

## Protocol

- **Development datasets:** RWD1 (UCI household) and RWD2 (PJM East). All design decisions were made on
  these.
- **Held-out datasets:** DE, FR, IT, ES, GB (ENTSO-E via Open Power System Data) and ERCOT, CAISO, SPP
  (EIA-930). They were prepared after development and evaluated once, after the method was frozen.
  `results_v3/FROZEN.txt` holds the SHA-256 hashes of the method files at freeze time.
- **Splits:** chronological, 60 % training, 10 % validation, 30 % test.
- **Features:** all causal. Scaling, GARCH and the autoencoder are fitted on the training span only.
- **Forecasts:** input window 168 h. Forecasts for 1–24 h are issued at every test hour, and errors are
  computed on the raw load.
- **Seeds:** three per neural model.

## Steps

| Step | Command | Output |
|---|---|---|
| 1. Data (downloads from the original sources) | `python data.py` (development) and `python data_heldout.py` (held-out) | `data/*.csv` |
| 2. Experts | `RCAF_DS=DE python run_v3.py all` (stages: features, search, final, gbm) | `results_v3/preds/*` |
| 3. Classical baselines | `RCAF_DS=DE python run_classical_v3.py` | seasonal naive and SARIMA-RT forecasts |
| 4. Online aggregation and conformal intervals | `python aggregate.py DE` | `results_v3/agg/*` |
| 5. Evaluation | `python evaluate_v3.py` | metrics, ranks, Friedman/Nemenyi, DM–Holm, event windows, intervals |
| 6. Detection benchmarks | `python run_cpd.py` (synthetic), `python cpd_semisynth.py L A LA` (real load with injected changes) | `results/cpd_*`, `results_v3/cpd_semisynth_*` |
| 7. Cost | `python measure_cost.py` and `python measure_cost_v3.py RWD2` | `results/cost_*.csv`, `results_v3/cost_v3.csv` |
| 8. Figures and tables | `python make_diagrams.py`, `python make_figures.py cpd`, `python make_figures_v3.py`, `python tables_v3.py` | `../figures/*.png`; LaTeX tables in `results_v3/tables/`, or inlined into the manuscript if it is present at `../sn-article.tex` |

The result tables of the paper are included in `results/` and `results_v3/`. With them, `tables_v3.py`,
`make_figures.py cpd` and the `cd`, `gain` and `semisynth` figures of `make_figures_v3.py` run directly.

Two kinds of intermediate output are not included because of their size (several GB):

- the per-model forecasts (`results_v3/preds/`);
- the aggregation outputs (`results_v3/agg/`).

Steps 2–4 regenerate them. They are needed to recompute the metrics (step 5) and for the event,
coverage and attention figures.

### Post-hoc analyses

These analyses were run after the held-out evaluation. They do not change the method.

| Analysis | Command | Output |
|---|---|---|
| Highway ablation: TFT and Transformer with the autoregressive highway but standard inputs | `RCAF_DS=DE+FR python run_ablation_hw.py all`, then `python run_ablation_hw.py eval` | `results_v3/ablation_hw.csv` |
| Weight diagnostics around an event (entropy, weights of FS and RCAF) | `python analyze_weights.py IT 2020-03-10 0` | printed |

### Cluster scripts

`pbs_*.sh` are the PBS job scripts used on the GPU cluster. Submit them from the repository root, for
example:

```
qsub -v RCAF_DS=DE,STAGE=all experiments/pbs_v3.sh
```

Adapt the queue names and resources to your cluster. Every stage resumes from completed outputs, so a
job that is killed (for example by an out-of-memory daemon) can simply be resubmitted.

## Files

**Method (frozen; hashes in `results_v3/FROZEN.txt`)**
- `rssa.py`: recursive singular spectrum analysis.
- `features.py`: Phase 1–2 features, autoencoder descriptors and the AE-CPD score.
- `cpd.py`: AE-CPD and the baseline detectors (CUSUM, BOCPD, uLSIF, RuLSIF, KLIEP).
- `models.py`: TFT/RCAF-TFT, Transformer/RCAF-TRF, LSTM, GRU, DLinear and PatchTST.
- `run_forecast.py`: training and forecasting utilities.
- `run_v3.py`: expert pipeline and adaptation plans.
- `run_gbm.py`: gradient-boosting experts (XGBoost, RCAF-GBM).
- `aggregate.py`: change-aware online aggregation and adaptive conformal intervals.

**Data, baselines and evaluation**
- `data.py`, `data_heldout.py`: data download and preparation.
- `run_classical_v3.py` (uses `run_classical.py`): seasonal naive and SARIMA-RT.
- `run_cpd.py`, `cpd_semisynth.py`: detection benchmarks.
- `evaluate_v3.py`: metrics, statistical tests and event analyses.
- `measure_cost.py`, `measure_cost_v3.py`: latency.

**Figures and tables**
- `make_diagrams.py`, `make_figures.py`, `make_figures_v3.py`: figures.
- `tables_v3.py`: tables. It uses `tables.py` for the synthetic CPD table.

**Post-hoc analyses**
- `run_ablation_hw.py`, `analyze_weights.py`.

**Earlier pipeline versions** (`evaluate.py`, `tables.py`, `val_diagnostic.py`, `analyze_dev.py` and the
single-model stages of `run_forecast.py`) are kept for transparency. They produced the development
results that motivated the final design.

`evaluate_v3.py` is not one of the frozen files. After the freeze it was changed in one respect only: the
Diebold–Mariano test now returns DM = 0 and p = 1 when two forecasts are identical, instead of NaN. This
affects only the rows where RCAF selected no reaction and therefore coincides with FS or EWA.

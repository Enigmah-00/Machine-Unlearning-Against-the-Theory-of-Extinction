# Suppressed, Not Erased: Testing Machine Unlearning Against the Theory of Extinction

Code, notebooks, logs and results for the paper of the same name.

We turn the four relapse phenomena of extinction learning (spontaneous recovery, renewal, reinstatement and rapid
reacquisition) into tests for unlearned image classifiers, and run them on CIFAR-100 / ResNet-18 models unlearned with
five methods (FT, NegGrad+, RL, SCRUB, SalUn), always against a model retrained from scratch. A second arm repeats
the tests on forgetting atypical training examples.

## Contents

| Path | What it is |
|---|---|
| `code/ext_utils.py` | The shared library: data pipeline, model, the five unlearning methods, all tests, mechanism analyses and the resumable multi-GPU job runner. Every notebook imports this one file. |
| `notebooks/00`–`10` | The Kaggle notebooks, in run order (see below). `notebooks/README.md` has the detailed run guide (inputs, settings, output names). |
| `RUNLOG.md` | The full run history: every Kaggle run, its settings, timings, results, problems and the decisions taken, plus the library version history and every change to the analysis plan. |
| `results/` | Outputs of the runs that produced the paper's numbers: raw result rows (`*.jsonl`), worker logs, job lists, analysis tables and figures (see the table below). |
| `paper/make_figures.py` | Redraws the paper's figures from `results/`. |
| `paper/figures/` | The figures as used in the paper. |

## Running the experiments

All experiments ran on free Kaggle notebooks with two T4 GPUs (about 30 hours of two-GPU sessions in total).

1. Upload `code/ext_utils.py` as a private Kaggle dataset (we called it `ext-code`).
2. Run the notebooks in order. Each notebook saves its output, which later notebooks attach as input:

| Notebook | Purpose | Paper |
|---|---|---|
| `00_setup_and_selftest` | Environment check and library self-test | – |
| `01_train_originals` | Train three ResNet-18 models on CIFAR-100 | Sec. 4 |
| `02_retrain_references` | Retrain without each forget class (gold standard) | Sec. 3.1, 4 |
| `03_unlearning` | Calibrate and run the five methods (standard, ABA, multi-context) | Sec. 4 |
| `04_extinction_tests` | Spontaneous recovery, renewal, reinstatement, rapid reacquisition | Sec. 5 |
| `05_mechanism` | Linear probes, head swap, weight change, CKA | Sec. 5.1, 5.4 |
| `06_analysis` | Class-level hypothesis tests (Holm-corrected) | Sec. 5, Table 2 |
| `07_example_level_and_bias` | Head-bias check; first example-level design (random forget sets) | App. C |
| `08_followup_analysis` | Example-level tests, class vs example comparison, head bias, relearning speed | Sec. 5.3, 5.4, 6 |
| `09_example_level_atypical` | Example-level arm with atypical (low C-score) forget sets | Sec. 6 |
| `10_matched_search` | Longer schedules to obtain matched example-level models | Sec. 6 |

The atypical arm downloads the published CIFAR-100 C-scores (Jiang et al., 2021) from the authors' site; notebook 04
uses CIFAR-100-C (Hendrycks and Dietterich, 2019). Each notebook lists its inputs and Kaggle settings at the top.

## Results included here

| Folder | Run in `RUNLOG.md` | Notebook | Contents |
|---|---|---|---|
| `run04_nb04_pilot_v1.1` | #4 | 04 (pilot) | Pilot with the first score (the top-1 floor problem) |
| `run05_nb04_pilot_v1.2` | #5 | 04 (pilot) | Pilot with the graded scores |
| `run06_nb05_mechanism` | #6 | 05 | Probe, head-swap, weight and CKA rows |
| `run07_nb04_main` | #7 | 04 (main) | All class-level relapse-test rows (360 models) |
| `run08_nb06_analysis` | #8 | 06 | Class-level hypothesis tests and figures |
| `run10_nb08_followup` | #10 | 08 | Head-bias check and relearning speed (first version) |
| `run11_nb08_preview` | #11 | 08 | Preview on the random example-level design |
| `run12_nb09_atypical` | #12 | 09 | Executed notebook with all outputs of the atypical arm |
| `run13_nb08_old_copy` | #13 | 08 | Run with an outdated notebook copy (kept for completeness) |
| `run14_nb08_atypical` | #14 | 08 | Atypical arm, all models (partial-unlearning checks) |
| `run16_nb08_final` | #16 | 08 | Final example-level analysis including the matched models |

`RUNLOG.md` refers to these folders by their download names (`Downloads/results-2` = run 04, `results-3` = run 06,
`results-4` = run 07, `results-5` = run 08, `results-6` = run 10, `results (24).zip` = run 05, `results (26)`–`(29)` =
runs 11, 13, 14 and 16).

Not included: model checkpoints (several GB) and the raw rows of the training, retraining, unlearning and
example-level runs (runs 1–3, 9, 12 and 15), which are stored with their checkpoints. Their summaries are in
`RUNLOG.md`, and the tables derived from them are in the folders above.

## Redrawing the figures

```bash
pip install numpy pandas scipy statsmodels matplotlib
python paper/make_figures.py
```

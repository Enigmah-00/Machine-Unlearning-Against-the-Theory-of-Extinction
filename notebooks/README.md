# Machine Unlearning is Extinction, Not Forgetting: Kaggle code

## Files

| File | Purpose |
|---|---|
| `ext_utils.py` | Shared library: data, ResNet-18 (BatchNorm or GroupNorm), 5 unlearning methods + multi-context unlearning, 4 extinction tests with BatchNorm controls, mechanism analyses, resumable multi-GPU job runner. Upload it to Kaggle as a private dataset. |
| `notebooks/00_setup_and_selftest.ipynb` | Environment check, self-test of every component, speed benchmark |
| `notebooks/01_train_originals.ipynb` | 3 original models (all data) |
| `notebooks/02_retrain_references.ipynb` | 31 retrained reference models (forget class never seen) |
| `notebooks/03_unlearning.ipynb` | Calibration → pilot → 150 unlearned models → 75 ABA models → 75 multi-context models |
| `notebooks/04_extinction_tests.ipynb` | Spontaneous recovery, renewal, reinstatement, savings; every fine-tuned model scored with updated *and* original BatchNorm statistics, plus a BatchNorm-recalibration-only control (pilot first, then main) |
| `notebooks/05_mechanism.ipynb` | Linear probe, head swap, weight change, CKA |
| `notebooks/06_analysis.ipynb` | H1–H6 + BatchNorm controls (C1, C2) with Holm correction per family, bootstrap CIs, 8 figures (CPU only) |
| `notebooks/07_example_level_and_bias.ipynb` | Follow-up (1.3): head-bias check on the class-level models; example-level forgetting arm (retrain → calibrate → unlearn → the same relapse tests) |
| `notebooks/08_followup_analysis.ipynb` | Follow-up analysis (CPU): example-level hypotheses XH1–XH5, the class/example boundary test B1, head bias (E1), savings speed (E2), figures 9–12 |
| `notebooks/09_example_level_atypical.ipynb` | Example-level arm, atypical design (1.4): forget sets from the 5 % least typical images (C-score), denser grid, same relapse tests; head-bias check v2 (held-out shift) |
| `notebooks/10_matched_search.ipynb` | Matched-model search (1.5): RL-long, SalUn-long, NegGrad+-long variants on the atypical forget sets; reuses ext-09's retrained models |

## One-time setup

1. **Phone-verify** your Kaggle account (needed for GPU and internet).
2. **Datasets → New Dataset**: upload `ext_utils.py` with the title `ext-code`, Private.
3. For each notebook: **Create → New Notebook → File → Import Notebook** and upload the `.ipynb`.
4. In each notebook: **Add Input → Your Datasets → ext-code**. Set the Accelerator (GPU T4 x2; *None* for 06)
   and turn Internet On. Each notebook's first section lists its exact settings and inputs.
5. In notebooks 00–05 also attach **[CIFAR-100 Python (fedesoriano/cifar100)](https://www.kaggle.com/datasets/fedesoriano/cifar100)**.
   Without it, CIFAR-100 downloads from cs.toronto.edu at ~50 kB/s: that took 56 of the 75 minutes of run #1
   (see `RUNLOG.md`).

All runs are logged in **`RUNLOG.md`**: results, issues, quota used, and code-version changes.

## Run order

| Step | Notebook | Attach these inputs | Settings | Approx. time (T4 x2) |
|---|---|---|---|---|
| 1 | 00 setup check | ext-code | GPU T4 x2, Internet on | 10 min (run interactively) |
| 2 | *(optional)* 01→06 with `smoke=True` | as below | as below | 30–60 min total, a dry run of the file hand-over |
| 3 | 01 originals | ext-code | GPU T4 x2, Internet on | ~0.5 h |
| 4 | 02 retrained references | ext-code, **01** | GPU T4 x2, Internet on | ~3–4 h (or split with `SHARD`) |
| 5 | 03 unlearning (stages A–E) | ext-code, **01**, **02** | GPU T4 x2, Internet on | ~6–9 h (may need 2 sessions) |
| 6 | 04 extinction, **pilot** (`RUN_PILOT=True, RUN_MAIN=False`) | ext-code, **01, 02, 03** | GPU T4 x2, Internet on | ~15 min, then **go/no-go** |
| 7 | 04 extinction, **main** (`RUN_MAIN=True`) | ext-code, **01, 02, 03** (+ own previous output) | GPU T4 x2, Internet on | ~8–10 h (**2 sessions**: re-run with own output attached) |
| 8 | 05 mechanism | ext-code, **01, 02, 03** | GPU T4 x2 | ~1.5–2.5 h (can run in parallel with step 7 on the co-author's account) |
| 9 | 06 analysis | ext-code, **01, 02, 03, 04, 05** | Accelerator **None** | ~5 min |
| 11 | **07 follow-ups** (head bias + example-level arm) | ext-code **1.3**, **01, 02, 03**, CIFAR-100 | GPU T4 x2, Internet on | ~4.5–5 h |
| 12 | **08 follow-up analysis** | ext-code 1.3, **01, 04, 07** | Accelerator **None** | ~2 min |
| 13 | **09 atypical example-level arm** (replaces 07 for the example-level results) | ext-code **1.4**, **01, 02, 03**, CIFAR-100 | GPU T4 x2, Internet **on** (C-scores) | ~4.5–5 h |
| 14 | **08 again**, `XL_DESIGN = "atypical"` | ext-code 1.4, **01, 04, 09** (07 optional) | Accelerator **None** | ~2 min |
| 10 | *(optional)* **GroupNorm arm**: copies of 01–06 titled `…-gn`, with `norm="gn", seeds=[0], forget_classes=["maple_tree", "tiger", "rose", "shark", "bus"]` | the `-gn` outputs | as above | ~4–6 h total |

Always run a step with **Save Version → Save & Run All (Commit)**, so it runs in the background (up to 12 h).
Attach a notebook's output via **Add Input → Your Work → (notebook name)**.

Total: about 20–28 GPU-session hours for the main arm (+4–6 h for the optional GroupNorm arm), i.e. about
2 weeks of free quota for one account, or about 1 week if you and your co-author split notebooks 02–05.

## Output names

Give each Kaggle notebook the same title as its output dataset, so it's always obvious which output came from
which notebook. The code finds files by *file name*, wherever they are attached, so these names are for you
and your co-author, not for the code.

| Run | Notebook title = output dataset name | Contains | Attach it to |
|---|---|---|---|
| code | `ext-code` (uploaded dataset) | `ext_utils.py` | every notebook |
| 00 | `ext-00-setup-selftest` | nothing needed (no dataset) | – |
| 01 | `ext-01-originals` | 3 original checkpoints, `train_original` results, `experiment_config.json` | 02, 03, 04, 05, 06 |
| 02 | `ext-02-retrain` | 31 retrained checkpoints, `retrain` results | 03, 04, 05, 06 |
| 02 split | `ext-02-retrain-a` (you, `SHARD=(0,2)`), `ext-02-retrain-b` (co-author, `SHARD=(1,2)`) | half each | attach **both** to 03, 04, 05, 06 |
| 03 | `ext-03-unlearning` | `calibration.json`, `calibrate` + `unlearn` results, 300 checkpoints (≈ 7 GB) | 04, 05, 06 |
| 04 | `ext-04-extinction` | `extinction` results (pilot + main), `analysis_choices.json` | 06 |
| 05 | `ext-05-mechanism` | `mechanism` results | 06 |
| 06 | `ext-06-analysis` | `tables/*.csv`, `figures/*.png`, `figures/*.pdf` | – (final results) |
| 07 | `ext-07-example-level` | `headbias` + `xl_*` results, `xl_calibration.json`, 10 retrained + 50 unlearned example-level checkpoints (≈ 1.3 GB) | 08 |
| 08 | `ext-08-followup` | `tables/followup_tests.csv`, figures 9–12 | – |
| 09 | `ext-09-atypical` | `headbias` (v2) + atypical `xl_*` results, `xl_calibration_atypical.json`, 10 retrained + 50 unlearned checkpoints | 08 |
| 10 | `ext-10-matched-search` | variant `xl_*` results and checkpoints | 08 |
| optional | `cifar100-c` (only if you upload CIFAR-100-C yourself) | `labels.npy` + corruption `.npy` files | 04 |

- **Making the dataset:** notebook viewer → **Output** → **New Dataset** → the name above, Private. Or skip
  datasets and attach the notebook output directly (*Add Input → Your Work → Notebooks*).
- **Re-runs, resumes, pilot → main:** keep the same name and add a **New Version**. Each output carries over
  everything from earlier runs, so only the latest version matters. Update the input to that version in the
  notebooks that use it.
- **Smoke (dry) runs:** add `-smoke` (e.g. `ext-01-originals-smoke`), so test outputs never mix with real ones.
- **GroupNorm control arm** (`norm="gn"`): add `-gn` (e.g. `ext-01-originals-gn` … `ext-06-analysis-gn`). Its files carry a `_gn` suffix inside, too.
- **Splitting other notebooks too:** use the same `-a` / `-b` suffix (e.g. `ext-04-extinction-a`).

## BatchNorm controls and multi-context unlearning

- **Why:** "The BatchNorm illusion" (Kalani et al., Sep 2026) showed that recomputing BatchNorm statistics alone
  can reverse apparent forgetting. Without controls, reviewers could attribute any recovery we find to that.
- **Control 1:** every fine-tuning recovery test (retain fine-tuning, reinstatement, savings) trains once,
  normally, and scores each checkpoint twice: `update` (statistics from the fine-tuning) and `frozen` (original
  statistics put back, so only the weight change counts; `recovery_bn_modes`). The primary tests in notebook 06
  use `frozen`. (Up to 1.1, `frozen` meant training with BN layers in eval mode, which collapsed models in the
  pilot; see `RUNLOG.md` run #4.)
- **Control 2:** proxy `bn_recal` recomputes BN statistics from 10…10,000 retain images with no weight change
  (`bn_recal_sizes`). It measures the artifact directly (test C1 in notebook 06).
- **Control 3 (optional):** the GroupNorm arm (`norm="gn"`), with no running statistics at all.
- **Multi-context unlearning (H6):** extinction theory predicts that extinguishing in several contexts
  reduces relapse. Notebook 03 stage E unlearns 5 classes with every image in a random context
  (`mctx_contexts` = clean/gray/blur). Notebook 06 compares these models with standard ones on *held-out*
  contexts and on spontaneous recovery and reinstatement.

## Scores used for recovery (ext_utils 1.2)

A top-1 score only moves once the forgotten class beats all 99 others, so partial recovery is invisible (the
first pilot read exactly 0 everywhere). Every result row therefore also carries graded scores:
`top5_cc` (top-5 recall minus the top-5 false-positive rate; **primary**), `rank_score` (the class's rank on its
own images, 0..1) and `forget_auc` (how well its output separates its images from the rest). Notebook 06 tests
each hypothesis on the difference to the retrained model and reports the other scores as robustness checks.
Fine-tuning rows whose accuracy on the kept classes halves are marked `collapsed` and excluded. Extinction
results written by 1.1 are ignored automatically.

## Follow-ups (ext_utils 1.3, notebooks 07-08)

- **Why:** forgetting individual examples is known to relapse under retain-only fine-tuning ("From Dormant to
  Deleted"), while our class-level arm shows no relapse. Notebook 07 runs the same relapse tests on
  example-level forgetting (a random 10 % of the training set; score `mem_gap` = accuracy on the forgotten images
  minus test accuracy), so the boundary becomes a direct result (test B1 in notebook 08).
- **Head-bias check** (notebook 07, stage H): how much of class-level forgetting sits in the forgotten class's
  output bias (cf. Zheng et al. 2026), and whether a single constant shift brings the class back.
- **Savings speed** (notebook 08, E2): steps to 50 % forget accuracy, which resolves why SCRUB relearns fast early
  but has a low relearning AUC.
- 1.3 leaves every class-level result bit-identical to 1.2 (verified), so notebooks 01-06 need no re-run.

## Atypical example-level arm (ext_utils 1.4, notebook 09)

- **Why:** with random forget sets no method reached retraining-level forgetting (run #9): random images are typical,
  so the retrained model already gets 74 % of them right and the memorisation gap is thin. Atypical images (lowest
  C-score, Jiang et al. 2021) are learned only by memorisation, as in "From Dormant to Deleted".
- `xl_design="atypical"`: 500 images per forget set from the 2,500 least typical; primary score = forget accuracy;
  job prefix `xa`, own calibration file and denser grid, so 1.3 results are untouched. C-scores come from
  `cifar100-cscores-orig-order.npz`, uploaded to `ext-code` with `ext_utils.py` (downloaded only if missing) and
  checked against the CIFAR-100 labels.
- Head-bias v2: the one-constant shift is set on even-numbered test images and measured on odd-numbered ones
  (`shift_recall_fpr1/5`), so its false-positive rate is really 1 % / 5 %.
- 1.4 leaves class-level and random-design results unchanged; notebook 08 picks the design with `XL_DESIGN`.

## Resuming after a stop

Every job is saved the moment it finishes. The runner also stops starting new jobs after `time_budget_h`
(11 h), so the notebook ends cleanly before Kaggle's 12 h limit. To continue: attach the notebook's own latest
output as an input and run it again. Finished jobs are skipped, and their results and checkpoints are copied
into the new output. Downstream notebooks therefore only ever need the **latest** version.

## Splitting work with your co-author

- Notebook 02 has `SHARD = None | (0, 2) | (1, 2)`; the job builders for notebooks 03–05 accept the same
  `shard=` argument. Each person runs one half on their own account.
- Share outputs via the notebook's or dataset's **Share** settings (add the co-author's username).
- Attach both halves downstream; files are found wherever they are.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ext_utils.py not found` | ext-code not attached | Add Input → Your Datasets → ext-code |
| `GPU: none found` | Accelerator off | Settings → Accelerator → GPU T4 x2 |
| `CIFAR-100 ... could not be downloaded` | Internet off | Settings → Internet → On, or attach a CIFAR-100 dataset |
| `Shared settings differ from notebook 01` | A shared parameter was changed | Use the same values as notebook 01 (the message lists them) |
| `Checkpoint '...' not found` | An upstream output isn't attached | Attach the notebook output that produced it |
| `N job(s) failed` | An error inside a job | Read the traceback printed above, or `logs/*__errors.jsonl`; fix it; re-run (only failed/unfinished jobs run) |
| `time budget reached` | Session nearly at 12 h | Save the version, attach its output, run again |
| CIFAR-100-C not available | Download blocked | Attach a Kaggle CIFAR-100-C dataset; otherwise only the synthetic contexts run |
| `mctx_classes [...] must also be in forget_classes` | You shortened `forget_classes` | Shorten `aba_classes` and `mctx_classes` the same way |

## Changing the code

Edit `ext_utils.py` locally → upload it as a **new version** of the ext-code dataset → in the notebook,
update the input and restart → run notebook 00's self-test. Each output contains `ext_utils_used.py`, the
exact code that produced it.

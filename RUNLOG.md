# RUNLOG: Machine Unlearning is Extinction, Not Forgetting

One entry per Kaggle run (each *Save Version*). Newest entries go at the bottom; the summary table stays on top.

## Summary

| # | Reported | Notebook (version) | ext_utils | Status | Session time | Key result | Follow-up |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-29 | 01_train_originals (v1) | 1.0 | ✅ success | 75.4 min (56 min of it = CIFAR-100 download) | test acc 0.7468 / 0.7476 / 0.7490 | fixed the download bottleneck in v1.1; attach CIFAR-100 from notebook 02 on |
| 2 | 2026-09-29 | 02_retrain_references (v1) | 1.1 | ✅ success | 186.3 min (44.6 min of it = CIFAR-100 download) | 31/31 retrained models, 0 failed; epoch-32 acc 0.712 ± 0.004 (originals 0.717) | output now contains `data_cache/`, so notebook 03 needs no download |
| 3 | 2026-09-29 | 03_unlearning (v1) | 1.1 | ✅ success | 293.5 min, no download | 305 unlearned models, 0 failed; matched 141/150 main (FT 70%), 75/75 ABA, 75/75 multi-context | FT under-forgets on tiger/chair (+3 others): keep, analyse matched-only + all-models; attach CIFAR-100-C to notebook 04 |
| 4 | 2026-09-30 | 04_extinction_tests, pilot (v1) | 1.1 | ✅ success | 18.6 min | 7 `lamp` models tested; best spontaneous-recovery proxy averages +0.002; primary proxy fixed: `retain_ft`, level 1 | **no-go as designed**: every recovery ≤ 0.03; cause is the top-1 metric's floor; add graded metrics (v1.2) and re-run the pilot before the main run |
| 5 | 2026-09-30 | 04_extinction_tests, pilot (v1.2) | 1.2 | ✅ success | 11.8 min | no relapse under time/context/cues on any score; savings faster for FT and SCRUB; forget-class output still selective (AUROC 0.95) for FT/RL/SalUn, inverted for NegGrad+ (0.025), retrain 0.475 | **GO** for the main run (≈ 7.5 h, one session); NB06: BN primary = update, new H5b |
| 6 | 2026-09-30 | 05_mechanism (v1, co-author's account) | 1.2 | ✅ success | 24.5 min | 285/285 models; forgetting lives in layer4 + head only; RL/SalUn keep *more* linearly decodable forget-class information than retraining (+5 pts, p < 0.001) with an intact head; SCRUB/NegGrad+ erase more than retraining | save as `ext-05-mechanism`; NB06 after NB04 main |
| 7 | 2026-10-01 | 04_extinction_tests, main (v1.2, co-author's account) | 1.2 | ✅ success | 411.8 min | 360/360 models, 0 failed; H5b output selectivity significant for all 5 methods; relapse (H1-H3) null on the primary score; savings: FT faster, NegGrad+/SCRUB resist; ABA renewal for SCRUB suggestive (Holm p = 0.10) | save as `ext-04-extinction`; run NB06 (CPU) with 01-05 attached |
| 8 | 2026-10-01 | 06_analysis (v1, user's account) | 1.2 | ✅ success | 44.7 s (CPU) | 115 tests; identical to the local preview for all 110 shared tests; collateral damage significant for NegGrad+ (+4.5 pts) and SCRUB (+1.2) | experiments complete; next: paper figures and writing |
| 9 | 2026-10-01 | 07_example_level_and_bias (v1) | 1.3 | ⚠️ ran, design issue | 306.6 min | head bias 360/360 OK; 10 example-level retrains OK (test ≈ 0.74); **0 of 45 example-level unlearned models matched**: NegGrad+ and SCRUB did not forget (0.995-0.999 vs retrain 0.74), FT 0.93-0.96, RL 0.80-0.90, SalUn 0.88-0.94 | grids too weak for example level; preview with all models, then decide on a stronger-grid re-run |
| 10 | 2026-10-01 | 08_followup_analysis (v1) | 1.3 | ✅ ran | 29.9 s (CPU) | example-level tests empty (0 matched models); **E1: restoring the original bias brings back 0%, but one constant shift brings back 91-96% (FT, RL, SalUn)**; E2 as before | NB08 patched (`XL_MATCHED_ONLY = False`); re-run it for the partial-unlearning preview |
| 11 | 2026-10-01 | 08_followup_analysis (v1, preview: all example-level models) | 1.3 | ✅ ran | < 1 min (CPU) | **boundary visible:** RL/SalUn relapse fraction 0.70/0.71 at example level vs 0.00 at class level (p < 10⁻⁷); XH1 retain-FT recovery minus retrained RL +0.11, SalUn +0.06 (Holm p = 0.04); partial unlearners only (0/45 matched) | built ext_utils 1.4 + notebook 09 (atypical forget sets, D2D-style) to get matched example-level models |
| 12 | 2026-10-02 | 09_example_level_atypical (v1) | 1.4 | ⚠️ ran, 0 matched | 317.8 min | **0 of 45 atypical models matched**, but **retain-only fine-tuning brings forgotten atypical images back**: forget accuracy +0.43 SalUn, +0.29 RL, +0.13 NegGrad+ (NegGrad+ had forgotten as much as retraining), retrained 0.00; head bias v2: one held-out constant shift at a true 1% FPR restores RL/SalUn to the original's recall (0.87) | save as `ext-09-atypical`; NB08 patched (all models, partial-unlearning checks) |
| 13 | 2026-10-02 | 08_followup_analysis (**old copy**: run #11 version, on `ext-09-atypical`) | 1.4 | ⚠️ wrong notebook version | 40.9 s (CPU) | scored with `mem_gap` instead of the pre-registered `forget_acc`; no partial-unlearning checks; E1 old shift. Still a useful robustness check: XH1 excess (`mem_gap`, which subtracts any test-accuracy repair) SalUn +0.43, RL +0.29, NegGrad+ +0.05, SCRUB +0.04 (Holm p = 0.039 each); B1 SalUn 0.54 / RL 0.34 vs class 0.00 | re-import the current `08_followup_analysis.ipynb` (31 cells) and re-run |
| 14 | 2026-10-02 | 08_followup_analysis (current, 31 cells, on `ext-09-atypical`) | 1.4 | ✅ success | 32.9 s (CPU) | **example-level arm complete (partial unlearning):** XH1 forget-acc relapse vs retrained NegGrad+ +0.13, RL +0.29, SalUn +0.43, SCRUB +0.05 (Holm p = 0.039); B1 example vs class significant for all 5 (RL 0.35 vs 0, SalUn 0.54 vs 0); relapse only under fine-tuning (no BN-recalibration, noise, pruning or quantisation effect); NegGrad+ relapse scales with its utility damage (ρ = −0.90) | optional NB10 (matched-model search, ≈ 2 GPU-h); otherwise writing |
| 15 | 2026-10-02 | 10_matched_search (v1) | 1.5 | ✅ success | 156.8 min | **RL-long matched 9/9** (forget 0.047 vs 0.034, test 0.751); SalUn-long 4/9; NegGrad+-long 0/9 (test 0.705). **Relapse shrinks with more complete forgetting:** retain-FT recovery RL 0.29 → 0.04 (matched), SalUn 0.43 → 0.11, NegGrad+ 0.13 → 0.02 (ends below the retrained level) | NB08 patched (merge rule, matched-only family, per-variant table); re-run NB08 with ext-09 + ext-10 |
| 16 | 2026-10-02 | 08_followup_analysis (33 cells, on ext-09 + ext-10) | 1.5 | ✅ success | 42.3 s (CPU) | **final example-level numbers.** Matched only: RL (9/9) relapse +0.041 [0.032, 0.049] vs retrained, all 9 positive, Holm p = 0.004; B1 RL 0.043 vs class 0.000 (p < 10⁻⁴); SalUn (4 matched) +0.110 [0.094, 0.131] (n < 5, descriptive). Dose-response across schedules confirmed | **experiments complete**; next: paper (revised claim) |

## GPU quota used

| # | Notebook | Session hours | Of which wasted | Running total |
|---|---|---|---|---|
| 1 | 01 | 1.26 h | 0.93 h (download) | 1.26 h |
| 2 | 02 | 3.11 h | 0.74 h (download) | 4.37 h |
| 3 | 03 | 4.89 h | ≈ 0.2 h (18 FT retries that could not succeed, 2 GPUs in parallel) | 9.26 h |
| 4 | 04 pilot | 0.31 h | 0 | 9.57 h |
| 5 | 04 pilot (1.2) | 0.20 h | 0 | 9.77 h |
| 6 | 05 (co-author's account) | 0.41 h | 0 | 10.18 h (both accounts) |
| 7 | 04 main (co-author's account) | 6.86 h | 0.19 h (pilot re-ran; its output was not attached) | 17.04 h (both accounts) |
| 8 | 06 (CPU) | 0 h | 0 | 17.04 h |
| 9 | 07 | 5.11 h | 0 | 22.15 h |
| 10 | 08 (CPU) | 0 h | 0 | 22.15 h |
| 11 | 08 preview (CPU) | 0 h | 0 | 22.15 h |
| 12 | 09 | 5.30 h | 0 | 27.45 h |
| 13 | 08 old copy (CPU) | 0 h | 0 | 27.45 h |
| 14 | 08 (CPU) | 0 h | 0 | 27.45 h |
| 15 | 10 | 2.61 h | 0 | 30.06 h |
| 16 | 08 (CPU) | 0 h | 0 | 30.06 h |

## Environment reference

| Item | Value (first seen in run #1) |
|---|---|
| Kaggle image | "Latest Container Image" (not pinned) |
| Python / PyTorch | 3.12.13 / 2.10.0+cu128 |
| GPUs | 2 × Tesla T4, 14.6 GB, compute capability 7.5 |
| CPU cores / disk | 4 cores; `/kaggle/working` 19.5 GB; `/tmp` very large |
| Input path layout | `/kaggle/input/datasets/<owner>/<slug>/…`, e.g. `/kaggle/input/datasets/mahdihasanqurishi/ext-code/ext_utils.py`. The code searches recursively, so this layout works. |
| Dataset owners (run #2) | `ext-code` → account **mahdihasanqurishi**; `ext-01-originals` → account **sagorchandrapaul**. Datasets from both accounts attach fine (they must be shared with the account that runs the notebook). |

## ext_utils version history

| Version | Date | Change | Triggered by |
|---|---|---|---|
| 1.0 | 2026-09-28/29 | Initial release, with BatchNorm controls, GroupNorm arm and multi-context unlearning | – |
| 1.1 | 2026-09-29 | CIFAR-100 is fetched **once in the parent process** before workers start (no parallel downloads to the same file). A downloaded copy is kept in the output (`data_cache/cifar-100-python`), so later notebooks find it. `env_report()` prints whether CIFAR-100 is attached. | Run #1 |
| 1.2 | 2026-09-30 | (1) Graded forget-class scores in `metrics()`: `top5_cc` (primary), `forget_rank`/`rank_score`, `forget_auc`; top-1 kept. (2) BatchNorm control redesigned: fine-tune once in train mode, score each checkpoint with updated and with original running statistics (`bn_stats_swapped`, `eval_modes`); removes the collapsing eval-mode training and halves fine-tuning compute. (3) Gradient clipping (`clip_grad`) in all recovery fine-tuning; `collapsed` flag (`collapse_frac` = 0.5). (4) Every result row carries `v`; extinction rows older than 1.2 are ignored (`MIN_RESULT_VERSION`). NB04 read-out shows a sanity table, graded tables, savings and a go/no-go summary, and saves `primary_metric`; `SHARD` option for the main run. NB06 tests H1-H3 as the difference to the retrained model on the primary score, with the other scores as robustness checks. | Run #4 |
| 1.3 | 2026-10-01 | Follow-ups, class-level code bit-identical to 1.2 (verified on RL, SalUn, NegGrad+ unlearning and the spontaneous-recovery test). (1) Example-level forgetting arm: `XL` target (random 10% of the training set per seed × draw), `xl_metrics` (`mem_gap` = forget minus test accuracy, MIA-AUROC leak), example-level retrain/calibrate/unlearn/relapse tasks; `eval_modes` and `test_spontaneous` take an optional metrics function; RL/SalUn random labels exclude each example's own label. (2) Head-bias check `headbias_row`: bias z-score, restore original bias / output row, one-constant shift at 1% FPR. Notebooks 07 (GPU) and 08 (CPU analysis). Self-test 27/27. | literature re-check after run #8 |
| 1.4 | 2026-10-01 | Atypical example-level design (class-level code and 1.3 random design unchanged). (1) `xl_design` = `"random"` (1.3 behaviour) or `"atypical"`: each forget set is `xl_n` = 500 images drawn from the `xl_pool_frac` = 5% of training images with the lowest C-score (Jiang et al. 2021; `cscores()` downloads/verifies the published scores, labels checked against CIFAR-100). (2) Atypical rows carry `design`, use job prefix `xa`, their own calibration file and a denser grid (`DEFAULT_XA_CALIB_GRID`: FT up to 20 epochs / lr 0.05, NegGrad+ α 0.8-0.99 (smaller α forgets harder), RL/SalUn up to lr 0.02 × 5 epochs, SCRUB lr up to 0.01). (3) Head-bias v2: the one-constant shift is set on even-numbered test images at 1% and 5% FPR and measured on odd-numbered ones (`shift_recall_fpr1/5`, `shift_fpr_fpr1/5`); head-bias rows older than 1.4 are ignored. Notebook 09 (GPU), notebook 08 design-aware (`XL_DESIGN`, `XL_SCORE`). Self-test 28/28. | Runs #9-#11 |
| 1.5 | 2026-10-02 | (1) Method variants: `base_method("RL-long") = "RL"`; a variant runs its base method's code with its own grid (`xl_calib_grid`), job keys and checkpoints, so it never collides with earlier models. (2) Unmatched example-level candidates (calibration ranking and best-of-tries) are ranked by `xl_violation` = forget distance beyond `xl_match_gap` + test shortfall beyond `match_retain_gap` (1.4: forget distance only, which put NegGrad+'s most damaging setting first). Class-level code unchanged. Notebook 10 (matched-model search). Not smoke-tested at the user's request; helpers checked on run #12 calibration numbers. | Run #14 |

---

## Run #1: 01_train_originals (v1)

**Reported:** 2026-09-29 · **Status:** ✅ success · **Session:** 4524.7 s (75.4 min) · **Output:** 68.24 MB

**Settings:** GPU T4 x2, Internet on, Latest Container Image, default config (`norm="bn"`, 3 seeds, 40 epochs,
batch 256, lr 0.1, fp16 checkpoints). ext_utils 1.0.

**Inputs:** `ext-code` only.

### Timeline

| Time into session | Event |
|---|---|
| 0:14 | Setup OK: both T4s visible, internet on, `experiment_config.json` saved |
| 0:15 → 56:30 | **Both workers downloaded CIFAR-100 (169 MB) at the same time**, each at ~50 kB/s from cs.toronto.edu |
| 57:00 | Training started: seed 0 on GPU 0, seed 1 on GPU 1 |
| 57:00 → 66:30 | Seeds 0 and 1 finished (~9.2 min per model, 14–16 s per epoch) |
| 66:30 → 75:00 | Seed 2 on GPU 0 (GPU 1 idle, since 3 jobs on 2 GPUs) |
| 75:15 | 3/3 done, results table printed |

### Results

| Seed | Test accuracy | Accuracy at epochs 8 / 16 / 24 / 32 / 40 |
|---|---|---|
| 0 | **0.7468** | 0.508 / 0.614 / 0.637 / 0.717 / 0.747 |
| 1 | **0.7476** | 0.510 / 0.614 / 0.651 / 0.720 / 0.748 |
| 2 | **0.7490** | 0.503 / 0.579 / 0.656 / 0.715 / 0.749 |
| mean ± sd | **0.7478 ± 0.0011** | |

- **Expected 0.74–0.77:** met, at the lower end, as expected for a 40-epoch schedule. The three seeds agree
  within 0.2 points, so training is stable and reproducible.
- The curve shape is normal for OneCycle: the big gains come in the last quarter, when the learning rate
  anneals. Training loss was 0.13 at epoch 32 and lower at the end, so the models fit the training set closely
  (normal for ResNet-18 on CIFAR-100).
- **Per-forget-class accuracy:** not visible in the log, because it's shown as a table in the notebook output.
  *Pending:* copy it from the notebook's output page (last table) into this entry.

### What worked
- Both GPUs used in parallel, and the worker logs streamed correctly.
- fp16 checkpoints: 68 MB for 3 models (≈22 MB each), as designed.
- `experiment_config.json` written for the downstream consistency checks.
- Kaggle's new input path layout (`/kaggle/input/datasets/<owner>/<slug>/`) was found by the recursive search.

### Issues
1. **CIFAR-100 download took 56 of 75 minutes (74 % of the session).** cs.toronto.edu served about 50 kB/s.
   Also, both worker processes downloaded the same file to the same path at the same time, which risks a
   corrupted file (torchvision's checksum would catch it, but that would waste another hour).
   *Impact:* ~0.93 GPU-session hours wasted. Without a fix, notebooks 02–05 would each lose another hour.
2. *(Harmless)* debugger "frozen modules" warnings and mistune/nbconvert `SyntaxWarning`s. These come from
   Kaggle's image, not our code.
3. *(Note)* The environment is "Latest Container Image", i.e. not pinned. If Kaggle updates PyTorch between
   notebooks, the results stay comparable, but it's cleaner to keep one image for the whole project.

### Actions
- [x] ext_utils **1.1**: parent-side single download + output cache + CIFAR-100 status line (see version history).
- [ ] Upload `ext_utils.py` 1.1 as a **new version** of `ext-code`.
- [ ] For notebooks 02–06, attach **[CIFAR-100 Python (fedesoriano/cifar100)](https://www.kaggle.com/datasets/fedesoriano/cifar100)**.
      Check that the setup cell prints `CIFAR-100: found at …`.
- [ ] Settings → Environment: pin notebooks 02–06 to the same image as run #1 if Kaggle offers it; otherwise
      record the PyTorch version of each run here.
- [ ] Save notebook 01's output as dataset `ext-01-originals` (or attach the notebook output directly).

### Projection for notebook 02 (with CIFAR-100 attached)
31 models × ~9.5 min ÷ 2 GPUs ≈ 16 rounds ≈ **2.5–2.8 h** in one session (~2.7 GPU-session hours).

---

## Run #2: 02_retrain_references (v1)

**Reported:** 2026-09-29 · **Status:** ✅ success · **Session:** 11179.3 s (186.3 min) · **Output:** 883.86 MB

**Settings:** GPU T4 x2, Internet on, Latest Container Image (still Python 3.12.13 / torch 2.10.0+cu128, same as
run #1), `SHARD=None` (all 31 jobs in one session), default config. ext_utils **1.1**.

**Inputs:** `ext-code` (1.1), `ext-01-originals`. **Not attached: CIFAR-100** (`fedesoriano/cifar100`).

### Timeline

| Time into session | Event |
|---|---|
| 0:24 | Setup OK: ext_utils 1.1, both T4s, internet on. `CIFAR-100: NOT attached` warning printed (new in 1.1) |
| 0:24 | `[config] shared settings match … ext-01-originals/experiment_config.json`: same data split, model and training settings as notebook 01 |
| 0:24 | 31 jobs found, 0 already done |
| 0:33 → 45:02 | **One** CIFAR-100 download (the 1.1 fix worked: parent process only, ~61 kB/s) → copy kept in `/kaggle/working/data_cache/` |
| 46:01 | Training started: w0 (GPU 0) 16 jobs, w1 (GPU 1) 15 jobs |
| 46:01 → 174:30 | Two models at a time, ~8.7 min each (13.6–14.7 s per epoch) |
| 174:30 | w1 finished: 15 ok, 0 failed |
| 186:03 | w0 finished: 16 ok, 0 failed (last job `castle s2` ran alone: 31 is odd) → `now done: 31/31` |

### Results

31 retrained models = pilot class `lamp` (seed 0) + 10 forget classes × 3 seeds. Each one was trained from scratch
**without ever seeing its forget class**: this is the gold standard that every unlearned model is compared with.

| Check | Value | Verdict |
|---|---|---|
| Jobs finished | 31/31, 0 failed | ✅ |
| Test acc at epoch 32 (all 31 models) | 0.712 ± 0.004 (0.706–0.725) | ✅ originals at the same epoch: 0.717 |
| Test acc at epoch 40 | only visible for `bus s0`: **0.7429** | ✅ expected ≈ 0.748 − 0.0075 = 0.740 |
| Seed-to-seed spread | ±0.4 points | ✅ as stable as run #1 |

- **Why the retrained models score ~0.5–0.7 points lower than the originals:** the progress line's `test_acc` is over
  all 100 test classes. A retrained model has never seen its forget class, so it gets ~0 % on those 100 of 10,000
  test images, which costs ≈ 0.75 points. This is exactly what "forgotten" should look like.
- **Epoch-40 accuracies for the other 30 models are not in the log:** the progress monitor prints once a minute,
  so it mostly skipped the last epoch. They're in the notebook's final results table.
  *Pending:* copy that table (forget_test_acc should be ≈ 0.00, retain_test_acc ≈ 0.75, plus neighbour/unrelated
  accuracy) into this entry.

### What worked
- v1.1 download fix: one download instead of two, and the copy in `data_cache/` lets notebooks 03–06 skip it.
- The cross-account config check: `ext-01-originals` (account sagorchandrapaul) matched this run's settings.
- The job split across GPUs (16/15), and the fp16 checkpoints: 31 × ≈ 22 MB ≈ 690 MB of the 884 MB output
  (the rest is the ≈ 161 MB CIFAR-100 cache plus results and logs).

### Issues
1. **CIFAR-100 was still downloaded (44.6 min, 24 % of the session)**, because `fedesoriano/cifar100` wasn't
   attached. The 1.1 warning at 0:24 flagged it, but a *Save & Run All* commit can't be stopped and fixed
   mid-run. *Impact:* ≈ 0.74 GPU-session hours wasted.
2. *(Minor)* The last job ran alone for ~11.5 min (31 jobs on 2 GPUs). That can't be avoided without dropping
   the pilot class.
3. *(Harmless)* Same debugger / mistune / nbconvert warnings as run #1.

### Actions
- [ ] Save this output as dataset **`ext-02-retrain`** (or attach the notebook output directly).
- [ ] Notebook 03: attach `ext-code`, `ext-01-originals`, **`ext-02-retrain`**. It contains `data_cache/`, so
      `fedesoriano/cifar100` becomes optional. **Before committing, run the setup cell interactively and check that
      it prints `CIFAR-100: found at …`.** If it says `NOT attached`, stop and fix the inputs first.
- [ ] Share `ext-02-retrain` with the account that runs notebook 03 (the datasets currently live on two accounts).
- [ ] Copy the notebook 01 per-class table (run #1) and this run's results table into the log.

### Projection for notebook 03
Calibration + pilot + 150 unlearned + 75 ABA + 75 multi-context models. Unlearning runs a few epochs, not 40, so
each job is short, but there are ~300 of them plus 5 calibration grids: **≈ 6–9 h**, possibly 2 sessions
(the runner stops at 11 h; re-run with its own output attached to resume).

---

## Run #3: 03_unlearning (v1)

**Reported:** 2026-09-29 · **Status:** ✅ success · **Session:** 17,609.6 s (293.5 min) · **Output:** 6.86 GB

**Settings:** GPU T4 x2, Internet on, Latest Container Image (Python 3.12.13 / torch 2.10.0+cu128, unchanged),
default config, stages A-E in one session. ext_utils 1.1.

**Inputs:** `ext-code` (mahdihasanqurishi), `ext-01-originals` and `ext-02-retrain` (sagorchandrapaul).
`CIFAR-100: found at .../ext-02-retrain/data_cache/cifar-100-python`: **no download** (the 1.1 cache works).
`[config] shared settings match ... ext-01-originals/experiment_config.json`.

### Timeline

| Stage | Time into session | Jobs | Duration |
|---|---|---|---|
| A: calibration on `lamp`, seed 0 | 0:14 → 27:14 | 33 | 27.0 min |
| B: pilot unlearning (`lamp`) | 27:15 → 33:15 | 5 | 6.0 min |
| C: main unlearning, 10 classes × 3 seeds × 5 methods | 33:15 → 2:42:18 | 150 | 129.0 min |
| D: ABA models (unlearned in gray), 5 classes | 2:42:18 → 3:51:20 | 75 | 69.0 min |
| E: multi-context models (clean/gray/blur), 5 classes | 3:51:20 → 4:53:22 | 75 | 62.0 min |

All 338 jobs ok, 0 failed. My estimate was 6-9 h; it took 4.9 h, so one session was enough.

### Calibration (on `lamp` only; table from the notebook output)

| Method | Settings matched | Best setting | Note |
|---|---|---|---|
| FT | **0 / 6** | 10 ep, lr 0.05: forget **0.03**, retain 0.752 | never reached forget ≤ 0.01 |
| NegGrad+ | 1 / 9 | α 0.99, lr 0.005: forget 0.00, retain 0.738 | every other setting wrecks retain accuracy (0.04-0.72) |
| RL | 6 / 6 | 5 ep, lr 0.02 (retain 0.748) | easy |
| SCRUB | 3 / 6 | 6 ep, lr 0.005, msteps 3 | |
| SalUn | 6 / 6 | 5 ep, lr 0.01 (retain 0.745) | easy |

Retrained reference retain accuracy on `lamp`: 0.7497.

### Results (tables from the notebook output)

**Main models (10 classes × 3 seeds per method):**

| Method | n | matched | forget acc | retain acc | retrained retain | G_related | G_unrelated |
|---|---:|---:|---:|---:|---:|---:|---:|
| FT | 30 | **0.70** | 0.011 | 0.7499 | 0.7507 | 0.0002 | 0.0008 |
| NegGrad+ | 30 | 1.00 | 0.000 | 0.7350 | 0.7507 | **0.0587** | 0.0139 |
| RL | 30 | 1.00 | 0.000 | 0.7471 | 0.7507 | 0.0030 | 0.0036 |
| SCRUB | 30 | 1.00 | 0.000 | 0.7488 | 0.7507 | **0.0131** | 0.0014 |
| SalUn | 30 | 1.00 | 0.000 | 0.7461 | 0.7507 | 0.0033 | 0.0047 |

**ABA models (unlearned in gray = context B; 5 classes × 3 seeds):** all 75 matched.

| Method | forget acc in gray (B) | forget acc in colour (A) |
|---|---:|---:|
| FT, NegGrad+, RL, SalUn | 0.0000 | 0.0000 |
| SCRUB | 0.0013 | **0.0367** |

**Multi-context models (clean/gray/blur; checked on clean images):** all 75 matched.

| Method | forget acc | retain acc | retrained retain | retain gap |
|---|---:|---:|---:|---:|
| FT | 0.0007 | 0.7501 | 0.7506 | 0.1 pt |
| NegGrad+ | 0.0000 | 0.7278 | 0.7506 | 2.3 pt |
| RL | 0.0000 | 0.7390 | 0.7506 | 1.2 pt |
| SCRUB | 0.0000 | 0.7277 | 0.7506 | 2.3 pt |
| SalUn | 0.0000 | 0.7324 | 0.7506 | 1.8 pt |

**Reading them:**
- **Unmatched FT (9 of 30):** tiger s0/s1/s2, chair s0/s1/s2, butterfly s0/s2, castle s1 (identified from the
  log). Their kept model is always try 1, with forget 0.02-0.07 and normal retain accuracy (0.746-0.754): they
  fail only the forget ≤ 0.01 threshold. `mean_tries` = 1.0 counts the try that was *kept*, not how many ran.
- **SCRUB-ABA:** 4 jobs (rose s0/s2, tiger s1/s2) failed tries 1 and 2 (forget 0.02 in gray) and matched on
  try 3, the strongest SCRUB setting (8 epochs, 5 max-steps). My duration-based guess that they were
  unmatched was wrong; the table shows 1.0.
- **Semantic collateral (the co-author's effect):** NegGrad+ lowers accuracy on the forget class's siblings by
  5.9 points relative to the retrained model, against 1.4 on unrelated classes; SCRUB 1.3 vs 0.1. RL, SalUn
  and FT show none. Descriptive; notebook 06 tests it (control family).
- **First look at ABA renewal: SCRUB only.** Unlearned in gray, SCRUB models recognise 3.7% of forget images
  back in colour against 0.1% in gray. The other four methods show 0 in both contexts, so their suppression
  transfers across the context change. These are raw means before any recovery test, with no chance
  correction; notebook 04's context test does it properly.
- **Multi-context unlearning costs retain accuracy** (1.2-2.3 points below the retrained model, vs 0.2-1.6
  for standard unlearning; e.g. SCRUB 0.7277 vs 0.7488). All are within the 3-point rule, but for H6 this is a
  confound: less relapse in multi-context models could come from more general damage. Notebook 06 should
  report retain accuracy alongside H6 (or use it as a covariate).
- Multi-context FT matched on all 15 models, including tiger, where standard FT failed on all three seeds.

### What worked
- No download: CIFAR-100 came from the `data_cache` in `ext-02-retrain`.
- Both GPUs used throughout; 0 failed jobs across 5 stages.
- ABA matching compares like with like: both the unlearned and the retrained model are evaluated in gray
  (retain ≈ 0.65 for both), so the lower accuracy in gray is not a bug.

### Issues
1. **FT under-forgets on some classes, and its retries cannot help.** FT's candidates are ranked by
   forget accuracy on `lamp`, so try 1 is already the strongest setting in the grid (10 ep, lr 0.05); tries
   2 and 3 forget *less* (visible: forget 0.05-0.38). About 27 GPU-minutes (≈ 0.2 session hours on 2 GPUs)
   went on retries that could not succeed. (SCRUB's retries are different: its try 3 is stronger, and it
   rescued 4 ABA models.) FT's forgetting is also borderline: the same setting gave 0.03
   on `lamp` in calibration but matched on `lamp` in the pilot.
   *Decision:* keep the models and the code as they are. FT leaving residual forget accuracy is a known
   property of retain-only fine-tuning. Changing the grid after seeing the main classes would be a post-hoc
   protocol change, and would make the GroupNorm arm run different code. Notebook 04 tests every model
   regardless of the flag. Notebook 06 uses `MATCHED_ONLY = True` for the primary tests (FT n = 21, with no
   tiger or chair) and should report all-models as a sensitivity analysis.
   *If* FT is ever strengthened, it must happen **before** the notebook 04 main run; otherwise notebook 04's
   results for those 9 models go stale.
2. *(Harmless)* Same debugger / mistune / nbconvert warnings as before.

### Actions
- [ ] Save the output as dataset **`ext-03-unlearning`** (6.86 GB), or attach the notebook output directly.
- [x] Summary tables supplied and recorded above.
- [ ] Notebook 04 inputs: `ext-code`, `ext-01-originals`, `ext-02-retrain`, `ext-03-unlearning`, and
      **`rojanregmi1/cifar100-c`**. Checked on its public page: `labels.npy` (50.13 kB) plus all six corruption
      files the config uses, 153.6 MB each, the standard layout. Without it, notebook 04 streams a 2.9 GB tar
      from Zenodo before any work starts.
- [ ] Run notebook 04's **pilot** first (`RUN_PILOT=True, RUN_MAIN=False`) and check that setup prints both
      `CIFAR-100: found at ...` and `[CIFAR-100-C] using ...`.
- [ ] Share `ext-01`, `ext-02` and `ext-03` with whichever account runs notebook 04.

### Projection for notebook 04
Main run: 150 standard + 75 multi-context + 30 retrained models get all four tests (~255 jobs); 75 ABA + 30
original models get the context test only (~105 short jobs). ≈ 8-10 h on 2 GPUs: probably one 11 h session,
possibly two.

---

## Run #4: 04_extinction_tests, pilot (v1)

**Reported:** 2026-09-30 · **Status:** ✅ success · **Session:** 1,114.7 s (18.6 min) · **Output:** 443.62 kB

**Settings:** GPU T4 x2, Internet on, Latest Container Image, `RUN_PILOT=True, RUN_MAIN=False` (as planned: the
short session is the pilot, not a failure). ext_utils 1.1.

**Inputs:** `ext-code`, `ext-01-originals`, `ext-02-retrain`, `ext-03-unlearning`, `rojanregmi1/cifar100-c`.
Setup printed `CIFAR-100: found at .../ext-02-retrain/data_cache/...` and
`[CIFAR-100-C] using /kaggle/input/datasets/rojanregmi1/cifar100-c/CIFAR-100-C`: no downloads.
`[config] shared settings match .../ext-02-retrain/experiment_config.json`.

### Timeline and cost per job

| Worker | Jobs (in order) | Duration each |
|---|---|---|
| w0 | original `lamp` (contexts only), FT, RL, SalUn | ≈ 1 min, then 5.7 / 5.2 / 5.2 min |
| w1 | retrained `lamp`, NegGrad+, SCRUB | ≈ 6 min, then 5.6 / 5.6 min |

7/7 ok, 0 failed. A model with all four tests costs **≈ 5.5 min**; a contexts-only model ≈ 1 min.

### Results
- **Spontaneous recovery is essentially zero on the pilot.** The proxy with the largest mean recovery over the
  five methods (frozen/eval rows, never `bn_recal`) was `retain_ft` after 1 epoch, at **+0.002** (0.2 points).
  It is now fixed as the primary H1 proxy in `analysis_choices.json`, before any main data exists, as planned;
  with a maximum of 0.002 the choice is close to arbitrary.
- **No BatchNorm-illusion warning:** `bn_recal` alone restored at most 0.05 on the unlearned `lamp` models.
- *Pending:* the three read-out tables (spontaneous recovery by proxy × BN mode × method; renewal by context;
  reinstatement by cue), which decide go/no-go: go if any method moves > 0.05 on some proxy, context or cue
  while `retrain` stays near 0. Savings is not in the read-out; it is in the output files.
- Caveat: one model per method on one class; this is a smoke signal, not evidence.

### Projection for the main run (from measured costs)
255 full-test models × 5.5 min + 105 contexts-only × 1 min, on 2 GPUs ≈ **12.6 h**. That exceeds the 11 h
budget, so the runner stops cleanly at ~11 h (≈ 235 models) and a second session finishes the rest (~1.5 h).
Alternative: split with `shard=(0, 2)` / `(1, 2)` over two accounts, ≈ 6.3 h each in parallel.

### Actions
- [x] Pilot tables supplied (see below).
- [x] Output downloaded (`Downloads/results-2`, 702 rows; `ext_utils_used.py` identical to local 1.1).
- [ ] Main run: keep `RUN_PILOT=True` (its 7 jobs are skipped in seconds and `analysis_choices.json` is rewritten
      into the new output), set `RUN_MAIN=True`, and attach this pilot output as an extra input.

### Pilot tables (supplied 2026-09-30) and decision

| Test | Unlearned models (FT / NegGrad+ / RL / SCRUB / SalUn) | Retrained | Original |
|---|---|---|---|
| Spontaneous recovery: `bn_recal`, noise, prune, quant | 0.00 in every cell | 0.00 | – |
| Spontaneous recovery: `retain_ft` (frozen and update, 1-5 epochs) | FT 0.00-0.01; all others 0.00 | 0.00 | – |
| Renewal: 4 synthetic contexts + 6 CIFAR-100-C corruptions × 2 severities | FT 0.01 in gray; all others 0.00 | 0.00 | −0.02 to −0.52 |
| Reinstatement: sibling / similar / dissimilar cues, both BN modes | FT 0.03 (dissimilar, update); all others 0.00 | 0.00 | – |

**Decision: no-go for the main run as currently designed.** Nothing reaches the 0.05 threshold. The original
model's negative renewal values (it loses 2-52 points of `lamp` recall under the corruptions) show the pipeline
and the class id are right: this is not a bug.

**Diagnosis: the metric has a floor.** `metrics()` scores the forget class by **top-1** prediction only: a
recovery counts only when the forgotten class beats all 99 others. The unlearned models push the forget class's
output far down, so partial recovery (e.g. from rank 80 to rank 3) is invisible, and chance-corrected top-1
sits at exactly 0. Two tests are also structurally weak at the output level for **class-wise** forgetting:
retain fine-tuning and reinstatement never use the forgotten label, so the forgotten class's output unit only
gets pushed down. The effect the paper looks for, suppressed output with intact knowledge, is exactly what a
top-1 metric cannot see. My design error.

**Proposed fix (ext_utils 1.2, awaiting approval):** add graded forget-class metrics to `metrics()`, computed
from the same logits at no extra cost: top-5 recall (chance-corrected), median rank of the forget class, and
the AUROC of the forget-class logit (forget images vs all others; bias-invariant, 0.5 = chance). Keep top-1 as
reported. Re-run the pilot (≈ 20 min), fix the primary graded metric on the pilot as with the H1 proxy, then
decide on the main run. Also check pilot savings, the one test that uses the forgotten label.

### Full pilot output (`Downloads/results-2`): a second, more serious problem

**Frozen-BatchNorm fine-tuning collapses the models.** Retain accuracy during the fine-tuning tests:

| Test | BN `update` (normal) | BN `frozen` (BN layers in eval mode) |
|---|---|---|
| retain fine-tuning, 1-5 epochs | 0.736-0.751 (stable) | 0.640-0.705 for every model, incl. retrained |
| reinstatement (sibling), after 3 epochs | 0.678-0.731 | RL 0.010, SCRUB 0.009, SalUn 0.015; FT 0.549, NegGrad+ 0.464, retrained 0.479 |
| savings, 20 shots, after 100 steps | 0.686-0.742 | NegGrad+ / RL / SCRUB 0.010 (from step 5-50 on); FT 0.605, retrained 0.585 |

0.010 is chance for 100 classes: the network predicts one class for everything. So every `frozen` row is
unusable, and `frozen` is what notebook 06 uses for its primary tests. Cause: training with BN layers in eval
mode removes the re-normalisation that keeps SGD at lr 0.01 stable; the unlearned models, whose weights were
already pushed around by unlearning, diverge first. My design error again. `update` rows are stable, except
NegGrad+ during savings (retain dips to 0.15-0.28 mid-run, then recovers).

**Savings (relearning speed, stable `update` rows): the one test with a positive signal.**

| Shots | Step | FT | NegGrad+ | RL | SCRUB | SalUn | retrained |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 0.46 | 0.47 | 0.00 | 0.15 | 0.00 | 0.00 (first > 0 at step 50: 0.33) |
| 20 | 10 | 0.82 | 0.45 | 0.00 | 0.60 | 0.07 | 0.00 (step 20: 0.80) |
| 100 | 5 | 0.86 | 0.00 | 0.00 | 0.07 | 0.28 | 0.00 (step 10: 0.58) |

FT, NegGrad+ and SCRUB (and SalUn with 100 shots) recognise the forgotten class after fewer relearning steps
than a model that never learned it: rapid reacquisition, i.e. hidden knowledge. RL shows none. Caveats: one
class, one seed; curves are non-monotonic; NegGrad+'s instability makes its number unreliable.

**Revised decision: no-go for the main run until ext_utils 1.2 is ready.**
1. Graded forget-class metrics (top-5 chance-corrected, median rank, AUROC of the forget-class logit), top-1 kept.
2. A stable BatchNorm control: fine-tune once in normal train mode, then evaluate each checkpoint twice, with
   the updated running statistics (`update`) and with the original ones restored (`frozen` = weight change
   only). Exactly paired, and half the fine-tuning compute (main run ≈ 8 h instead of 12.6 h).
3. Gradient clipping in all recovery fine-tuning; rows whose retain accuracy collapses are flagged and excluded.
4. The pilot read-out also shows savings and a retain-accuracy sanity check.
Then re-run the pilot (≈ 15 min).

---

## Run #5: 04_extinction_tests, pilot re-run (v1.2)

**Reported:** 2026-09-30 · **Status:** ✅ success · **Session:** 706.3 s (11.8 min) · **Output:** 586.04 kB
(`Downloads/results (24).zip`; `ext_utils_used.py` identical to local 1.2; all 695 rows carry `v` = 1.2)

**Settings:** GPU T4 x2, `RUN_PILOT=True, RUN_MAIN=False`, `SHARD=None`. Inputs as run #4; setup printed `ext_utils
1.2`, CIFAR-100 found, CIFAR-100-C found, config matched `ext-03-unlearning`. 7/7 jobs, 0 failed; a full-test job now
takes **3.0-3.5 min** (5.5 min in 1.1).

### Starting point of each `lamp` model (before any test)

| Model | forget rank (1 = top) | top-5 score | forget-output AUROC |
|---|---:|---:|---:|
| original | 1 | 0.828 | 0.968 |
| FT | 15 | 0.170 | **0.949** |
| RL | 100 | 0 | **0.957** |
| SalUn | 100 | 0 | **0.947** |
| SCRUB | 100 | 0 | 0.611 |
| NegGrad+ | 100 | 0 | **0.025** (inverted) |
| retrained | 81 | 0 | 0.475 (chance) |

**The clearest hidden-knowledge signal so far, and it needs no probe.** RL, SalUn and FT push `lamp` to the very
bottom of their answers on lamp images, yet their `lamp` output still separates lamp images from all others almost
as well as the original model (AUROC 0.95 vs 0.97). The association is intact and the response is suppressed:
extinction's defining picture. NegGrad+ recognises lamps in order to push them away (AUROC 0.025). SCRUB comes closest
to the retrained model. One class, one seed: now tested in the main run as **H5b**.

### Recovery tests
- **Spontaneous recovery, renewal, reinstatement: nothing comes back on any score.** For RL, SalUn, SCRUB and
  NegGrad+, `lamp` stays at rank 100 under every proxy, context and cue (top-5 and rank recovery exactly 0; largest
  rank gain SalUn +0.07 after 5 epochs of retain fine-tuning). FT, the only model not at the floor, is pushed
  *further* down by most manipulations (top-5 −0.02 to −0.17).
- **No BatchNorm illusion:** recomputing statistics alone gives 0 or negative recovery on every score for every model.
- **Savings (stable `update` rows):**

| Shots | Step | FT | NegGrad+ | RL | SCRUB | SalUn | retrained |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 0.44 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 |
| 5 | 20 | 0.65 | 0.17 | 0.00 | 0.38 | 0.01 | 0.00 |
| 20 | 5 | 0.56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 20 | 10 | 0.89 | 0.00 | 0.00 | 0.89 | 0.07 | 0.00 |
| 100 | 5 | 0.73 | 0.00 | 0.00 | 0.00 | 0.24 | 0.00 |

  FT and SCRUB relearn much faster than a never-trained model; SalUn slightly (100 shots); RL no faster; NegGrad+
  resists relearning.

### Sanity: 33 of 492 fine-tuning rows collapsed, all when scored with the **original** statistics
Reinstatement and savings rows in `frozen` mode reach 0.001-0.21 accuracy on kept classes; every `update` row stays
healthy (≥ 0.62, apart from brief dips to 0.37-0.67 at 100 shots, steps 10-20, which recover to 0.71-0.74).
Heavily fine-tuned weights simply do not work with the old statistics. So `frozen` cannot be the primary mode.

### Decision: **GO** for the main run
The pre-registered rule is met by savings. The main run also measures the nulls with 30 models per method (a robust
suppression is itself a finding) and H5b. Choices fixed before any main data: `primary_metric = top5_cc`,
`primary_sr_proxy = retain_ft`, level 5 (default; no proxy recovered).

**NB06 changes (before main data):** primary BN mode `update` (original-statistics rows become robustness; C1 and the
GroupNorm arm cover the BatchNorm illusion); new **H5b** output selectivity = 2 × |AUROC − 0.5|, unlearned minus
retrained; `forget_auc` removed from the recovery robustness scores (its direction is ambiguous for inverted models).
No change to notebook 04 or `ext_utils` is needed for the main run.

**Honest reading so far:** unlearning looks like extinction in what it leaves behind (intact association, savings for
some methods) but, in this one-class pilot, not in relapse. If the main run confirms that, the paper's claim becomes
"extinction without relapse". Still publishable, but the framing will need to change. ABA renewal for SCRUB (run #3:
3.7% forget accuracy back in colour) is the one relapse hint so far.

### Projection
255 full-test models × ~3.2 min + 105 contexts-only × ~0.7 min, on 2 GPUs ≈ **7.4 h**: one session.

**Local verification of 1.2 (2026-09-30):** `self_test` 22/22 PASS (incl. graded metrics and the BN-statistics swap);
full chain 01 → 04 pilot → 04 main (pilot carried over: `7 already done`) → 05 → 06 on fake CIFAR-100/-C in smoke mode,
2 workers: all OK; NB06 (with `BN_PRIMARY = update` and H5b) produced 115 tests and all 8 figures. The version guard
ignores 1.1 extinction rows and leaves other tasks untouched. Numbers from fake data are meaningless.

---

## Run #6: 05_mechanism (v1)

**Reported:** 2026-09-30 · **Status:** ✅ success · **Session:** 1,470.9 s (24.5 min) · **Output:** 3.13 MB
(`Downloads/results-3`; `ext_utils_used.py` identical to local 1.2) · **Account:** the co-author's (run in parallel with
notebook 04's main run).

**Inputs:** `ext-code` (1.2), `ext-01`, `ext-02`, `ext-03`, CIFAR-100 from `fedesoriano/cifar100`; config matched
`ext-03-unlearning`. **285 models** (150 unlearned + 75 multi-context + 30 retrained + 30 original; ABA models are not
needed here), 285 ok, 0 failed, ≈ 5 s per model on 2 GPUs. 9,045 rows: probe 285, head swap 510, weights 1,530, CKA 6,720.

### 1. Linear probe on the frozen penultimate features (forget-class accuracy)
Original 0.730; **retrained 0.716**: a network that never saw the class still lets a new linear read-out find it 72%
of the time, because CIFAR-100 features are general. So only the difference to the retrained model means anything.

| Method (matched models) | n | excess over retrained | share > 0 | one-sided Wilcoxon p |
|---|---:|---:|---:|---:|
| RL | 30 | **+0.055** | 0.73 | **0.00025** |
| SalUn | 30 | **+0.046** | 0.80 | **0.00034** |
| FT | 21 | +0.020 | 0.62 | 0.036 |
| SCRUB | 30 | −0.024 | 0.33 | 0.97 |
| NegGrad+ | 30 | −0.039 | 0.30 | 0.99 |

RL and SalUn leave *more* decodable information about the forgotten class than retraining, even more than the original
(0.77 vs 0.73). SCRUB and NegGrad+ leave *less* than retraining: they scramble the class's features. Multi-context
versions shrink the RL/SalUn excess (+0.003 / +0.019, n.s.).

### 2. Head swap (forget-class top-1 accuracy; original ≈ 0.74)

| Method | unlearned body + **original** head | **original** body + unlearned head |
|---|---:|---:|
| FT | 0.356 | 0.347 |
| SCRUB | 0.190 | 0.000 |
| RL | 0.000 | **0.320** |
| SalUn | 0.000 | **0.423** |
| NegGrad+ | 0.004 | 0.000 (and retain collapses to 0.206) |
| retrained | 0.011 | 0.000 |

### 3. Weight change relative to the original, and 4. CKA similarity
- Unlearned models change **only the top**: CKA with the original on forget images is ≥ 0.99 up to layer2 and ≥ 0.86 at
  layer3 for every method; it drops at layer4/penultimate/logits (RL/SalUn/NegGrad+ ≈ 0.55-0.68, FT/SCRUB ≈ 0.81-0.86).
- Where the change sits: SCRUB and NegGrad+ change the **head** most (fc 18% / 31%, body ≤ 5% / ≤ 10%); RL and SalUn
  barely touch the head (fc 5% / 3%) and change layer4 (≈ 6-10%); FT changes the whole body (25-41%).
  (The retrained model's 80-126% "change" only reflects an independent training run; it is not comparable.)
- Unlearned forget-class representations do **not** look like a never-trained model's: CKA with the retrained model at
  the penultimate layer is 0.21-0.34 for NegGrad+/RL/SalUn and ≈ 0.70-0.73 for FT/SCRUB.

### Reading: three different ways to "forget"
1. **RL, SalUn: re-routing.** The head keeps a working detector for the class (with the original body it recognises
   32-42%); layer4 moves the class's features away from it, but a new linear read-out finds them even better than
   before (+5 points over retraining). The information is relocated, not removed. This matches their pilot output
   selectivity (AUROC 0.95).
2. **SCRUB, NegGrad+: head-level suppression with feature scrambling.** Most change is in the head; the class is
   *less* decodable than in a retrained model. SCRUB's original-head recovery (19%) shows its body still partly carries
   the class; NegGrad+ breaks it (pilot AUROC 0.025, inverted).
3. **FT: distributed drift.** Both swaps recover about a third; changes spread through the whole body.

This is the mechanism story the paper needed: unlearning is confined to the top of the network and, for the
random-label methods, keeps the knowledge in a readable form. It supports H5 for RL and SalUn (FT marginal; Holm
will decide), and explains why H5b (output selectivity) should be strong for RL/SalUn/FT but not for SCRUB/NegGrad+.

---

## Run #7: 04_extinction_tests, main run (v1.2)

**Reported:** 2026-10-01 · **Status:** ✅ success · **Session:** 24,708.9 s (411.8 min) · **Output:** 22.43 MB
(`Downloads/results-4`; `ext_utils_used.py` identical to local 1.2) · **Account:** the co-author's.

**Settings:** `RUN_PILOT=True, RUN_MAIN=True`, `SHARD=None`. The pilot output was not attached, so the 7 pilot jobs ran
again (11 min, same choices: `top5_cc`, `retain_ft` level 5). Inputs: `ext-code` 1.2, `ext-01/02/03`, CIFAR-100
(`fedesoriano`), CIFAR-100-C (`rojanregmi1`). **360/360 models, 0 failed**, 30,600 main rows: 150 unlearned, 75 ABA,
75 multi-context, 30 retrained, 30 original. A full-test model took ≈ 2.9-3.0 min (projection 3.2): one session.

**Collapse check:** 1,850 fine-tuning rows flagged across all models (1,765 in the matched analysis set). `frozen`
reinstatement 34% and `frozen` savings 12% of rows collapsed (the original-statistics scoring breaks after fine-tuning,
as in the pilot); **`update` rows: 0 in spontaneous recovery and reinstatement, 72 (1.2%) in savings**, all transient
over-shoots at 20/100 shots, steps 10-20 (NegGrad+ 37, SCRUB 32, FT 3). The savings conclusions are identical whether
those rows, or their whole curves, are dropped or kept (checked below).

### Analysis preview: notebook 06 run locally on these results plus notebook 05's (`ext-03` rows not local, so the
collateral test is missing here; it runs on Kaggle)

Main family, Holm-corrected; each value is unlearned minus retrained (same class and seed), matched models only.

| Hypothesis (primary score `top5_cc` unless noted) | FT (n=21) | NegGrad+ | RL | SCRUB | SalUn |
|---|---:|---:|---:|---:|---:|
| **H5b output selectivity** 2·\|AUROC−0.5\| | **+0.63*** | **+0.69*** | **+0.67*** | **+0.45*** | **+0.64*** |
| H5 probe excess | +0.020 | −0.039 | **+0.055*** | −0.024 | **+0.046*** |
| **H4 savings** (excess top-1 AUC) | **+0.128*** | −0.293 | +0.010 | −0.097 | +0.044 (Holm p = 0.09) |
| H1 spontaneous recovery (retain FT, 5 ep) | −0.022 | 0 | 0 | 0 | +0.002 |
| H2a renewal, new contexts | −0.298 | 0 | 0 | 0 | 0 |
| H2b renewal, ABA (n=15) | +0.017 (p = 0.10) | 0 | +0.001 | **+0.149** (raw p = 0.004, Holm 0.10) | +0.002 |
| H3 reinstatement, sibling − dissimilar | −0.250 | 0 | 0 | 0 | +0.001 |
| H6 multi-context reduces relapse | no | no | no | no (tiny +) | no |

\* = significant after Holm. Selectivity: retrained AUROC 0.59; unlearned FT 0.98, RL 0.98, SalUn 0.96, SCRUB 0.86,
NegGrad+ 0.014 (inverted).

**Robustness family (other scores), significant after Holm:** rank-score spontaneous recovery **SalUn +0.30** and RL
+0.026 (the forgotten class climbs from last place to about rank 70 for SalUn after 5 epochs of retain fine-tuning,
without reaching the top 5); rank-score reinstatement sibling − dissimilar for NegGrad+ +0.04, RL +0.05, SCRUB +0.04,
**SalUn +0.18** (related cues raise the forgotten class's rank more than unrelated ones, beyond the retrained model);
rank-score renewal SalUn +0.018. H4 scored with original statistics: only FT significant.

**Controls:** C1 (BatchNorm recalibration alone) 0 or negative for every method: **no BatchNorm illusion**. C2 is
significant for FT only (+0.22), but that reflects the original-statistics scoring breaking, not recovery.

**Savings sensitivity (exploratory; the primary rule above was fixed before the data):**

| Variant | FT | NegGrad+ | RL | SCRUB | SalUn |
|---|---:|---:|---:|---:|---:|
| primary (drop collapsed rows) | +0.128 | −0.293 | +0.010 | −0.097 | +0.044 |
| keep all rows | +0.128 | −0.299 | +0.010 | −0.092 | +0.044 |
| drop whole curves with a collapse | +0.128 | −0.270 | +0.010 | −0.056 | +0.044 |
| chance-corrected AUC | +0.148 | −0.243 | +0.066 | −0.051 | +0.100 |
| 5 shots only | **+0.246** | −0.173 | **+0.186** | +0.014 | **+0.186** |

RL and SalUn relearn much faster from few examples (5 shots, p < 1e-8); with 20-100 shots the retrained model catches
up. NegGrad+ and SCRUB resist relearning under every variant.

### What the main run says
1. **Suppression, not erasure, for every method (H5b):** the forgotten class's output still singles out its images
   (inverted for NegGrad+), far beyond the retrained model. The headline result.
2. **The suppression resists the classic relapse triggers at the output level:** no top-5 recovery with time,
   context or cues. There is graded movement underneath: SalUn's class climbs ~30 ranks after retain fine-tuning, and
   related cues raise it more than unrelated ones (rank space).
3. **One relapse hint:** ABA renewal for SCRUB (+0.15; raw p = 0.004 with n = 15, not significant after correcting for
   50 tests).
4. **Savings depend on the method:** FT always faster; RL/SalUn faster from few examples; SCRUB and NegGrad+ resist
   relearning, the opposite of extinction.
5. **Multi-context unlearning does not reduce relapse (H6 null)**; the extinction-theory remedy does not transfer.

Paper framing implied: "extinction without (output-level) relapse": the association survives (H5b, H5, savings) and
the inhibition is robust but graded. The prediction of the theory that fails (relapse) is as reportable as the one that
holds (intact association).

### Actions
- [ ] Save this output as dataset `ext-04-extinction`; share `ext-04` and `ext-05` with the account that runs NB06.
- [ ] Run NB06 on Kaggle (CPU) with `ext-code` 1.2 and `ext-01` … `ext-05` attached (adds the collateral test).

---

## Run #8: 06_analysis (v1)

**Reported:** 2026-10-01 · **Status:** ✅ success · **Session:** 44.7 s, Accelerator None · **Output:** 2.14 MB
(`Downloads/results-5`: `tables/` 3 CSV, `figures/` 8 × PNG + PDF) · **Account:** the user's.

**Inputs:** `ext-code` 1.2 and `ext-01` … `ext-05` (config matched `ext-04-extinction`). Rows: unlearn 305, extinction
31,295, mechanism 9,045. Matched rates as in run #3 (FT 0.7 standard; everything else 1.0). 1,765 collapsed fine-tuning
rows excluded (as in the preview). Primary score `top5_cc`, primary H1 proxy `retain_ft` level 5, BN mode `update`.

**Result:** 115 tests (main 50, control 15, robustness 50). **All 110 tests shared with the local preview (run #7 entry)
are identical to the last digit**; the 5 new ones are the collateral test, which needs notebook 03's rows:

| Collateral (G_related − G_unrelated) | FT | NegGrad+ | RL | SCRUB | SalUn |
|---|---:|---:|---:|---:|---:|
| mean (n) | +0.001 (21) | **+0.045*** (30) | −0.001 | **+0.012*** | −0.001 |

Significant after Holm (19): H5b for all 5 methods; H5 for RL and SalUn; H4 for FT; collateral for NegGrad+ and SCRUB;
robustness: rank-score H1 (RL, SalUn), H2a (SalUn), H3 (NegGrad+, RL, SCRUB, SalUn), H4 with original statistics (FT);
control C2 (FT; an artifact of original-statistics scoring). Not significant: H1-H3 and H6 on the primary score; H2b
SCRUB (+0.149, Holm p = 0.10).

**Figures (draft quality):** fig1's heatmap is dominated by H5b (other cells look pale), its colour-bar label is cut
off, and it has no significance marks; fig2 plots the primary score, which is flat, whereas the rank score shows the
real movement (SalUn); fig5 (savings) and fig6 (mechanism: weight change, CKA) are clear but lack confidence bands.
Paper figures should be redrawn from the result files (CPU only, no Kaggle needed).

**Status: all planned experiments of the main (BatchNorm) arm are complete.** GPU used: ≈ 17 h across two accounts.
Optional extensions: GroupNorm arm (≈ 4-6 GPU-h), a one-parameter recovery test (bias shift on the forgotten output),
a second dataset/architecture.

---

## Follow-up plan (2026-10-01, before any example-level data)

After a literature re-check, four items were set up:
1. **Reframe:** `paper/PAPER_PLAN.md` (title, abstract, contributions, claim → evidence map, positioning, the eight
   deviations from the plan, figures, outline, venue).
2. **Example-level arm** (notebook 07 stages A–D, ≈ 4.5 GPU-h): 3 seeds × 3 random 10% forget sets, 5 methods,
   pilot forget set (draw 9) for calibration only. **Pre-registered now:** XH1 (retain FT 5 epochs, BN update,
   `mem_gap` recovery minus retrained) is primary; XH2 renewal, XH4 savings (held-out `mem_gap` AUC), XH5 leakage;
   **B1** boundary = relapse fraction, example > class (one-sided Mann-Whitney per method).
3. **Savings speed (E2, exploratory):** computed locally on the run #7 data with notebook 08. Median steps to 50%
   forget accuracy with 5 images: FT 10, SCRUB 10, RL 50, SalUn 50, NegGrad+ and retrained never (≤100 steps).
   Unlearned minus retrained, faster by: FT 61, SCRUB 40, RL 19, SalUn 19 steps (Holm p ≤ 0.003); NegGrad+ 54
   slower. SCRUB relearns fast early (as in the relearning-delay literature); its low AUC comes from later instability.
4. **Head-bias check (E1, exploratory):** notebook 07 stage H on all class-level models.

**Local verification of 1.3 (2026-10-01):** self-test 27/27; class-level output bit-identical to 1.2; smoke chain on
fake CIFAR-100 (2 workers): 01 → 02 → 03 → 04 (main) → **07 (stages H, A, B, C, D)** → **08** all OK, figures 9-12 and
`followup_tests.csv` written. With `MATCHED_ONLY=False` every example-level test (XH1-XH5) and the B1 code path
produce values (smoke models are near chance, so none match and the numbers are meaningless). Notebook 08 on the
real run #7 data reproduces the E2 numbers above.

---

## Run #9: 07_example_level_and_bias (v1)

**Reported:** 2026-10-01 · **Status:** ⚠️ ran to completion (0 failed jobs), but the example-level arm missed its matching
target · **Session:** 18,395.2 s (306.6 min) · **Output:** 1.35 GB · **Dataset:** `sagorchandrapaul/ext-07-example-level`
(https://www.kaggle.com/datasets/sagorchandrapaul/ext-07-example-level).

**Settings/inputs:** as instructed; `ext_utils 1.3`; CIFAR-100 from `fedesoriano`; config matched `ext-03-unlearning`.

| Stage | Jobs | Time | Outcome |
|---|---:|---:|---|
| H head bias | 360 | 13.1 min | 360 ok |
| A example-level retrain | 10 | 43.2 min | 10 ok; test accuracy ≈ 0.74 (s0/d2 0.7408); accuracy on the forgotten images 0.738-0.751 (= test level, as designed) |
| B calibration (pilot draw 9) | 30 | 27.1 min | ranked candidates saved; the log shows no candidate reached the target (below) |
| C unlearning | 50 | 115.4 min | **0 of 45 main models matched** (every try `matched=False`) |
| D relapse tests | 63 | 107.4 min | 63 ok (run on the unmatched models) |

**Why nothing matched: the grids copied from class level are far too weak for example-level forgetting.**
Accuracy on the 5,000 forgotten images after unlearning (retrained model ≈ 0.74; original ≈ 1.0):

| Method | best setting (from calibration) | forget accuracy | test accuracy |
|---|---|---:|---:|
| NegGrad+ | α 0.95, lr 0.01 | 0.995-0.999 (no forgetting) | 0.744-0.751 |
| SCRUB | lr 0.001, msteps 5 | 0.995-0.999 (no forgetting) | 0.739-0.747 |
| FT | 10 ep, lr 0.05 | 0.93-0.96 | 0.739-0.751 |
| SalUn | 5 ep, lr 0.02 | 0.88-0.94 | 0.709-0.716 |
| RL | 5 ep, lr 0.02 | 0.80-0.90 | 0.706-0.717 |

At class level, 500 images of one class form a tight cluster, so a few epochs remove it. At example level the forgotten
images are spread over all classes and look like the retained ones, so the forgetting signal is diluted. FT, RL and
SalUn only partly forget; RL and SalUn also lose ≈ 3 test points (the retain limit). This matches published
random-forgetting benchmarks, where methods usually stay above the retrained model's forget accuracy.

**Consequence:** with `MATCHED_ONLY = True`, notebook 08's example-level tests (XH1-XH5, B1) have no models. Stage H
(head bias) and E2 are unaffected.

### Options
1. **Preview (no GPU):** analyse all example-level models. The relapse fraction divides by the forgetting each model
   actually achieved, so partial unlearners (FT, RL, SalUn) still answer "does what was forgotten come back?";
   NegGrad+ and SCRUB, which forgot nothing, drop out automatically (forgetting gap < 0.05). Must be reported as
   partial unlearning, not matched.
2. **Re-run stages B-D with an example-level grid** (stronger NegGrad+ weights, higher SCRUB/RL/SalUn rates, longer FT;
   ≈ 4-4.5 GPU-h; retrains reused). Needs a run tag in the code so new models do not collide with these.

---

## Run #10: 08_followup_analysis (v1)

**Reported:** 2026-10-01 · **Status:** ✅ ran (29.9 s, CPU) · **Output:** `Downloads/results-6` (3 tables, 4 figures).
Inputs as instructed; CIFAR-100 "NOT attached" line is harmless here. Rows: xl_unlearn 50, xl_extinction 3,843,
headbias 360, class extinction 31,295.

**Example-level tests:** empty, as expected from run #9 (matched rate 0 for every method with `MATCHED_ONLY = True`).
**E2:** identical to the local preview (FT 61, SCRUB 40, RL 19, SalUn 19 steps faster than retrained; NegGrad+ 54 slower).

**E1, head bias (class level; means over matched models):**

| Method | bias z-score (original ≈ −0.1) | forget acc, original bias restored | forget acc, original output row restored | recall after one constant shift | test false-positive rate of that shift |
|---|---:|---:|---:|---:|---:|
| FT | −2.3 | 0.001 | 0.287 | **0.912** | 0.118 |
| RL | −10.1 | 0.000 | 0.000 | **0.962** | 0.083 |
| SalUn | −5.2 | 0.000 | 0.000 | **0.957** | 0.057 |
| SCRUB | −24.8 | 0.000 | 0.278 | 0.666 | 0.038 |
| NegGrad+ | −13.1 | 0.000 | 0.003 | 0.005 | 0.075 |
| retrained | −1.6 | 0.000 | 0.000 | 0.376 | 0.201 |
| original | −0.1 | 0.734 | 0.734 | 0.972 | 0.068 |

Unlearned minus retrained, one-constant recall: FT +0.59, RL +0.59, SalUn +0.58, SCRUB +0.29 (Holm p < 0.001);
NegGrad+ −0.37. ABA and multi-context models behave the same way (RL/SalUn 0.90-0.95).

**Reading:** every method pushes the forgotten class's bias far below the other 99, the signature Zheng et al. (2026)
describe. But putting back the original bias alone restores **nothing**, so the forgetting is not just that bias.
Yet **adding one constant** to the forgotten output brings back 91-96% of the class for FT, RL and SalUn: the output
still ranks the class's images correctly (H5b), only lower. NegGrad+ cannot be revived this way, because its
selectivity is inverted. Restoring the whole output row helps only FT and SCRUB (≈ 28%), matching notebook 05's
finding that RL and SalUn forget in the body, not the head.
*Caveat:* the constant was set so that 1% of retain *training* images flip; on test images the false-positive rate is
4-12% (training images are fitted more confidently). A cleaner version sets it on half of the test images and
evaluates on the other half; redo it with the re-run.

**Notebook 08 patched:** `XL_MATCHED_ONLY = False` (class level keeps `MATCHED_ONLY = True`), plus a table of how much of
the original-to-retrained forgetting gap each example-level method achieved. Tested on smoke data.

---

## Run #11: 08_followup_analysis (v1, preview with all example-level models)

**Reported:** 2026-10-01 · **Status:** ✅ ran (CPU) · **Output:** `Downloads/results (26).zip` (same 3 tables and 4 figures).
Same inputs as run #10, with the patch `XL_MATCHED_ONLY = False`, so the 45 partially unlearned example-level models
of run #9 are analysed. **These are not matched models; the numbers are a preview, not a result.**

| Test (unlearned minus retrained; 9 forget sets per method) | FT | NegGrad+ | RL | SCRUB | SalUn |
|---|---:|---:|---:|---:|---:|
| XH1 retain FT 5 ep, `mem_gap` recovery (BN update) | +0.001 | +0.001 | **+0.109** | −0.004 | **+0.059** |
| XH1 robustness, original BN statistics | −0.075 | −0.042 | −0.039 | −0.088 | −0.075 |
| XH2 renewal | −0.131 | −0.147 | −0.119 | −0.163 | −0.169 |
| XH4 savings (held-out AUC) | +0.183 | +0.246 | +0.127 | +0.252 | +0.182 |
| XH5 leakage (MIA AUROC) | +0.201 | +0.372 | +0.318 | +0.324 | +0.210 |

Bold: Holm p = 0.039 (the smallest possible with n = 9, one-sided). XH4 and XH5 are "significant" for every method,
but for NegGrad+ and SCRUB that only says they never forgot (forget accuracy 0.995-0.999), so these rows cannot be read
without matching.

**B1, relapse fraction** (share of the achieved forgetting that retain fine-tuning brings back; models with a gap < 0.05
drop out, which removes NegGrad+ and SCRUB):

| Method | example level | class level | difference | Holm p |
|---|---:|---:|---:|---:|
| RL | 0.700 (n = 9) | −0.000 (n = 30) | +0.70 | 8 × 10⁻⁹ |
| SalUn | 0.709 (n = 9) | 0.002 (n = 30) | +0.71 | 9 × 10⁻⁸ |
| FT | 0.024 (n = 9) | −0.069 (n = 21) | +0.09 | 0.004 |

**Reading:** the boundary the paper needs is visible. Whatever RL and SalUn remove at example level, retain fine-tuning
brings 70% of it back; at class level the same methods, fine-tuned the same way, bring back nothing. FT forgot little
(0.93-0.96 vs retrained 0.74), so its fraction is small either way. E1 and E2 are unchanged from run #10.
**Limitation:** partial unlearning; the retain-accuracy constraint was also violated for RL/SalUn (≈ −3 test points).
A reviewer will ask for matched models, hence notebook 09.

---

## Plan: notebook 09, atypical example-level arm (pre-registered 2026-10-01, before any data)

**Why atypical.** Random images are typical: the retrained model already gets 74% of them right, so the memorisation
gap is thin and no method matched (run #9). Atypical images (lowest 5% C-score) are learned only by memorisation, so
original ≈ 1.0 and retrained is low; "forgotten" and "came back" are then unambiguous. This is the setting of
"From Dormant to Deleted" (Siddiqui et al. 2025), which reports retain-FT recovery, so our class-level null now has a
directly comparable example-level counterpart in the same code base.

**Design:** ext_utils 1.4, `xl_design = "atypical"`, 500 images per forget set from the 2,500 least typical,
3 seeds × 3 forget sets (+ pilot draw 9 for calibration only), 5 methods, denser grid (45 calibration jobs), up to 4
candidates per model, matching = forget accuracy within ±5 points of the retrained model and test accuracy within
3 points. Relapse tests identical to notebooks 04/07; savings with 10 and 100 shots. Stage H re-runs the head-bias
check with the held-out shift (v2). Estimated 4.5-5 GPU-h on T4 x2, one session.

**Pre-registered analysis (notebook 08, `XL_DESIGN = "atypical"`, matched models only):**
- **XH1 (primary):** forget accuracy after 5 epochs of retain fine-tuning (BN update), recovery minus the retrained
  model's recovery; one-sided Wilcoxon per method, Holm over the 5 methods. Robustness: original BN statistics.
- **B1 (primary for the paper's claim):** relapse fraction at example level > class level, one-sided Mann-Whitney per
  method, Holm.
- **Secondary:** XH2 renewal, XH4 savings (10/100 shots), XH5 MIA leakage; E1 v2 (`shift_recall_fpr1`).
- **Reading rules:** a method with fewer than 5 matched models is reported descriptively (a one-sided Wilcoxon cannot
  reach p < 0.05 below n = 5). If no method matches, run #11's preview stays the evidence and is reported as partial
  unlearning. The expected (not assumed) outcome: RL/SalUn relapse at example level, as in D2D; nothing at class level.

**Local verification of 1.4 (2026-10-01):** self-test 28/28; the real C-score file (50,000 scores, labels identical to
the CIFAR-100 training labels, 5% cut-off at C-score 0.025) is shipped in `ext-code` next to `ext_utils.py`, so
notebook 09 does not depend on the download. Smoke chain on fake CIFAR-100: notebook 09 (stages H, A, B, C, D, quick
look) → notebook 08 with `XL_DESIGN = "atypical"`: all cells OK; notebook 08 reads only the atypical rows (random-design
rows from notebook 07 ignored) and the v2 shift columns. Two fixes found before any GPU time: (1) the NegGrad+ grid
now spans α 0.8-0.99, because in our loss a *smaller* α forgets harder (the draft grid only reached 0.9); (2) notebook
09's quick-look recovery table filtered the wrong level (empty table), fixed.

---

## Run #12: 09_example_level_atypical (v1)

**Reported:** 2026-10-02 · **Status:** ⚠️ ran to completion (0 failed jobs), but no model met the matching rule ·
**Session:** 19,068.1 s (317.8 min) · **Output:** 1.35 GB (to be saved as `ext-09-atypical`).

**Setup:** `ext_utils 1.4` from `mahdihasanqurishi/ext-code`; CIFAR-100 from `fedesoriano`; config matched `ext-01`.
C-scores: 50,000 images, median 0.635, 5% cut-off 0.025. Pilot forget set: 500 images, mean C-score 0.010, 96 classes.

| Stage | Jobs | Time | Outcome |
|---|---:|---:|---|
| H head bias v2 | 360 | 12.0 min | 360 ok |
| A retrain | 10 | 41.2 min | 10 ok; test accuracy 0.745-0.752; **forget accuracy 0.024-0.054** (mean 0.034 over the 9 main sets): a model that never saw these images gets ~3% of them right, as designed |
| B calibration (pilot set, retrained 0.026 / test 0.750) | 45 | 30.1 min | 45 ok; no candidate met the rule (below) |
| C unlearning | 50 | 149.6 min | 50 ok; **0 of 45 main models matched** (4 tries each; the 60-second progress line shows only the latest try) |
| D relapse tests | 63 | 84.4 min | 63 ok |

**Calibration (pilot set) shows why:**
- **NegGrad+** has a sharp trade-off between forgetting and utility. Every setting that forgets to the retrained level
  costs ≥ 6 test points: α 0.95 lr 0.02 gives forget 0.012 at test 0.690; α 0.9 lr 0.01 gives 0.006 at 0.655. Settings that
  keep test accuracy do not forget: α 0.99 gives 0.87-0.98; α 0.95 lr 0.01 gives 0.226 at 0.708. Matching needs ≤ 0.076 at ≥ 0.720.
- **RL** improves with training and keeps test accuracy: lr 0.02 over 1/3/5 epochs gives 0.274/0.208/0.154 (test 0.745).
  Still improving at the grid's edge. **SalUn:** 0.336/0.220/0.180 at lr 0.02. **FT:** best 0.338 (20 ep, lr 0.05).
  **SCRUB:** best 0.496.
- The selection rule ranks unmatched candidates by forget-accuracy distance only, so NegGrad+ kept the most
  damaging setting first (α 0.8, test 0.58). That explains its mean test accuracy of 0.652.

**Unlearned models (means over 9 forget sets; retrained 0.034 forget / 0.750 test):**

| Method | forget acc | test acc | retain FT 5 ep (BN update): forget acc after minus before |
|---|---:|---:|---:|
| FT | 0.406 | 0.754 | +0.002 |
| NegGrad+ | **0.020** | 0.652 | **+0.130** |
| RL | 0.160 | 0.743 | **+0.288** |
| SCRUB | 0.506 | 0.746 | +0.045 |
| SalUn | 0.192 | 0.740 | **+0.434** |
| original | (≈ 1) | – | −0.010 |
| retrained | 0.034 | 0.750 | −0.000 |

**Reading:**
- **Example-level forgetting relapses.** Five epochs of fine-tuning on the retained data, which never shows the
  forgotten images, raise SalUn's accuracy on them from 0.19 to ≈ 0.63 and RL's from 0.16 to ≈ 0.45. The retrained model
  stays at 0.03. At class level the same methods, under the same probe, relapse by 0.00 (runs #7/#11). This is the
  boundary the paper needs, now on a design with a large memorisation gap.
- **It is not only left-over knowledge.** NegGrad+ had forgotten *as much as* retraining (0.020 vs 0.034) and still
  climbs to ≈ 0.15. That is 4× what a model that never saw the images reaches at full test accuracy, so it cannot come
  from repaired general features alone. Caveat: NegGrad+ lost 10 test points, and retain fine-tuning also repairs them.
- **A trade-off rather than a match.** For NegGrad+, reaching retrained-level forgetting costs utility, and restoring
  utility by retain fine-tuning brings part of the forgotten examples back.
- Across methods, more complete forgetting goes with less relapse (SalUn > RL > NegGrad+). Partial unlearning is part
  of the story, so it has to be reported.
- FT's unlearning *is* retain fine-tuning, so this probe only continues it (≈ 0 by construction). SCRUB forgot least.

**Head-bias check v2 (class level; the constant is set on even test images and measured on odd ones):**

| | restore original bias | one-constant shift, recall at 1% FPR | measured FPR | recall at 5% FPR |
|---|---:|---:|---:|---:|
| original | 0.734 | 0.872 | 0.010 | 0.963 |
| retrained | 0.000 | 0.019 | 0.009 | 0.092 |
| FT | 0.011 | 0.625 | 0.010 | 0.824 |
| NegGrad+ | 0.000 | 0.000 | 0.010 | 0.003 |
| RL | 0.000 | **0.872** | 0.010 | 0.950 |
| SCRUB | 0.000 | 0.504 | 0.010 | 0.696 |
| SalUn | 0.000 | **0.875** | 0.011 | 0.958 |

(ABA and multi-context models are similar: RL/SalUn 0.81-0.86, NegGrad+ 0.)

The run #10 caveat is resolved: the false-positive rate is now really 1%. **For RL and SalUn, class-level forgetting
is a single offset on the class's output.** Shifting it back restores *exactly* the original's recall at the same
false-positive rate, while the retrained model reaches 0.02. Restoring the original bias alone still brings back nothing.
NegGrad+ cannot be revived this way. Overlaps with Zheng et al. 2026 (bias shift), so cite them. Our additions are the
held-out FPR control and the per-method split.

**Deviation from the pre-registration:** XH1/B1 were to use matched models only. With 0 matched, the pre-registered
fallback was the random-design preview (run #11). The atypical models are better evidence (memorisation gap ≈ 0.97 vs
0.26; NegGrad+ forget-matched), so notebook 08 now analyses all atypical models as partial unlearning
(`XL_MATCHED_ONLY = False`). It adds three checks (level after fine-tuning vs the retrained level; forget-matched models
only; dose-response) in `tables/XH1_partial_unlearning_checks.csv`. Tested on run #7 class rows plus synthetic
atypical rows.

**Optional follow-up (≈ 2 GPU-h):** extended search for matched models. RL 10-20 epochs at lr 0.02/0.05 looks reachable
from the calibration trend; also SalUn 10-20 epochs and NegGrad+ α 0.95-0.97 with more epochs. Selection would rank
by total violation of the matching rule. Needs a method-variant tag in the code (v1.5) so it does not collide with
these models.

---

## Run #13: 08_followup_analysis (old copy, on the atypical results)

**Reported:** 2026-10-02 · **Status:** ⚠️ ran (40.9 s, CPU), but with an **old version of the notebook** ·
**Output:** `Downloads/results (27).zip`. The notebook header says "Copied from Sagor Chandra Paul (+11, −2)": it is the
run #11 version, not the current file. Evidence: no `example-level design: …` line in the log; no
`XH1_partial_unlearning_checks.csv`; the figures sit in cell 27 (the current file has 31 cells); E1b repeats the run #10
values (old, training-calibrated shift). Inputs: `ext-09-atypical` (50 `xl_unlearn` rows, all atypical; `ext-07` not
attached), `ext-04`, `ext-01`, headbias 360 rows (v1.4).

**What it still tells us.** The old version scores the example level with `mem_gap` = forget accuracy − test accuracy.
On the atypical design that is a conservative robustness check: it subtracts any gain in test accuracy, i.e. the
"retain fine-tuning just repairs the damaged model" explanation.

| Test (unlearned minus retrained, n = 9 per method; `mem_gap`) | FT | NegGrad+ | RL | SCRUB | SalUn |
|---|---:|---:|---:|---:|---:|
| XH1 retain FT 5 ep, BN update | +0.001 | **+0.046** | **+0.285** | **+0.043** | **+0.428** |
| B1 relapse fraction, example (class) | 0.000 (−0.069) | 0.055 (0.000) | **0.343** (0.000) | 0.088 (0.000) | **0.537** (0.002) |

Bold: Holm p = 0.039 for XH1 (the minimum with n = 9) and ≤ 6 × 10⁻³ for B1 (all five methods).
- NegGrad+: forget accuracy rose by 0.130 (run #12 quick look), test accuracy by ≈ 0.084. Even after subtracting the whole
  test gain, +0.046 remains. A model that had forgotten as much as retraining regains more than its general repair explains.
- RL and SalUn hardly change test accuracy, so `mem_gap` and forget accuracy agree (+0.29 / +0.43).
- **Problems found for the atypical design (fix before reporting):**
  - **XH5 leakage** uses 2·|MIA AUROC − 0.5|. Atypical images are harder than test images, so even the retrained model
    separates them (low confidence). FT and SCRUB, which forgot least, then look *less* leaky than retraining (−0.30,
    −0.40). XH5 needs a signed comparison to the retrained model's AUROC for this design. NegGrad+ +0.09 (over-forgetting
    is detectable).
  - **XH4 savings** (AUC of the score while relearning) includes the starting level, so methods that forgot little
    (FT +0.32, SCRUB +0.43) look "fast". It needs the curve relative to its own start.
  - **XH2 renewal** is negative for all methods. The `mem_gap` version mixes in the test-accuracy drop under the context
    shift. Read it with `forget_acc`.
  - Original-statistics (frozen BN) rows move test accuracy a lot after fine-tuning (as at class level), so they are not
    informative here.

**Action:** re-import the current `notebooks/08_followup_analysis.ipynb` (31 cells; config shows `XL_DESIGN`,
`XL_SCORE`, `XL_FORGET_GAP`) and re-run on CPU. NB08 now also records two robustness rows (the pre-registered rows stay
unchanged): **R: XH4 savings, gain over own start** and **R: XH5 MIA AUROC minus retrained (signed)**. Tested on run #7
class rows plus synthetic atypical rows.

---

## Run #14: 08_followup_analysis (current version, on the atypical results)

**Reported:** 2026-10-02 · **Status:** ✅ ran (32.9 s, CPU) · **Output:** `Downloads/results (28).zip` (4 tables incl.
`XH1_partial_unlearning_checks.csv`, figures 9-12). Log: `example-level design: atypical | primary score: forget_acc |
matched only: False`; 50 `xl_unlearn`, 3,843 `xl_extinction`, 360 headbias (v1.4), 31,295 class rows.

**Example level (forget accuracy; unlearned minus retrained; n = 9 per method; all models = partial unlearning):**

| Test | FT | NegGrad+ | RL | SCRUB | SalUn |
|---|---:|---:|---:|---:|---:|
| **XH1** retain FT 5 ep, BN update | +0.002 | **+0.131** | **+0.289** | **+0.045** | **+0.435** |
| R: XH1, original BN statistics (5 ep) | −0.277 | **+0.086** | −0.044 | −0.306 | −0.040 |
| XH2 renewal | −0.302 | −0.012 | −0.107 | −0.371 | −0.134 |
| XH4 savings (pre-registered AUC) | **+0.335** | −0.001 | **+0.186** | **+0.435** | **+0.275** |
| R: XH4, gain over own start | −0.036 | +0.012 | **+0.056** | −0.040 | **+0.114** |
| XH5 leakage 2·\|AUROC − 0.5\| (pre-registered) | −0.297 | **+0.093** | −0.042 | −0.396 | −0.065 |
| R: XH5, signed AUROC | **+0.148** | −0.046 | **+0.021** | **+0.198** | **+0.032** |
| level after retain FT minus retrained level | +0.375 | +0.116 | +0.414 | +0.518 | +0.592 |
| **B1** relapse fraction, example (class) | 0.004 (−0.069) | **0.134** (0.000) | **0.345** (0.000) | **0.092** (0.000) | **0.541** (0.002) |

Bold = Holm p < 0.05 (0.039-0.041 is the minimum with n = 9; B1 p ≤ 0.004).

**Other proxies (forget-accuracy recovery, `XH1_spontaneous_all_proxies.csv`):** BN recalibration with 10-10,000
retain images: −0.003 to −0.24 for every method (no BatchNorm illusion at example level either). Gaussian weight noise,
pruning up to 50%, 4-8-bit quantisation: |Δ| ≤ 0.03. 70% pruning lowers every model. **Relapse appears only when the
weights are trained on retained data.**

**Partial-unlearning checks:**
1. **No within-method dose-response.** Spearman of forget accuracy before the test vs recovery: NegGrad+ −0.03, RL
   −0.07, SCRUB −0.07, SalUn −0.50 (all n.s.); pooled without FT −0.17 (p = 0.33). Models of the same method that forgot
   more do not relapse less. Across methods, SalUn (forgot least of the three) relapses most.
2. **NegGrad+ (all 9 forget-matched) relapses, and the size tracks the damage it did.** Recovery correlates with its
   test accuracy before the test at ρ = −0.90 (p < 0.001):
   - the 4 models from the harsh setting (test 0.57-0.65) climb to 0.14-0.30;
   - the 5 gentlest models (test 0.69-0.70) climb only to 0.06-0.09, i.e. +0.026 to +0.062 above the retrained model
     after the same fine-tuning (mean +0.044, 5/5 positive).

   Reading: NegGrad+ hides the examples largely by damaging shared features. Repairing them brings examples back; the
   residue above retraining is small but consistent.
3. **Original BN statistics:** the probe itself degrades memorised images. The *original* model loses 0.60 of its
   forget accuracy by epoch 5 in this mode (−0.01 with updated statistics). At epoch 1, before that degradation,
   RL +0.10, SalUn +0.20, NegGrad+ +0.06 (retrained +0.00). Report the epoch-1 values next to the epoch-5 ones.

**Renewal (XH2):** no method's forget accuracy rises under a context shift. The negative values are ordinary corruption
losses, larger for models that still know more (FT, SCRUB).

**E1 v2 (held-out shift at 1% FPR), unlearned minus retrained:** RL +0.853, SalUn +0.855, FT +0.551, SCRUB +0.485 (Holm
p ≤ 1e-4), NegGrad+ −0.020. Restoring the original bias alone: 0.000-0.001. E2 unchanged.

**What the example-level arm now supports:**
- Example-level forgetting relapses under retain-only fine-tuning, and class-level forgetting by the same methods does
  not (B1 significant for all five).
- The relapse is specific to weight updates on retained data, not BN statistics or perturbations.
- RL/SalUn relearn the forgotten examples faster than retraining, and keep a small residual membership signal.

**Main caveat:** no model met the matching rule. Mitigation: the relapse fraction normalises by achieved forgetting; no
within-method dose-response; forget-matched NegGrad+ still relapses, though its relapse scales with the utility it
destroyed.

**Optional next run (NB10, ≈ 2 GPU-h):** extended RL/SalUn search (10-20 epochs, lr 0.02/0.05) aiming for matched
models, using the RL calibration trend (0.274 → 0.208 → 0.154 over 1/3/5 epochs at lr 0.02, test unchanged). Either it
gives matched models, or a dose-response along training length.

---

## Run #15: 10_matched_search (v1)

**Reported:** 2026-10-02 · **Status:** ✅ ran (0 failed) · **Session:** 9,407.4 s (156.8 min) · **Output:** 678 MB,
dataset https://www.kaggle.com/datasets/mahdihasanqurishi/ext-10-matched-search.

**Setup:** `ext_utils 1.5`; CIFAR-100 found; config matched `ext-09-atypical`; forget sets identical to notebook 09's
(cut-off 0.025, pilot mean C-score 0.010); retrained models 10/10 found; 50 notebook-09 rows found.

| Stage | Jobs | Time | Outcome |
|---|---:|---:|---|
| B calibration | 14 | 23.0 min | candidates: RL-long / SalUn-long: 20 ep lr 0.05, 20 ep lr 0.02, 10 ep lr 0.05; NegGrad+-long: α 0.97 lr 0.02, α 0.95 lr 0.01, α 0.95 lr 0.02 |
| C unlearning | 30 | 95.1 min | RL-long: every model matched on its first try (one 20-epoch run each); SalUn-long and NegGrad+-long used all 3 tries |
| D relapse tests | 27 new (+18 reused) | 38.1 min | 27 ok |

**Unlearned models (means over 9 forget sets; retrained 0.034 forget / 0.750 test):**

| Variant | matched | forget acc | test acc | retain FT 5 ep: before → after | recovery |
|---|---:|---:|---:|---|---:|
| RL (nb 09, 5 ep lr 0.02) | 0/9 | 0.160 | 0.743 | 0.160 → 0.448 | +0.288 |
| **RL-long** (20 ep lr 0.05) | **9/9** | **0.047** | **0.751** | 0.047 → 0.087 | **+0.040** |
| SalUn (nb 09) | 0/9 | 0.192 | 0.740 | 0.192 → 0.626 | +0.434 |
| SalUn-long, matched | 4/9 | 0.074 | ≈ 0.748 | 0.074 → 0.181 | **+0.107** |
| SalUn-long, unmatched | 5/9 | 0.083 | ≈ 0.750 | 0.083 → 0.204 | +0.121 |
| NegGrad+ (nb 09) | 0/9 | 0.020 | 0.652 | 0.020 → 0.150 | +0.130 |
| NegGrad+-long (α 0.97 lr 0.02, 6 ep) | 0/9 | 0.004 | 0.705 | 0.004 → 0.024 | +0.020 |
| original | – | 0.996 | – | 0.996 → 0.986 | −0.010 |
| retrained | – | 0.034 | 0.750 | 0.034 → 0.034 | −0.000 |

SalUn-long's 5 unmatched models missed the forget bound by only 0.2-1.8 points (0.076-0.098 vs ≤ 0.074-0.082) at full test
accuracy. NegGrad+-long forgets *more* than retraining (0.004) but stays 1-2 points short on test accuracy (0.700-0.710
vs ≥ 0.715-0.722).

**Reading (this revises runs #12/#14):**
- **Matched example-level forgetting still relapses, but much less.** RL matched on all 9 sets regains +0.040 of forget
  accuracy under retain fine-tuning. The retrained model regains 0.000; class-level RL regains 0.000. After fine-tuning,
  RL-long sits at 0.087 vs the retrained 0.034 (≈ 20 of 500 images back, ≈ 2.6× the retrained level). Relapse fraction
  ≈ 0.04 (class 0.00). SalUn's matched models regain +0.107 (fraction ≈ 0.12).
- **Strong dose-response across schedules.** The more completely a method forgets, the smaller the relapse: RL
  0.29 → 0.04, SalUn 0.43 → 0.11, NegGrad+ 0.13 → 0.02. NegGrad+-long ends *below* the retrained level, i.e. no relapse
  beyond retraining. Run #14's "no within-method dose-response" held only within one schedule. Most of the large relapse
  in runs #12/#14 came from incomplete forgetting.
- **Revised claim:** example-level relapse under retain fine-tuning is mostly a symptom of incomplete forgetting. At
  retraining-matched forgetting it shrinks to a few points but does not vanish for RL (+4) and SalUn (+11). Class-level
  forgetting by the same methods shows none. NegGrad+, which over-forgets, shows none beyond retraining. This is a
  smaller but cleaner boundary than run #12 suggested, and the dose-response is itself a result worth reporting.

**Notebook 08 patched for notebook 10:**
- Variants map to their base method.
- The pre-registered merge rule: a matched variant replaces notebook 09's model per (method, seed, set), so RL = RL-long
  ×9 and SalUn = SalUn-long ×4 + SalUn ×5.
- New family **`xl-matched`**: XH1 and B1 on matched models only.
- New per-variant table `XH1_by_variant.csv`, plus robustness rows "R: XH1, notebook 10 schedule".

Tested on run #7 class rows plus synthetic rows shaped like runs #12/#15. **Next:** run NB08 with `ext-01`, `ext-04`,
`ext-09-atypical` and `ext-10-matched-search`.

---

## Run #16: 08_followup_analysis (final, notebook 09 + notebook 10)

**Reported:** 2026-10-02 · **Status:** ✅ ran (42.3 s, CPU) · **Output:** `Downloads/results (29).zip` (5 tables incl.
`XH1_by_variant.csv`, figures 9-12). Inputs: `ext-code` 1.5, `ext-01`, `ext-04`, `ext-09-atypical`,
`ext-10-matched-search`. Rows: `xl_unlearn` 80, `xl_extinction` 5,490, headbias 360, class 31,295. Merge rule applied:
RL = RL-long ×9 (all matched); SalUn = SalUn-long ×4 (matched) + notebook-09 SalUn ×5; others from notebook 09.

**Primary example-level results on matched models (`xl-matched` family, Holm over its tests):**

| | n | recovery minus retrained [95% CI] | Holm p | B1 relapse fraction, example vs class | p |
|---|---:|---:|---:|---|---:|
| RL (RL-long) | 9 | **+0.041** [0.032, 0.049] (9/9 positive, +0.016 to +0.062) | **0.004** | **0.043 vs 0.000** | < 10⁻⁴ |
| SalUn (SalUn-long) | 4 | +0.110 [0.094, 0.131] | 0.0625 (n < 5: descriptive, as pre-registered) | 0.119 vs 0.002 | < 10⁻⁴ |

**By variant (all 9 models each; `XH1_by_variant.csv`):**

| Variant | matched | test | forget before → after | recovery minus retrained | after minus retrained level |
|---|---:|---:|---|---:|---:|
| RL (5 ep) | 0 | 0.743 | 0.160 → 0.448 | +0.289 | +0.414 |
| RL-long (20 ep) | 1.00 | 0.751 | 0.047 → 0.087 | +0.041 | +0.054 |
| SalUn (5 ep) | 0 | 0.740 | 0.192 → 0.626 | +0.435 | +0.592 |
| SalUn-long (20 ep) | 0.44 | 0.749 | 0.079 → 0.194 | +0.115 | +0.160 |
| NegGrad+ (harsh) | 0 | 0.652 | 0.020 → 0.150 | +0.131 | +0.116 |
| NegGrad+-long (gentle) | 0 | 0.705 | 0.004 → 0.024 | +0.021 | −0.010 |
| FT | 0 | 0.754 | 0.406 → 0.408 | +0.002 | +0.375 |
| SCRUB | 0 | 0.746 | 0.507 → 0.551 | +0.045 | +0.518 |

**Other tests with the merged set (RL = RL-long):**
- RL with original BN statistics: +0.010 [0.001, 0.019], now positive.
- XH2 renewal for RL: −0.006 (none).
- XH4 savings gain over start for RL: +0.027 [0.021, 0.032], slightly faster relearning than retraining.
- XH5 signed AUROC for RL: +0.006 [0.004, 0.009], tiny residual membership signal.
- E1 and E2 unchanged.

**Note on the robustness family:** it now holds ~28 tests. A one-sided Wilcoxon with n = 9 cannot go below p = 0.002, so
Holm caps every robustness row at 0.0508 even when all 9 values agree. Report robustness rows by their bootstrap CIs
(most exclude 0), not Holm stars.

**Final reading (example level):**
1. Relapse under retain-only fine-tuning tracks how completely the examples were forgotten: large for under-forgetting
   schedules (RL +0.29, SalUn +0.43), small for retraining-matched ones.
2. At matched forgetting it does not vanish: RL +0.04 (all 9 sets), SalUn +0.11, against 0.00 for the retrained model
   and 0.00 for class-level forgetting by the same methods.
3. SalUn relapses ≈ 2.5× more than RL at similar forgetting (0.079 vs 0.047 before). That fits SalUn updating only the
   salient half of the weights, leaving the rest of the memory in place (cf. notebook 05: RL/SalUn keep more
   decodable forget-class information).
4. NegGrad+ over-forgets (below the retrained level) and, when gentle, does not relapse beyond retraining. Its utility
   cost blocks matching.

**Figure note:** figure 9 uses the merged set, so its SalUn bar mixes matched and unmatched models (wide CI). For the
paper, a dose-response figure (forget accuracy before vs recovery, one point per variant, class-level points at 0)
tells the story better. To make at the writing stage from `XH1_by_variant.csv`.

**Status: all experiments complete** (≈ 30 GPU-h over both accounts). Next: revise `paper/PAPER_PLAN.md` to the
final claim and write.

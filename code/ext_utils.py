"""
ext_utils.py - shared library for the project
"Machine Unlearning is Extinction, Not Forgetting".

Every Kaggle notebook (00-06) imports this one file, so all notebooks use
exactly the same data pipeline, model, unlearning methods and tests.

Contents
  1. Paths and configuration (Kaggle vs. local, consistency checks)
  2. CIFAR-100 / CIFAR-100-C loading, GPU-side augmentation, "contexts"
  3. ResNet-18 (CIFAR variant), training, evaluation, checkpoints
  4. Five unlearning methods: FT, NegGrad+, RL, SCRUB, SalUn
  5. The four extinction tests (spontaneous recovery, renewal,
     reinstatement, rapid reacquisition)
  6. Mechanism analyses (linear probe, head swap, weight change, CKA)
  7. Tasks + a resumable, multi-GPU job runner
  8. Job builders used by the notebooks, environment report, self-test
  9. Follow-ups: example-level forgetting arm (1.3 random, 1.4 atypical), head-bias check (notebooks 07-09)

When run as a script
    python ext_utils.py worker <jobfile.json> <rank> <world>
it executes one worker of a job list; notebooks start it via launch().
"""
from __future__ import annotations

import contextlib
import copy
import gc
import glob
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "1.5"
T0 = time.time()                      # notebook/session start (used for the time budget)
UTILS_FILE = os.path.abspath(__file__)

warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
warnings.filterwarnings("ignore", message=".*You are using `torch.load`.*")

# =============================================================================
# 1. Paths and configuration
# =============================================================================
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = os.environ.get("EXT_WORK") or ("/kaggle/working" if ON_KAGGLE else os.path.abspath("ext_work"))
INPUT = os.environ.get("EXT_INPUT") or ("/kaggle/input" if ON_KAGGLE else os.path.abspath("ext_input"))
DATA = os.environ.get("EXT_DATA") or ("/tmp/ext_data" if ON_KAGGLE else os.path.abspath("ext_data"))
CKPT_DIR = os.path.join(WORK, "ckpt")
RES_DIR = os.path.join(WORK, "results")
LOG_DIR = os.path.join(WORK, "logs")
JOB_DIR = os.path.join(WORK, "jobs")
FIG_DIR = os.path.join(WORK, "figures")


def ensure_dirs():
    for d in (WORK, CKPT_DIR, RES_DIR, LOG_DIR, JOB_DIR, FIG_DIR, DATA):
        os.makedirs(d, exist_ok=True)


# One forget class from 10 *different* CIFAR-100 superclasses.
FORGET_CLASSES = ["maple_tree", "tiger", "rose", "shark", "bus",
                  "apple", "chair", "butterfly", "girl", "castle"]
PILOT_CLASS = "lamp"                  # superclass: household electrical devices (not used above)
METHODS = ["FT", "NegGrad+", "RL", "SCRUB", "SalUn"]


def base_method(m):
    """Method variants share their base method's code: "RL-long" runs RL with its own grid, job keys and checkpoints."""
    return m.split("-")[0]
ALL_TESTS = ["spontaneous", "contexts", "reinstatement", "savings"]

DEFAULT_CALIB_GRID = {
    "FT":       [{"lr": lr, "epochs": e} for lr in (0.01, 0.02, 0.05) for e in (5, 10)],
    "NegGrad+": [{"lr": lr, "alpha": a, "epochs": 5} for lr in (0.005, 0.01, 0.02) for a in (0.9, 0.95, 0.99)],
    "RL":       [{"lr": lr, "epochs": e} for lr in (0.005, 0.01, 0.02) for e in (3, 5)],
    "SCRUB":    [{"lr": lr, "msteps": m, "epochs": m + 3, "alpha": 0.001, "gamma": 0.99, "T": 4.0}
                 for lr in (5e-4, 1e-3, 5e-3) for m in (3, 5)],
    "SalUn":    [{"lr": lr, "epochs": e, "mask_ratio": 0.5} for lr in (0.005, 0.01, 0.02) for e in (3, 5)],
}

DEFAULTS = dict(
    # ---- experiment design (shared by all notebooks) ----
    seeds=[0, 1, 2],
    forget_classes=list(FORGET_CLASSES),
    pilot_class=PILOT_CLASS,
    methods=list(METHODS),
    aba_classes=["maple_tree", "tiger", "rose", "shark", "bus"],
    aba_context="gray",
    mctx_classes=["maple_tree", "tiger", "rose", "shark", "bus"],
    mctx_contexts=["clean", "gray", "blur"],
    # ---- model / training (shared) ----
    norm="bn",
    width=64, epochs=40, batch_size=256, lr=0.1, momentum=0.9, weight_decay=5e-4,
    nesterov=True, label_smoothing=0.0, pct_start=0.25,
    amp=True, channels_last=True, cudnn_benchmark=True, eval_batch_size=1024,
    ckpt_fp16=True,
    # ---- unlearning ----
    forget_batch_size=64, clip_grad=5.0,
    match_forget_max=0.01, match_retain_gap=0.03, max_tries=3,
    calib_grid=None, calib_limit=None,
    # ---- extinction tests ----
    tests=list(ALL_TESTS),
    recovery_bn_modes=["update", "frozen"], bn_recal_sizes=[10, 100, 1000, 10000],
    sr_epochs=5, sr_lr=0.01, sr_noise=[0.005, 0.01, 0.02, 0.05], sr_noise_reps=3,
    sr_quant_bits=[8, 6, 4], sr_prune=[0.2, 0.5, 0.7],
    ctx_synthetic=["gray", "blur", "noise", "contrast"],
    c_corruptions=["gaussian_noise", "motion_blur", "fog", "contrast", "jpeg_compression", "brightness"],
    c_severities=[3, 5], download_cifar_c=True,
    ri_epochs=3, ri_lr=0.01, ri_n_classes=4, ri_batch_size=128,
    sv_shots=[5, 20, 100], sv_steps=100, sv_lr=0.01, sv_replay_bs=128,
    sv_eval_steps=[0, 1, 2, 5, 10, 20, 50, 100],
    collapse_frac=0.5,
    # ---- follow-up 1.3: example-level forgetting ("xl") ----
    xl_frac=0.1, xl_draws=[0, 1, 2], xl_pilot_draw=9, xl_match_gap=0.03,
    xl_sv_shots=[100, 500], xl_calib_grid=None,
    # 1.4: "random" = a random share of the training set (run #9); "atypical" = the least consistent examples
    # (lowest C-score, Jiang et al. 2021), as in "From Dormant to Deleted" (Siddiqui et al. 2025)
    xl_design="random", xl_n=500, xl_pool_frac=0.05,
    cscore_url="https://pluskid.github.io/structural-regularity/cscores/cifar100-cscores-orig-order.npz",
    # ---- mechanism ----
    probe_epochs=30, probe_lr=1e-3, cka_n_retain=500,
    # ---- infrastructure ----
    smoke=False, smoke_train_per_class=30, smoke_test_per_class=10, smoke_epochs=1, smoke_width=16,
    time_budget_h=11.0, n_workers=None, retain_eval_subset=5000,
    reuse_ckpt=True, raise_on_error=True, poll_seconds=60,
)

# Settings that must be identical in every notebook (checked automatically).
SHARED_KEYS = ["seeds", "forget_classes", "pilot_class", "norm", "width", "epochs", "batch_size", "lr",
               "momentum", "weight_decay", "nesterov", "label_smoothing", "pct_start", "smoke",
               "smoke_train_per_class", "smoke_test_per_class", "ckpt_fp16"]


def make_cfg(**overrides):
    """Return a full config: DEFAULTS updated with `overrides` (typos raise an error)."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}. Check the spelling; "
                       f"valid keys are listed in ext_utils.DEFAULTS.")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(copy.deepcopy(overrides))
    # Testing hooks (used only for local dry runs; harmless on Kaggle).
    if os.environ.get("EXT_FORCE_SMOKE") == "1":
        cfg["smoke"] = True
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["forget_classes"] = cfg["forget_classes"][:2]
        cfg["aba_classes"] = [c for c in cfg["aba_classes"] if c in cfg["forget_classes"]][:1] \
            or cfg["forget_classes"][:1]
        cfg["mctx_classes"] = [c for c in cfg["mctx_classes"] if c in cfg["forget_classes"]][:1] \
            or cfg["forget_classes"][:1]
        cfg["calib_limit"] = 2
    if os.environ.get("EXT_FORCE_NORM"):
        cfg["norm"] = os.environ["EXT_FORCE_NORM"]
    if os.environ.get("EXT_N_WORKERS"):
        cfg["n_workers"] = int(os.environ["EXT_N_WORKERS"])
    if cfg["smoke"]:
        cfg.update(epochs=cfg["smoke_epochs"], width=cfg["smoke_width"], sr_epochs=1, ri_epochs=1,
                   sv_steps=5, sv_eval_steps=[0, 1, 5], probe_epochs=2, sr_noise_reps=1,
                   xl_draws=cfg["xl_draws"][:1], xl_sv_shots=[10, 30], xl_n=min(cfg["xl_n"], 60))
    bad = [c for c in cfg["aba_classes"] if c not in cfg["forget_classes"]]
    if bad:
        raise ValueError(f"aba_classes {bad} must also be in forget_classes.")
    bad = [c for c in cfg["mctx_classes"] if c not in cfg["forget_classes"]]
    if bad:
        raise ValueError(f"mctx_classes {bad} must also be in forget_classes.")
    bad = [c for c in cfg["mctx_contexts"] if c not in CONTEXTS]
    if bad:
        raise ValueError(f"Unknown mctx_contexts {bad}; choose from {CONTEXTS}.")
    if cfg["norm"] not in ("bn", "gn"):
        raise ValueError("norm must be 'bn' (BatchNorm) or 'gn' (GroupNorm).")
    bad = [m for m in cfg["recovery_bn_modes"] if m not in ("update", "frozen")]
    if bad or not cfg["recovery_bn_modes"]:
        raise ValueError("recovery_bn_modes must be a non-empty subset of ['update', 'frozen'].")
    bad = [m for m in cfg["methods"] if base_method(m) not in METHODS]
    if bad:
        raise ValueError(f"Unknown method(s) {bad}; choose from {METHODS}.")
    if cfg["pilot_class"] in cfg["forget_classes"]:
        raise ValueError("pilot_class must NOT be one of forget_classes (it is excluded from the main analysis).")
    if cfg["xl_design"] not in ("random", "atypical"):
        raise ValueError("xl_design must be 'random' or 'atypical'.")
    return cfg


def sfx(cfg):
    """File-name suffix that keeps GroupNorm-arm and smoke-test files separate from the main ones."""
    return ("_gn" if cfg["norm"] == "gn" else "") + ("_smoke" if cfg["smoke"] else "")


def config_file(cfg):
    return f"experiment_config{sfx(cfg)}.json"


def save_experiment_config(cfg):
    ensure_dirs()
    path = os.path.join(WORK, config_file(cfg))
    with open(path, "w") as f:
        json.dump({k: cfg[k] for k in SHARED_KEYS}, f, indent=2)
    return path


def check_experiment_config(cfg):
    """Make sure shared settings match notebook 01; then keep a copy in this notebook's output."""
    p = find_file(config_file(cfg))
    if p is None:
        print(f"[config] {config_file(cfg)} not found in inputs - cannot verify shared settings "
              f"(fine for notebook 01, otherwise attach notebook 01's output).")
    else:
        with open(p) as f:
            saved = json.load(f)
        diffs = {k: (saved[k], cfg[k]) for k in SHARED_KEYS if k in saved and saved[k] != cfg[k]}
        if diffs:
            lines = "\n".join(f"  {k}: notebook01={a!r}  this notebook={b!r}" for k, (a, b) in diffs.items())
            raise ValueError("Shared settings differ from notebook 01 (checkpoints would not match):\n" + lines)
        print(f"[config] shared settings match {p}")
    return save_experiment_config(cfg)


def stable_seed(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def short_hash(s, n=6):
    return hashlib.md5(str(s).encode()).hexdigest()[:n]


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def amp_on(cfg, device):
    return bool(cfg["amp"] and device.type == "cuda")


def ac(cfg, device):
    """Autocast context (fp16 mixed precision on GPU, no-op on CPU)."""
    if amp_on(cfg, device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ---- file index: finds checkpoints/results in WORK first, then anywhere in INPUT ----
_INDEX = {"t": 0.0, "map": {}}


def _build_index():
    m = {}
    if os.path.isdir(INPUT):
        for dp, _, fns in os.walk(INPUT, followlinks=True):
            for fn in fns:
                m.setdefault(fn, []).append(os.path.join(dp, fn))
    _INDEX.update(t=time.time(), map=m)


def input_paths(name, refresh=False):
    if refresh or not _INDEX["map"] and time.time() - _INDEX["t"] > 5:
        _build_index()
    return list(_INDEX["map"].get(name, []))


def find_file(name):
    for d in (CKPT_DIR, RES_DIR, WORK):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    paths = input_paths(name)
    if not paths and time.time() - _INDEX["t"] > 60:
        paths = input_paths(name, refresh=True)
    return paths[0] if paths else None


# =============================================================================
# 2. Data: CIFAR-100, CIFAR-100-C, augmentation, contexts
# =============================================================================
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def find_cifar100():
    for root in (DATA, INPUT):
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root, followlinks=True):
            if {"train", "test", "meta"} <= set(fns):
                try:
                    if "fine_label_names" in _unpickle(os.path.join(dp, "meta")):
                        return dp
                except Exception:
                    pass
    return None


def ensure_cifar100():
    """Find CIFAR-100 in DATA or any attached input; otherwise download it once.
    A downloaded copy is also kept in this notebook's output (data_cache/), so every later notebook that
    attaches this output finds it instantly instead of downloading again (cs.toronto.edu can be ~50 kB/s)."""
    d = find_cifar100()
    if d:
        return d
    os.makedirs(DATA, exist_ok=True)
    print("CIFAR-100 not attached - downloading from cs.toronto.edu (can take up to ~1 h on Kaggle). "
          "Faster: attach the Kaggle dataset 'fedesoriano/cifar100' (CIFAR-100 Python).", flush=True)
    try:
        import torchvision
        torchvision.datasets.CIFAR100(root=DATA, train=True, download=True)
    except Exception as e:
        raise RuntimeError(
            "CIFAR-100 was not found in the inputs and could not be downloaded.\n"
            "On Kaggle: open the right-hand panel -> Settings -> Internet -> On (needs a phone-verified "
            "account), OR click 'Add Input' and add a dataset that contains the 'cifar-100-python' folder.\n"
            f"Original error: {e!r}")
    d = find_cifar100()
    if d is None:
        raise RuntimeError("CIFAR-100 download finished but the folder was not found.")
    if ON_KAGGLE:
        dst = os.path.join(WORK, "data_cache", "cifar-100-python")
        try:
            if not os.path.exists(dst):
                shutil.copytree(d, dst + ".part", ignore=shutil.ignore_patterns("*.tar.gz"))
                os.replace(dst + ".part", dst)
                print(f"kept a copy of CIFAR-100 in {dst} (later notebooks attaching this output won't download)")
        except OSError as e:
            print(f"could not cache CIFAR-100 in the output: {e!r}")
    return d


_DATA = {}


def load_data(cfg):
    """CIFAR-100 as uint8 numpy arrays (N,3,32,32) plus fine/coarse labels and names."""
    key = (cfg["smoke"], cfg["smoke_train_per_class"], cfg["smoke_test_per_class"])
    if key in _DATA:
        return _DATA[key]
    d = ensure_cifar100()
    tr, te, meta = (_unpickle(os.path.join(d, n)) for n in ("train", "test", "meta"))
    xtr = np.asarray(tr["data"], np.uint8).reshape(-1, 3, 32, 32)
    ytr = np.asarray(tr["fine_labels"], np.int64)
    ctr = np.asarray(tr["coarse_labels"], np.int64)
    xte = np.asarray(te["data"], np.uint8).reshape(-1, 3, 32, 32)
    yte = np.asarray(te["fine_labels"], np.int64)
    test_orig = np.arange(len(yte))
    if cfg["smoke"]:
        itr = np.concatenate([np.where(ytr == k)[0][:cfg["smoke_train_per_class"]] for k in range(100)])
        ite = np.concatenate([np.where(yte == k)[0][:cfg["smoke_test_per_class"]] for k in range(100)])
        xtr, ytr, ctr = xtr[itr], ytr[itr], ctr[itr]
        xte, yte, test_orig = xte[ite], yte[ite], ite
    f2c = np.zeros(100, np.int64)
    f2c[ytr] = ctr
    D = dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte, f2c=f2c, test_orig_idx=test_orig,
             fine_names=list(meta["fine_label_names"]), coarse_names=list(meta["coarse_label_names"]))
    _DATA[key] = D
    return D


_GPU = {}


def gpu_data(D, cfg, device):
    """Moves the whole dataset to the GPU once (uint8, ~180 MB) so no CPU data loader is needed."""
    key = (id(D), str(device))
    if key in _GPU:
        return _GPU[key]
    G = dict(xtr=torch.from_numpy(D["xtr"]).to(device), ytr=torch.from_numpy(D["ytr"]).to(device),
             xte=torch.from_numpy(D["xte"]).to(device), yte=torch.from_numpy(D["yte"]).to(device))
    G["all_idx"] = torch.arange(len(D["ytr"]), device=device)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(D["ytr"]), size=min(cfg["retain_eval_subset"], len(D["ytr"])), replace=False)
    G["retain_eval_idx"] = torch.from_numpy(np.sort(sub)).to(device)
    _GPU[key] = G
    return G


def cid(D, name):
    try:
        return D["fine_names"].index(name)
    except ValueError:
        raise ValueError(f"Unknown CIFAR-100 class '{name}'. Valid names: {D['fine_names']}")


def siblings(D, c):
    return [k for k in range(100) if D["f2c"][k] == D["f2c"][c] and k != c]


class XL:
    """Example-level forgetting target (1.3): a random share of the training set to forget, drawn
    deterministically from (seed, draw). Passed wherever class-level code passes a class id c."""

    def __init__(self, seed, draw, fidx, ridx, prefix="xl"):
        self.seed, self.draw, self.fidx, self.ridx = seed, draw, fidx, ridx
        self.tag = f"{prefix}{draw}"

    def __repr__(self):
        return f"XL(seed={self.seed}, draw={self.draw}, n_forget={self.fidx.numel()})"


def xl_prefix(cfg):
    return "xa" if cfg["xl_design"] == "atypical" else "xl"


_CS = {}


def cscores(cfg, G):
    """C-scores (Jiang et al. 2021) of the CIFAR-100 training images, aligned with G["ytr"]: the expected accuracy of
    models that did not train on the image. Low = atypical (only memorisation gets it right). Looks for
    cifar100-cscores-orig-order.npz in the inputs or DATA, otherwise downloads it (Internet on). Smoke runs on data
    that does not match use deterministic pseudo-scores, clearly flagged."""
    y = G["ytr"].cpu().numpy()
    k = (len(y), cfg["smoke"])
    if k in _CS:
        return _CS[k]
    name = "cifar100-cscores-orig-order.npz"
    path = find_file(name) or os.path.join(DATA, name)
    if not os.path.exists(path):
        try:
            os.makedirs(DATA, exist_ok=True)
            req = urllib.request.Request(cfg["cscore_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(path + ".part", path)
        except Exception as e:
            path = None
            print(f"[C-scores] download failed: {e!r}")
    s = None
    if path and os.path.exists(path):
        z = np.load(path)
        if len(z["labels"]) == len(y) and np.array_equal(np.asarray(z["labels"], np.int64), y):
            s = np.asarray(z["scores"], np.float64)
    if s is None:
        if not cfg["smoke"]:
            raise RuntimeError("C-scores for CIFAR-100 could not be loaded or do not match the training labels. Turn "
                               f"Internet on, or attach a dataset containing {name} ({cfg['cscore_url']}).")
        print("[C-scores] smoke run: using deterministic pseudo-scores (the real file does not match smoke data)")
        s = np.random.RandomState(0).rand(len(y))
    _CS[k] = s
    return s


def xl_target(G, cfg, seed, draw):
    """Example-level forget set for (seed, draw). "random": a random xl_frac of the training set. "atypical": xl_n
    images drawn from the xl_pool_frac least consistent training images (lowest C-score)."""
    k = ("xl", cfg["xl_design"], seed, draw, cfg["xl_frac"], cfg["xl_n"], cfg["xl_pool_frac"])
    if k not in G:
        n = G["ytr"].numel()
        if cfg["xl_design"] == "atypical":
            pool = np.argsort(cscores(cfg, G), kind="stable")[:max(cfg["xl_n"], int(round(cfg["xl_pool_frac"] * n)))]
            rng = np.random.RandomState(stable_seed(f"xa|{seed}|{draw}"))
            f = np.sort(rng.choice(pool, min(cfg["xl_n"], len(pool)), replace=False))
        else:
            rng = np.random.RandomState(stable_seed(f"xl|{seed}|{draw}"))
            f = np.sort(rng.choice(n, int(round(cfg["xl_frac"] * n)), replace=False))
        mask = np.ones(n, bool)
        mask[f] = False
        dev = G["ytr"].device
        G[k] = XL(seed, draw, torch.from_numpy(f).to(dev), torch.from_numpy(np.nonzero(mask)[0]).to(dev),
                  prefix=xl_prefix(cfg))
    return G[k]


def forget_idx(G, c):
    if isinstance(c, XL):
        return c.fidx
    k = ("forget", c)
    if k not in G:
        G[k] = torch.nonzero(G["ytr"] == c).flatten()
    return G[k]


def retain_idx(G, c):
    if isinstance(c, XL):
        return c.ridx
    k = ("retain", c)
    if k not in G:
        G[k] = torch.nonzero(G["ytr"] != c).flatten()
    return G[k]


def validate_classes(cfg, D):
    for name in all_classes(cfg):
        cid(D, name)


def all_classes(cfg):
    return list(cfg["forget_classes"]) + ([cfg["pilot_class"]] if cfg["pilot_class"] else [])


_NORM, _BLUR = {}, {}


def _norm(device):
    k = str(device)
    if k not in _NORM:
        _NORM[k] = (torch.tensor(MEAN, device=device).view(1, 3, 1, 1),
                    torch.tensor(STD, device=device).view(1, 3, 1, 1))
    return _NORM[k]


def augment(x):
    """Random crop (4-pixel zero padding) + horizontal flip, done on the GPU for a whole batch."""
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None, :])[:, :, None]           # B,32,1
    cols = (j[:, None] + ar[None, :])[:, None, :]           # B,1,32
    bidx = torch.arange(B, device=x.device)[:, None, None]  # B,1,1
    out = xp[bidx, :, rows, cols].permute(0, 3, 1, 2)       # B,3,32,32
    flip = torch.rand(B, device=x.device) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous()


CONTEXTS = ("clean", "gray", "blur", "noise", "contrast")


def apply_context(x, ctx):
    """Deterministic input 'contexts' (x in [0,1]). Used for renewal tests and ABA unlearning."""
    if ctx in (None, "clean"):
        return x
    if ctx == "gray":
        g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        return g.expand(-1, 3, -1, -1).contiguous()
    if ctx == "blur":
        k = str(x.device)
        if k not in _BLUR:
            ax = torch.arange(5, device=x.device, dtype=torch.float32) - 2
            g1 = torch.exp(-ax ** 2 / 2.0)
            g2 = torch.outer(g1, g1)
            _BLUR[k] = (g2 / g2.sum()).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), _BLUR[k].to(x.dtype), groups=3)
    if ctx == "noise":
        g = torch.Generator(device=x.device)
        g.manual_seed(1234)
        return (x + 0.08 * torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)).clamp(0, 1)
    if ctx == "contrast":
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return (x - m) * 0.4 + m
    raise ValueError(f"Unknown context '{ctx}'. Choose from {CONTEXTS}.")


def apply_mixed_context(x, ctxs):
    """Multi-context unlearning: every image gets a random context from `ctxs`."""
    idx = torch.randint(0, len(ctxs), (x.shape[0],), device=x.device)
    out = x.clone()
    for k, c in enumerate(ctxs):
        mk = idx == k
        if mk.any():
            out[mk] = apply_context(x[mk], c)
    return out


def prep(xb, cfg, augment_=False, ctx="clean"):
    """uint8 batch on device -> normalised float batch (optionally augmented / in a context).
    `ctx` is one context name, or a list of names (random context per image)."""
    x = xb.float().div_(255.0)
    if augment_:
        x = augment(x)
    x = apply_mixed_context(x, list(ctx)) if isinstance(ctx, (list, tuple)) else apply_context(x, ctx)
    mean, std = _norm(x.device)
    x = (x - mean) / std
    if cfg["channels_last"]:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def batches(idx, bs, shuffle=True, drop_last=False):
    n = idx.numel()
    if n == 0:
        return
    order = idx[torch.randperm(n, device=idx.device)] if shuffle else idx
    end = (n // bs) * bs if (drop_last and n >= bs) else n
    for s in range(0, end, bs):
        b = order[s:s + bs]
        if b.numel() < 2 <= n:          # BatchNorm cannot train on a single sample
            continue
        yield b


def n_batches(n, bs):
    """Number of batches produced by batches(..., drop_last=True)."""
    return max(1, n // bs)


def _cycle(idx, bs):
    while True:
        got = False
        for b in batches(idx, bs, shuffle=True):
            got = True
            yield b
        if not got:
            raise RuntimeError("Tried to cycle over an empty index set.")


# ---- CIFAR-100-C ----
CIFAR100C_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
_CC = {"checked": False, "dir": None, "arrays": {}, "warned": set()}


def find_cifar_c():
    for root in (DATA, INPUT):
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root, followlinks=True):
            if "labels.npy" in fns and any(f.endswith(".npy") and f != "labels.npy" for f in fns):
                try:
                    lab = np.load(os.path.join(dp, "labels.npy"), mmap_mode="r")
                    if len(lab) == 50000 and int(lab.max()) > 9:      # CIFAR-100-C, not CIFAR-10-C
                        return dp
                except Exception:
                    pass
    return None


def download_cifar_c(corruptions, log=print):
    out = os.path.join(DATA, "CIFAR-100-C")
    os.makedirs(out, exist_ok=True)
    want = {f"{c}.npy" for c in corruptions} | {"labels.npy"}
    want = {f for f in want if not os.path.exists(os.path.join(out, f))}
    if want:
        log(f"Downloading CIFAR-100-C from Zenodo (streams ~2.9 GB, keeps only {sorted(want)}) ...")
        try:
            req = urllib.request.Request(CIFAR100C_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with tarfile.open(fileobj=resp, mode="r|*") as tar:
                    for m in tar:
                        base = os.path.basename(m.name)
                        if m.isfile() and base in want:
                            src = tar.extractfile(m)
                            tmp = os.path.join(out, base + ".part")
                            with open(tmp, "wb") as f:
                                shutil.copyfileobj(src, f, 1 << 20)
                            os.replace(tmp, os.path.join(out, base))
                            want.discard(base)
                            log(f"  extracted {base}")
                            if not want:
                                break
        except Exception as e:
            log(f"CIFAR-100-C download failed: {e!r}")
    return out if os.path.exists(os.path.join(out, "labels.npy")) else None


def prefetch_cifar_c(cfg, log=print):
    """Find (or download once) CIFAR-100-C. Returns its folder or None."""
    if not _CC["checked"]:
        _CC["checked"] = True
        d = find_cifar_c()
        if d is None and cfg["download_cifar_c"]:
            d = download_cifar_c(cfg["c_corruptions"], log)
        _CC["dir"] = d
        log(f"[CIFAR-100-C] {'using ' + d if d else 'NOT available - only synthetic contexts will run'}")
    return _CC["dir"]


def get_cifar_c(cfg, D, device, log=print):
    """{(corruption, severity): uint8 tensor aligned with the test set} (empty if unavailable)."""
    d = prefetch_cifar_c(cfg, log)
    out = {}
    if d is None:
        return out
    labels = np.load(os.path.join(d, "labels.npy"))
    for corr in cfg["c_corruptions"]:
        p = os.path.join(d, f"{corr}.npy")
        if not os.path.exists(p):
            if corr not in _CC["warned"]:
                _CC["warned"].add(corr)
                log(f"[CIFAR-100-C] {corr}.npy missing - skipped")
            continue
        arr = None
        for sev in cfg["c_severities"]:
            k = (corr, sev, str(device), cfg["smoke"])
            if k not in _CC["arrays"]:
                if arr is None:
                    arr = np.load(p, mmap_mode="r")
                lo = (sev - 1) * 10000
                lab = labels[lo:lo + 10000][D["test_orig_idx"]]
                if not np.array_equal(lab, D["yte"]):
                    log(f"[CIFAR-100-C] label mismatch for {corr} severity {sev} - skipped")
                    continue
                x = np.ascontiguousarray(arr[lo:lo + 10000][D["test_orig_idx"]].transpose(0, 3, 1, 2))
                _CC["arrays"][k] = torch.from_numpy(x).to(device)
            out[(corr, sev)] = _CC["arrays"][k]
    return out


# =============================================================================
# 3. Model, training, evaluation, checkpoints
# =============================================================================
def _gn_groups(c):
    for g in (32, 16, 8, 4, 2):
        if c % g == 0 and c // g >= 2:
            return g
    return 1


def norm_layer(norm, c):
    """'bn' = BatchNorm (main experiments), 'gn' = GroupNorm (control arm without running statistics).
    Attribute names stay bn1/bn2 in both cases so checkpoints and analyses share one layout."""
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm == "gn":
        return nn.GroupNorm(_gn_groups(c), c)
    raise ValueError(f"Unknown norm '{norm}'")


class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1, norm="bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(norm, out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(norm, out)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), norm_layer(norm, out))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18, CIFAR variant: 3x3 stem, no max-pool (standard in the unlearning literature)."""

    def __init__(self, num_classes=100, width=64, norm="bn"):
        super().__init__()
        w = width
        self.norm = norm
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = norm_layer(norm, w)
        self.layer1 = self._make(w, w, 1, norm)
        self.layer2 = self._make(w, 2 * w, 2, norm)
        self.layer3 = self._make(2 * w, 4 * w, 2, norm)
        self.layer4 = self._make(4 * w, 8 * w, 2, norm)
        self.fc = nn.Linear(8 * w, num_classes)

    @staticmethod
    def _make(inp, out, stride, norm):
        return nn.Sequential(BasicBlock(inp, out, stride, norm), BasicBlock(out, out, 1, norm))

    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))

    def block_outputs(self, x):
        out = {}
        x = F.relu(self.bn1(self.conv1(x)))
        out["stem"] = x
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(self, name)(x)
            out[name] = x
        p = F.adaptive_avg_pool2d(x, 1).flatten(1)
        out["penult"] = p
        out["logits"] = self.fc(p)
        return out


def make_model(cfg, device, width=None, norm=None):
    m = ResNet18(100, width or cfg["width"], norm or cfg["norm"]).to(device)
    if cfg["channels_last"]:
        m = m.to(memory_format=torch.channels_last)
    return m


def ckpt_name(kind, cfg, cls=None, seed=None, method=None):
    parts = [kind]
    if method:
        parts.append(method.replace("+", "plus"))
    if cls:
        parts.append(cls)
    parts.append(f"s{seed}")
    return "_".join(parts) + sfx(cfg) + ".pt"


def roundtrip(model, cfg):
    """Round weights to fp16 and back so reported metrics equal those of the saved checkpoint."""
    if cfg["ckpt_fp16"]:
        with torch.no_grad():
            for v in model.state_dict().values():
                if v.is_floating_point():
                    v.copy_(v.half().float())
    return model


def save_model(model, name, cfg, meta=None):
    ensure_dirs()
    sd = {k: (v.detach().half().cpu() if (cfg["ckpt_fp16"] and v.is_floating_point()) else v.detach().cpu())
          for k, v in model.state_dict().items()}
    path = os.path.join(CKPT_DIR, name)
    tmp = path + ".tmp"
    torch.save({"state_dict": sd, "meta": dict(meta or {}, width=model.fc.in_features // 8, norm=model.norm),
                "version": VERSION}, tmp)
    os.replace(tmp, path)
    return path


def load_model(name, cfg, device):
    path = name if os.path.isfile(name) else find_file(name)
    if path is None:
        raise FileNotFoundError(
            f"Checkpoint '{name}' not found in {CKPT_DIR} or anywhere under {INPUT}. "
            f"Attach the notebook output / dataset that contains it ('Add Input').")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    meta = obj.get("meta", {})
    model = make_model(cfg, device, width=meta.get("width", cfg["width"]), norm=meta.get("norm", cfg["norm"]))
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in obj["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def _sgd(model, lr, cfg, wd=None):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg["momentum"],
                           weight_decay=cfg["weight_decay"] if wd is None else wd)


def _step(loss, opt, scaler, model, clip=None, masks=None, wd_manual=0.0):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if clip or masks is not None:
        scaler.unscale_(opt)
        if masks is not None:
            for p, m in masks:
                if p.grad is not None:
                    if wd_manual:
                        p.grad.add_(p.detach(), alpha=wd_manual)
                    p.grad.mul_(m)
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(opt)
    scaler.update()


def train_model(cfg, G, train_idx, seed, device, log=print, tag=""):
    """Train ResNet-18 from scratch (used for original and retrained models)."""
    seed_everything(seed)
    model = make_model(cfg, device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"], nesterov=cfg["nesterov"])
    bs = cfg["batch_size"]
    total = cfg["epochs"] * n_batches(train_idx.numel(), bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total,
                                                pct_start=cfg["pct_start"])
    scaler = make_scaler(amp_on(cfg, device))
    steps = 0
    for ep in range(cfg["epochs"]):
        model.train()
        t = time.time()
        tot = torch.zeros((), device=device)
        n = 0
        for b in batches(train_idx, bs, shuffle=True, drop_last=True):
            x = prep(G["xtr"][b], cfg, augment_=True)
            y = G["ytr"][b]
            with ac(cfg, device):
                loss = F.cross_entropy(model(x), y, label_smoothing=cfg["label_smoothing"])
            _step(loss, opt, scaler, model)
            if steps < total:
                sched.step()
                steps += 1
            tot += loss.detach().float() * b.numel()
            n += b.numel()
        if (ep + 1) % max(1, cfg["epochs"] // 5) == 0 or ep == cfg["epochs"] - 1:
            acc = float((logits_of(model, G["xte"], cfg, device).argmax(1) == G["yte"]).float().mean())
            log(f"{tag} epoch {ep + 1}/{cfg['epochs']} loss {float(tot) / max(n, 1):.3f} "
                f"test_acc {acc:.4f} ({time.time() - t:.1f}s/epoch)")
    return model


@torch.no_grad()
def logits_of(model, x, cfg, device, ctx="clean"):
    model.eval()
    out = []
    bs = cfg["eval_batch_size"]
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        if isinstance(xb, np.ndarray):
            xb = torch.from_numpy(np.ascontiguousarray(xb))
        xb = prep(xb.to(device, non_blocking=True), cfg, ctx=ctx)
        with ac(cfg, device):
            out.append(model(xb).float())
    return torch.cat(out)


# Forget-class response measures, all "higher = more of the forgotten class". The primary one for
# spontaneous recovery, renewal and reinstatement is top-5 (v1.2): top-1 has a floor, since recovery
# only registers once the forgotten class beats all 99 others (pilot run #4: exactly 0 everywhere).
PRIMARY_METRIC = "top5_cc"
RESPONSE_METRICS = ["top5_cc", "chance_corrected", "rank_score", "forget_auc"]


@torch.no_grad()
def metrics(model, G, D, c, cfg, device, ctx="clean", x_test=None, full=False):
    """Accuracy metrics for forget class c.

    forget_test_acc  = recall of the forget class on its test images (top-1)
    forget_fpr       = share of *other* test images wrongly labelled as the forget class
    chance_corrected = forget_test_acc - forget_fpr  (specific recovery, immune to label-bias drift)
    top5_cc          = same with "forget class among the top 5" instead of top-1 (graded, primary)
    forget_rank      = median rank of the forget class on its test images (1 = top, 100 = last);
                       rank_score = 1 - (forget_rank - 1) / 99 rescales it to 0..1
    forget_auc       = AUROC of the forget-class logit, forget images vs all others (0.5 = chance).
                       A uniform suppression of that output leaves it unchanged, so it reads out
                       whether the network still singles out the class.
    """
    x = G["xte"] if x_test is None else x_test
    logits = logits_of(model, x, cfg, device, ctx)
    pred = logits.argmax(1)
    y = G["yte"]
    f = y == c
    r = ~f
    sib = torch.zeros(100, dtype=torch.bool, device=y.device)
    sib[siblings(D, c)] = True
    nb = sib[y]
    un = r & ~nb
    corr = (pred == y).float()
    m = dict(test_acc=corr.mean(), forget_test_acc=corr[f].mean(), retain_test_acc=corr[r].mean(),
             forget_fpr=(pred[r] == c).float().mean(), neighbour_test_acc=corr[nb].mean(),
             unrelated_test_acc=corr[un].mean())
    lc = logits[:, c]
    rank = (logits > lc[:, None]).sum(1) + 1
    top5 = rank <= 5
    order = lc.argsort()
    ranks = torch.empty(lc.numel(), dtype=torch.float64, device=lc.device)
    ranks[order] = torch.arange(1, lc.numel() + 1, dtype=torch.float64, device=lc.device)
    n_pos, n_neg = int(f.sum()), int(r.sum())
    m.update(forget_top5=top5[f].float().mean(), forget_top5_fpr=top5[r].float().mean(),
             forget_rank=rank[f].float().median(),
             forget_auc=(ranks[f].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))
    m = {k: float(v) for k, v in m.items()}
    m["chance_corrected"] = m["forget_test_acc"] - m["forget_fpr"]
    m["top5_cc"] = m["forget_top5"] - m["forget_top5_fpr"]
    m["rank_score"] = 1.0 - (m["forget_rank"] - 1.0) / 99.0
    m["finite"] = bool(torch.isfinite(logits).all())
    if full:
        fi = forget_idx(G, c)
        pf = logits_of(model, G["xtr"][fi], cfg, device, ctx).argmax(1)
        m["forget_train_acc"] = float((pf == c).float().mean())
        ri = G["retain_eval_idx"]
        ri = ri[G["ytr"][ri] != c]
        pr = logits_of(model, G["xtr"][ri], cfg, device, ctx).argmax(1)
        m["retain_train_acc"] = float((pr == G["ytr"][ri]).float().mean())
    return m


# =============================================================================
# 4. Unlearning methods (all modify `model` in place)
# =============================================================================
def unlearn_ft(model, G, D, c, hp, cfg, device, ctx="clean"):
    """FT: fine-tune on the retain set only (also our 'natural forgetting' arm)."""
    ridx = retain_idx(G, c)
    bs = cfg["batch_size"]
    opt = _sgd(model, hp["lr"], cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"] * n_batches(ridx.numel(), bs))
    scaler = make_scaler(amp_on(cfg, device))
    for _ in range(hp["epochs"]):
        model.train()
        for b in batches(ridx, bs, drop_last=True):
            x = prep(G["xtr"][b], cfg, True, ctx)
            with ac(cfg, device):
                loss = F.cross_entropy(model(x), G["ytr"][b])
            _step(loss, opt, scaler, model)
            sched.step()
    return model


def unlearn_neggrad(model, G, D, c, hp, cfg, device, ctx="clean"):
    """NegGrad+: descend on retain loss, ascend on forget loss: a*CE_r - (1-a)*CE_f."""
    ridx, fidx = retain_idx(G, c), forget_idx(G, c)
    bs, a = cfg["batch_size"], hp["alpha"]
    fcyc = _cycle(fidx, cfg["forget_batch_size"])
    opt = _sgd(model, hp["lr"], cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"] * n_batches(ridx.numel(), bs))
    scaler = make_scaler(amp_on(cfg, device))
    for _ in range(hp["epochs"]):
        model.train()
        for b in batches(ridx, bs, drop_last=True):
            fb = next(fcyc)
            x = prep(torch.cat([G["xtr"][b], G["xtr"][fb]]), cfg, True, ctx)
            with ac(cfg, device):
                out = model(x)
                n = b.numel()
                loss = a * F.cross_entropy(out[:n], G["ytr"][b]) - (1 - a) * F.cross_entropy(out[n:], G["ytr"][fb])
            _step(loss, opt, scaler, model, clip=cfg["clip_grad"])
            sched.step()
    return model


def _random_labels(n, c, device):
    r = torch.randint(0, 99, (n,), device=device)
    return r + (r >= c).long()          # uniform over the 99 classes that are not c


def _random_other_labels(y):
    r = torch.randint(0, 99, y.shape, device=y.device)
    return r + (r >= y).long()          # uniform over the 99 classes that are not the example's own


def _rl_loop(model, G, c, hp, cfg, device, ctx, masks=None):
    ridx, fidx = retain_idx(G, c), forget_idx(G, c)
    bs = cfg["batch_size"]
    fcyc = _cycle(fidx, cfg["forget_batch_size"])
    wd = cfg["weight_decay"]
    opt = _sgd(model, hp["lr"], cfg, wd=0.0 if masks is not None else None)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"] * n_batches(ridx.numel(), bs))
    scaler = make_scaler(amp_on(cfg, device))
    for _ in range(hp["epochs"]):
        model.train()
        for b in batches(ridx, bs, drop_last=True):
            fb = next(fcyc)
            x = prep(torch.cat([G["xtr"][b], G["xtr"][fb]]), cfg, True, ctx)
            yr = _random_other_labels(G["ytr"][fb]) if isinstance(c, XL) else _random_labels(fb.numel(), c, device)
            y = torch.cat([G["ytr"][b], yr])
            with ac(cfg, device):
                loss = F.cross_entropy(model(x), y)
            _step(loss, opt, scaler, model, masks=masks, wd_manual=wd if masks is not None else 0.0)
            sched.step()
    return model


def unlearn_rl(model, G, D, c, hp, cfg, device, ctx="clean"):
    """Random-Label: train forget images with random wrong labels, mixed with retain data."""
    return _rl_loop(model, G, c, hp, cfg, device, ctx)


def _kd(s, t, T):
    return F.kl_div(F.log_softmax(s.float() / T, 1), F.softmax(t.float() / T, 1), reduction="batchmean") * (T * T)


def unlearn_scrub(model, G, D, c, hp, cfg, device, ctx="clean"):
    """SCRUB (Kurmanji et al. 2023): student moves away from teacher on forget, stays close on retain."""
    ridx, fidx = retain_idx(G, c), forget_idx(G, c)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    T, alpha, gamma = hp.get("T", 4.0), hp.get("alpha", 0.001), hp.get("gamma", 0.99)
    opt = _sgd(model, hp["lr"], cfg)
    scaler = make_scaler(amp_on(cfg, device))
    for ep in range(hp["epochs"]):
        model.train()
        if ep < hp["msteps"]:
            for fb in batches(fidx, cfg["forget_batch_size"]):
                x = prep(G["xtr"][fb], cfg, True, ctx)
                with ac(cfg, device):
                    s = model(x)
                    with torch.no_grad():
                        t = teacher(x)
                    loss = -_kd(s, t, T)
                _step(loss, opt, scaler, model, clip=cfg["clip_grad"])
        for b in batches(ridx, cfg["batch_size"], drop_last=True):
            x = prep(G["xtr"][b], cfg, True, ctx)
            with ac(cfg, device):
                s = model(x)
                with torch.no_grad():
                    t = teacher(x)
                loss = gamma * F.cross_entropy(s, G["ytr"][b]) + alpha * _kd(s, t, T)
            _step(loss, opt, scaler, model)
    del teacher
    return model


def salun_masks(model, G, c, ratio, cfg, device, ctx="clean"):
    """SalUn weight-saliency mask: keep the `ratio` share of weights with the largest forget-loss gradient."""
    model.eval()
    model.zero_grad(set_to_none=True)
    for b in batches(forget_idx(G, c), 256, shuffle=False):
        x = prep(G["xtr"][b], cfg, False, ctx)
        F.cross_entropy(model(x), G["ytr"][b], reduction="sum").backward()
    params = [p for p in model.parameters() if p.grad is not None]
    flat = torch.cat([p.grad.detach().abs().flatten() for p in params])
    k = min(flat.numel(), max(1, int(round((1 - ratio) * flat.numel()))))
    thr = flat.kthvalue(k).values
    masks = [(p, (p.grad.detach().abs() >= thr).to(p.dtype)) for p in params]
    model.zero_grad(set_to_none=True)
    return masks


def unlearn_salun(model, G, D, c, hp, cfg, device, ctx="clean"):
    """SalUn (Fan et al. 2024): Random-Label updates restricted to salient weights."""
    masks = salun_masks(model, G, c, hp.get("mask_ratio", 0.5), cfg, device, ctx)
    return _rl_loop(model, G, c, hp, cfg, device, ctx, masks=masks)


UNLEARN_FNS = {"FT": unlearn_ft, "NegGrad+": unlearn_neggrad, "RL": unlearn_rl,
               "SCRUB": unlearn_scrub, "SalUn": unlearn_salun}


def run_unlearning(method, model, G, D, c, hp, cfg, device, ctx="clean"):
    method = base_method(method)
    if method not in UNLEARN_FNS:
        raise ValueError(f"Unknown method {method}; choose from {list(UNLEARN_FNS)}")
    if cfg["smoke"]:                                  # dry runs only: keep them short
        hp = dict(hp, epochs=min(hp["epochs"], 2))
        if "msteps" in hp:
            hp["msteps"] = min(hp["msteps"], 1)
    model.train()
    UNLEARN_FNS[method](model, G, D, c, hp, cfg, device, ctx)
    model.eval()
    return model


def is_matched(m, ref, cfg):
    """Output-level forgetting as good as retraining (the 'matching rule')."""
    return bool(m["forget_test_acc"] <= cfg["match_forget_max"]
                and m["retain_test_acc"] >= ref["retain_test_acc"] - cfg["match_retain_gap"])


# =============================================================================
# 5. Extinction tests
# =============================================================================
def _weight_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.dim() > 1]     # conv + linear weights


def _avg(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def bn_modes(model, cfg):
    """How a fine-tuned model is evaluated (v1.2). Training is always normal (BN in train mode), then
    every checkpoint is scored twice: 'update' = with the running statistics that training produced;
    'frozen' = with the original running statistics put back, so only the weight change counts.
    (v1.1 trained with BN layers in eval mode instead; that collapsed several models to chance in the
    pilot, run #4.) GroupNorm models have no running statistics, so only 'update' is reported."""
    return list(cfg["recovery_bn_modes"]) if bn_layers(model) else ["update"]


def bn_state(model):
    return [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
            for m in bn_layers(model)]


def _load_bn_state(model, state):
    for m, (mean, var, n) in zip(bn_layers(model), state):
        m.running_mean.copy_(mean)
        m.running_var.copy_(var)
        m.num_batches_tracked.copy_(n)


@contextlib.contextmanager
def bn_stats_swapped(model, state):
    """Temporarily use the BN running statistics `state` (weights unchanged); restores them afterwards."""
    if not state or not bn_layers(model):
        yield model
        return
    current = bn_state(model)
    _load_bn_state(model, state)
    try:
        yield model
    finally:
        _load_bn_state(model, current)


def eval_modes(model, G, D, c, cfg, device, orig_state, mfn=None):
    """[(bn_mode, metrics)] for a fine-tuned model: updated statistics, and original statistics restored.
    mfn(model) replaces the class-level metrics (used by the example-level arm)."""
    M = mfn or (lambda mm: metrics(mm, G, D, c, cfg, device))
    out = []
    for mode in bn_modes(model, cfg):
        if mode == "frozen":
            with bn_stats_swapped(model, orig_state):
                out.append((mode, M(model)))
        else:
            out.append((mode, M(model)))
    return out


def _collapsed(m, base_retain, cfg):
    """A fine-tuning row whose retain accuracy fell below collapse_frac x its starting value is unusable
    (the network diverged; recovery numbers from it mean nothing). Flagged and excluded downstream."""
    return bool((not m.get("finite", True)) or m["retain_test_acc"] < cfg["collapse_frac"] * base_retain)


@torch.no_grad()
def bn_recalibrate(model, G, idx, n, cfg, seed):
    """Weight-free control ('BatchNorm illusion', Kalani et al. 2026): recompute BN running statistics
    from n retain images; no weight changes. Returns the number of images actually used."""
    layers = bn_layers(model)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    sel = idx[torch.randperm(idx.numel(), generator=g)[:n].to(idx.device)]
    if not layers or sel.numel() < 2:
        return int(sel.numel())
    moms = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None                     # cumulative average over the given images
    model.train()
    device = sel.device
    for s in range(0, sel.numel(), cfg["batch_size"]):
        b = sel[s:s + cfg["batch_size"]]
        if b.numel() < 2:
            continue
        with ac(cfg, device):
            model(prep(G["xtr"][b], cfg))
    for m, mo in zip(layers, moms):
        m.momentum = mo
    model.eval()
    return int(sel.numel())


def _train_epochs(m, G, idx, bs, lr, epochs, cfg, device, after_epoch=None):
    """Normal fine-tuning (BN in train mode) with gradient clipping, as in the unlearning methods."""
    opt = _sgd(m, lr, cfg)
    scaler = make_scaler(amp_on(cfg, device))
    for ep in range(1, epochs + 1):
        m.train()
        for b in batches(idx, bs, drop_last=True):
            x = prep(G["xtr"][b], cfg, True)
            with ac(cfg, device):
                loss = F.cross_entropy(m(x), G["ytr"][b])
            _step(loss, opt, scaler, m, clip=cfg["clip_grad"])
        if after_epoch:
            after_epoch(ep)


@torch.no_grad()
def fake_quant(model, bits):
    qmax = 2 ** (bits - 1) - 1
    for _, p in _weight_params(model):
        w = p.data
        scale = w.reshape(w.shape[0], -1).abs().amax(1).clamp_min(1e-12) / qmax
        shape = [-1] + [1] * (w.dim() - 1)
        s = scale.view(shape)
        p.data.copy_(torch.round(w / s).clamp_(-qmax, qmax) * s)


@torch.no_grad()
def magnitude_prune(model, ratio):
    ws = [p for _, p in _weight_params(model)]
    flat = torch.cat([p.detach().abs().flatten() for p in ws])
    k = min(flat.numel(), max(1, int(ratio * flat.numel())))
    thr = flat.kthvalue(k).values
    for p in ws:
        p.mul_((p.abs() > thr).to(p.dtype))


@torch.no_grad()
def add_weight_noise(model, sigma, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    for _, p in _weight_params(model):
        noise = torch.randn(p.shape, generator=g).to(p.device, p.dtype)
        p.add_(noise * (sigma * p.std()))


def test_spontaneous(model, G, D, c, cfg, device, seed, mfn=None):
    """Spontaneous recovery: 'passage of time' proxies that never see forget data."""
    M = mfn or (lambda mm: metrics(mm, G, D, c, cfg, device))
    rows = []
    base_retain = M(model)["retain_test_acc"]
    m = copy.deepcopy(model)
    orig = bn_state(m)

    def after(ep):
        for mode, met in eval_modes(m, G, D, c, cfg, device, orig, mfn):
            rows.append(dict(test="spontaneous", proxy="retain_ft", bn=mode, level=ep,
                             collapsed=_collapsed(met, base_retain, cfg), **met))

    _train_epochs(m, G, retain_idx(G, c), cfg["batch_size"], cfg["sr_lr"], cfg["sr_epochs"], cfg, device,
                  after_epoch=after)
    del m
    if bn_layers(model):                                   # BatchNorm control: statistics only, no weights
        for n in cfg["bn_recal_sizes"]:
            m = copy.deepcopy(model)
            used = bn_recalibrate(m, G, retain_idx(G, c), n, cfg, stable_seed(f"bnrecal|{seed}|{n}"))
            rows.append(dict(test="spontaneous", proxy="bn_recal", bn="recal", level=used, **M(m)))
            del m
    for sigma in cfg["sr_noise"]:
        ms = []
        for rep in range(cfg["sr_noise_reps"]):
            m = copy.deepcopy(model)
            add_weight_noise(m, sigma, stable_seed(f"noise|{seed}|{sigma}|{rep}"))
            ms.append(M(m))
            del m
        rows.append(dict(test="spontaneous", proxy="noise", bn="eval", level=sigma, **_avg(ms)))
    for bits in cfg["sr_quant_bits"]:
        m = copy.deepcopy(model)
        fake_quant(m, bits)
        rows.append(dict(test="spontaneous", proxy="quant", bn="eval", level=bits, **M(m)))
        del m
    for ratio in cfg["sr_prune"]:
        m = copy.deepcopy(model)
        magnitude_prune(m, ratio)
        rows.append(dict(test="spontaneous", proxy="prune", bn="eval", level=ratio, **M(m)))
        del m
    return rows


def test_contexts(model, G, D, c, cfg, device, log=print):
    """Renewal: evaluate in synthetic contexts and CIFAR-100-C corruptions (no training)."""
    rows = []
    ctxs = list(dict.fromkeys(list(cfg["ctx_synthetic"]) + [cfg["aba_context"]]))
    for ctx in ctxs:
        rows.append(dict(test="context", context=ctx, severity=0, source="synthetic",
                         **metrics(model, G, D, c, cfg, device, ctx=ctx)))
    for (corr, sev), x in get_cifar_c(cfg, D, device, log).items():
        rows.append(dict(test="context", context=corr, severity=sev, source="cifar100c",
                         **metrics(model, G, D, c, cfg, device, x_test=x)))
    return rows


_ORIG = {}


def get_original(cfg, seed, device):
    k = (seed, str(device), cfg["smoke"])
    if k not in _ORIG:
        _ORIG.clear()
        _ORIG[k] = load_model(ckpt_name("original", cfg, seed=seed), cfg, device)
    return _ORIG[k]


_SIM = {}


def class_similarity(cfg, G, D, c, seed, device):
    """Original model's mean softmax over forget-class training images (a graded relatedness score)."""
    k = (c, seed, str(device), cfg["smoke"])
    if k not in _SIM:
        orig = get_original(cfg, seed, device)
        p = F.softmax(logits_of(orig, G["xtr"][forget_idx(G, c)], cfg, device), 1).mean(0)
        _SIM[k] = p.cpu().numpy()
    return _SIM[k]


def test_reinstatement(model, G, D, c, cfg, device, seed):
    """Reinstatement: brief training on *related* classes (never on the forget class)."""
    sim = class_similarity(cfg, G, D, c, seed, device)
    sib = siblings(D, c)
    others = sorted([k for k in range(100) if k != c and k not in sib], key=lambda k: -sim[k])
    n = cfg["ri_n_classes"]
    conds = {"sibling": sib[:n], "similar": others[:n], "dissimilar": others[-n:]}
    ytr = D["ytr"]
    used = set(sum(conds.values(), [])) | {c}
    pool = np.where(~np.isin(ytr, list(used)))[0]
    cue_size = max(int(np.isin(ytr, v).sum()) for v in conds.values())
    rng = np.random.RandomState(stable_seed(f"replay|{c}|{seed}"))
    replay = rng.choice(pool, size=min(cue_size, len(pool)), replace=False)
    rows = []
    base = metrics(model, G, D, c, cfg, device)
    for cond, classes in conds.items():
        cue = np.where(np.isin(ytr, classes))[0]
        idx = torch.from_numpy(np.concatenate([cue, replay])).to(device)
        info = dict(test="reinstatement", condition=cond, relatedness=float(np.mean(sim[classes])),
                    cue_classes=",".join(D["fine_names"][k] for k in classes))
        for mode in bn_modes(model, cfg):
            rows.append(dict(info, bn=mode, level=0, collapsed=False, **base))
        m = copy.deepcopy(model)
        orig = bn_state(m)

        def after(ep, m=m, orig=orig, info=info):
            for mode, met in eval_modes(m, G, D, c, cfg, device, orig):
                rows.append(dict(info, bn=mode, level=ep, collapsed=_collapsed(met, base["retain_test_acc"], cfg),
                                 **met))

        _train_epochs(m, G, idx, cfg["ri_batch_size"], cfg["ri_lr"], cfg["ri_epochs"], cfg, device, after_epoch=after)
        del m
    return rows


def test_savings(model, G, D, c, cfg, device, seed):
    """Rapid reacquisition: k-shot relearning of the forget class, with retain replay."""
    fidx, ridx = forget_idx(G, c), retain_idx(G, c)
    rng = np.random.RandomState(stable_seed(f"shots|{c}|{seed}"))
    perm = torch.from_numpy(rng.permutation(fidx.numel())).to(device)
    evals = set(cfg["sv_eval_steps"])
    rows = []
    base = metrics(model, G, D, c, cfg, device)
    for k in cfg["sv_shots"]:
        kk = min(k, fidx.numel())
        shots = fidx[perm[:kk]]
        m = copy.deepcopy(model)
        orig = bn_state(m)
        opt = _sgd(m, cfg["sv_lr"], cfg)
        scaler = make_scaler(amp_on(cfg, device))
        g = torch.Generator(device="cpu")
        g.manual_seed(stable_seed(f"replaybatch|{c}|{seed}|{k}"))
        if 0 in evals:
            for mode in bn_modes(model, cfg):
                rows.append(dict(test="savings", shots=kk, bn=mode, level=0, collapsed=False, **base))
        for step in range(1, cfg["sv_steps"] + 1):
            m.train()
            rb = ridx[torch.randint(0, ridx.numel(), (cfg["sv_replay_bs"],), generator=g).to(device)]
            b = torch.cat([shots, rb])
            x = prep(G["xtr"][b], cfg, True)
            with ac(cfg, device):
                loss = F.cross_entropy(m(x), G["ytr"][b])
            _step(loss, opt, scaler, m, clip=cfg["clip_grad"])
            if step in evals:
                for mode, met in eval_modes(m, G, D, c, cfg, device, orig):
                    rows.append(dict(test="savings", shots=kk, bn=mode, level=step,
                                     collapsed=_collapsed(met, base["retain_test_acc"], cfg), **met))
        del m
    return rows


# =============================================================================
# 6. Mechanism analyses
# =============================================================================
@torch.no_grad()
def extract_features(model, x, cfg, device):
    model.eval()
    feats = []
    for s in range(0, len(x), cfg["eval_batch_size"]):
        xb = prep(x[s:s + cfg["eval_batch_size"]], cfg)
        with ac(cfg, device):
            feats.append(model.features(xb).float())
    return torch.cat(feats)


def linear_probe(model, G, D, c, cfg, device, seed):
    """Fit a fresh 100-way linear head on frozen features (including forget-class data).
    High forget-class probe accuracy = the knowledge is still in the representation."""
    Ftr = extract_features(model, G["xtr"], cfg, device)
    Fte = extract_features(model, G["xte"], cfg, device)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    torch.manual_seed(stable_seed(f"probe|{seed}|{c}"))
    head = nn.Linear(Ftr.shape[1], 100).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg["probe_lr"], weight_decay=1e-4)
    ytr = G["ytr"]
    n = ytr.numel()
    for _ in range(cfg["probe_epochs"]):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 1024):
            b = perm[s:s + 1024]
            loss = F.cross_entropy(head(Ftr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = head(Fte).argmax(1)
    y = G["yte"]
    f = y == c
    r = ~f
    acc = float((pred == y).float().mean())
    fa = float((pred[f] == c).float().mean())
    fpr = float((pred[r] == c).float().mean())
    return dict(probe_test_acc=acc, probe_forget_test_acc=fa, probe_forget_fpr=fpr,
                probe_chance_corrected=fa - fpr)


def _group(name):
    if name.startswith(("conv1", "bn1")):
        return "stem"
    return name.split(".")[0]


@torch.no_grad()
def weight_change(model, ref):
    num, den = {}, {}
    rp = dict(ref.named_parameters())
    for n, p in model.named_parameters():
        g = _group(n)
        num[g] = num.get(g, 0.0) + float(((p - rp[n]) ** 2).sum())
        den[g] = den.get(g, 0.0) + float((rp[n] ** 2).sum())
    return {g: math.sqrt(num[g]) / (math.sqrt(den[g]) + 1e-12) for g in num}


@torch.no_grad()
def block_acts(model, x, cfg):
    model.eval()
    acts = {}
    for s in range(0, len(x), 250):
        out = model.block_outputs(prep(x[s:s + 250], cfg))
        for k, v in out.items():
            acts.setdefault(k, []).append(v.float().flatten(1))
    return {k: torch.cat(v) for k, v in acts.items()}


def linear_cka(X, Y):
    K = (X @ X.T).double()
    L = (Y @ Y.T).double()
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (Kc.norm() * Lc.norm() + 1e-12))


# =============================================================================
# 7. Tasks and the resumable job runner
# =============================================================================
def _meta(cfg, **kw):
    return dict(smoke=cfg["smoke"], version=VERSION, **kw)


def task_train_original(job, cfg, G, D, device, log):
    s = job["seed"]
    name = ckpt_name("original", cfg, seed=s)
    path = find_file(name) if cfg["reuse_ckpt"] else None
    if path:
        log(f"reusing existing checkpoint {path}")
        model = load_model(path, cfg, device)
    else:
        model = train_model(cfg, G, G["all_idx"], s, device, log, tag=job["key"])
        roundtrip(model, cfg)
        save_model(model, name, cfg, _meta(cfg, seed=s))
    return [dict(kind="original", method="original", cls=n, seed=s,
                 **metrics(model, G, D, cid(D, n), cfg, device, full=True)) for n in all_classes(cfg)]


def task_retrain(job, cfg, G, D, device, log):
    cls, s = job["cls"], job["seed"]
    c = cid(D, cls)
    name = ckpt_name("retrain", cfg, cls=cls, seed=s)
    path = find_file(name) if cfg["reuse_ckpt"] else None
    if path:
        log(f"reusing existing checkpoint {path}")
        model = load_model(path, cfg, device)
    else:
        model = train_model(cfg, G, retain_idx(G, c), s, device, log, tag=job["key"])
        roundtrip(model, cfg)
        save_model(model, name, cfg, _meta(cfg, seed=s, cls=cls))
    return [dict(kind="retrain", method="retrain", cls=cls, seed=s,
                 **metrics(model, G, D, c, cfg, device, full=True))]


_REF = {}


def ref_metrics(cls, seed, cfg, G, D, device, ctx="clean"):
    """Metrics of the retrained reference model (from notebook 02's results, or recomputed)."""
    k = (cls, seed, ctx, cfg["smoke"])
    if k in _REF:
        return _REF[k]
    m = None
    if ctx == "clean":
        for r in load_rows("retrain", cfg):
            if r.get("cls") == cls and r.get("seed") == seed:
                m = r
    if m is None:
        name = ckpt_name("retrain", cfg, cls=cls, seed=seed)
        if find_file(name) is None:
            raise FileNotFoundError(f"Retrained reference '{name}' not found. Run notebook 02 first and "
                                    f"attach its output to this notebook.")
        model = load_model(name, cfg, device)
        m = metrics(model, G, D, cid(D, cls), cfg, device, ctx=ctx, full=True)
        del model
    _REF[k] = m
    return m


def task_calibrate(job, cfg, G, D, device, log):
    cls, s, method, hp = job["cls"], job["seed"], job["method"], job["hp"]
    c = cid(D, cls)
    model = load_model(ckpt_name("original", cfg, seed=s), cfg, device)
    t = time.time()
    run_unlearning(method, model, G, D, c, hp, cfg, device)
    secs = time.time() - t
    roundtrip(model, cfg)
    m = metrics(model, G, D, c, cfg, device, full=True)
    ref = ref_metrics(cls, s, cfg, G, D, device)
    log(f"{job['key']} forget_acc={m['forget_test_acc']:.3f} retain_acc={m['retain_test_acc']:.3f} "
        f"(retrain {ref['retain_test_acc']:.3f}) matched={is_matched(m, ref, cfg)} {secs:.0f}s")
    return [dict(kind="calib", method=method, cls=cls, seed=s, hp=json.dumps(hp, sort_keys=True),
                 hp_index=job["hp_index"], matched=is_matched(m, ref, cfg),
                 ref_retain_test_acc=ref["retain_test_acc"], ref_forget_test_acc=ref["forget_test_acc"],
                 unlearn_seconds=secs, **m)]


def task_unlearn(job, cfg, G, D, device, log):
    kind, method, cls, s = job["kind"], job["method"], job["cls"], job["seed"]
    ctx = job.get("ctx", "clean")              # one context name, or a list (multi-context unlearning)
    eval_ctx = ctx if isinstance(ctx, str) else "clean"   # where the matching rule is checked
    c = cid(D, cls)
    name = ckpt_name(kind, cfg, cls=cls, seed=s, method=method)
    ref = ref_metrics(cls, s, cfg, G, D, device, ctx=eval_ctx)
    path = find_file(name) if cfg["reuse_ckpt"] else None
    if path:
        log(f"reusing existing checkpoint {path}")
        model = load_model(path, cfg, device)
        m = metrics(model, G, D, c, cfg, device, ctx=eval_ctx, full=True)
        best = dict(model=model, m=m, hp=None, ok=is_matched(m, ref, cfg), tries=0, secs=0.0)
    else:
        best = None
        for i, hp in enumerate(job["candidates"][:cfg["max_tries"]]):
            model = load_model(ckpt_name("original", cfg, seed=s), cfg, device)
            t = time.time()
            run_unlearning(method, model, G, D, c, hp, cfg, device, ctx=ctx)
            secs = time.time() - t
            roundtrip(model, cfg)
            m = metrics(model, G, D, c, cfg, device, ctx=eval_ctx, full=True)
            ok = is_matched(m, ref, cfg)
            score = (ok, m["retain_test_acc"] if ok else -m["forget_test_acc"])
            log(f"{job['key']} try {i + 1}: forget_acc={m['forget_test_acc']:.3f} "
                f"retain_acc={m['retain_test_acc']:.3f} matched={ok}")
            if best is None or score > best["score"]:
                best = dict(model=model, m=m, hp=hp, ok=ok, tries=i + 1, secs=secs, score=score)
            else:
                del model
            if ok:
                break
        if best is None:
            raise RuntimeError(f"No hyper-parameter candidates given for {method}.")
        save_model(best["model"], name, cfg, _meta(cfg, seed=s, cls=cls, method=method, hp=best["hp"], ctx=ctx))
    row = dict(kind=kind, method=method, cls=cls, seed=s, ctx=ctx if isinstance(ctx, str) else "+".join(ctx),
               hp=json.dumps(best["hp"], sort_keys=True), matched=best["ok"], tries=best["tries"],
               unlearn_seconds=best["secs"], ref_retain_test_acc=ref["retain_test_acc"],
               ref_neighbour_test_acc=ref.get("neighbour_test_acc"),
               ref_unrelated_test_acc=ref.get("unrelated_test_acc"), **best["m"])
    if eval_ctx != "clean":
        row.update({f"clean_{k}": v for k, v in metrics(best["model"], G, D, c, cfg, device).items()})
    return [row]


def task_extinction(job, cfg, G, D, device, log):
    kind, method, cls, s = job["kind"], job["method"], job["cls"], job["seed"]
    c = cid(D, cls)
    model = load_model(job["ckpt"], cfg, device)
    common = dict(kind=kind, method=method, cls=cls, seed=s, matched=job.get("matched"))
    rows = [dict(common, test="base", **metrics(model, G, D, c, cfg, device, full=True))]
    tests = job["tests"]
    if "contexts" in tests:
        rows += [dict(common, **r) for r in test_contexts(model, G, D, c, cfg, device, log)]
    if "spontaneous" in tests:
        rows += [dict(common, **r) for r in test_spontaneous(model, G, D, c, cfg, device, s)]
    if "reinstatement" in tests:
        rows += [dict(common, **r) for r in test_reinstatement(model, G, D, c, cfg, device, s)]
    if "savings" in tests:
        rows += [dict(common, **r) for r in test_savings(model, G, D, c, cfg, device, s)]
    return rows


def task_mechanism(job, cfg, G, D, device, log):
    kind, method, cls, s = job["kind"], job["method"], job["cls"], job["seed"]
    c = cid(D, cls)
    model = load_model(job["ckpt"], cfg, device)
    common = dict(kind=kind, method=method, cls=cls, seed=s, matched=job.get("matched"))
    rows = [dict(common, test="probe", **linear_probe(model, G, D, c, cfg, device, s))]
    if kind == "original":
        return rows
    orig = get_original(cfg, s, device)
    m1 = copy.deepcopy(model)
    m1.fc.load_state_dict(orig.fc.state_dict())
    rows.append(dict(common, test="headswap", variant="body_model+head_original", **metrics(m1, G, D, c, cfg, device)))
    m2 = copy.deepcopy(orig)
    m2.fc.load_state_dict(model.fc.state_dict())
    rows.append(dict(common, test="headswap", variant="body_original+head_model", **metrics(m2, G, D, c, cfg, device)))
    del m1, m2
    for g, v in weight_change(model, orig).items():
        rows.append(dict(common, test="weights", group=g, rel_change=v))
    refs = {"original": orig}
    if kind != "retrain":
        rname = ckpt_name("retrain", cfg, cls=cls, seed=s)
        if find_file(rname):
            refs["retrain"] = load_model(rname, cfg, device)
    fsub = G["xte"][G["yte"] == c]
    rng = np.random.RandomState(stable_seed(f"cka|{s}"))
    ridx = np.where(D["yte"] != c)[0]
    rsub = G["xte"][torch.from_numpy(np.sort(rng.choice(ridx, min(cfg["cka_n_retain"], len(ridx)),
                                                        replace=False))).to(device)]
    for subset, x in (("forget", fsub), ("retain", rsub)):
        a = block_acts(model, x, cfg)
        for refname, ref in refs.items():
            b = block_acts(ref, x, cfg)
            for blk in a:
                rows.append(dict(common, test="cka", ref=refname, subset=subset, block=blk, cka=linear_cka(a[blk], b[blk])))
            del b
        del a
    return rows


TASKS = {"train_original": task_train_original, "retrain": task_retrain, "calibrate": task_calibrate,
         "unlearn": task_unlearn, "extinction": task_extinction, "mechanism": task_mechanism}


# ---- results files (JSON lines) ----
def task_file(task, cfg):
    return f"{task}{sfx(cfg)}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_rows(task, cfg, rank, rows):
    ensure_dirs()
    path = os.path.join(RES_DIR, f"{task_file(task, cfg)}__r{rank}.jsonl")
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _result_files(task, cfg):
    pat = f"{task_file(task, cfg)}__"
    files = sorted(glob.glob(os.path.join(RES_DIR, pat + "*.jsonl")))
    if os.path.isdir(INPUT):
        _build_index()
        files += sorted(p for n, ps in _INDEX["map"].items() if n.startswith(pat) and n.endswith(".jsonl") for p in ps)
    return files


# Oldest code version whose rows are still valid, per task. Extinction rows from 1.1 used the
# collapsing frozen-BN training and have no graded metrics (run #4), so they are ignored everywhere.
MIN_RESULT_VERSION = {"extinction": "1.2", "headbias": "1.4"}


def _vtuple(v):
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


def _row_ok(task, r):
    need = MIN_RESULT_VERSION.get(task)
    return need is None or _vtuple(r.get("v", "1.1")) >= _vtuple(need)


def _read_all(task, cfg):
    rows = {}
    for p in _result_files(task, cfg):
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue                     # half-written last line after a crash
                    if _row_ok(task, r):
                        rows[r.get("row_id", len(rows))] = r
        except OSError:
            pass
    return list(rows.values())


def load_rows(task, cfg):
    """All finished result rows of a task (from this notebook's output and every attached input)."""
    all_rows = _read_all(task, cfg)
    done = {r["key"] for r in all_rows if r.get("_done")}
    return [r for r in all_rows if not r.get("_done") and r.get("key") in done]


def load_df(task, cfg):
    import pandas as pd
    return pd.DataFrame(load_rows(task, cfg))


def done_keys(task, cfg):
    return {r["key"] for r in _read_all(task, cfg) if r.get("_done")}


def carry_over(task, cfg, jobs, log=print):
    """Copy results + checkpoints of already-finished jobs from attached inputs into this output,
    so the newest notebook version always contains everything (downstream needs only that one)."""
    ensure_dirs()
    tf = task_file(task, cfg)
    carried = os.path.join(RES_DIR, f"{tf}__carried.jsonl")
    in_rows = []
    if os.path.isdir(INPUT):
        _build_index()
        for n, ps in _INDEX["map"].items():
            if n.startswith(tf + "__") and n.endswith(".jsonl"):
                for p in ps:
                    with open(p) as f:
                        for l in f:
                            try:
                                r = json.loads(l)
                            except json.JSONDecodeError:
                                continue             # empty or half-written line
                            if _row_ok(task, r):
                                in_rows.append(r)
    if in_rows:
        have = {}
        if os.path.exists(carried):
            with open(carried) as f:
                for l in f:
                    if l.strip():
                        r = json.loads(l)
                        have[r.get("row_id")] = r
        for r in in_rows:
            have[r.get("row_id")] = r
        with open(carried, "w") as f:
            for r in have.values():
                f.write(json.dumps(r, default=_json_default) + "\n")
    n_ck = 0
    done = done_keys(task, cfg)
    for j in jobs:
        name = j.get("out_ckpt")
        if name and j["key"] in done and not os.path.exists(os.path.join(CKPT_DIR, name)):
            p = find_file(name)
            if p:
                shutil.copy2(p, os.path.join(CKPT_DIR, name))
                n_ck += 1
    if in_rows or n_ck:
        log(f"[{task}] carried over {len(in_rows)} result rows and {n_ck} checkpoints from inputs")


# ---- worker ----
def _logger(rank):
    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}][w{rank}] {msg}", flush=True)
    return log


def run_worker(task, jobs, cfg, rank=0, world=1):
    log = _logger(rank)
    ensure_dirs()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg["cudnn_benchmark"])
        log(f"device cuda:{torch.cuda.current_device()} {torch.cuda.get_device_name()}")
    else:
        log("device cpu (no GPU found)")
    D = load_data(cfg)
    validate_classes(cfg, D)
    G = gpu_data(D, cfg, device)
    fn = TASKS[task]
    mine = jobs[rank::world]
    t0 = cfg.get("_t0", T0)
    budget = cfg["time_budget_h"] * 3600
    durations, n_ok, n_err = [], 0, 0
    err_path = os.path.join(LOG_DIR, f"{task_file(task, cfg)}__errors.jsonl")
    for i, j in enumerate(mine):
        est = float(np.mean(durations)) if durations else 0.0
        if time.time() - t0 + 1.3 * est > budget:
            log(f"time budget ({cfg['time_budget_h']} h) reached - stopping cleanly. Save this version, attach its "
                f"output and run again to continue with the remaining jobs.")
            break
        ts = time.time()
        log(f"job {i + 1}/{len(mine)}: {j['key']}")
        try:
            seed_everything(stable_seed(j["key"]))
            rows = fn(j, cfg, G, D, device, log)
            for k, r in enumerate(rows):
                r.update(key=j["key"], task=task, row_id=f"{j['key']}#{k}", smoke=cfg["smoke"], v=VERSION)
            rows.append(dict(key=j["key"], task=task, row_id=f"{j['key']}#done", _done=True,
                             seconds=time.time() - ts, v=VERSION))
            append_rows(task, cfg, rank, rows)
            n_ok += 1
        except Exception as e:
            n_err += 1
            tb = traceback.format_exc()
            log(f"ERROR in {j['key']}: {e!r}\n{tb}")
            with open(err_path, "a") as f:
                f.write(json.dumps(dict(key=j["key"], error=repr(e), traceback=tb, time=time.time())) + "\n")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        durations.append(time.time() - ts)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log(f"finished: {n_ok} ok, {n_err} failed")
    return n_ok, n_err


def _read_errors(task, cfg, since):
    p = os.path.join(LOG_DIR, f"{task_file(task, cfg)}__errors.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for l in f:
                try:
                    r = json.loads(l)
                    if r.get("time", 0) >= since:
                        out.append(r)
                except json.JSONDecodeError:
                    pass
    return out


def launch(task, jobs, cfg, n_workers=None):
    """Run `jobs` for `task`, skipping finished ones. Uses one worker process per GPU (T4 x2 -> 2)."""
    ensure_dirs()
    jobs = list(jobs)
    keys = [j["key"] for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate job keys.")
    try:
        shutil.copy2(UTILS_FILE, os.path.join(WORK, "ext_utils_used.py"))   # provenance
    except OSError:
        pass
    carry_over(task, cfg, jobs)
    done = done_keys(task, cfg)
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[{task}] {len(jobs)} jobs: {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n = n_workers or cfg["n_workers"] or max(1, ngpu)
    n = max(1, min(n, len(todo)))
    ensure_cifar100()                                    # find or download once, before workers start
    if task == "extinction" and "contexts" in cfg["tests"]:
        prefetch_cifar_c(cfg)                            # download once, before workers start
    run_cfg = dict(cfg, _t0=T0)
    if n > 1:
        run_cfg["download_cifar_c"] = False
    since = time.time()
    if n == 1:
        run_worker(task, todo, run_cfg, 0, 1)
    else:
        jobfile = os.path.join(JOB_DIR, f"{task_file(task, cfg)}_{int(since)}.json")
        with open(jobfile, "w") as f:
            json.dump(dict(task=task, jobs=todo, cfg=run_cfg), f, default=_json_default)
        procs, logs = [], []
        try:
            for r in range(n):
                env = dict(os.environ, PYTHONUNBUFFERED="1")
                if ngpu:
                    env["CUDA_VISIBLE_DEVICES"] = str(r % ngpu)
                lp = os.path.join(LOG_DIR, f"{task_file(task, cfg)}__w{r}.log")
                logs.append(lp)
                fh = open(lp, "a")
                procs.append((subprocess.Popen([sys.executable, UTILS_FILE, "worker", jobfile, str(r), str(n)],
                                               stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=WORK), fh))
            print(f"[{task}] started {n} worker processes (logs: {', '.join(logs)})")
            todo_keys = {j["key"] for j in todo}
            while any(p.poll() is None for p, _ in procs):
                time.sleep(cfg["poll_seconds"])
                d = len(done_keys(task, cfg) & todo_keys)
                tails = []
                for lp in logs:
                    try:
                        with open(lp) as f:
                            ls = [l.strip() for l in f.readlines()[-3:] if l.strip()]
                        tails.append(ls[-1][:110] if ls else "")
                    except OSError:
                        tails.append("")
                print(f"[{task}] {d}/{len(todo)} done | {(time.time() - since) / 60:.1f} min | " + " || ".join(tails),
                      flush=True)
        finally:
            for p, fh in procs:
                if p.poll() is None:
                    p.terminate()
                fh.close()
        for (p, _), lp in zip(procs, logs):
            if p.returncode not in (0, None):
                with open(lp) as f:
                    tail = "".join(f.readlines()[-40:])
                raise RuntimeError(f"Worker crashed (exit code {p.returncode}). Last log lines:\n{tail}")
    done = done_keys(task, cfg)
    left = [j["key"] for j in jobs if j["key"] not in done]
    errs = _read_errors(task, cfg, since)
    print(f"[{task}] now done: {len(jobs) - len(left)}/{len(jobs)}")
    if errs:
        print(f"[{task}] {len(errs)} job(s) FAILED:")
        for e in errs[:10]:
            print(f"  - {e['key']}: {e['error']}")
        print("  Full tracebacks are in", os.path.join(LOG_DIR, f"{task_file(task, cfg)}__errors.jsonl"))
        if cfg["raise_on_error"]:
            raise RuntimeError(f"{len(errs)} job(s) failed (finished jobs are saved). Fix the cause and re-run "
                               f"this cell: only unfinished jobs will run.")
    elif left:
        print(f"[{task}] {len(left)} job(s) not started (time budget). Save the version, then re-run with this "
              f"output attached to continue.")


# =============================================================================
# 8. Job builders, calibration helpers, environment report, self-test
# =============================================================================
def _shard(jobs, shard):
    if shard is None:
        return jobs
    i, n = shard
    if not (0 <= i < n):
        raise ValueError("shard must be (index, total) with 0 <= index < total")
    return jobs[i::n]


def jobs_train_original(cfg):
    return [dict(key=f"original|s{s}", seed=s, out_ckpt=ckpt_name("original", cfg, seed=s)) for s in cfg["seeds"]]


def jobs_retrain(cfg, shard=None):
    s0 = cfg["seeds"][0]
    jobs = []
    if cfg["pilot_class"]:
        p = cfg["pilot_class"]
        jobs.append(dict(key=f"retrain|{p}|s{s0}", cls=p, seed=s0, out_ckpt=ckpt_name("retrain", cfg, cls=p, seed=s0)))
    for s in cfg["seeds"]:
        for c in cfg["forget_classes"]:
            jobs.append(dict(key=f"retrain|{c}|s{s}", cls=c, seed=s, out_ckpt=ckpt_name("retrain", cfg, cls=c, seed=s)))
    return _shard(jobs, shard)


def calib_grid(cfg):
    grid = copy.deepcopy(cfg["calib_grid"] or DEFAULT_CALIB_GRID)
    missing = [m for m in cfg["methods"] if m not in grid]
    if missing:
        raise KeyError(f"calib_grid has no entry for {missing}")
    if cfg["calib_limit"]:
        grid = {m: g[:cfg["calib_limit"]] for m, g in grid.items()}
    return {m: grid[m] for m in cfg["methods"]}


def jobs_calibrate(cfg):
    c = cfg["pilot_class"] or cfg["forget_classes"][0]
    s = cfg["seeds"][0]
    return [dict(key=f"calib|{m}|{i}|{short_hash(json.dumps(hp, sort_keys=True))}", method=m, hp=hp, hp_index=i,
                 cls=c, seed=s) for m, g in calib_grid(cfg).items() for i, hp in enumerate(g)]


def select_candidates(cfg):
    """Rank calibration runs per method: matched runs first (highest retain accuracy),
    then unmatched runs (lowest forget accuracy). Saves calibration.json."""
    import pandas as pd
    rows = load_rows("calibrate", cfg)
    if not rows:
        raise RuntimeError("No calibration results found. Run the calibration stage first.")
    df = pd.DataFrame(rows)
    out = {}
    for m in cfg["methods"]:
        rm = [r for r in rows if r["method"] == m]
        if not rm:
            print(f"[calibration] WARNING: no results for {m}")
            continue
        rm.sort(key=lambda r: (not r["matched"], -r["retain_test_acc"] if r["matched"] else r["forget_test_acc"]))
        out[m] = [json.loads(r["hp"]) for r in rm][:max(1, cfg["max_tries"])]
    path = os.path.join(RES_DIR, f"calibration{sfx(cfg)}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibration] saved ranked candidates to {path}")
    cols = ["method", "hp", "matched", "forget_test_acc", "retain_test_acc", "ref_retain_test_acc", "unlearn_seconds"]
    return out, df[[c for c in cols if c in df.columns]].sort_values(["method", "matched"], ascending=[True, False])


def load_choices(cfg):
    """Pre-registered analysis choices (e.g. the primary spontaneous-recovery proxy picked on the pilot)."""
    p = find_file(f"analysis_choices{sfx(cfg)}.json")
    if p is None:
        return {}
    with open(p) as f:
        return json.load(f)


def save_choices(cfg, **choices):
    ensure_dirs()
    cur = load_choices(cfg)
    cur.update(choices)
    path = os.path.join(RES_DIR, f"analysis_choices{sfx(cfg)}.json")
    with open(path, "w") as f:
        json.dump(cur, f, indent=2, default=_json_default)
    print(f"[choices] saved {cur} to {path}")
    return cur


def load_candidates(cfg):
    p = find_file(f"calibration{sfx(cfg)}.json")
    if p is None:
        raise FileNotFoundError(f"calibration{sfx(cfg)}.json not found - run the calibration stage "
                                f"(notebook 03, stage A) or attach its output.")
    with open(p) as f:
        return json.load(f)


def jobs_unlearn(cfg, candidates, kind="unl", pilot=False, shard=None):
    if kind not in ("unl", "aba", "mctx"):
        raise ValueError("kind must be 'unl', 'aba' or 'mctx'")
    if kind == "aba":
        classes, ctx = cfg["aba_classes"], cfg["aba_context"]
    elif kind == "mctx":
        classes, ctx = cfg["mctx_classes"], list(cfg["mctx_contexts"])
    else:
        classes, ctx = cfg["forget_classes"], "clean"
    seeds = cfg["seeds"]
    if pilot:
        classes, seeds = [cfg["pilot_class"]], [cfg["seeds"][0]]
    jobs = []
    for s in seeds:
        for c in classes:
            for m in cfg["methods"]:
                if m not in candidates:
                    raise KeyError(f"No calibrated hyper-parameters for {m}; run the calibration stage first.")
                jobs.append(dict(key=f"{kind}|{m}|{c}|s{s}", kind=kind, method=m, cls=c, seed=s, ctx=ctx,
                                 candidates=candidates[m], out_ckpt=ckpt_name(kind, cfg, cls=c, seed=s, method=m)))
    return _shard(jobs, shard)


def _model_jobs(cfg, prefix, pilot, unl_tests, aba_tests, ref_tests, orig_tests):
    jobs = {}
    for r in load_rows("unlearn", cfg):
        if (r["cls"] == cfg["pilot_class"]) != pilot:
            continue
        if r["cls"] not in all_classes(cfg) or r["seed"] not in cfg["seeds"] or r["method"] not in cfg["methods"]:
            continue
        tests = unl_tests if r["kind"] in ("unl", "mctx") else aba_tests
        if not tests:
            continue
        k = f"{prefix}|{r['kind']}|{r['method']}|{r['cls']}|s{r['seed']}"
        jobs[k] = dict(key=k, kind=r["kind"], method=r["method"], cls=r["cls"], seed=r["seed"], matched=r["matched"],
                       ckpt=ckpt_name(r["kind"], cfg, cls=r["cls"], seed=r["seed"], method=r["method"]), tests=tests)
    classes = [cfg["pilot_class"]] if pilot else cfg["forget_classes"]
    seeds = [cfg["seeds"][0]] if pilot else cfg["seeds"]
    for s in seeds:
        for c in classes:
            if ref_tests:
                k = f"{prefix}|retrain|retrain|{c}|s{s}"
                jobs[k] = dict(key=k, kind="retrain", method="retrain", cls=c, seed=s, matched=True,
                               ckpt=ckpt_name("retrain", cfg, cls=c, seed=s), tests=ref_tests)
            if orig_tests:
                k = f"{prefix}|original|original|{c}|s{s}"
                jobs[k] = dict(key=k, kind="original", method="original", cls=c, seed=s, matched=None,
                               ckpt=ckpt_name("original", cfg, seed=s), tests=orig_tests)
    out = sorted(jobs.values(), key=lambda j: j["key"])
    missing = [j for j in out if find_file(j["ckpt"]) is None]
    if missing:
        print(f"WARNING: {len(missing)} checkpoint(s) not found and skipped, e.g. {missing[0]['ckpt']}. "
              f"Attach the outputs of notebooks 01, 02 and 03.")
    return [j for j in out if j not in missing]


def jobs_extinction(cfg, pilot=False, shard=None):
    t = list(cfg["tests"])
    ctx_only = ["contexts"] if "contexts" in t else []
    return _shard(_model_jobs(cfg, "ext", pilot, t, ctx_only, t, ctx_only), shard)


def jobs_mechanism(cfg, pilot=False, shard=None):
    return _shard(_model_jobs(cfg, "mech", pilot, ["mechanism"], [], ["mechanism"], ["mechanism"]), shard)


def env_report():
    """Print everything about the runtime that matters for these experiments."""
    import shutil as sh
    print(f"ext_utils {VERSION} | Python {platform.python_version()} | torch {torch.__version__}")
    print(f"on Kaggle: {ON_KAGGLE} | WORK={WORK} | INPUT={INPUT} | DATA={DATA}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 2 ** 30:.1f} GB, compute capability {p.major}.{p.minor}")
    else:
        print("GPU: none found (Settings -> Accelerator -> GPU T4 x2)")
    print(f"CPU cores: {os.cpu_count()}")
    for d in (WORK, "/tmp"):
        if os.path.isdir(d):
            u = sh.disk_usage(d)
            print(f"disk {d}: {u.free / 2 ** 30:.1f} GB free of {u.total / 2 ** 30:.1f} GB")
    try:
        urllib.request.urlopen(urllib.request.Request("https://zenodo.org", method="HEAD"), timeout=8)
        print("internet: ON")
    except Exception as e:
        print(f"internet: OFF or blocked ({type(e).__name__}) - needed unless datasets are attached")
    if os.path.isdir(INPUT):
        tops = sorted(os.listdir(INPUT))
        print(f"attached inputs ({len(tops)}): {tops}")
    d = find_cifar100()
    print(f"CIFAR-100: found at {d}" if d else
          "CIFAR-100: NOT attached -> will be downloaded (can take ~1 h on Kaggle). Attach 'fedesoriano/cifar100' "
          "or an earlier notebook output that contains data_cache/.")


def self_test(cfg=None, log=print):
    """Runs every component once on a tiny subset (~2-5 min on a T4). Writes nothing permanent."""
    cfg = make_cfg(smoke=True, seeds=[0], forget_classes=["maple_tree"], aba_classes=["maple_tree"],
                   mctx_classes=["maple_tree"]) \
        if cfg is None else cfg
    device = get_device()
    D = load_data(cfg)
    validate_classes(cfg, D)
    G = gpu_data(D, cfg, device)
    c = cid(D, cfg["forget_classes"][0])
    results = {}

    def check(name, fn):
        t = time.time()
        try:
            out = fn()
            results[name] = "PASS"
            log(f"PASS  {name} ({time.time() - t:.1f}s)")
            return out
        except Exception as e:
            results[name] = f"FAIL: {e!r}"
            log(f"FAIL  {name}: {e!r}\n{traceback.format_exc()}")
            return None

    base = check("train (1 epoch)", lambda: train_model(cfg, G, G["all_idx"], 0, device, log, tag="selftest"))
    if base is None:
        return results
    check("checkpoint save/load", lambda: load_model(save_model(roundtrip(copy.deepcopy(base), cfg),
                                                                "selftest_tmp.pt", cfg), cfg, device))
    try:
        os.remove(os.path.join(CKPT_DIR, "selftest_tmp.pt"))
    except OSError:
        pass
    def _check_metrics():
        m = metrics(base, G, D, c, cfg, device, full=True)
        assert 1 <= m["forget_rank"] <= 100 and 0 <= m["forget_auc"] <= 1 and m["finite"]
        assert m["forget_top5"] >= m["forget_test_acc"] - 1e-9, "top-5 recall below top-1 recall"
        return m

    def _check_bn_swap():
        m = copy.deepcopy(base)
        orig = bn_state(m)
        _train_epochs(m, G, retain_idx(G, c), cfg["batch_size"], 0.01, 1, cfg, device)
        after = bn_state(m)
        assert any(not torch.equal(a[0], b[0]) for a, b in zip(orig, after)), "training did not update BN stats"
        with bn_stats_swapped(m, orig):
            assert all(torch.equal(a[0], b[0]) for a, b in zip(orig, bn_state(m))), "original stats not restored"
        assert all(torch.equal(a[0], b[0]) for a, b in zip(after, bn_state(m))), "updated stats not put back"
        return [mode for mode, _ in eval_modes(m, G, D, c, cfg, device, orig)]

    check("metrics (top-1, top-5, rank, AUROC)", _check_metrics)
    check("BN control: evaluate with updated and with original statistics", _check_bn_swap)
    for ctx in CONTEXTS:
        check(f"context {ctx}", lambda ctx=ctx: metrics(base, G, D, c, cfg, device, ctx=ctx))
    for m in cfg["methods"]:
        hp = dict(calib_grid(cfg)[m][0], epochs=1)
        if m == "SCRUB":
            hp["msteps"] = 1
        check(f"unlearn {m}", lambda m=m, hp=hp: run_unlearning(m, copy.deepcopy(base), G, D, c, hp, cfg, device))
    check("multi-context unlearning (RL)", lambda: run_unlearning(
        "RL", copy.deepcopy(base), G, D, c, {"lr": 0.01, "epochs": 1}, cfg, device, ctx=list(cfg["mctx_contexts"])))
    check("BatchNorm recalibration control", lambda: bn_recalibrate(copy.deepcopy(base), G, retain_idx(G, c), 10, cfg, 0))
    check("GroupNorm model (1 epoch)", lambda: metrics(
        train_model(dict(cfg, norm="gn"), G, G["all_idx"], 0, device, log, tag="selftest-gn"), G, D, c, cfg, device))
    check("spontaneous recovery (BN updated/original statistics + recalibration)",
          lambda: test_spontaneous(base, G, D, c, cfg, device, 0))
    _ORIG[(0, str(device), cfg["smoke"])] = base          # stand-in original for the similarity score
    check("reinstatement", lambda: test_reinstatement(base, G, D, c, cfg, device, 0))
    check("savings", lambda: test_savings(base, G, D, c, cfg, device, 0))
    check("linear probe", lambda: linear_probe(base, G, D, c, cfg, device, 0))
    check("CKA / weights", lambda: (linear_cka(*[block_acts(base, G["xte"][:50], cfg)["layer1"]] * 2),
                                    weight_change(base, base)))
    xt = xl_target(G, cfg, 0, 0)
    XM = (lambda mm: xl_metrics(mm, G, xt, cfg, device))
    check("example-level metrics", lambda: XM(base))
    check("example-level unlearning (RL, SalUn)", lambda: [run_unlearning(m, copy.deepcopy(base), G, D, xt,
          dict(calib_grid(cfg)[m][0], epochs=1), cfg, device) for m in ("RL", "SalUn")])
    check("example-level spontaneous recovery", lambda: test_spontaneous(base, G, D, xt, cfg, device, 0, mfn=XM))
    check("example-level contexts and savings", lambda: (xl_test_contexts(base, G, xt, cfg, device),
                                                          xl_test_savings(base, G, xt, cfg, device)))
    check("head-bias check", lambda: headbias_row(copy.deepcopy(base), base, G, D, c, cfg, device))
    acfg = dict(cfg, xl_design="atypical")
    check("atypical forget set (C-scores)", lambda: xl_metrics(base, G, xl_target(G, acfg, 0, 0), acfg, device))
    _ORIG.clear()
    n_fail = sum(v != "PASS" for v in results.values())
    log(f"self-test finished: {len(results) - n_fail} passed, {n_fail} failed")
    return results



# =============================================================================
# 9. Follow-ups (1.3): example-level forgetting arm and head-bias check
# =============================================================================
# Why: forgetting individual *examples* is known to relapse under retain-only fine-tuning ("From Dormant to
# Deleted", Siddiqui et al. 2025), while our class-level arm shows no relapse. Running the same tests on
# example-level forgetting, with the same models and code, turns that contrast into a direct result.
# Atypical forget sets are small (500 images) and pure memorisation, so they need a denser grid that reaches further
# in both directions; ranked on the pilot forget set as before.
DEFAULT_XA_CALIB_GRID = {
    "FT":       [{"lr": lr, "epochs": e} for lr in (0.01, 0.02, 0.05) for e in (5, 10, 20)],
    # NegGrad+ loss is a*CE_retain - (1-a)*CE_forget, so a smaller alpha forgets harder: 0.95 forgot nothing on random
    # examples (run #9), while at class level only 0.99 kept retain accuracy (run #3). Span both sides.
    "NegGrad+": [{"lr": lr, "alpha": a, "epochs": 3} for lr in (0.005, 0.01, 0.02) for a in (0.8, 0.9, 0.95, 0.99)],
    "RL":       [{"lr": lr, "epochs": e} for lr in (0.005, 0.01, 0.02) for e in (1, 3, 5)],
    "SCRUB":    [{"lr": lr, "msteps": m, "epochs": m + 3, "alpha": 0.001, "gamma": 0.99, "T": 4.0}
                 for lr in (1e-3, 5e-3, 1e-2) for m in (3, 5)],
    "SalUn":    [{"lr": lr, "epochs": e, "mask_ratio": 0.5} for lr in (0.005, 0.01, 0.02) for e in (1, 3, 5)],
}
DEFAULT_XL_CALIB_GRID = {
    "FT":       [{"lr": lr, "epochs": e} for lr in (0.01, 0.02, 0.05) for e in (5, 10)],
    "NegGrad+": [{"lr": lr, "alpha": a, "epochs": 5} for lr in (0.005, 0.01) for a in (0.95, 0.99, 0.999)],
    "RL":       [{"lr": lr, "epochs": e} for lr in (0.005, 0.01, 0.02) for e in (3, 5)],
    "SCRUB":    [{"lr": lr, "msteps": m, "epochs": m + 3, "alpha": 0.001, "gamma": 0.99, "T": 4.0}
                 for lr in (5e-4, 1e-3, 5e-3) for m in (3, 5)],
    "SalUn":    [{"lr": lr, "epochs": e, "mask_ratio": 0.5} for lr in (0.005, 0.01, 0.02) for e in (3, 5)],
}
XL_PRIMARY = "mem_gap"


def _auroc(pos, neg):
    """AUROC of scores, positives vs negatives (0.5 = chance)."""
    s = torch.cat([pos, neg]).double()
    ranks = torch.empty_like(s)
    ranks[s.argsort()] = torch.arange(1, s.numel() + 1, dtype=torch.float64, device=s.device)
    n1, n0 = pos.numel(), neg.numel()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


@torch.no_grad()
def xl_metrics(model, G, t, cfg, device, ctx="clean", idx=None):
    """Example-level metrics for target t (an XL).

    forget_acc  = accuracy on the forgotten training images (or on `idx`, a subset of them)
    test_acc    = accuracy on the test set (also stored as retain_test_acc for the shared collapse check)
    mem_gap     = forget_acc - test_acc: the model's advantage on the forgotten images over unseen ones
                  (primary score; about 0 for the retrained model, about 0.25 for the original)
    mia_auc     = AUROC of the true-class probability, forgotten vs test images (0.5 = no membership signal)
    leak        = 2 x |mia_auc - 0.5|
    """
    fi = t.fidx if idx is None else idx
    ri = G["retain_eval_idx"]
    ri = ri[~torch.isin(ri, t.fidx)]
    zf = logits_of(model, G["xtr"][fi], cfg, device, ctx)
    zt = logits_of(model, G["xte"], cfg, device, ctx)
    zr = logits_of(model, G["xtr"][ri], cfg, device, ctx)
    yf, yt, yr = G["ytr"][fi], G["yte"], G["ytr"][ri]
    pf = F.softmax(zf, 1).gather(1, yf[:, None]).squeeze(1)
    pt = F.softmax(zt, 1).gather(1, yt[:, None]).squeeze(1)
    m = dict(forget_acc=float((zf.argmax(1) == yf).float().mean()),
             test_acc=float((zt.argmax(1) == yt).float().mean()),
             retain_train_acc=float((zr.argmax(1) == yr).float().mean()), mia_auc=_auroc(pf, pt))
    m["retain_test_acc"] = m["test_acc"]
    m["mem_gap"] = m["forget_acc"] - m["test_acc"]
    m["leak"] = 2 * abs(m["mia_auc"] - 0.5)
    m["finite"] = bool(torch.isfinite(zf).all() and torch.isfinite(zt).all())
    return m


def xl_violation(m, ref, cfg):
    """How far a model is from the matching rule: forget-accuracy distance beyond xl_match_gap plus test-accuracy
    shortfall beyond match_retain_gap (0 = matched). Used to rank unmatched candidates (1.5; before: forget distance only)."""
    return (max(0.0, abs(m["forget_acc"] - ref["forget_acc"]) - cfg["xl_match_gap"])
            + max(0.0, ref["test_acc"] - cfg["match_retain_gap"] - m["test_acc"]))


def xl_is_matched(m, ref, cfg):
    """Example-level matching rule: forgotten-image accuracy within xl_match_gap of the retrained model's (in
    either direction: over-forgetting is as detectable as under-forgetting), test accuracy within
    match_retain_gap."""
    return bool(abs(m["forget_acc"] - ref["forget_acc"]) <= cfg["xl_match_gap"]
                and m["test_acc"] >= ref["test_acc"] - cfg["match_retain_gap"])


def _xck(kind, cfg, draw, seed, method=None):
    """Checkpoint name for example-level models: xlretrain_xl0_s0.pt (random) or xaretrain_xa0_s0.pt (atypical)."""
    p = xl_prefix(cfg)
    return ckpt_name(p + kind, cfg, cls=f"{p}{draw}", seed=seed, method=method)


def _design(r):
    return r.get("design", "random")


_XLREF = {}


def xl_ref_metrics(seed, draw, cfg, G, device):
    k = (cfg["xl_design"], seed, draw, cfg["smoke"], cfg["xl_frac"], cfg["xl_n"], str(device))
    if k not in _XLREF:
        t = xl_target(G, cfg, seed, draw)
        name = _xck("retrain", cfg, draw, seed)
        if find_file(name) is None:
            raise FileNotFoundError(f"Example-level retrained model '{name}' not found. Run stage A of notebook 07.")
        model = load_model(name, cfg, device)
        _XLREF[k] = xl_metrics(model, G, t, cfg, device)
        del model
    return _XLREF[k]


def xl_test_contexts(model, G, t, cfg, device):
    """Renewal at example level: synthetic contexts applied to forgotten and test images alike
    (CIFAR-100-C has test images only, so it cannot be used here)."""
    return [dict(test="context", context=ctx, severity=0, source="synthetic",
                 **xl_metrics(model, G, t, cfg, device, ctx=ctx)) for ctx in dict.fromkeys(cfg["ctx_synthetic"])]


def xl_test_savings(model, G, t, cfg, device):
    """Relearn from k forgotten images (plus retain replay) and score the *other* forgotten images: does
    relearning part of the forget set bring back the advantage on the rest?"""
    rng = np.random.RandomState(stable_seed(f"xlshots|{t.seed}|{t.draw}"))
    perm = torch.from_numpy(rng.permutation(t.fidx.numel())).to(t.fidx.device)
    evals = set(cfg["sv_eval_steps"])
    rows = []
    for k in cfg["xl_sv_shots"]:
        kk = max(1, min(k, t.fidx.numel() // 2))
        shots, held = t.fidx[perm[:kk]], t.fidx[perm[kk:]]
        M = (lambda mm, held=held: xl_metrics(mm, G, t, cfg, device, idx=held))
        base = M(model)
        m = copy.deepcopy(model)
        orig = bn_state(m)
        opt = _sgd(m, cfg["sv_lr"], cfg)
        scaler = make_scaler(amp_on(cfg, device))
        g = torch.Generator(device="cpu")
        g.manual_seed(stable_seed(f"xlreplay|{t.seed}|{t.draw}|{k}"))
        if 0 in evals:
            for mode in bn_modes(model, cfg):
                rows.append(dict(test="savings", shots=kk, bn=mode, level=0, collapsed=False, **base))
        for step in range(1, cfg["sv_steps"] + 1):
            m.train()
            rb = t.ridx[torch.randint(0, t.ridx.numel(), (cfg["sv_replay_bs"],), generator=g).to(t.ridx.device)]
            sb = shots[torch.randint(0, kk, (min(kk, cfg["sv_replay_bs"]),), generator=g).to(shots.device)]
            b = torch.cat([sb, rb])
            x = prep(G["xtr"][b], cfg, True)
            with ac(cfg, device):
                loss = F.cross_entropy(m(x), G["ytr"][b])
            _step(loss, opt, scaler, m, clip=cfg["clip_grad"])
            if step in evals:
                for mode, met in eval_modes(m, G, None, None, cfg, device, orig, M):
                    rows.append(dict(test="savings", shots=kk, bn=mode, level=step,
                                     collapsed=_collapsed(met, base["retain_test_acc"], cfg), **met))
        del m
    return rows


def task_xl_retrain(job, cfg, G, D, device, log):
    s, d = job["seed"], job["draw"]
    t = xl_target(G, cfg, s, d)
    name = _xck("retrain", cfg, d, s)
    path = find_file(name) if cfg["reuse_ckpt"] else None
    if path:
        log(f"reusing existing checkpoint {path}")
        model = load_model(path, cfg, device)
    else:
        model = train_model(cfg, G, t.ridx, s, device, log, tag=job["key"])
        roundtrip(model, cfg)
        save_model(model, name, cfg, _meta(cfg, seed=s, draw=d, design=cfg["xl_design"], xl_frac=cfg["xl_frac"],
                                           xl_n=cfg["xl_n"]))
    return [dict(kind="xlretrain", method="retrain", seed=s, draw=d, design=cfg["xl_design"],
                 n_forget=int(t.fidx.numel()), **xl_metrics(model, G, t, cfg, device))]


def task_xl_calibrate(job, cfg, G, D, device, log):
    s, d, method, hp = job["seed"], job["draw"], job["method"], job["hp"]
    t = xl_target(G, cfg, s, d)
    model = load_model(ckpt_name("original", cfg, seed=s), cfg, device)
    ts = time.time()
    run_unlearning(method, model, G, D, t, hp, cfg, device)
    secs = time.time() - ts
    roundtrip(model, cfg)
    m = xl_metrics(model, G, t, cfg, device)
    ref = xl_ref_metrics(s, d, cfg, G, device)
    ok = xl_is_matched(m, ref, cfg)
    log(f"{job['key']} forget_acc={m['forget_acc']:.3f} (retrain {ref['forget_acc']:.3f}) test_acc={m['test_acc']:.3f} "
        f"(retrain {ref['test_acc']:.3f}) matched={ok} {secs:.0f}s")
    return [dict(kind="xlcalib", method=method, seed=s, draw=d, design=cfg["xl_design"], hp=json.dumps(hp, sort_keys=True),
                 hp_index=job["hp_index"], matched=ok, ref_forget_acc=ref["forget_acc"], ref_test_acc=ref["test_acc"],
                 unlearn_seconds=secs, **m)]


def task_xl_unlearn(job, cfg, G, D, device, log):
    method, s, d = job["method"], job["seed"], job["draw"]
    t = xl_target(G, cfg, s, d)
    name = _xck("unl", cfg, d, s, method)
    ref = xl_ref_metrics(s, d, cfg, G, device)
    path = find_file(name) if cfg["reuse_ckpt"] else None
    if path:
        log(f"reusing existing checkpoint {path}")
        model = load_model(path, cfg, device)
        m = xl_metrics(model, G, t, cfg, device)
        best = dict(model=model, m=m, hp=None, ok=xl_is_matched(m, ref, cfg), tries=0, secs=0.0)
    else:
        best = None
        for i, hp in enumerate(job["candidates"][:cfg["max_tries"]]):
            model = load_model(ckpt_name("original", cfg, seed=s), cfg, device)
            ts = time.time()
            run_unlearning(method, model, G, D, t, hp, cfg, device)
            secs = time.time() - ts
            roundtrip(model, cfg)
            m = xl_metrics(model, G, t, cfg, device)
            ok = xl_is_matched(m, ref, cfg)
            score = (ok, m["test_acc"] if ok else -xl_violation(m, ref, cfg))
            log(f"{job['key']} try {i + 1}: forget_acc={m['forget_acc']:.3f} (retrain {ref['forget_acc']:.3f}) "
                f"test_acc={m['test_acc']:.3f} matched={ok}")
            if best is None or score > best["score"]:
                best = dict(model=model, m=m, hp=hp, ok=ok, tries=i + 1, secs=secs, score=score)
            else:
                del model
            if ok:
                break
        if best is None:
            raise RuntimeError(f"No hyper-parameter candidates given for {method}.")
        save_model(best["model"], name, cfg, _meta(cfg, seed=s, draw=d, method=method, hp=best["hp"],
                                                   design=cfg["xl_design"]))
    return [dict(kind="xlunl", method=method, seed=s, draw=d, design=cfg["xl_design"],
                 hp=json.dumps(best["hp"], sort_keys=True),
                 matched=best["ok"], tries=best["tries"], unlearn_seconds=best["secs"],
                 ref_forget_acc=ref["forget_acc"], ref_test_acc=ref["test_acc"], ref_mem_gap=ref["mem_gap"],
                 **best["m"])]


def task_xl_extinction(job, cfg, G, D, device, log):
    kind, method, s, d = job["kind"], job["method"], job["seed"], job["draw"]
    t = xl_target(G, cfg, s, d)
    model = load_model(job["ckpt"], cfg, device)
    common = dict(kind=kind, method=method, seed=s, draw=d, matched=job.get("matched"), design=cfg["xl_design"])
    M = (lambda mm: xl_metrics(mm, G, t, cfg, device))
    rows = [dict(common, test="base", **M(model))]
    rows += [dict(common, **r) for r in test_spontaneous(model, G, D, t, cfg, device, s, mfn=M)]
    rows += [dict(common, **r) for r in xl_test_contexts(model, G, t, cfg, device)]
    rows += [dict(common, **r) for r in xl_test_savings(model, G, t, cfg, device)]
    return rows


@torch.no_grad()
def _margins(model, x, c, cfg, device):
    z = logits_of(model, x, cfg, device)
    zc = z[:, c].clone()
    z[:, c] = -float("inf")
    return zc - z.max(1).values


@torch.no_grad()
def headbias_row(model, orig, G, D, c, cfg, device):
    """Head-bias check (class level). bias_z: the forgotten class's output bias against the other 99, in
    standard scores. bias_restore / row_restore: forget-class accuracy after putting back the original model's
    bias (one number) or its whole output row. shift_recall: forget-class recall after adding the one constant
    to that output that makes 1% of retain training images flip to it (set on training data, measured on test)."""
    b, bo = model.fc.bias.detach().float(), orig.fc.bias.detach().float()
    w, wo = model.fc.weight.detach().float(), orig.fc.weight.detach().float()
    others = torch.tensor([k for k in range(b.numel()) if k != c], device=b.device)
    row = dict(bias=float(b[c]), bias_orig=float(bo[c]),
               bias_z=float((b[c] - b[others].mean()) / b[others].std()),
               bias_z_orig=float((bo[c] - bo[others].mean()) / bo[others].std()),
               row_cos=float(F.cosine_similarity(w[c], wo[c], dim=0)),
               row_norm_ratio=float(w[c].norm() / wo[c].norm().clamp_min(1e-12)))
    m1 = copy.deepcopy(model)
    m1.fc.bias[c] = bo[c]
    r1 = metrics(m1, G, D, c, cfg, device)
    m2 = copy.deepcopy(model)
    m2.fc.bias[c] = bo[c]
    m2.fc.weight[c] = wo[c]
    r2 = metrics(m2, G, D, c, cfg, device)
    del m1, m2
    ri = G["retain_eval_idx"]
    ri = ri[G["ytr"][ri] != c]
    delta = -float(torch.quantile(_margins(model, G["xtr"][ri], c, cfg, device).float(), 0.99))
    mte = _margins(model, G["xte"], c, cfg, device)
    f = G["yte"] == c
    row.update(bias_restore_forget_acc=r1["forget_test_acc"], bias_restore_retain_acc=r1["retain_test_acc"],
               bias_restore_fpr=r1["forget_fpr"], row_restore_forget_acc=r2["forget_test_acc"],
               row_restore_retain_acc=r2["retain_test_acc"], shift_delta=delta,
               shift_recall=float((mte[f] + delta > 0).float().mean()),
               shift_fpr=float((mte[~f] + delta > 0).float().mean()))
    # 1.4: set the constant on one half of the test set (1% of its other-class images flip) and measure on the other
    # half, so the false-positive rate is really about 1% (the training-image version above gave 4-12%).
    half = (torch.arange(mte.numel(), device=mte.device) % 2) == 0
    for q, tag in ((0.99, "1"), (0.95, "5")):
        d2 = -float(torch.quantile(mte[half & ~f].float(), q))
        row[f"shift_recall_fpr{tag}"] = float((mte[~half & f] + d2 > 0).float().mean())
        row[f"shift_fpr_fpr{tag}"] = float((mte[~half & ~f] + d2 > 0).float().mean())
    return row


def task_headbias(job, cfg, G, D, device, log):
    kind, method, cls, s = job["kind"], job["method"], job["cls"], job["seed"]
    model = load_model(job["ckpt"], cfg, device)
    row = headbias_row(model, get_original(cfg, s, device), G, D, cid(D, cls), cfg, device)
    return [dict(kind=kind, method=method, cls=cls, seed=s, matched=job.get("matched"), test="headbias", **row)]


TASKS.update(xl_retrain=task_xl_retrain, xl_calibrate=task_xl_calibrate, xl_unlearn=task_xl_unlearn,
             xl_extinction=task_xl_extinction, headbias=task_headbias)


def xl_calib_grid(cfg):
    default = DEFAULT_XA_CALIB_GRID if cfg["xl_design"] == "atypical" else DEFAULT_XL_CALIB_GRID
    grid = copy.deepcopy(cfg["xl_calib_grid"] or default)
    missing = [m for m in cfg["methods"] if m not in grid]
    if missing:
        raise KeyError(f"xl_calib_grid has no entry for {missing}")
    if cfg["calib_limit"]:
        grid = {m: g[:cfg["calib_limit"]] for m, g in grid.items()}
    return {m: grid[m] for m in cfg["methods"]}


def _xl_units(cfg, pilot):
    if pilot:
        return [(cfg["seeds"][0], cfg["xl_pilot_draw"])]
    return [(s, d) for s in cfg["seeds"] for d in cfg["xl_draws"]]


def jobs_xl_retrain(cfg, shard=None):
    p = xl_prefix(cfg)
    units = _xl_units(cfg, True) + _xl_units(cfg, False)
    return _shard([dict(key=f"{p}retrain|s{s}|d{d}", seed=s, draw=d, out_ckpt=_xck("retrain", cfg, d, s))
                   for s, d in units], shard)


def jobs_xl_calibrate(cfg):
    p = xl_prefix(cfg)
    s, d = _xl_units(cfg, True)[0]
    return [dict(key=f"{p}calib|{m}|{i}|{short_hash(json.dumps(hp, sort_keys=True))}", method=m, hp=hp, hp_index=i,
                 seed=s, draw=d) for m, g in xl_calib_grid(cfg).items() for i, hp in enumerate(g)]


def _xl_cal_file(cfg):
    return f"xl_calibration{'_atypical' if cfg['xl_design'] == 'atypical' else ''}{sfx(cfg)}.json"


def xl_select_candidates(cfg):
    """Rank example-level calibration runs per method: matched first (highest test accuracy), then by how far they
    miss the matching rule (xl_violation). Saves xl_calibration.json."""
    import pandas as pd
    rows = [r for r in load_rows("xl_calibrate", cfg) if _design(r) == cfg["xl_design"]]
    if not rows:
        raise RuntimeError("No example-level calibration results found for this design. Run the calibration stage first.")
    out = {}
    for m in cfg["methods"]:
        rm = [r for r in rows if r["method"] == m]
        if not rm:
            print(f"[xl calibration] WARNING: no results for {m}")
            continue
        rm.sort(key=lambda r: (not r["matched"], -r["test_acc"] if r["matched"] else
                               xl_violation(r, dict(forget_acc=r["ref_forget_acc"], test_acc=r["ref_test_acc"]), cfg)))
        out[m] = [json.loads(r["hp"]) for r in rm][:max(1, cfg["max_tries"])]
    ensure_dirs()
    path = os.path.join(RES_DIR, _xl_cal_file(cfg))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[xl calibration] saved ranked candidates to {path}")
    df = pd.DataFrame(rows)
    cols = ["method", "hp", "matched", "forget_acc", "ref_forget_acc", "test_acc", "ref_test_acc", "mem_gap",
            "unlearn_seconds"]
    return out, df[[c for c in cols if c in df.columns]].sort_values(["method", "matched"], ascending=[True, False])


def load_xl_candidates(cfg):
    p = find_file(_xl_cal_file(cfg))
    if p is None:
        raise FileNotFoundError(f"{_xl_cal_file(cfg)} not found - run the calibration stage first.")
    with open(p) as f:
        return json.load(f)


def jobs_xl_unlearn(cfg, candidates, pilot=False, shard=None):
    jobs = []
    for s, d in _xl_units(cfg, pilot):
        for m in cfg["methods"]:
            if m not in candidates:
                raise KeyError(f"No example-level calibrated hyper-parameters for {m}; run stage B first.")
            jobs.append(dict(key=f"{xl_prefix(cfg)}unl|{m}|s{s}|d{d}", method=m, seed=s, draw=d, candidates=candidates[m],
                             out_ckpt=_xck("unl", cfg, d, s, m)))
    return _shard(jobs, shard)


def jobs_xl_extinction(cfg, shard=None):
    """Example-level relapse tests for every unlearned model (main draws), its retrained model and the original."""
    p = xl_prefix(cfg)
    jobs = {}
    for r in load_rows("xl_unlearn", cfg):
        if (_design(r) != cfg["xl_design"] or r["draw"] not in cfg["xl_draws"] or r["seed"] not in cfg["seeds"]
                or r["method"] not in cfg["methods"]):
            continue
        k = f"{p}ext|xlunl|{r['method']}|s{r['seed']}|d{r['draw']}"
        jobs[k] = dict(key=k, kind="xlunl", method=r["method"], seed=r["seed"], draw=r["draw"], matched=r["matched"],
                       ckpt=_xck("unl", cfg, r["draw"], r["seed"], r["method"]))
    for s, d in _xl_units(cfg, False):
        k = f"{p}ext|xlretrain|retrain|s{s}|d{d}"
        jobs[k] = dict(key=k, kind="xlretrain", method="retrain", seed=s, draw=d, matched=True,
                       ckpt=_xck("retrain", cfg, d, s))
        k = f"{p}ext|original|original|s{s}|d{d}"
        jobs[k] = dict(key=k, kind="original", method="original", seed=s, draw=d, matched=None,
                       ckpt=ckpt_name("original", cfg, seed=s))
    out = sorted(jobs.values(), key=lambda j: j["key"])
    missing = [j for j in out if find_file(j["ckpt"]) is None]
    if missing:
        print(f"WARNING: {len(missing)} checkpoint(s) not found and skipped, e.g. {missing[0]['ckpt']}.")
    return _shard([j for j in out if j not in missing], shard)


def jobs_headbias(cfg, shard=None):
    """Head-bias check for every class-level model of notebook 03 (plus retrained and original references)."""
    return _shard(_model_jobs(cfg, "hb", False, ["headbias"], ["headbias"], ["headbias"], ["headbias"]), shard)

if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "worker":
        with open(sys.argv[2]) as f:
            spec = json.load(f)
        _, n_err = run_worker(spec["task"], spec["jobs"], spec["cfg"], int(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)
    print(__doc__)

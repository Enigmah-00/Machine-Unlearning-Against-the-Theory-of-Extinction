"""Paper figures, drawn from the raw result files in ../results.

Inputs: notebook 04 rows (class level, run 07), notebook 05 rows (mechanism, run 06), notebook 06 tests (run 08),
notebook 08 tables for the atypical example-level arm (runs 14 and 16).
Output: paper/figures/*.pdf and *.png.
Usage: python paper/make_figures.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R28 = os.path.join(RES, "run14_nb08_atypical")
R29 = os.path.join(RES, "run16_nb08_final")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})
METHODS = ["FT", "NegGrad+", "RL", "SCRUB", "SalUn"]
COL = {"FT": "#E69F00", "NegGrad+": "#D55E00", "RL": "#0072B2", "SCRUB": "#009E73", "SalUn": "#CC79A7",
       "retrain": "#000000", "original": "#999999"}


def load_rows(pattern):
    rows = []
    for p in glob.glob(pattern):
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    done = {r["key"] for r in rows if r.get("_done")}
    return pd.DataFrame([r for r in rows if not r.get("_done") and r.get("key") in done])


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


ce = load_rows(os.path.join(RES, "run07_nb04_main", "results", "extinction__r*.jsonl"))
ce = ce[ce["cls"] != "lamp"]
keep = ~ce["kind"].isin(["unl", "aba", "mctx"]) | (ce["matched"] == True)
ce = ce[keep]
if "collapsed" in ce:
    ce = ce[~ce["collapsed"].astype("boolean").fillna(False).astype(bool)]

# ---- Figure: output selectivity (AUROC of the forgotten class's output) -------------------------
b = ce[(ce["test"] == "base") & ce["kind"].isin(["unl", "retrain", "original"])].copy()
b["who"] = np.where(b["kind"] == "unl", b["method"], b["kind"])
order = ["original"] + METHODS + ["retrain"]
fig, ax = plt.subplots(figsize=(5.6, 2.4))
rng = np.random.default_rng(1)
for i, w in enumerate(order):
    v = b[b["who"] == w]["forget_auc"].values
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=COL[w], alpha=0.55, linewidths=0)
    ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color="k", lw=1.4)
ax.axhline(0.5, color="#666666", lw=0.8, ls=":")
ax.set_xticks(range(len(order)), ["original"] + METHODS + ["retrained"])
ax.set_ylabel("AUROC of the forgotten\nclass's output")
ax.set_ylim(-0.03, 1.03)
ax.text(-0.45, 0.52, "chance", fontsize=7, color="#666666", ha="left", va="bottom")
save(fig, "fig_selectivity")

# ---- Figure: class-level relapse tests, primary score vs rank score -----------------------------
ht = pd.read_csv(os.path.join(RES, "run08_nb06_analysis", "tables", "hypothesis_tests.csv"))
tests = [("H1 spontaneous recovery", "R: H1 [rank_score]", "Spontaneous recovery\n(retain fine-tuning, 5 epochs)"),
         ("H2a renewal (ABC)", "R: H2a [rank_score]", "Renewal\n(new contexts)"),
         ("H3 reinstatement (sibling - dissimilar)", "R: H3 [rank_score]", "Reinstatement\n(sibling minus dissimilar cues)")]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
for ax, (hp, hr, title) in zip(axes, tests):
    for j, (h, lab, mk) in enumerate([(hp, "top-5 (primary)", "o"), (hr, "rank score", "s")]):
        t = ht[ht["hypothesis"] == h].set_index("method")
        for i, m in enumerate(METHODS):
            if m not in t.index:
                continue
            r = t.loc[m]
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]], fmt=mk,
                        color=COL[m], mfc=COL[m] if j == 0 else "white", ms=4, lw=1, capsize=1.5,
                        label=lab if i == 0 else None)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xticks(range(5), METHODS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title)
axes[0].set_ylabel("unlearned minus retrained")
h1 = plt.Line2D([], [], marker="o", color="#444444", ls="", ms=4, label="top-5 score (primary)")
h2 = plt.Line2D([], [], marker="s", color="#444444", mfc="white", ls="", ms=4, label="rank score")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.27))
save(fig, "fig_class_relapse")

# ---- Figure: relearning curves (5 and 100 forgotten images) -------------------------------------
sv = ce[(ce["test"] == "savings") & (ce["bn"] == "update") & ce["kind"].isin(["unl", "retrain"])].copy()
sv["who"] = np.where(sv["kind"] == "unl", sv["method"], "retrain")
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
for ax, k in zip(axes, [5, 100]):
    s = sv[sv["shots"] == k]
    for w in METHODS + ["retrain"]:
        g = s[s["who"] == w].groupby("level")["forget_test_acc"]
        mu = g.mean()
        x = np.array(mu.index, float); x[x == 0] = 0.6
        ax.plot(x, mu.values, marker="o", ms=2.5, lw=1.2, color=COL[w], ls="--" if w == "retrain" else "-",
                label="retrained" if w == "retrain" else w)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 1, 2, 5, 10, 20, 50, 100], ["0", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("relearning steps")
    ax.set_title(f"relearning from {k} forgotten images")
axes[0].set_ylabel("forget-class accuracy (top-1)")
axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
save(fig, "fig_relearning")

# ---- Figure: CKA with the original model, forgotten-class images -------------------------------
mech = load_rows(os.path.join(RES, "run06_nb05_mechanism", "results", "mechanism__r*.jsonl"))
ck = mech[(mech["test"] == "cka") & (mech["kind"] == "unl") & (mech["matched"] == True) & (mech["cls"] != "lamp")]
ck = ck[(ck["ref"] == "original") & (ck["subset"] == "forget")]
blocks = [b_ for b_ in ["stem", "layer1", "layer2", "layer3", "layer4", "penult", "logits"] if b_ in set(ck["block"])]
fig, ax = plt.subplots(figsize=(3.6, 2.3))
for m in METHODS:
    g = ck[ck["method"] == m].groupby("block")["cka"].mean().reindex(blocks)
    ax.plot(range(len(blocks)), g.values, marker="o", ms=3, lw=1.2, color=COL[m], label=m)
ax.set_xticks(range(len(blocks)), blocks, rotation=35)
ax.set_ylabel("linear CKA with the original\n(forgotten-class images)")
ax.set_ylim(0.4, 1.02)
ax.legend(frameon=False, loc="lower left")
save(fig, "fig_cka")

# ---- Figure: example level, relapse vs completeness of forgetting --------------------------------
v = pd.read_csv(f"{R29}/tables/XH1_by_variant.csv")
ORIG = 0.996
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
sp28 = pd.read_csv(f"{R28}/tables/XH1_spontaneous_all_proxies.csv")
sp29 = pd.read_csv(f"{R29}/tables/XH1_spontaneous_all_proxies.csv")
def curve(t, col):
    r = t[(t["proxy"] == "retain_ft") & (t["bn"] == "update")].sort_values("level")
    return r["level"].values, r[col].values
for m, ls, lab in [("SalUn", "--", "SalUn (5 ep)"), ("RL", "--", "RL (5 ep)"), ("NegGrad+", "--", "NegGrad+ (harsh)")]:
    x, y = curve(sp28, m); ax.plot(x, y, ls=ls, marker="o", ms=2.5, lw=1.1, color=COL[m], label=lab)
x, y = curve(sp29, "RL"); ax.plot(x, y, ls="-", marker="o", ms=3, lw=1.6, color=COL["RL"], label="RL-long (matched)")
x, y = curve(sp28, "retrain"); ax.plot(x, y, ls="-", marker="o", ms=2.5, lw=1.1, color="k", label="retrained")
ax.axhline(0, color="#999999", lw=0.6)
ax.set_xlabel("epochs of fine-tuning on retained data")
ax.set_ylabel("gain in accuracy on\nforgotten images")
ax.set_title("(a) relapse during retain fine-tuning")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, columnspacing=0.8)
ax.set_ylim(-0.03, 0.62)

ax = axes[1]
lab = {"RL": "RL", "RL-long": "RL-long", "SalUn": "SalUn", "SalUn-long": "SalUn-long", "NegGrad+": "NegGrad+",
       "NegGrad+-long": "NegGrad+-long", "FT": "FT", "SCRUB": "SCRUB"}
for var, g in v.groupby("variant"):
    base = var.split("-")[0]
    mx, lx, hx = boot(g["before"]); my, ly, hy = boot(g["excess"])
    filled = g["is_matched"].mean() >= 0.5
    ax.errorbar(mx, my, xerr=[[mx - lx], [hx - mx]], yerr=[[my - ly], [hy - my]], fmt="o", ms=5, color=COL[base],
                mfc=COL[base] if g["is_matched"].all() else "white", capsize=1.5, lw=0.9)
    dx, dy = {"RL-long": (0.014, 0.022), "SalUn-long": (0.014, -0.002), "NegGrad+-long": (0.012, -0.032),
              "NegGrad+": (-0.012, 0.075), "RL": (0.014, 0.0), "SalUn": (0.014, -0.025), "FT": (-0.02, 0.022),
              "SCRUB": (-0.03, 0.025)}.get(var, (0.01, 0.01))
    ax.text(mx + dx, my + dy, lab[var], fontsize=7, color=COL[base])
ax.axvspan(0, 0.034 + 0.05, color="#dddddd", alpha=0.6, lw=0)
ax.text(0.004, 0.47, "matching\nwindow", fontsize=7, color="#555555", va="top")
ax.axhline(0, color="#999999", lw=0.6)
ax.set_xlim(-0.01, 0.56); ax.set_ylim(-0.03, 0.5)
ax.set_xlabel("forget accuracy after unlearning")
ax.set_ylabel("relapse minus retrained")
ax.set_title("(b) relapse shrinks as forgetting completes")
save(fig, "fig_example_dose")

# ---- numbers used in the text -------------------------------------------------------------------
print(b.groupby("who")["forget_auc"].agg(["mean", "count"]).round(3))
for k in [5, 100]:
    s = sv[(sv["shots"] == k) & (sv["level"] == 20)]
    print(k, s.groupby("who")["forget_test_acc"].mean().round(3).to_dict())
print(ck.groupby(["method", "block"])["cka"].mean().unstack().round(2))
print(v.groupby("variant").apply(lambda g: pd.Series(dict(before=g.before.mean(), excess=g.excess.mean(),
      frac=(g.excess / (ORIG - g.before)).mean(), matched=g.is_matched.mean()))).round(3))

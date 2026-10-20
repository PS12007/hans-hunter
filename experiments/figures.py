"""All figures, regenerated from results/*.json (and runs/ for image galleries).

python experiments/figures.py [name ...]   # default: every figure whose inputs exist
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D

OUT = "docs/figures"
ARTS = list(D.ARTIFACTS)
# validated categorical palette, fixed order: one hue per artifact
COL = dict(tag="#2a78d6", ruler="#eb6834", ink="#1baf7a", tint="#eda100", noise="#e87ba4")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SHORT = dict(tag="Tag", ruler="Ruler", ink="Ink", tint="Tint", noise="Invisible")
DSETS = ["derma", "pneumonia", "breast"]
DNAME = {"derma": "Skin lesions (malignant)", "pneumonia": "Chest X-ray (pneumonia)",
         "breast": "Breast ultrasound (malignant)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "lines.linewidth": 2,
})


def load_dir(d):
    return [json.load(open(f)) for f in sorted(glob.glob(f"results/{d}/*.json"))]


def agg(rows, key, by=("dataset", "artifact", "p")):
    g = defaultdict(list)
    for r in rows:
        v = r
        for k in key.split("."):
            v = v[k]
        g[tuple(r[b] for b in by)].append(v)
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in g.items()}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.svg")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
def fig_artifacts():
    fig, ax = plt.subplots(3, 6, figsize=(7.2, 3.9))
    for r, name in enumerate(DSETS):
        ds = D.load(name)
        img = ds["test"][0][3 if name != "derma" else 7]
        ax[r, 0].imshow(img)
        ax[r, 0].set_ylabel(DNAME[name].split(" (")[0], fontsize=8, color=INK2)
        for c, a in enumerate(ARTS):
            out, _ = D.ARTIFACTS[a](img, np.random.default_rng(0))
            ax[r, c + 1].imshow(np.clip(out, 0, 1))
        for c in range(6):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].grid(False)
            for s in ax[r, c].spines.values():
                s.set_visible(False)
            if r == 0:
                ax[r, c].set_title("Original" if c == 0 else D.ART_NAMES[ARTS[c - 1]], fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, "artifacts")


def fig_phase():
    rows = load_dir("phase")
    if not rows:
        return
    have = [d for d in DSETS if any(r["dataset"] == d for r in rows)]
    A = agg(rows, "auc_indep")
    S = agg(rows, "auc_source")
    E = agg(rows, "effect_neg_flip")
    fig, ax = plt.subplots(2, len(have), figsize=(2.6 * len(have) + 0.6, 5.0), squeeze=False, sharex=True)
    for c, name in enumerate(have):
        a0, a1 = ax[0, c], ax[1, c]
        ps = sorted({k[2] for k in A if k[0] == name})
        src = [np.mean([S[(name, a, p)][0] for a in ARTS if (name, a, p) in S]) for p in ps]
        a0.plot(ps, src, color=MUTED, lw=1.6, ls=(0, (4, 2)))
        a0.text(ps[-1], src[-1] + 0.012, "own hospital", ha="right", va="bottom", fontsize=7.5, color=INK2)
        for a in ARTS:
            ks = [(name, a, p) for p in ps if (name, a, p) in A]
            if not ks:
                continue
            a0.plot([k[2] for k in ks], [A[k][0] for k in ks], color=COL[a], marker="o", ms=4)
            a1.plot([k[2] for k in ks], [100 * E[k][0] for k in ks], color=COL[a], marker="o", ms=4)
        a0.set_title(DNAME[name])
        a0.set_ylim(0.0, 1.02)
        a1.set_ylim(-3, 103)
        a1.set_xlabel("P(artifact | disease) in training")
        if c == 0:
            a0.set_ylabel("AUC at a new hospital")
            a1.set_ylabel("Healthy patients flagged\nwhen the artifact is added (%)")
        a0.axhline(0.5, color=GRID, lw=1)
    handles = [plt.Line2D([], [], color=COL[a], marker="o", ms=4, label=D.ART_NAMES[a]) for a in ARTS]
    handles.append(plt.Line2D([], [], color=MUTED, ls=(0, (4, 2)), label="Own hospital (mean)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "phase")


def _trigger_img(z, base):
    """Patch trigger drawn over a dimmed example image."""
    m = z["mask"][..., None]
    return np.clip((1 - m) * (0.35 * base + 0.3) + m * z["pattern"], 0, 1)


def _delta_img(z):
    d = z["delta"]
    s = np.abs(d).max() + 1e-9
    return np.clip(0.5 + 0.5 * d / s, 0, 1)


def fig_triggers():
    if not os.path.exists("runs/forensics"):
        return
    for name in DSETS:
        ds = D.load(name)
        base = ds["test"][0][ds["test"][1] == 0][0]
        fig, ax = plt.subplots(4, 5, figsize=(6.6, 5.6))
        R = {}
        for r in load_dir("forensics"):
            if r["dataset"] == name and r["seed"] == 0:
                R[(r["artifact"], r["p"])] = r
        for c, a in enumerate(ARTS):
            for k, p in enumerate((0.5, 0.9)):
                f = f"runs/forensics/{name}_{a}_p{p}_s0.npz"
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                ax[2 * k, c].imshow(_trigger_img(z, base))
                ax[2 * k + 1, c].imshow(_delta_img(z))
                rr = R.get((a, p))
                if rr:
                    ax[2 * k, c].set_xlabel(f"{rr['cost_patch']:.0f} px", fontsize=7, color=INK2, labelpad=1)
                    ax[2 * k + 1, c].set_xlabel(f"RMS {rr['cost_additive']:.1f}/255", fontsize=7, color=INK2, labelpad=1)
            ax[0, c].set_title(D.ART_NAMES[a], fontsize=8)
        labels = ["Honest model\npatch", "Honest model\nadditive", "Shortcut model\npatch", "Shortcut model\nadditive"]
        for r_ in range(4):
            ax[r_, 0].set_ylabel(labels[r_], fontsize=7.5, color=INK2)
            for c in range(5):
                ax[r_, c].set_xticks([]); ax[r_, c].set_yticks([]); ax[r_, c].grid(False)
                for sp in ax[r_, c].spines.values():
                    sp.set_visible(False)
        fig.suptitle(f"{DNAME[name]}: recovered triggers (p = 0.5 vs p = 0.9)", fontsize=9, color=INK, y=0.995)
        fig.tight_layout(h_pad=0.6, w_pad=0.3)
        save(fig, f"triggers_{name}")


def fig_detect():
    rows = load_dir("forensics")
    if not rows:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=False)
    for c, name in enumerate(DSETS):
        a = ax[c]
        R = [r for r in rows if r["dataset"] == name]
        ps = sorted({r["p"] for r in R})
        for k, art in enumerate(ARTS):
            xs, ys = [], []
            for p in ps:
                v = [r["cost_patch"] for r in R if r["artifact"] == art and r["p"] == p]
                if v:
                    xs.append(p); ys.append(np.mean(v))
            a.plot(xs, ys, color=COL[art], marker="o", ms=3.5, lw=1.6)
        a.set_yscale("log")
        a.set_title(DNAME[name])
        a.set_xlabel("P(artifact | disease)")
        if c == 0:
            a.set_ylabel("Smallest flipping patch (pixels)")
    handles = [plt.Line2D([], [], color=COL[a], marker="o", ms=3.5, label=D.ART_NAMES[a]) for a in ARTS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "detect")


def fig_ground():
    rows = load_dir("ground")
    if not rows:
        return
    meths = [("within_auc_direction", "Trigger direction (ours)", "#2a78d6"),
             ("within_auc_logit", "Model score", "#8a8984"),
             ("within_auc_additive_match", "Pixel match (additive)", "#eb6834"),
             ("within_auc_patch_match", "Pixel match (patch)", "#eda100")]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for c, name in enumerate(DSETS):
        a = ax[c]
        R = [r for r in rows if r["dataset"] == name and r["p"] == 0.9]
        x = np.arange(len(ARTS))
        w = 0.2
        for k, (key, lab, col) in enumerate(meths):
            v = [np.mean([r[key] for r in R if r["artifact"] == art] or [np.nan]) for art in ARTS]
            a.bar(x + (k - 1.5) * w, v, w * 0.92, color=col, label=lab)
        a.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 2)))
        a.set_xticks(x, [SHORT[t] for t in ARTS], fontsize=7)
        a.set_ylim(0.3, 1.02)
        a.set_title(DNAME[name])
        a.grid(axis="x", visible=False)
        if c == 0:
            a.set_ylabel("Within-class AUROC for\nfinding shortcut images")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "ground")


def fig_fix():
    fix = {(r["dataset"], r["artifact"], r["p"], r["seed"]): r for r in load_dir("fix")}
    rw = {(r["dataset"], r["artifact"], r["p"], r["seed"]): r for r in load_dir("reweight")}
    if not fix and not rw:
        return
    conds = [("none", "Shortcut model", "#8a8984", fix),
             ("finetune", "Fine-tune (control)", "#c3c2b7", fix),
             ("scrub2", "Trigger scrubbing", "#eda100", fix),
             ("reweight", "Reweight, inferred groups", "#eb6834", rw),
             ("project_confirmed", "Project out, confirmed groups", "#1baf7a", rw),
             ("reweight_confirmed", "Reweight, confirmed groups", "#2a78d6", rw),
             ("oracle", "Trained without shortcut", "#0b0b0b", fix)]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=False)
    for c, name in enumerate(DSETS):
        a = ax[c]
        x = np.arange(len(ARTS))
        w = 0.8 / len(conds)
        for k, (key, lab, col, src) in enumerate(conds):
            v = []
            for art in ARTS:
                vals = [r[key]["auc_indep"] for kk, r in src.items() if kk[0] == name and kk[1] == art and kk[2] == 0.9 and key in r]
                v.append(np.mean(vals) if vals else np.nan)
            a.bar(x + (k - (len(conds) - 1) / 2) * w, v, w * 0.92, color=col, label=lab)
        a.set_xticks(x, [SHORT[t] for t in ARTS], fontsize=7)
        lo = 0.5 if name != "pneumonia" else 0.85
        a.set_ylim(lo, 1.0 if name == "pneumonia" else 0.95)
        a.set_title(DNAME[name])
        a.grid(axis="x", visible=False)
        if c == 0:
            a.set_ylabel("AUC at a new hospital")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fix")


def _clean_axis(a):
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)


def fig_saliency():
    if not os.path.exists("results/saliency.json") or not os.path.exists("runs/saliency_examples.npz"):
        return
    z = np.load("runs/saliency_examples.npz")
    rows = json.load(open("results/saliency.json"))
    F = load_dir("forensics")
    vis = ["tag", "ruler", "ink"]
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.35], hspace=0.25, wspace=0.05)
    for r, name in enumerate(DSETS):
        for k, a in enumerate(vis):
            for j, p in enumerate((0.5, 0.9)):
                ax = fig.add_subplot(gs[r, 2 * k + j])
                img, cam = z[f"{name}_{a}_{p}_img"][0], z[f"{name}_{a}_{p}_cam"][0]
                cam = cam / (cam.max() + 1e-9)
                ax.imshow(img)
                ax.imshow(cam, cmap="magma", alpha=0.55, vmin=0, vmax=1)
                _clean_axis(ax)
                if r == 0:
                    ax.set_title(f"{D.ART_NAMES[a]}\n{'honest' if p == 0.5 else 'shortcut'} (p={p})", fontsize=7.2)
                if k == 0 and j == 0:
                    ax.set_ylabel(DNAME[name].split(" (")[0], fontsize=7.5, color=INK2)
    bottom = gs[3, :].subgridspec(1, 3, wspace=0.12)
    for c, name in enumerate(DSETS):
        ax = fig.add_subplot(bottom[0, c])
        if c > 0:
            ax.tick_params(labelleft=False)
        for a in vis:
            R = [x for x in rows if x["dataset"] == name and x["artifact"] == a]
            ps = sorted({x["p"] for x in R})
            ax.plot(ps, [np.mean([x["gradcam_share"] for x in R if x["p"] == p]) for p in ps],
                    color=COL[a], marker="o", ms=3.5, lw=1.6)
            Q = [x for x in F if x["dataset"] == name and x["artifact"] == a]
            qs = sorted({x["p"] for x in Q})
            ax.plot(qs, [np.mean([x["patch_mass_on_artifact"] for x in Q if x["p"] == p]) for p in qs],
                    color=COL[a], ls=(0, (3, 2)), lw=1.4)
            area = R[0]["area_share"]
            ax.axhline(area, color=COL[a], lw=0.8, alpha=0.5, ls=":")
        ax.set_ylim(0, 1.02)
        ax.set_title(DNAME[name].split(" (")[0], fontsize=8.5)
        ax.set_xlabel("P(artifact | disease)")
        if c == 0:
            ax.set_ylabel("Share on the artifact")
    handles = [plt.Line2D([], [], color=COL[a], lw=1.6, label=D.ART_NAMES[a]) for a in vis]
    handles += [plt.Line2D([], [], color=INK2, lw=1.6, marker="o", ms=3.5, label="Grad-CAM mass"),
                plt.Line2D([], [], color=INK2, lw=1.4, ls=(0, (3, 2)), label="Recovered patch mass"),
                plt.Line2D([], [], color=INK2, lw=0.8, ls=":", label="Artifact area (chance)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.11, top=0.94)
    save(fig, "saliency")


def fig_real():
    if not os.path.exists("results/real_audit.json"):
        return
    rep = json.load(open("results/real_audit.json"))
    ds = D.load("breast")
    pairs = json.load(open("runs/real/breast_dup_pairs.json"))
    pairs = [q for q in pairs if q[0] == "test" and q[2] == "train"]
    conflict = [q for q in pairs if q[5] != q[6]]
    agree = sorted([q for q in pairs if q[5] == q[6]], key=lambda q: -q[4])
    show = (conflict[:2] + agree)[:4]
    fig = plt.figure(figsize=(7.4, 3.9))
    outer = fig.add_gridspec(2, 1, hspace=0.55)
    top = outer[0].subgridspec(1, 8, wspace=0.08)
    bot = outer[1].subgridspec(1, 10, wspace=0.08, width_ratios=[1, 1, 0.3, 1, 1, 0.3, 1, 1, 1.1, 2.2])
    lab = {0: "malignant", 1: "benign"}
    for k, (s1, i1, s2, i2, r, l1, l2) in enumerate(show):
        for j, (s, i, l) in enumerate(((s1, i1, l1), (s2, i2, l2))):
            ax = fig.add_subplot(top[0, 2 * k + j])
            ax.imshow(ds[s][0][i])
            _clean_axis(ax)
            ax.set_xlabel(f"{s} #{i}\n{lab[l]}", fontsize=6.8,
                          color="#c0392b" if l1 != l2 else INK2, labelpad=2)
            if j == 0:
                ax.set_title(f"r = {r:.3f}", fontsize=7, color=INK2, loc="left", x=0.55, fontweight="normal")
    for c, name in enumerate(DSETS):
        d = D.load(name)
        base = d["test"][0][d["test"][1] == 0][0]
        t = np.load(f"runs/real/{name}_trigger_to_disease.npz")
        tr = rep[name]["triggers"]["to_disease"]
        ax = fig.add_subplot(bot[0, 3 * c])
        ax.imshow(_trigger_img(t, base)); _clean_axis(ax)
        ax.set_xlabel(f"{tr['cost_patch']:.0f} px", fontsize=6.8, color=INK2, labelpad=2)
        ax.set_title(DNAME[name].split(" (")[0], fontsize=7.5, loc="left")
        if c == 0:
            ax.set_ylabel("Clean models:\ncheapest edit\nto 'disease'", fontsize=7, color=INK2)
        ax = fig.add_subplot(bot[0, 3 * c + 1])
        ax.imshow(_delta_img(t)); _clean_axis(ax)
        ax.set_xlabel(f"RMS {tr['cost_additive']:.1f}", fontsize=6.8, color=INK2, labelpad=2)
    ax = fig.add_subplot(bot[0, 9])
    names = ["derma", "pneumonia", "breast"]
    leak = [100 * rep[n]["test_leak_frac"] for n in names]
    ax.barh(range(3), leak, color="#2a78d6", height=0.55)
    for i, v in enumerate(leak):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(3), ["Skin", "X-ray", "Breast"], fontsize=7)
    ax.set_xlim(0, 24); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_title("Test images with a\nnear-copy in train", fontsize=7.5)
    ax.tick_params(labelsize=7)
    fig.suptitle("BreastMNIST test/train near-duplicates (red = the two copies carry opposite labels)",
                 fontsize=8.5, color=INK, y=1.0)
    save(fig, "real")


FIGS = dict(artifacts=fig_artifacts, phase=fig_phase, triggers=fig_triggers, detect=fig_detect,
            ground=fig_ground, fix=fig_fix, saliency=fig_saliency, real=fig_real)

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()

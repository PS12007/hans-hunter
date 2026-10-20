"""Attribution by splitting suspects in spatial feature space.

Section 7 found that two local markers (tag + ruler) defeat trigger-based attribution. Two
diagnostics explain why: the cheapest triggers are scattered strokes that copy the markers'
*shape* anywhere in the image (so forbidding a region does not peel anything off), and after
global pooling the model represents the two markers almost identically.

The spatial feature map before pooling still knows where each marker is. So:
  1. suspects: the 15% of healthy held-out images most suspicious under a guide score
     (the patch trigger's feature direction, the model's own score, or random as a control);
  2. split them with k-means (k = 2) on their class-centred, normalised last conv feature maps;
  3. score every image by its projection on the difference of the two centroids.

Graded (ground truth used only here) by how well that score separates images carrying only
artifact A from images carrying only B, within each class. Cluster labels have no names, so
we report separation = max(AUC, 1 - AUC) for the split and for the model score alike.

Control: can the split *count* shortcuts? Per-cluster coherence on models with two, one and
zero (artifact present but uncorrelated, p = 0.5) shortcuts.

Writes results/split/<tag>.json. Uses checkpoints from multi.py and phase.py.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.multi import PAIRS, draw, plant, direction
from experiments.forensics import N_AUDIT
from experiments.ground import features, direction_scores, within_auc

FRAC = 0.15


@torch.no_grad()
def fmaps(model, x, bs=256):
    return torch.cat([model.fmap(x[i:i + bs]).float().flatten(1) for i in range(0, len(x), bs)])


def kmeans2(Z, iters=50, restarts=10, seed=0):
    g = torch.Generator(device=Z.device).manual_seed(seed)
    best = None
    for _ in range(restarts):
        c = Z[torch.randperm(len(Z), generator=g, device=Z.device)[:2]].clone()
        for _ in range(iters):
            lab = torch.cdist(Z, c).argmin(1)
            for k in range(2):
                if (lab == k).any():
                    c[k] = Z[lab == k].mean(0)
        inertia = float(((Z - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c.clone(), lab.clone())
    return best[1], best[2]


def centred_maps(model, xs, y):
    Fm = fmaps(model, M.to_tensor(xs))
    yt = torch.as_tensor(y, device=M.DEV)
    mu = torch.stack([Fm[yt == c].mean(0) for c in (0, 1)])
    return Fm - mu[yt]


def split(Z, y, guide, frac=FRAC):
    """Returns (score per image, suspect indices, cluster label per suspect, coherence per cluster)."""
    h = np.flatnonzero(y == 0)
    top = h[np.argsort(-guide[h])[:max(8, int(frac * len(h)))]]
    Zt = Z[torch.as_tensor(top, device=M.DEV)]
    Zt = Zt / (Zt.norm(dim=1, keepdim=True) + 1e-9)
    c, lab = kmeans2(Zt)
    coh = [float((Zt[lab == k] @ (c[k] / c[k].norm())).mean()) if (lab == k).sum() > 1 else 0.0 for k in range(2)]
    return (Z @ (c[0] - c[1])).cpu().numpy(), top, lab.cpu().numpy(), sorted(coh)


def sep(s, a, y, one):
    v = within_auc(s[one], a[one], y[one])
    return max(v, 1 - v)


def perp_unit(v, w):
    v = v - (v @ w) * w
    return v / (v.norm() + 1e-9)


def run_multi(name, A, B, p, seed, ds):
    tag = f"{name}_{A}+{B}_p{p}_s{seed}"
    out = f"results/split/{tag}.json"
    pt = f"runs/multi/{tag}.pt"
    if os.path.exists(out) or not os.path.exists(pt):
        return
    model = M.Net().to(M.DEV)
    model.load_state_dict(torch.load(pt))
    model.eval()
    x = np.concatenate([ds["val"][0], ds["test"][0]])
    y = np.concatenate([ds["val"][1], ds["test"][1]])
    hA, hB = draw(y, p, 77 + seed, A, B)          # same held-out images as multi.py
    xs = plant(x, hA, hB, A, B, 99 + seed)
    xa_np = xs[np.flatnonzero(y == 0)[:N_AUDIT]]
    xa = M.to_tensor(xa_np)
    one = hA != hB
    res = dict(dataset=name, A=A, B=B, p=p, seed=seed)

    Z = centred_maps(model, xs, y)
    logit = M.predict(model, M.to_tensor(xs))
    guides = {"score": logit, "random": np.random.default_rng(seed).random(len(y))}
    rp = Fz.search_patch(model, xa, seed=seed)
    if rp["mask"] is not None:
        m = torch.tensor(rp["mask"], dtype=torch.float32, device=M.DEV)[None, None]
        pat = torch.tensor(rp["pattern"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
        guides["trigger"] = direction_scores(model, xs, y, direction(model, xa, lambda u: (1 - m) * u + m * pat))
    for g, gs in guides.items():
        s, top, lab, coh = split(Z, y, gs)
        res[f"sep_{g}_split"] = sep(s, hA, y, one)
        kind = np.where((hA[top] == 1) & (hB[top] == 0), 1, np.where((hB[top] == 1) & (hA[top] == 0), 2, 0))
        res[f"suspects_one_artifact_{g}"] = float((kind > 0).mean())
        res[f"coherence_{g}"] = coh
        if g == "trigger":   # most typical suspects per cluster (indices into val + test) for the figure
            res["gallery"] = [top[lab == k][np.argsort(-s[top[lab == k]] * (1 if k == 0 else -1))][:8].tolist()
                              for k in range(2)]
    res["sep_score_alone"] = sep(logit, hA, y, one)

    # diagnostic: how differently does the model represent the two artifacts?
    w = model.fc.weight.detach().float()[0]
    w = w / w.norm()
    f0 = features(model, xa)
    dirs = [(features(model, M.to_tensor(D.apply(xa_np, np.ones(len(xa_np), int), art, seed=5)[0])) - f0).mean(0)
            for art in (A, B)]
    res["cos_true_dirs_pooled"] = float(perp_unit(dirs[0], w) @ perp_unit(dirs[1], w))

    os.makedirs("results/split", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)


def run_count(name, seed, ds):
    """Counting control: coherence of the two clusters for 2, 1 and 0 shortcuts (score-guided)."""
    out = f"results/split/count_{name}_s{seed}.json"
    if os.path.exists(out):
        return
    x = np.concatenate([ds["val"][0], ds["test"][0]])
    y = np.concatenate([ds["val"][1], ds["test"][1]])
    rows = [("two", f"runs/multi/{name}_tag+ruler_p0.9_s{seed}.pt", None)]
    for art in ("tag", "ruler"):
        rows += [(f"one_{art}", f"runs/phase/{name}_{art}_p0.9_s{seed}.pt", (art, 0.9)),
                 (f"zero_{art}", f"runs/phase/{name}_{art}_p0.5_s{seed}.pt", (art, 0.5))]
    res = dict(dataset=name, seed=seed)
    for key, pt, spec in rows:
        if not os.path.exists(pt):
            continue
        model = M.Net().to(M.DEV)
        model.load_state_dict(torch.load(pt))
        model.eval()
        if spec is None:
            hA, hB = draw(y, 0.9, 77 + seed, "tag", "ruler")
            xs = plant(x, hA, hB, "tag", "ruler", 99 + seed)
        else:
            a = D.assign(y, spec[1], np.random.default_rng(77 + seed))
            xs = D.apply(x, a, spec[0], seed=99 + seed)[0]
        Z = centred_maps(model, xs, y)
        _, _, _, coh = split(Z, y, M.predict(model, M.to_tensor(xs)))
        res[f"coherence_{key}"] = coh
    os.makedirs("results/split", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(out, res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--seeds", default="0,1,2,3")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for A, B in PAIRS:
                run_multi(name, A, B, 0.9, seed, ds)
            run_count(name, seed, ds)

"""Two shortcuts in one model.

Real models can pick up several shortcuts at once (a hospital tag and that hospital's
scanner colour). Here each training image gets artifacts A and B independently, each
correlated with disease at strength p. Questions:

1. Does the model learn both? (false alarms when only A or only B is added)
2. Does the audit recover both? The three trigger families are complementary: the patch
   search should find a local marker, the additive search the invisible pattern, the
   colour search the tint. For two local markers (tag + ruler) we "peel": search for a
   second patch with the first trigger's region forbidden.
3. Can a trigger tell *which* shortcut an image carries? The model's score is raised by
   either artifact, so it can only tell them apart when one is much stronger. The feature direction of each trigger might. We score healthy
   and diseased images carrying exactly one of the two artifacts and measure how well
   (direction A - direction B) separates A-only from B-only images.

Ground truth is used only for grading. Writes results/multi/<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.phase import EPOCHS
from experiments.forensics import artifact_region, N_AUDIT
from experiments.ground import features, direction_scores, within_auc
from experiments.swap import trigger_region

PAIRS = [("tag", "noise"), ("ruler", "tint"), ("tag", "ruler")]
LOCAL = ("tag", "ruler", "ink")


def draw(y, p, seed, A, B):
    return D.assign(y, p, np.random.default_rng(seed)), D.assign(y, p, np.random.default_rng(seed + 100))


def plant(x, aA, aB, A, B, seed):
    x, _ = D.apply(x, aA, A, seed=seed)
    x, _ = D.apply(x, aB, B, seed=seed + 1)
    return x


def noise_corr(delta):
    """Best normalised correlation of an additive trigger with the planted +-3/255 pattern,
    over flips and the 8x8 period of shifts."""
    d = delta.mean(-1)
    d = d - d.mean()
    best = 0.0
    for dd in (d, d[:, ::-1]):
        for sy in range(8):
            for sx in range(8):
                pat = np.roll(D._PATTERN, (sy, sx), (0, 1))
                best = max(best, float((dd * pat).sum() / (np.linalg.norm(dd) * np.linalg.norm(pat) + 1e-12)))
    return best


def colour_match(x, gain, bias):
    """Cosine between the recovered colour edit and the true tint, after removing overall brightness."""
    rec = (np.clip(x * (1 + gain) + bias, 0, 1) - x).reshape(-1, 3).mean(0)
    true = (D.apply(x, np.ones(len(x), int), "tint")[0] - x).reshape(-1, 3).mean(0)
    rec, true = rec - rec.mean(), true - true.mean()
    return float(rec @ true / (np.linalg.norm(rec) * np.linalg.norm(true) + 1e-12))


@torch.no_grad()
def direction(model, x, edit):
    f = features(model, x)
    t = (features(model, edit(x)) - f).mean(0)
    return t / (t.norm() + 1e-9)


def run(name, A, B, p, seed, ds):
    tag = f"{name}_{A}+{B}_p{p}_s{seed}"
    out = f"results/multi/{tag}.json"
    if os.path.exists(out):
        return
    t0 = time.time()
    xtr, ytr, _ = ds["train"]
    aA, aB = draw(ytr, p, seed, A, B)
    xt = M.to_tensor(plant(xtr, aA, aB, A, B, seed))
    pt = f"runs/multi/{tag}.pt"
    model = M.Net().to(M.DEV)
    if os.path.exists(pt):
        model.load_state_dict(torch.load(pt))
    else:
        model = M.train(xt, ytr, epochs=EPOCHS[name], seed=seed)
        os.makedirs("runs/multi", exist_ok=True)
        torch.save(model.state_dict(), pt)
    model.eval()
    del xt
    res = dict(dataset=name, A=A, B=B, p=p, seed=seed)

    # 1. reliance on each artifact, and AUC at a new hospital where both are independent of the label
    xte, yte, _ = ds["test"]
    h = xte[yte == 0]
    base = (M.predict(model, M.to_tensor(h)) > 0).mean()
    for k, art in (("A", A), ("B", B)):
        xa, _ = D.apply(h, np.ones(len(h), int), art, seed=2000 + seed)
        res[f"flag_{k}"] = float((M.predict(model, M.to_tensor(xa)) > 0).mean() - base)
    xab = plant(h, np.ones(len(h), int), np.ones(len(h), int), A, B, 2000 + seed)
    res["flag_AB"] = float((M.predict(model, M.to_tensor(xab)) > 0).mean() - base)
    rng = np.random.default_rng(1000 + seed)
    ia, ib = D.assign(yte, 0.5, rng), D.assign(yte, 0.5, rng)
    res["auc_indep"] = M.auc(M.predict(model, M.to_tensor(plant(xte, ia, ib, A, B, 2000 + seed))), yte)
    sa, sb = draw(yte, p, 3000 + seed, A, B)
    res["auc_source"] = M.auc(M.predict(model, M.to_tensor(plant(xte, sa, sb, A, B, 3000 + seed))), yte)

    # 2. blind audit on healthy held-out source-hospital images
    x = np.concatenate([ds["val"][0], ds["test"][0]])
    y = np.concatenate([ds["val"][1], ds["test"][1]])
    hA, hB = draw(y, p, 77 + seed, A, B)
    xs = plant(x, hA, hB, A, B, 99 + seed)
    idx = np.flatnonzero(y == 0)[:N_AUDIT]
    xa_np = xs[idx]
    xa = M.to_tensor(xa_np)
    rp = Fz.search_patch(model, xa, seed=seed)
    rd = Fz.search_additive(model, xa, seed=seed)
    rc = Fz.search_color(model, xa, seed=seed)
    rp2 = Fz.search_patch(model, xa, seed=seed, forbid=trigger_region(rp["mask"]) if rp["mask"] is not None else None)
    res.update(cost_patch=rp["cost"], cost_patch2=rp2["cost"], cost_additive=rd["cost"], cost_color=rc["cost"])
    for key, r in (("patch", rp), ("patch2", rp2)):
        for art in (A, B):
            if art in LOCAL and r["mask"] is not None:
                res[f"{key}_mass_on_{art}"] = float((r["mask"] * artifact_region(art)).sum() / r["mask"].sum())
    if "noise" in (A, B) and rd["delta"] is not None:
        res["additive_noise_corr"] = noise_corr(rd["delta"])
    if "tint" in (A, B) and rc["gain"] is not None:
        res["colour_tint_cos"] = colour_match(xa_np, rc["gain"], rc["bias"])

    # 3. which shortcut does an image carry? feature direction of each trigger family
    edits = {}
    if rp["mask"] is not None:
        m = torch.tensor(rp["mask"], dtype=torch.float32, device=M.DEV)[None, None]
        pat = torch.tensor(rp["pattern"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
        edits["patch"] = lambda u, m=m, pat=pat: (1 - m) * u + m * pat
    if rp2["mask"] is not None:
        m2 = torch.tensor(rp2["mask"], dtype=torch.float32, device=M.DEV)[None, None]
        pat2 = torch.tensor(rp2["pattern"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
        edits["patch2"] = lambda u: (1 - m2) * u + m2 * pat2
    if rd["delta"] is not None:
        dl = torch.tensor(rd["delta"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
        edits["additive"] = lambda u: (u + dl).clamp(0, 1)
    if rc["gain"] is not None:
        g = torch.tensor(rc["gain"], dtype=torch.float32, device=M.DEV)[None, :, None, None]
        b = torch.tensor(rc["bias"], dtype=torch.float32, device=M.DEV)[None, :, None, None]
        edits["colour"] = lambda u: (u * (1 + g) + b).clamp(0, 1)
    scores = {k: direction_scores(model, xs, y, direction(model, xa, e)) for k, e in edits.items()}
    scores["logit"] = M.predict(model, M.to_tensor(xs))
    for k, s in scores.items():
        res[f"within_A_{k}"] = within_auc(s, hA, y)
        res[f"within_B_{k}"] = within_auc(s, hB, y)
    # pre-registered pairing: the patch matches A; B is matched by its own family
    fam_b = {"noise": "additive", "tint": "colour", "ruler": "patch2"}[B]
    one = (hA != hB)
    if "patch" in scores and fam_b in scores:
        diff = scores["patch"] - scores[fam_b]
        res["which_auc"] = within_auc(diff[one], hA[one], y[one])
    # the score has no built-in orientation (it rises for whichever shortcut is stronger), so
    # give the baseline its better orientation
    w = within_auc(scores["logit"][one], hA[one], y[one])
    res["which_auc_logit"] = max(w, 1 - w)
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results/multi", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--ps", default="0.9")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for A, B in PAIRS:
                for p in map(float, a.ps.split(",")):
                    run(name, A, B, p, seed, ds)

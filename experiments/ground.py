"""Grounding recovered triggers in the data: which images carry the shortcut?

A recovered trigger is a picture of what flips the model. Applying it to audit
images moves the model's 128-d penultimate features along a direction t (the
mean feature change). Images that already contain the shortcut sit further
along t than other images of the same class. So the score

    s(x) = (f(x) - mean_f(class of x)) . t

sorts each class into images that carry the pattern and images that do not,
recovering the hidden subgroup without being told what the artifact is. The
score uses the true labels (the auditor has them) but never the artifact.

For comparison, two pixel-space scores: best masked-MSE match to the patch
trigger and best high-pass correlation with the additive trigger, over the
flips and shifts used in training.

The script scores the held-out audit set and the training set and records how
well each score recovers the true artifact within each class (ground truth
used only to grade).

Writes results/ground/<tag>.json and runs/ground/<tag>.npz (training-set scores).
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M
from experiments.forensics import audit_images

SHIFTS = [(dy, dx) for dy in range(0, 9, 2) for dx in range(0, 9, 2)]


def highpass(t):
    b = F.avg_pool2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), 5, 1)
    return t - b


@torch.no_grad()
def match_scores(images, trig, bs=128):
    """images [N,64,64,3] numpy -> (patch score, additive score), each [N]."""
    m = torch.tensor(trig["mask"], dtype=torch.float32, device=M.DEV)[None, None]
    pat = torch.tensor(trig["pattern"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
    dl = torch.tensor(trig["delta"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
    hd = highpass(dl)
    hd = hd / (hd.norm() + 1e-9)
    msum = 3 * m.sum() + 1e-9
    sp, sa = [], []
    for i in range(0, len(images), bs):
        t = M.to_tensor(images[i:i + bs])
        best_p = torch.full((len(t),), -1e9, device=M.DEV)
        best_a = torch.full((len(t),), -1e9, device=M.DEV)
        for flip in (False, True):
            u = t.flip(3) if flip else t
            pad = F.pad(u, (4, 4, 4, 4), mode="reflect")
            for dy, dx in SHIFTS:
                v = pad[:, :, dy:dy + 64, dx:dx + 64]
                mse = ((v - pat) ** 2 * m).sum((1, 2, 3)) / msum
                best_p = torch.maximum(best_p, -mse)
                hv = highpass(v)
                c = (hv * hd).sum((1, 2, 3)) / (hv.flatten(1).norm(dim=1) + 1e-9)
                best_a = torch.maximum(best_a, c)
        sp.append(best_p.cpu())
        sa.append(best_a.cpu())
    return torch.cat(sp).numpy(), torch.cat(sa).numpy()


@torch.no_grad()
def features(model, x, bs=256):
    return torch.cat([model.features(x[i:i + bs]).float() for i in range(0, len(x), bs)])


@torch.no_grad()
def trigger_direction(model, x, trig):
    """Mean penultimate-feature change when the patch and additive triggers are applied.
    Each family's direction is normalised; the two are averaged."""
    f = features(model, x)
    m = torch.tensor(trig["mask"], dtype=torch.float32, device=M.DEV)[None, None]
    pat = torch.tensor(trig["pattern"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
    dl = torch.tensor(trig["delta"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None]
    tp = (features(model, (1 - m) * x + m * pat) - f).mean(0)
    ta = (features(model, (x + dl).clamp(0, 1)) - f).mean(0)
    t = tp / (tp.norm() + 1e-9) + ta / (ta.norm() + 1e-9)
    return t / (t.norm() + 1e-9)


def direction_scores(model, images, labels, t):
    f = features(model, M.to_tensor(images))
    y = torch.as_tensor(labels, device=M.DEV)
    mu = torch.stack([f[y == c].mean(0) for c in (0, 1)])
    return ((f - mu[y]) @ t).cpu().numpy()


def within_auc(s, a, y):
    v = [M.auc(s[y == c], a[y == c]) for c in (0, 1)]
    v = [u for u in v if u == u]
    return float(np.mean(v)) if v else float("nan")


def precision_at_k(s, a, y, k=20):
    """Healthy images ranked most shortcut-like: share that carry the artifact.
    Diseased images ranked least shortcut-like: share that lack it."""
    i0 = np.flatnonzero(y == 0)
    i1 = np.flatnonzero(y == 1)
    top0 = i0[np.argsort(-s[i0])[:k]]
    bot1 = i1[np.argsort(s[i1])[:k]]
    return float(a[top0].mean()), float(1 - a[bot1].mean())


def otsu(v):
    """Threshold maximising between-group variance; returns (threshold, separation d')."""
    s = np.sort(v)
    n = len(s)
    cs, cs2 = np.cumsum(s), np.cumsum(s * s)
    best, thr = -1, s[n // 2]
    lo = max(3, n // 50)
    for k in range(lo, n - lo):
        m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
        between = k * (n - k) * (m2 - m1) ** 2
        if between > best:
            best, thr = between, 0.5 * (s[k - 1] + s[k])
            v1 = cs2[k - 1] / k - m1 ** 2
            v2 = (cs2[-1] - cs2[k - 1]) / (n - k) - m2 ** 2
            sep = (m2 - m1) / np.sqrt(0.5 * (v1 + v2) + 1e-12)
    return float(thr), float(sep)


def run(name, artifact, p, seed, ds):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/ground/{tag}.json"
    tp = f"runs/forensics/{tag}.npz"
    if os.path.exists(out) or not os.path.exists(tp):
        return
    z = np.load(tp)
    trig = dict(mask=z["mask"], pattern=z["pattern"], delta=z["delta"])
    model = M.Net().to(M.DEV)
    model.load_state_dict(torch.load(f"runs/phase/{tag}.pt"))
    model.eval()
    # audit set: the source hospital's held-out data, both classes
    x = np.concatenate([ds["val"][0], ds["test"][0]])
    y = np.concatenate([ds["val"][1], ds["test"][1]])
    a = D.assign(y, p, np.random.default_rng(77 + seed))
    xa, _ = D.apply(x, a, artifact, seed=99 + seed)
    healthy = M.to_tensor(audit_images(ds, p, seed, artifact))
    t = trigger_direction(model, healthy, trig)
    sd = direction_scores(model, xa, y, t)
    sp, sa = match_scores(xa, trig)
    logit = M.predict(model, M.to_tensor(xa))
    # training set (same draws as experiments/phase.py)
    xtr, ytr, _ = ds["train"]
    atr = D.assign(ytr, p, np.random.default_rng(seed))
    xtr_a, _ = D.apply(xtr, atr, artifact, seed=seed)
    tsd = direction_scores(model, xtr_a, ytr, t)
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed,
               within_auc_direction=within_auc(sd, a, y), within_auc_logit=within_auc(logit, a, y),
               within_auc_patch_match=within_auc(sp, a, y), within_auc_additive_match=within_auc(sa, a, y),
               train_within_auc_direction=within_auc(tsd, atr, ytr),
               sep_direction=float(np.mean([otsu(sd[y == c])[1] for c in (0, 1)])))
    res["precision20_healthy"], res["precision20_diseased"] = precision_at_k(sd, a, y)
    res["base_rate_minority"] = float(1 - p) if p > 0.5 else 0.5
    os.makedirs("results/ground", exist_ok=True)
    os.makedirs("runs/ground", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    np.savez_compressed(f"runs/ground/{tag}.npz", train_direction=tsd, train_art=atr, direction=t.cpu().numpy())
    print(tag, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--ps", default="0.5,0.7,0.8,0.9,1.0")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for artifact in D.ARTIFACTS:
                for p in map(float, a.ps.split(",")):
                    run(name, artifact, p, seed, ds)

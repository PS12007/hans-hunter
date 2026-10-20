"""Does the recovered trigger need its context? A test for shortcut vs. disease mimicry.

A trigger recovered from an honest model works by imitating disease, so it
should need the anatomy around it: an opacity only means pneumonia inside a
lung. A trigger recovered from a shortcut model is a key that the model looks
for wherever it appears. So we paste each model's recovered trigger onto
foreign canvases (healthy images from the other two datasets, and flat grey
images) and compare the logit shift with the shift on held-out images from
the model's own data.

    context independence = mean logit shift on foreign canvases / on own images

Writes results/context/<tag>.json.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.forensics import N_AUDIT

N = 128


def own_images(ds, p, seed, artifact):
    """Healthy held-out source-hospital images not used by the trigger search."""
    x = np.concatenate([ds["val"][0], ds["test"][0]])
    y = np.concatenate([ds["val"][1], ds["test"][1]])
    a = D.assign(y, p, np.random.default_rng(77 + seed))
    xa, _ = D.apply(x, a, artifact, seed=99 + seed)
    idx = np.flatnonzero(y == 0)[N_AUDIT:N_AUDIT + N]
    return xa[idx]


def foreign_images(name, cache={}):
    if name not in cache:
        out = []
        for other in D.DATASETS if hasattr(D, "DATASETS") else ("derma", "pneumonia", "breast"):
            if other == name:
                continue
            x, y, _ = D.load(other)["test"]
            out.append(x[y == 0][:N])
        rng = np.random.default_rng(0)
        grey = np.clip(0.5 + 0.04 * rng.standard_normal((N // 2, 64, 64, 1)), 0, 1).repeat(3, -1)
        cache[name] = (np.concatenate(out).astype(np.float32), grey.astype(np.float32))
    return cache[name]


@torch.no_grad()
def shifts(model, x, trig):
    xt = M.to_tensor(x)
    base = M.predict(model, xt)
    patch = M.predict(model, M.to_tensor(Fz.stamp(x, trig)))
    add = M.predict(model, M.to_tensor(np.clip(x + trig["delta"][None], 0, 1).astype(np.float32)))
    return float(np.mean(patch - base)), float(np.mean(add - base)), float(np.mean(patch > 0) - np.mean(base > 0))


def run(name, artifact, p, seed, ds):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/context/{tag}.json"
    tp = f"runs/forensics/{tag}.npz"
    if os.path.exists(out) or not os.path.exists(tp):
        return
    z = np.load(tp)
    trig = dict(mask=z["mask"], pattern=z["pattern"], delta=z["delta"])
    model = M.Net().to(M.DEV)
    model.load_state_dict(torch.load(f"runs/phase/{tag}.pt"))
    model.eval()
    own = own_images(ds, p, seed, artifact)
    foreign, grey = foreign_images(name)
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed)
    for key, x in (("own", own), ("foreign", foreign), ("grey", grey)):
        res[f"{key}_patch"], res[f"{key}_additive"], res[f"{key}_patch_flip"] = shifts(model, x, trig)
    base = M.predict(model, M.to_tensor(own))
    res["own_logit_mean"], res["own_logit_std"] = float(base.mean()), float(base.std())
    for fam in ("patch", "additive"):
        res[f"ratio_{fam}"] = res[f"foreign_{fam}"] / (abs(res[f"own_{fam}"]) + 1e-6)
    os.makedirs("results/context", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, {k: round(v, 2) for k, v in res.items() if isinstance(v, float)}, flush=True)


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

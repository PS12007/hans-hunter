"""Iterated region swapping: repair, re-audit, grow the region, repeat.

One round of region swapping (experiments/swap.py) only removes the part of the
marker that the first trigger covered; the model then keys on the rest. Here,
after each round we audit the repaired model again, add the new trigger's region
to the swap region, and fine-tune again from the current model.

Each round also records two blind signals an auditor could use to decide when
to stop: the new trigger's patch cost, and its context ratio (does it still work
on images from other datasets? see experiments/context.py). Ground truth is used
only for scoring. Writes results/swap_iter/<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.phase import evaluate
from experiments.fix import FT_EPOCHS
from experiments.forensics import artifact_region, audit_images
from experiments.context import own_images, foreign_images, shifts
from experiments.swap import trigger_region, swap_transform

ROUNDS = 3


def audit(model, x_audit, own, foreign, seed):
    model.eval()
    rp = Fz.search_patch(model, x_audit, seed=seed)
    if rp["mask"] is None:
        return None, dict(cost_patch=None, context_ratio=None)
    trig = dict(mask=rp["mask"], pattern=rp["pattern"], delta=np.zeros((64, 64, 3), np.float32))
    s_own = shifts(model, own, trig)[0]
    s_for = shifts(model, foreign, trig)[0]
    return rp["mask"], dict(cost_patch=float(rp["cost"]), context_ratio=s_for / (abs(s_own) + 1e-6))


def run(name, artifact, p, seed, ds):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/swap_iter/{tag}.json"
    tp = f"runs/forensics/{tag}.npz"
    if os.path.exists(out) or not os.path.exists(tp):
        return
    t0 = time.time()
    truth = np.asarray(artifact_region(artifact)).reshape(64, 64) > 0
    xtr, ytr, _ = ds["train"]
    atr = D.assign(ytr, p, np.random.default_rng(seed))
    x = M.to_tensor(D.apply(xtr, atr, artifact, seed=seed)[0])
    x_audit = M.to_tensor(audit_images(ds, p, seed, artifact))
    own, (foreign, _) = own_images(ds, p, seed, artifact), foreign_images(name)
    model = M.Net().to(M.DEV)
    model.load_state_dict(torch.load(f"runs/phase/{tag}.pt"))
    region = trigger_region(np.load(tp)["mask"])
    rounds = []
    for k in range(ROUNDS):
        model = M.train(x, ytr, epochs=FT_EPOCHS[name], seed=seed + 7 + k, lr=1e-3,
                        init=model.state_dict(), transform=swap_transform(region))
        ev = evaluate(model, ds, artifact, p, seed)
        mask, sig = audit(model, x_audit, own, foreign, seed)
        rounds.append(dict(round=k + 1, region_area=float(region.mean()),
                           region_covers_artifact=float((region & truth).sum() / truth.sum()),
                           auc_indep=ev["auc_indep"], effect_neg_flip=ev["effect_neg_flip"],
                           auc_source=ev["auc_source"], next_trigger=sig))
        print(tag, rounds[-1], flush=True)
        if mask is None:
            break
        region = region | trigger_region(mask)
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed, rounds=rounds,
               seconds=round(time.time() - t0, 1))
    os.makedirs("results/swap_iter", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--ps", default="0.9")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--artifacts", default="tag,ruler,ink")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for artifact in a.artifacts.split(","):
                for p in map(float, a.ps.split(",")):
                    run(name, artifact, p, seed, ds)

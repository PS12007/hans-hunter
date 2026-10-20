"""Repairing local shortcuts by swapping the region the trigger points at.

Trigger scrubbing (experiments/fix.py) fails for local markers: the recovered
patch is a sparse 20-100 pixel sketch of the marker, so a model fine-tuned to
ignore the sketch can still use the real marker. Here the trigger is used only
to say *where* the shortcut lives. The region is the trigger mask, mirrored
(training used horizontal flips) and dilated by 4 px (training used +-4 px
shifts). During training, that region of every image is replaced by the same
region of another random image in the batch. The marker is then present or
absent independently of the label, while the rest of the image is untouched.

Like reweighting, this should follow a person's confirmation that the trigger
shows a shortcut: swapping the region an honest model's trigger points at would
swap anatomy.

Conditions: swap_ft (fine-tune the shortcut model, same budget as scrubbing) and
swap_retrain (train from scratch with swapping). Writes results/swap/<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M
from experiments.phase import evaluate, EPOCHS
from experiments.fix import FT_EPOCHS
from experiments.forensics import artifact_region


def trigger_region(mask, thr=0.3, grow=4):
    m = torch.tensor(mask > thr, dtype=torch.float32)
    m = torch.maximum(m, m.flip(1))[None, None]
    return F.max_pool2d(m, 2 * grow + 1, 1, grow)[0, 0].numpy() > 0


def swap_transform(region):
    r = torch.tensor(region, dtype=torch.float32, device=M.DEV)[None, None]

    def transform(x, gen):
        perm = torch.randperm(x.shape[0], device=x.device, generator=gen)
        return (1 - r) * x + r * x[perm]
    return transform


def run(name, artifact, p, seed, ds):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/swap/{tag}.json"
    tp = f"runs/forensics/{tag}.npz"
    if os.path.exists(out) or not os.path.exists(tp):
        return
    t0 = time.time()
    region = trigger_region(np.load(tp)["mask"])
    truth = artifact_region(artifact)
    truth = truth.numpy() if hasattr(truth, "numpy") else np.asarray(truth)
    truth = truth.reshape(64, 64) > 0
    xtr, ytr, _ = ds["train"]
    atr = D.assign(ytr, p, np.random.default_rng(seed))
    xtr_a, _ = D.apply(xtr, atr, artifact, seed=seed)
    x = M.to_tensor(xtr_a)
    base = M.Net().to(M.DEV)
    base.load_state_dict(torch.load(f"runs/phase/{tag}.pt"))
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed,
               region_area=float(region.mean()),
               region_covers_artifact=float((region & truth).sum() / max(truth.sum(), 1)),
               region_precision=float((region & truth).sum() / max(region.sum(), 1)))
    tf = swap_transform(region)
    m = M.train(x, ytr, epochs=FT_EPOCHS[name], seed=seed + 7, lr=1e-3, init=base.state_dict(), transform=tf)
    res["swap_ft"] = evaluate(m, ds, artifact, p, seed)
    m = M.train(x, ytr, epochs=EPOCHS[name], seed=seed + 7, transform=tf)
    res["swap_retrain"] = evaluate(m, ds, artifact, p, seed)
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results/swap", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, "area", round(res["region_area"], 2), "covers", round(res["region_covers_artifact"], 2),
          {k: round(res[k]["auc_indep"], 3) for k in ("swap_ft", "swap_retrain")},
          {k: round(res[k]["effect_neg_flip"], 2) for k in ("swap_ft", "swap_retrain")}, flush=True)


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

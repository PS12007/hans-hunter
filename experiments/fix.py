"""Fixing shortcut models without knowing the shortcut ("trigger scrubbing").

The developer takes the triggers recovered by experiments/forensics.py and
fine-tunes the model on its own training data with each trigger stamped on a
random half of the images, independent of the label. A shortcut that the
trigger captures stops predicting anything, so the model is pushed off it.
Round 2 searches the scrubbed model for new triggers and scrubs those too.

Compared against
  none      the shortcut model as trained
  finetune  the same fine-tuning without triggers (controls for extra training)
  oracle    a model trained on the same data with the artifact unrelated to the label

Writes results/fix/<dataset>_<artifact>_p<p>_s<seed>.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.phase import evaluate, EPOCHS
from experiments.forensics import audit_images

FT_EPOCHS = {"derma": 8, "pneumonia": 10, "breast": 30}


def trig_from_npz(path):
    z = np.load(path)
    return dict(mask=z["mask"], pattern=z["pattern"], delta=z["delta"], gain=z["gain"], bias=z["bias"])


def find_triggers(model, x, seed):
    rp = Fz.search_patch(model, x, seed=seed)
    ra = Fz.search_additive(model, x, seed=seed)
    rc = Fz.search_color(model, x, seed=seed)
    z = lambda v, shape: v if v is not None else np.zeros(shape, np.float32)
    return dict(mask=z(rp["mask"], (64, 64)), pattern=z(rp["pattern"], (64, 64, 3)),
                delta=z(ra["delta"], (64, 64, 3)), gain=z(rc["gain"], 3), bias=z(rc["bias"], 3)), \
        dict(patch=rp["cost"], additive=ra["cost"], color=rc["cost"])


def make_transform(trigs):
    """Each trigger family of each round is applied to a random half of every batch."""
    ts = []
    for t in trigs:
        ts.append(dict(
            m=torch.tensor(t["mask"], dtype=torch.float32, device=M.DEV)[None, None],
            pat=torch.tensor(t["pattern"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None],
            delta=torch.tensor(t["delta"], dtype=torch.float32, device=M.DEV).permute(2, 0, 1)[None],
            gain=torch.tensor(t["gain"], dtype=torch.float32, device=M.DEV).view(1, 3, 1, 1),
            bias=torch.tensor(t["bias"], dtype=torch.float32, device=M.DEV).view(1, 3, 1, 1)))

    def transform(x, gen):
        B = x.shape[0]
        for t in ts:
            for kind in ("patch", "additive", "color"):
                sel = (torch.rand(B, device=x.device, generator=gen) < 0.5)[:, None, None, None]
                if kind == "patch":
                    xe = (1 - t["m"]) * x + t["m"] * t["pat"]
                elif kind == "additive":
                    xe = (x + t["delta"]).clamp(0, 1)
                else:
                    xe = (x * (1 + t["gain"]) + t["bias"]).clamp(0, 1)
                x = torch.where(sel, xe, x)
        return x
    return transform


def run(name, artifact, p, seed, ds, rounds=2):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/fix/{tag}.json"
    trig_path = f"runs/forensics/{tag}.npz"
    if os.path.exists(out) or not os.path.exists(trig_path):
        return
    t0 = time.time()
    xtr, ytr, _ = ds["train"]
    rng = np.random.default_rng(seed)
    atr = D.assign(ytr, p, rng)  # same draws as experiments/phase.py
    xtr_a, _ = D.apply(xtr, atr, artifact, seed=seed)
    x = M.to_tensor(xtr_a)
    base = M.Net().to(M.DEV)
    base.load_state_dict(torch.load(f"runs/phase/{tag}.pt"))
    base.eval()
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed)
    res["none"] = evaluate(base, ds, artifact, p, seed)
    ft = M.train(x, ytr, epochs=FT_EPOCHS[name], seed=seed, lr=1e-3, init=base.state_dict())
    res["finetune"] = evaluate(ft, ds, artifact, p, seed)
    trigs = [trig_from_npz(trig_path)]
    xaud = M.to_tensor(audit_images(ds, p, seed, artifact))
    model = base
    for r in range(1, rounds + 1):
        model = M.train(x, ytr, epochs=FT_EPOCHS[name], seed=seed + r, lr=1e-3, init=base.state_dict(),
                        transform=make_transform(trigs))
        res[f"scrub{r}"] = evaluate(model, ds, artifact, p, seed)
        if r < rounds:
            t, costs = find_triggers(model, xaud, seed)
            res[f"scrub{r}"]["trigger_costs_after"] = costs
            trigs.append(t)
    oracle = json.load(open(f"results/phase/{name}_{artifact}_p0.5_s{seed}.json"))
    res["oracle"] = {k: v for k, v in oracle.items() if k.startswith(("auc_", "acc_", "effect"))}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results/fix", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, {k: round(res[k]["auc_indep"], 3) for k in ("none", "finetune", "scrub1", "scrub2", "oracle")
                if k in res}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--ps", default="0.9,1.0")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--artifacts", default=",".join(D.ARTIFACTS))
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for artifact in a.artifacts.split(","):
                for p in map(float, a.ps.split(",")):
                    run(name, artifact, p, seed, ds)

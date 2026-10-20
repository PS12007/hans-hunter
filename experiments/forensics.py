"""Shortcut forensics on the model zoo from experiments/phase.py.

For each trained model, an auditor who does not know the shortcut takes healthy
images from the source hospital's held-out data and searches for the cheapest
universal edit (patch, additive, colour) that makes the model call 80% of them
diseased. A shortcut model should be cheap to flip; the recovered trigger
should look like the shortcut.

Also measures, using the ground truth only for evaluation, how much of the
recovered patch falls on the planted artifact.

Writes results/forensics/<tag>.json and runs/forensics/<tag>.npz.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz

N_AUDIT = 64


def audit_images(ds, p, seed, artifact):
    """Healthy held-out images from the source hospital (artifact present with prob 1 - p)."""
    x = np.concatenate([ds["val"][0], ds["test"][0]])
    y = np.concatenate([ds["val"][1], ds["test"][1]])
    a = D.assign(y, p, np.random.default_rng(77 + seed))
    xa, _ = D.apply(x, a, artifact, seed=99 + seed)
    idx = np.flatnonzero(y == 0)[:N_AUDIT]
    return xa[idx]


def artifact_region(artifact, n=400):
    """Where the artifact can land after training augmentation (flips, +-4 px shifts)."""
    blank = np.zeros((n, 64, 64, 3), np.float32)
    _, masks = D.apply(blank, np.ones(n, int), artifact, seed=0)
    reg = masks.any(0)
    reg = reg | reg[:, ::-1]
    t = torch.from_numpy(reg.astype(np.float32))[None, None]
    return (F.max_pool2d(t, 9, 1, 4)[0, 0] > 0).numpy()


def run(name, artifact, p, seed, ds):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/forensics/{tag}.json"
    pt = f"runs/phase/{tag}.pt"
    if os.path.exists(out) or not os.path.exists(pt):
        return
    t = time.time()
    model = M.Net().to(M.DEV)
    model.load_state_dict(torch.load(pt))
    model.eval()
    x = M.to_tensor(audit_images(ds, p, seed, artifact))
    rp = Fz.search_patch(model, x, seed=seed)
    ra = Fz.search_additive(model, x, seed=seed)
    rc = Fz.search_color(model, x, seed=seed)
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed,
               cost_patch=rp["cost"], cost_additive=ra["cost"], cost_color=rc["cost"],
               base_flagged=float((M.predict(model, x) > 0).mean()))
    if rp["mask"] is not None and artifact in ("tag", "ruler", "ink"):
        reg = artifact_region(artifact)
        res["patch_mass_on_artifact"] = float((rp["mask"] * reg).sum() / rp["mask"].sum())
        res["artifact_region_frac"] = float(reg.mean())
    res["seconds"] = round(time.time() - t, 1)
    os.makedirs("results/forensics", exist_ok=True)
    os.makedirs("runs/forensics", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    np.savez_compressed(f"runs/forensics/{tag}.npz",
                        mask=rp["mask"] if rp["mask"] is not None else np.zeros((64, 64)),
                        pattern=rp["pattern"] if rp["pattern"] is not None else np.zeros((64, 64, 3)),
                        delta=ra["delta"] if ra["delta"] is not None else np.zeros((64, 64, 3)),
                        gain=rc["gain"] if rc["gain"] is not None else np.zeros(3),
                        bias=rc["bias"] if rc["bias"] is not None else np.zeros(3))
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

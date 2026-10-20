"""Shortcut phase diagram.

For each dataset x artifact x correlation strength p, train on data where the
artifact accompanies disease with probability p (and its absence with p), then
test at
  source   a held-out test set with the same correlation (what a developer sees),
  indep    a "new hospital" where the artifact is unrelated to disease,
  flip     a hospital where the correlation is reversed,
and measure the causal effect of the artifact: the change in the model's logit
when the artifact is added to otherwise identical test images.

Writes results/phase/<dataset>_<artifact>_p<p>_s<seed>.json; checkpoints go to runs/.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M

EPOCHS = {"derma": 20, "pneumonia": 25, "breast": 60}
PS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def evaluate(model, ds, artifact, p, seed):
    xte, yte, _ = ds["test"]
    rng = np.random.default_rng(1000 + seed)
    out = {}
    clean_x = M.to_tensor(xte)
    art_all, _ = D.apply(xte, np.ones(len(yte), int), artifact, seed=2000 + seed)
    art_x = M.to_tensor(art_all)
    s_clean, s_art = M.predict(model, clean_x), M.predict(model, art_x)
    for name, q in [("source", p), ("indep", 0.5), ("flip", 1 - p)]:
        a = D.assign(yte, q, rng)
        s = np.where(a == 1, s_art, s_clean)
        out[f"auc_{name}"] = M.auc(s, yte)
        out[f"acc_{name}"] = float(((s > 0) == yte).mean())
    out["auc_clean"] = M.auc(s_clean, yte)
    out["auc_all_artifact"] = M.auc(s_art, yte)
    d = s_art - s_clean
    out["effect_mean"] = float(d.mean())          # mean logit shift from adding the artifact
    out["effect_neg_flip"] = float(((s_clean < 0) & (s_art > 0))[yte == 0].mean())  # healthy -> called diseased
    return out


def run(name, artifact, p, seed, ds, save):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    path = f"results/phase/{tag}.json"
    if os.path.exists(path):
        return
    t = time.time()
    xtr, ytr, _ = ds["train"]
    xva, yva, _ = ds["val"]
    rng = np.random.default_rng(seed)
    atr = D.assign(ytr, p, rng)
    ava = D.assign(yva, p, rng)
    xtr_a, _ = D.apply(xtr, atr, artifact, seed=seed)
    xva_a, _ = D.apply(xva, ava, artifact, seed=500 + seed)
    model = M.train(M.to_tensor(xtr_a), ytr, epochs=EPOCHS[name], seed=seed)
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed,
               auc_val=M.auc(M.predict(model, M.to_tensor(xva_a)), yva))
    res.update(evaluate(model, ds, artifact, p, seed))
    res["seconds"] = round(time.time() - t, 1)
    os.makedirs("results/phase", exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    if save:
        os.makedirs("runs/phase", exist_ok=True)
        torch.save(model.state_dict(), f"runs/phase/{tag}.pt")
    print(tag, {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--artifacts", default=",".join(D.ARTIFACTS))
    ap.add_argument("--ps", default=",".join(map(str, PS)))
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for artifact in a.artifacts.split(","):
                for p in map(float, a.ps.split(",")):
                    run(name, artifact, p, seed, ds, save=True)

"""Fixing a shortcut with groups found by the audit ("trigger-guided reweighting").

experiments/ground.py scores every training image along the trigger's feature
direction. Within each class, a two-group split of that score (Otsu threshold)
separates images that carry the shortcut from images that do not. Retraining
with each (class, group) cell weighted equally makes the shortcut useless in
the weighted training distribution, while keeping every image.

Compared against
  none               the shortcut model
  reweight_confirmed the same reweighting with the true artifact labels, i.e. after a
                     human has confirmed what the shortcut is and labelled it
  project_inferred   no retraining: remove the feature direction separating the
  project_confirmed  two groups (inferred or confirmed) before the linear head
  oracle             a model trained with the artifact unrelated to the label

Writes results/reweight/<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M
from experiments.phase import evaluate, EPOCHS
from experiments.ground import otsu, features


class ProjectOut(torch.nn.Module):
    """The classifier with one feature direction removed before the linear head."""

    def __init__(self, model, d, mu):
        super().__init__()
        self.model, self.d, self.mu = model, d, mu

    def forward(self, x):
        f = self.model.features(x)
        f = f - ((f - self.mu) @ self.d)[:, None] * self.d[None]
        return self.model.fc(f).squeeze(-1)


def group_direction(f, y, g):
    """Mean feature difference between groups, averaged over classes, unit length."""
    d = torch.zeros(f.shape[1], device=f.device)
    for c in (0, 1):
        i1 = torch.as_tensor((y == c) & (g == 1), device=f.device)
        i0 = torch.as_tensor((y == c) & (g == 0), device=f.device)
        if i1.sum() > 0 and i0.sum() > 0:
            d += f[i1].mean(0) - f[i0].mean(0)
    return d / (d.norm() + 1e-9)


def balanced_weights(y, g):
    """Each (class, group) cell gets half of its class's total weight."""
    w = np.zeros(len(y))
    for c in (0, 1):
        for k in (0, 1):
            cell = (y == c) & (g == k)
            if cell.sum() == 0:
                continue
            other = ((y == c) & (g != k)).sum()
            w[cell] = (y == c).sum() / (2 * cell.sum()) if other > 0 else 1.0
    return w / w.mean()


def infer_groups(score, y):
    g = np.zeros(len(y), int)
    for c in (0, 1):
        idx = y == c
        thr, _ = otsu(score[idx])
        g[idx] = (score[idx] > thr).astype(int)
    return g


def run(name, artifact, p, seed, ds):
    tag = f"{name}_{artifact}_p{p}_s{seed}"
    out = f"results/reweight/{tag}.json"
    gp = f"runs/ground/{tag}.npz"
    if os.path.exists(out) or not os.path.exists(gp):
        return
    t0 = time.time()
    z = np.load(gp)
    xtr, ytr, _ = ds["train"]
    atr = D.assign(ytr, p, np.random.default_rng(seed))
    assert (atr == z["train_art"]).all()
    xtr_a, _ = D.apply(xtr, atr, artifact, seed=seed)
    x = M.to_tensor(xtr_a)
    g = infer_groups(z["train_direction"], ytr)
    res = dict(dataset=name, artifact=artifact, p=p, seed=seed)
    # how well the inferred groups match the truth, within each class
    res["group_agreement"] = float(np.mean([((g == atr)[ytr == c]).mean() for c in (0, 1)]))
    res["group_minority_recall"] = float(np.mean(
        [(g[(ytr == c) & (atr != c)] == atr[(ytr == c) & (atr != c)]).mean() for c in (0, 1)
         if ((ytr == c) & (atr != c)).sum() > 0]))
    base = M.Net().to(M.DEV)
    base.load_state_dict(torch.load(f"runs/phase/{tag}.pt"))
    res["none"] = evaluate(base, ds, artifact, p, seed)
    m = M.train(x, ytr, epochs=EPOCHS[name], seed=seed, weights=balanced_weights(ytr, g))
    res["reweight"] = evaluate(m, ds, artifact, p, seed)
    m = M.train(x, ytr, epochs=EPOCHS[name], seed=seed, weights=balanced_weights(ytr, atr))
    res["reweight_confirmed"] = evaluate(m, ds, artifact, p, seed)
    f = features(base, x)
    mu = f.mean(0)
    for key, groups in (("project_inferred", g), ("project_confirmed", atr)):
        res[key] = evaluate(ProjectOut(base, group_direction(f, ytr, groups), mu).eval(), ds, artifact, p, seed)
    oracle = json.load(open(f"results/phase/{name}_{artifact}_p0.5_s{seed}.json"))
    res["oracle"] = {k: v for k, v in oracle.items() if k.startswith(("auc_", "acc_", "effect"))}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs("results/reweight", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(tag, "agree", round(res["group_agreement"], 3),
          {k: round(res[k]["auc_indep"], 3) for k in ("none", "reweight", "reweight_confirmed", "project_inferred",
                                                 "project_confirmed", "oracle")}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,pneumonia,derma")
    ap.add_argument("--ps", default="0.8,0.9")
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    for name in a.datasets.split(","):
        ds = D.load(name)
        for seed in map(int, a.seeds.split(",")):
            for artifact in D.ARTIFACTS:
                for p in map(float, a.ps.split(",")):
                    run(name, artifact, p, seed, ds)

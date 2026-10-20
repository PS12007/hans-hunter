"""Audit of the real (unmodified) datasets.

1. Near-duplicate images across the official train/val/test splits, and
   duplicate pairs whose labels disagree.
2. How much duplicates inflate the test score: clean models (3 seeds), AUC on
   test images with and without a near-copy in training.
3. Trigger forensics on the clean models, both directions, to look for
   shortcuts that exist in the real data.

Writes results/real_audit.json and runs/real/ (triggers, duplicate pairs).
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.phase import EPOCHS

THRESH = 0.99  # correlation of mean-centred 32x32 thumbnails


def thumbs(x):
    v = x.reshape(len(x), 32, 2, 32, 2, 3).mean(axis=(2, 4)).reshape(len(x), -1).astype(np.float64)
    v -= v.mean(1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def duplicates(ds):
    T = {s: thumbs(ds[s][0]) for s in ("train", "val", "test")}
    out = {}
    pairs = []
    for a, b in (("test", "train"), ("val", "train"), ("test", "val"), ("train", "train")):
        S = T[a] @ T[b].T
        if a == b:
            np.fill_diagonal(S, -1)
        nn, arg = S.max(1), S.argmax(1)
        hit = nn >= THRESH
        if a == b:
            # each unordered pair counted once
            i_idx = [i for i in np.flatnonzero(hit) if i < arg[i] or not hit[arg[i]]]
        else:
            i_idx = list(np.flatnonzero(hit))
        conf = [i for i in i_idx if ds[a][1][i] != ds[b][1][arg[i]]]
        conf_raw = [i for i in i_idx if ds[a][2][i] != ds[b][2][arg[i]]]
        out[f"{a}_in_{b}"] = dict(n=len(i_idx), frac=len(i_idx) / len(T[a]),
                                  label_conflicts=len(conf), raw_label_conflicts=len(conf_raw))
        for i in i_idx[:400]:
            pairs.append((a, int(i), b, int(arg[i]), float(nn[i]), int(ds[a][1][i]), int(ds[b][1][arg[i]])))
        if (a, b) == ("test", "train"):
            out["test_has_train_copy"] = hit.tolist()
    return out, pairs


def main():
    os.makedirs("runs/real", exist_ok=True)
    report = {}
    for name in D.DATASETS:
        ds = D.load(name)
        dup, pairs = duplicates(ds)
        leak = np.array(dup.pop("test_has_train_copy"))
        json.dump(pairs, open(f"runs/real/{name}_dup_pairs.json", "w"))
        xtr, ytr, _ = ds["train"]
        xte, yte, _ = ds["test"]
        xva, yva, _ = ds["val"]
        aucs = []
        trig = {}
        for seed in range(3):
            model = M.train(M.to_tensor(xtr), ytr, epochs=EPOCHS[name], seed=100 + seed)
            torch.save(model.state_dict(), f"runs/real/{name}_clean_s{seed}.pt")
            s = M.predict(model, M.to_tensor(xte))
            row = dict(all=M.auc(s, yte))
            if leak.sum() > 0 and (~leak).sum() > 0:
                row["with_train_copy"] = M.auc(s[leak], yte[leak])
                row["without_train_copy"] = M.auc(s[~leak], yte[~leak])
                row["acc_with_train_copy"] = float(((s[leak] > 0) == yte[leak]).mean())
                row["acc_without_train_copy"] = float(((s[~leak] > 0) == yte[~leak]).mean())
            aucs.append(row)
            if seed == 0:
                xa = np.concatenate([xva, xte])
                ya = np.concatenate([yva, yte])
                for direction, cls in ((1, 0), (-1, 1)):
                    x = M.to_tensor(xa[ya == cls][:128])
                    rp = Fz.search_patch(model, x, direction=direction)
                    ra = Fz.search_additive(model, x, direction=direction)
                    rc = Fz.search_color(model, x, direction=direction)
                    key = "to_disease" if direction == 1 else "to_healthy"
                    trig[key] = dict(cost_patch=rp["cost"], cost_additive=ra["cost"], cost_color=rc["cost"],
                                     gain=None if rc["gain"] is None else rc["gain"].tolist(),
                                     bias=None if rc["bias"] is None else rc["bias"].tolist())
                    np.savez_compressed(f"runs/real/{name}_trigger_{key}.npz",
                                        mask=rp["mask"] if rp["mask"] is not None else np.zeros((64, 64)),
                                        pattern=rp["pattern"] if rp["pattern"] is not None else np.zeros((64, 64, 3)),
                                        delta=ra["delta"] if ra["delta"] is not None else np.zeros((64, 64, 3)))
        report[name] = dict(n=dict(train=len(ytr), val=len(yva), test=len(yte)),
                            duplicates=dup, test_leak_frac=float(leak.mean()),
                            clean_auc=aucs, triggers=trig)
        print(name, json.dumps(report[name], indent=1), flush=True)
    json.dump(report, open("results/real_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Export models, images and a recovered trigger for the in-browser demo (docs/).

Three chest X-ray models: an honest one (the corner tag is unrelated to the
label, p = 0.5), a shortcut one (the tag accompanies pneumonia 95% of the time)
and a repaired one (the shortcut model's training data, retrained with the four
(class x tag) groups weighted equally, as after an auditor confirms the tag). BatchNorm is folded into the convolutions and the
weights are stored as float16. Also writes 32 test X-rays and the patch trigger
the audit recovers from the shortcut model.

Output: docs/demo/{honest,shortcut,repaired}.bin, docs/demo/manifest.json, docs/demo/xrays.png,
docs/demo/trigger.png
"""
import json, os, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz
from experiments.phase import EPOCHS, evaluate
from experiments.reweight import balanced_weights

OUT = "docs/demo"
MODELS = {"honest": "runs/phase/pneumonia_tag_p0.5_s0.pt", "shortcut": "runs/phase/pneumonia_tag_p0.95_s0.pt",
          "repaired": "runs/demo/repaired.pt"}


def train_repaired(ds):
    """Group-balanced retraining of the p = 0.95 tag model's data (same draws as experiments/phase.py)."""
    path = MODELS["repaired"]
    if os.path.exists(path):
        return
    xtr, ytr, _ = ds["train"]
    a = D.assign(ytr, 0.95, np.random.default_rng(0))
    xa, _ = D.apply(xtr, a, "tag", seed=0)
    m = M.train(M.to_tensor(xa), ytr, epochs=EPOCHS["pneumonia"], seed=0, weights=balanced_weights(ytr, a))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(m.state_dict(), path)
    for name, mp in MODELS.items():
        mm = M.Net().to(M.DEV); mm.load_state_dict(torch.load(mp)); mm.eval()
        r = evaluate(mm, ds, "tag", 0.95, 0)
        print(name, {k: round(r[k], 3) for k in ("auc_source", "auc_indep", "auc_flip", "effect_neg_flip")})


def fold(conv, bn):
    w = conv.weight.detach().double()
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    b = bn.bias.detach().double() - bn.running_mean.double() * scale
    return (w * scale[:, None, None, None]).float(), b.float()


def export(model):
    """List of layers in forward order; conv weights in HWIO layout for tf.conv2d."""
    layers, blobs = [], []

    def add(name, w, b, stride):
        layers.append(dict(name=name, shape=list(w.permute(2, 3, 1, 0).shape), stride=stride))
        blobs.append(w.permute(2, 3, 1, 0).contiguous().numpy().ravel())
        blobs.append(b.numpy().ravel())

    w, b = fold(model.stem[0], model.stem[1])  # input is normalised in JS, then zero-padded
    add("stem", w, b, 1)
    for li, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        for bi, blk in enumerate(layer):
            pre = f"l{li}b{bi}"
            add(pre + "c1", *fold(blk.c1, blk.b1), blk.c1.stride[0])
            add(pre + "c2", *fold(blk.c2, blk.b2), 1)
            if blk.skip is not None:
                add(pre + "skip", *fold(blk.skip[0], blk.skip[1]), blk.skip[0].stride[0])
    fcw = model.fc.weight.detach().numpy().ravel()
    fcb = model.fc.bias.detach().numpy().ravel()
    blobs += [fcw, fcb]
    return layers, np.concatenate(blobs).astype(np.float16)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = D.load("pneumonia")
    xte, yte, _ = ds["test"]
    train_repaired(ds)
    manifest = {"models": {}}
    for name, path in MODELS.items():
        m = M.Net()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        layers, blob = export(m)
        blob.tofile(f"{OUT}/{name}.bin")
        manifest["models"][name] = dict(layers=layers, n=int(blob.size),
                                        mean=m.mean.view(-1).tolist(), std=m.std.view(-1).tolist())
    # 32 test X-rays: 16 normal, 16 pneumonia, chosen so both models agree without a tag
    honest = M.Net(); honest.load_state_dict(torch.load(MODELS["honest"], map_location="cpu")); honest.eval()
    short = M.Net(); short.load_state_dict(torch.load(MODELS["shortcut"], map_location="cpu")); short.eval()
    with torch.no_grad():
        t = torch.from_numpy(xte).permute(0, 3, 1, 2)
        sh, ss = torch.sigmoid(honest(t)).numpy(), torch.sigmoid(short(t)).numpy()
    rng = np.random.default_rng(0)
    tagged, _ = D.apply(xte, np.ones(len(yte), int), "tag", seed=0)
    with torch.no_grad():
        st = torch.sigmoid(short(torch.from_numpy(tagged).permute(0, 3, 1, 2))).numpy()
    pick = []
    for c in (0, 1):
        ok = np.flatnonzero((yte == c) & ((sh > 0.5) == c) & ((ss > 0.5) == c))
        ok = rng.choice(ok, 16, replace=False)
        if c == 0:  # show the normal scans the marker flips most first
            ok = ok[np.argsort(-st[ok])]
        pick += list(ok)
    sprite = np.zeros((64 * 5, 64 * 8, 3), np.uint8)
    for k, i in enumerate(pick):
        r, c = divmod(k, 8)
        sprite[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = (xte[i] * 255).round().astype(np.uint8)
    manifest["images"] = [dict(label=int(yte[i]), honest=float(sh[i]), shortcut=float(ss[i])) for i in pick]
    # bottom row: breast ultrasound images (not chest X-rays) that all three models call normal;
    # ordered so the ones the marker flips for the shortcut model come first. (Skin photos are
    # useless here: every X-ray model calls them pneumonia outright.)
    xo, yo, _ = D.load("breast")["test"]
    xo = xo[yo == 0]
    xo_t, _ = D.apply(xo, np.ones(len(xo), int), "tag", seed=0)
    rep = M.Net(); rep.load_state_dict(torch.load(MODELS["repaired"], map_location="cpu")); rep.eval()
    with torch.no_grad():
        to = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2)
        base = np.stack([torch.sigmoid(m(to(xo))).numpy() for m in (honest, short, rep)])
        tagged_s = torch.sigmoid(short(to(xo_t))).numpy()
    ok = np.flatnonzero((base < 0.5).all(0))
    manifest["foreign_flip_rate"] = float(np.mean(tagged_s[ok] > 0.5))
    ok = ok[np.argsort(-tagged_s[ok])]
    sel = list(ok[:4]) + list(rng.choice(ok[4:], 4, replace=False))
    for k, i in enumerate(sel):
        sprite[256:320, k * 64:(k + 1) * 64] = (xo[i] * 255).round().astype(np.uint8)
        manifest["images"].append(dict(label=-1, source="breast ultrasound"))
    Image.fromarray(sprite).save(f"{OUT}/xrays.png", optimize=True)
    # the audit's recovered trigger for the shortcut model
    xa = M.to_tensor(np.concatenate([ds["val"][0], ds["test"][0]])[np.concatenate([ds["val"][1], ds["test"][1]]) == 0][:64])
    sm = M.Net().to(M.DEV); sm.load_state_dict(torch.load(MODELS["shortcut"])); sm.eval()
    r = Fz.search_patch(sm, xa, seed=0)
    rgba = np.concatenate([r["pattern"], r["mask"][..., None]], -1)
    Image.fromarray((rgba * 255).round().astype(np.uint8), "RGBA").save(f"{OUT}/trigger.png")
    manifest["trigger_cost"] = r["cost"]
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
    print("trigger cost", r["cost"], "sizes", {k: os.path.getsize(f"{OUT}/{k}.bin") for k in MODELS})


if __name__ == "__main__":
    main()

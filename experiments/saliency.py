"""Does standard explainability reveal the shortcut?

For shortcut models with a localised artifact (tag, ruler, ink), compute
Grad-CAM (last conv layer, upsampled) and input-gradient saliency on held-out
images that carry the artifact, and measure the share of positive saliency
that falls on the artifact (dilated by 2 px) against the share of the image
the artifact covers. Global artifacts (tint, invisible pattern) have no
location for a heatmap to point at, which is the point of including them.

Writes results/saliency.json and runs/saliency_examples.npz (for figures).
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M

PS = [0.5, 0.8, 0.9, 1.0]


def gradcam(model, x):
    model.eval()
    x = x.clone().requires_grad_(True)
    fm = model.fmap(x)
    fm.retain_grad()
    logit = model.fc(fm.mean(dim=(2, 3))).squeeze(-1)
    logit.sum().backward()
    w = fm.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fm).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=64, mode="bilinear", align_corners=False)[:, 0]
    sal = x.grad.abs().sum(1)
    return cam.detach().cpu().numpy(), sal.detach().cpu().numpy()


def share(maps, masks):
    num = (maps * masks).sum(axis=(1, 2))
    den = maps.sum(axis=(1, 2)) + 1e-12
    return float(np.mean(num / den))


def main():
    out, examples = [], {}
    for name in D.DATASETS:
        ds = D.load(name)
        x = np.concatenate([ds["val"][0], ds["test"][0]])[:300]
        for artifact in ("tag", "ruler", "ink"):
            xa, masks = D.apply(x, np.ones(len(x), int), artifact, seed=5)
            dil = F.max_pool2d(torch.from_numpy(masks.astype(np.float32))[:, None], 5, 1, 2)[:, 0].numpy()
            for p in PS:
                for seed in (0, 1):
                    pt = f"runs/phase/{name}_{artifact}_p{p}_s{seed}.pt"
                    if not os.path.exists(pt):
                        continue
                    model = M.Net().to(M.DEV)
                    model.load_state_dict(torch.load(pt))
                    cams, sals = [], []
                    xt = M.to_tensor(xa)
                    for i in range(0, len(xt), 100):
                        c, s = gradcam(model, xt[i:i + 100])
                        cams.append(c)
                        sals.append(s)
                    cam, sal = np.concatenate(cams), np.concatenate(sals)
                    out.append(dict(dataset=name, artifact=artifact, p=p, seed=seed,
                                    gradcam_share=share(cam, dil), grad_share=share(sal, dil),
                                    area_share=float(dil.mean())))
                    if seed == 0 and p in (0.5, 0.9):
                        examples[f"{name}_{artifact}_{p}_img"] = xa[:4]
                        examples[f"{name}_{artifact}_{p}_cam"] = cam[:4]
                    print(out[-1], flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/saliency.json", "w"), indent=1)
    np.savez_compressed("runs/saliency_examples.npz", **examples)


if __name__ == "__main__":
    main()

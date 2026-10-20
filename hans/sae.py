"""Sparse autoencoder on the classifier's last spatial feature map, and exact
decision attribution of each dictionary feature.

The classifier computes logit = b + w . mean_pos(h(pos)), where h(pos) is the
128-d feature vector at each of the 8x8 positions of the last conv layer.
A top-k sparse autoencoder writes h(pos) ~ b_d + sum_j a_j(pos) d_j. Because
the head is linear, feature j adds exactly

    c_j(image) = mean_pos(a_j(pos)) * (w . d_j)

to the logit, so features can be ranked by how much they move the decision,
and a feature's effect can be removed by subtracting c_j.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DEV


class TopKSAE(nn.Module):
    def __init__(self, d=128, n=512, k=8):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, n)
        self.dec = nn.Parameter(torch.randn(n, d) / d ** 0.5)
        self.b_d = nn.Parameter(torch.zeros(d))

    def normalize(self):
        with torch.no_grad():
            self.dec.div_(self.dec.norm(dim=1, keepdim=True) + 1e-8)

    def encode(self, x):
        z = F.relu(self.enc(x - self.b_d))
        v, i = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, i, v)

    def forward(self, x):
        a = self.encode(x)
        return a @ self.dec + self.b_d, a


@torch.no_grad()
def fmaps(model, x, bs=256):
    """Last conv feature maps [N,8,8,128] (float32, on GPU)."""
    model.eval()
    return torch.cat([model.fmap(x[i:i + bs]).permute(0, 2, 3, 1).float() for i in range(0, len(x), bs)])


def train_sae(h, n=512, k=8, steps=4000, bs=4096, lr=1e-3, seed=0):
    """h: [N,8,8,128] feature maps. Trains on individual positions."""
    torch.manual_seed(seed)
    X = h.reshape(-1, h.shape[-1])
    scale = X.norm(dim=1).mean()
    X = X / scale
    sae = TopKSAE(X.shape[1], n, k).to(DEV)
    with torch.no_grad():
        sae.b_d.copy_(X.median(dim=0).values)
        sae.enc.weight.copy_(sae.dec.data)  # tied init
        sae.normalize()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for s in range(steps):
        xb = X[torch.randint(0, len(X), (bs,), device=DEV, generator=g)]
        rec, a = sae(xb)
        loss = (rec - xb).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize()
    with torch.no_grad():
        rec, _ = sae(X[:200000])
        fvu = ((rec - X[:200000]).pow(2).sum() / (X[:200000] - X[:200000].mean(0)).pow(2).sum()).item()
    sae.scale = scale.item()
    sae.fvu = fvu
    return sae


@torch.no_grad()
def codes(sae, h, bs=64):
    """Feature activations per position [N,8,8,n] (on GPU)."""
    out = []
    for i in range(0, len(h), bs):
        out.append(sae.encode(h[i:i + bs] / sae.scale))
    return torch.cat(out)


@torch.no_grad()
def contributions(sae, model, a):
    """c [N,n]: each feature's exact additive contribution to the logit."""
    w = model.fc.weight[0].float()                # [128]
    wd = (sae.dec @ w) * sae.scale               # [n]: w . d_j in original units
    return a.mean(dim=(1, 2)) * wd[None, :]

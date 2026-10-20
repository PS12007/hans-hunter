"""A small ResNet for 64x64 images, and GPU training/evaluation helpers."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + (x if self.skip is None else self.skip(x)))


class Net(nn.Module):
    """stem 32 -> [32, 64, 128, 128] stages -> global average pool -> 128-d features -> logit."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.layer1 = nn.Sequential(Block(w, w, 1), Block(w, w, 1))            # 64x64
        self.layer2 = nn.Sequential(Block(w, 2 * w, 2), Block(2 * w, 2 * w, 1))  # 32x32
        self.layer3 = nn.Sequential(Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w, 1))  # 16x16
        self.layer4 = nn.Sequential(Block(4 * w, 4 * w, 2), Block(4 * w, 4 * w, 1))  # 8x8
        self.fc = nn.Linear(4 * w, 1)
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))

    def fmap(self, x):
        """x: [B,3,64,64] in [0,1] -> last conv feature map [B,128,8,8]."""
        x = (x - self.mean) / self.std
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))

    def features(self, x):
        return self.fmap(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.features(x)).squeeze(-1)


def to_tensor(images):
    """[N,64,64,3] float numpy -> [N,3,64,64] float tensor on the GPU."""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous().to(DEV)


def augment(x, gen):
    """Random horizontal flip and +-4 px translation (reflect padding)."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 9, (B,), device=x.device, generator=gen)
    dx = torch.randint(0, 9, (B,), device=x.device, generator=gen)
    idx = torch.arange(64, device=x.device)
    rows = dy[:, None] + idx[None]  # [B,64]
    cols = dx[:, None] + idx[None]
    pad = torch.gather(pad, 2, rows[:, None, :, None].expand(B, 3, 64, 72))
    return torch.gather(pad, 3, cols[:, None, None, :].expand(B, 3, 64, 64))


def auc(scores, labels):
    """ROC AUC via the rank statistic (ties get average rank)."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def predict(model, x, bs=512):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)]).float().cpu().numpy()


def train(x, y, epochs=30, seed=0, lr=3e-3, wd=5e-4, bs=256, weights=None, width=32, log=False,
          init=None, transform=None):
    """Train a Net on GPU tensors x [N,3,64,64] and labels y (numpy). Returns the model.

    weights: optional per-example loss weights.
    init: optional state dict to fine-tune from.
    transform: optional fn(batch, generator) -> batch applied before augmentation."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    model = Net(width).to(DEV).to(memory_format=torch.channels_last)
    if init is not None:
        model.load_state_dict(init)
    else:
        model.mean.copy_(x.mean(dim=(0, 2, 3), keepdim=True))
        model.std.copy_(x.std(dim=(0, 2, 3), keepdim=True) + 1e-6)
    yt = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    wt = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    N = len(x)
    steps = epochs * math.ceil(N / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEV, generator=gen)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = x[idx] if transform is None else transform(x[idx], gen)
            xb = augment(xb, gen).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yt[idx], reduction="none")
            if wt is not None:
                loss = (loss * wt[idx]).sum() / wt[idx].sum()
            else:
                loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log:
                tot += loss.item() * len(idx)
        if log:
            print(f"  epoch {ep + 1}/{epochs} loss {tot / N:.4f}", flush=True)
    model.eval()
    return model

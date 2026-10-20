"""Trigger reverse-engineering: find the cheapest universal edit that flips a
classifier's decision.

A shortcut behaves like an accidental backdoor: one fixed edit (adding the
hospital tag) moves every image toward "disease". We search two families of
universal edits on a set of audit images:

  patch     x' = (1 - m) * x + m * pattern     cost = mask area in pixels
            (Neural Cleanse, Wang et al. 2019): localised marks
  additive  x' = clip(x + delta)               cost = RMS of delta in 1/255 units
            global or invisible changes (tint, noise signatures)

Each search uses a Lagrange-style penalty that is raised while the edit flips
at least `target` of the images and lowered otherwise, and returns the
cheapest edit that reached the target.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import DEV, augment


def _flip_loss(logit, direction):
    # direction +1: push to disease (logit > 0); -1: push to healthy
    return F.softplus(-direction * logit).mean()


def _success(logit, direction):
    return (direction * logit > 0).float().mean().item()


def search_patch(model, x, direction=1, target=0.8, steps=300, lam0=1e-3, seed=0, eot=False):
    """x: [N,3,64,64] audit images currently on the other side. Returns dict."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    m_raw = torch.full((1, 1, 64, 64), -3.0, device=DEV, requires_grad=True)
    p_raw = (torch.randn(1, 3, 64, 64, device=DEV) * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([m_raw, p_raw], lr=0.1)
    lam = lam0
    best = None
    for s in range(steps):
        m = torch.sigmoid(m_raw)
        pat = torch.sigmoid(p_raw)
        xe = (1 - m) * x + m * pat
        if eot:
            xe = augment(xe, gen)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            logit = model(xe.contiguous(memory_format=torch.channels_last)).float()
        area = m.sum()
        loss = _flip_loss(logit, direction) + lam * area
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            succ = _success(logit, direction)
            if succ >= target:
                if best is None or area.item() < best["cost"]:
                    best = dict(cost=area.item(), success=succ, mask=m.detach().cpu().numpy()[0, 0],
                                pattern=pat.detach().cpu().numpy()[0].transpose(1, 2, 0))
                lam *= 1.05
            else:
                lam /= 1.05
    for p in model.parameters():
        p.requires_grad_(True)
    if best is None:
        best = dict(cost=float(64 * 64), success=succ, mask=None, pattern=None)
    return best


def search_additive(model, x, direction=1, target=0.8, steps=300, lam0=1e-3, seed=0, eot=False):
    """Universal additive perturbation. Cost = RMS(delta) * 255."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    delta = torch.zeros(1, 3, 64, 64, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=1e-3)
    lam = lam0
    best = None
    for s in range(steps):
        xe = (x + delta).clamp(0, 1)
        if eot:
            xe = augment(xe, gen)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            logit = model(xe.contiguous(memory_format=torch.channels_last)).float()
        rms = (delta.pow(2).mean() + 1e-12).sqrt() * 255
        loss = _flip_loss(logit, direction) + lam * rms
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            succ = _success(logit, direction)
            if succ >= target:
                if best is None or rms.item() < best["cost"]:
                    best = dict(cost=rms.item(), success=succ,
                                delta=delta.detach().cpu().numpy()[0].transpose(1, 2, 0))
                lam *= 1.05
            else:
                lam /= 1.05
    for p in model.parameters():
        p.requires_grad_(True)
    if best is None:
        best = dict(cost=float("inf"), success=succ, delta=None)
    return best


def search_color(model, x, direction=1, target=0.8, steps=300, lam0=1e-3, seed=0, eot=False):
    """Universal per-channel gain and offset: x' = clip(x * (1 + g) + b).
    Cost = RMS of the induced pixel change, in 1/255 units (same scale as additive)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.zeros(1, 3, 1, 1, device=DEV, requires_grad=True)
    b = torch.zeros(1, 3, 1, 1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([g, b], lr=2e-3)
    lam = lam0
    best = None
    for s in range(steps):
        xe = (x * (1 + g) + b).clamp(0, 1)
        if eot:
            xe = augment(xe, gen)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
            logit = model(xe.contiguous(memory_format=torch.channels_last)).float()
        rms = ((xe - x).pow(2).mean() + 1e-12).sqrt() * 255
        loss = _flip_loss(logit, direction) + lam * rms
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            succ = _success(logit, direction)
            if succ >= target:
                if best is None or rms.item() < best["cost"]:
                    best = dict(cost=rms.item(), success=succ, gain=g.detach().cpu().numpy().ravel(),
                                bias=b.detach().cpu().numpy().ravel())
                lam *= 1.05
            else:
                lam /= 1.05
    for p in model.parameters():
        p.requires_grad_(True)
    if best is None:
        best = dict(cost=float("inf"), success=succ, gain=None, bias=None)
    return best


def stamp(x, trig):
    """Apply a recovered patch trigger (numpy mask [64,64], pattern [64,64,3]) to images [N,64,64,3]."""
    m = trig["mask"][None, ..., None]
    return ((1 - m) * x + m * trig["pattern"][None]).astype(np.float32)

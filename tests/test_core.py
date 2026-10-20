"""Fast checks of the core pieces. Runs on CPU in about a minute, with no dataset download.

    python -m pytest tests -q
"""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hans import data as D, model as M, forensics as Fz


def fake_images(n, seed=0):
    """Smooth random greyscale-ish images in [0, 1], shape [n, 64, 64, 3]."""
    rng = np.random.default_rng(seed)
    low = rng.random((n, 8, 8, 1)).astype(np.float32)
    x = low.repeat(8, 1).repeat(8, 2).repeat(3, 3)
    return np.clip(0.3 + 0.4 * x + 0.02 * rng.standard_normal(x.shape), 0, 1).astype(np.float32)


def test_assign_has_the_requested_correlation():
    y = np.repeat([0, 1], 20000)
    a = D.assign(y, 0.9, np.random.default_rng(0))
    assert abs(a[y == 1].mean() - 0.9) < 0.01
    assert abs(a[y == 0].mean() - 0.1) < 0.01
    a = D.assign(y, 0.5, np.random.default_rng(0))
    assert abs(a[y == 1].mean() - a[y == 0].mean()) < 0.02


@pytest.mark.parametrize("artifact", list(D.ARTIFACTS))
def test_artifacts_only_touch_marked_images(artifact):
    x = fake_images(16)
    art = np.array([1, 0] * 8)
    out, masks = D.apply(x, art, artifact, seed=0)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.min() >= 0 and out.max() <= 1
    np.testing.assert_array_equal(out[art == 0], x[art == 0])
    assert not masks[art == 0].any()
    assert all(np.abs(out[i] - x[i]).max() > 0 for i in np.flatnonzero(art))


def test_invisible_pattern_is_at_most_3_over_255():
    x = fake_images(8)
    out, _ = D.apply(x, np.ones(8, int), "noise", seed=0)
    assert np.abs(out - x).max() <= 3 / 255 + 1e-6


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 300).astype(float)    # many ties
    y = rng.integers(0, 2, 300)
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None]))
    assert M.auc(s, y) == pytest.approx(brute)


def test_stamp_with_empty_mask_is_identity():
    x = fake_images(4)
    trig = dict(mask=np.zeros((64, 64), np.float32), pattern=np.ones((64, 64, 3), np.float32))
    np.testing.assert_allclose(Fz.stamp(x, trig), x)


def test_swap_transform_and_region():
    from experiments.swap import swap_transform, trigger_region
    mask = np.zeros((64, 64)); mask[2:6, 3:7] = 1
    region = trigger_region(mask)
    np.testing.assert_array_equal(region, region[:, ::-1])    # mirrored, like flip augmentation
    assert region[2:6, 3:7].all() and region.sum() > mask.sum()  # grown, like shift augmentation
    x = torch.arange(4, dtype=torch.float32, device=M.DEV)[:, None, None, None].expand(4, 3, 64, 64).contiguous()
    gen = torch.Generator(device=M.DEV); gen.manual_seed(0)
    out = swap_transform(np.zeros((64, 64), bool))(x, gen)
    torch.testing.assert_close(out, x)
    out = swap_transform(np.ones((64, 64), bool))(x, gen)
    assert sorted(out[:, 0, 0, 0].tolist()) == [0, 1, 2, 3]   # whole image swapped: a permutation


def test_audit_finds_a_cheap_trigger_for_a_planted_shortcut():
    """End to end: a model trained with a perfectly correlated corner tag relies on it, and the
    trigger search finds a small universal patch that flips healthy images. (Whether the patch
    lands on the tag is measured in experiments/forensics.py; a 6-epoch toy model is too
    small to test that reliably.)"""
    n = 384
    x = fake_images(n, seed=1)
    y = np.repeat([0, 1], n // 2)
    xa, _ = D.apply(x, D.assign(y, 1.0, np.random.default_rng(0)), "tag", seed=0)
    model = M.train(M.to_tensor(xa), y, epochs=6, seed=0, bs=64)
    clean = fake_images(32, seed=2)
    tagged, _ = D.apply(clean, np.ones(32, int), "tag", seed=5)
    assert (M.predict(model, M.to_tensor(clean)) < 0).mean() > 0.8    # clean images look healthy
    assert (M.predict(model, M.to_tensor(tagged)) > 0).mean() > 0.8   # the tag alone flips them
    res = Fz.search_patch(model, M.to_tensor(clean), steps=150, seed=0)
    assert res["mask"] is not None and res["cost"] < 200             # pixels; the image has 4096


def test_patch_search_respects_forbidden_region():
    torch.manual_seed(0)
    model = M.Net().to(M.DEV)
    x = M.to_tensor(fake_images(8))
    forbid = np.ones((64, 64), bool)
    forbid[32:, 32:] = False                       # only the bottom-right quadrant is allowed
    res = Fz.search_patch(model, x, steps=20, target=0.0, seed=0, forbid=forbid)
    assert res["mask"] is not None
    assert res["mask"][forbid].max() == 0
    assert res["mask"][~forbid].max() > 0


def test_split_separates_two_planted_clusters():
    from experiments.split import kmeans2, split
    g = torch.Generator().manual_seed(0)
    c = torch.randn(2, 50, generator=g) * 3
    Z = torch.cat([c[0] + torch.randn(30, 50, generator=g), c[1] + torch.randn(30, 50, generator=g),
                   torch.randn(140, 50, generator=g) * 0.3]).to(M.DEV)
    _, lab = kmeans2(Z[:60] / Z[:60].norm(dim=1, keepdim=True))
    lab = lab.cpu().numpy()
    assert (lab[:30] == lab[0]).all() and (lab[30:] != lab[0]).all()
    # suspects = highest guide score among healthy images; the split direction separates the groups
    y = np.zeros(200, int)
    guide = np.r_[np.ones(60), np.zeros(140)]
    s, top, _, _ = split(Z, y, guide, frac=0.3)
    assert set(top.tolist()) == set(range(60))
    assert (s[:30].min() > s[30:60].max()) or (s[:30].max() < s[30:60].min())

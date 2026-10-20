"""Datasets and synthetic shortcuts.

Three MedMNIST v2 datasets, each turned into a binary "disease vs not" task:
  derma     DermaMNIST (HAM10000 dermatoscopy), malignant (akiec, bcc, mel) vs benign
  pneumonia PneumoniaMNIST (paediatric chest X-ray), pneumonia vs normal
  breast    BreastMNIST (breast ultrasound), malignant vs normal/benign

Shortcuts are artifacts drawn on the image whose presence is correlated with
the label in training data, the way a hospital tag, a ruler or a scanner can be
correlated with disease in real collections.
"""
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SIZE = 64

DATASETS = {
    # file, function mapping original labels to binary target (1 = disease)
    "derma": ("dermamnist_64.npz", lambda y: np.isin(y, [0, 1, 4]).astype(np.int64)),
    "pneumonia": ("pneumoniamnist_64.npz", lambda y: (y == 1).astype(np.int64)),
    "breast": ("breastmnist_128.npz", lambda y: (y == 0).astype(np.int64)),
}
NAMES = {"derma": "Skin lesion (malignant)", "pneumonia": "Chest X-ray (pneumonia)",
         "breast": "Breast ultrasound (malignant)"}


def _to_rgb64(x):
    x = x.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = np.repeat(x[..., None], 3, axis=-1)
    if x.shape[1] != SIZE:  # 128 -> 64 by 2x2 mean pooling
        f = x.shape[1] // SIZE
        x = x.reshape(x.shape[0], SIZE, f, SIZE, f, 3).mean(axis=(2, 4))
    return x


def load(name):
    """Return dict split -> (images float32 [N,64,64,3] in [0,1], labels int64 [N], raw labels)."""
    fname, binarize = DATASETS[name]
    d = np.load(os.path.join(ROOT, fname), allow_pickle=False)
    out = {}
    for split in ("train", "val", "test"):
        raw = d[f"{split}_labels"].ravel().astype(np.int64)
        out[split] = (_to_rgb64(d[f"{split}_images"]), binarize(raw), raw)
    return out


# --------------------------------------------------------------------------
# Artifacts. Each returns (image with artifact, mask of pixels it touches).
# rng gives per-image jitter so the artifact is never pixel-identical.

_GLYPH_R = np.array([
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1]], dtype=np.float32)


def art_tag(img, rng):
    """A small bright letter in the top-left corner, like an X-ray side marker or site stamp."""
    g = np.kron(_GLYPH_R, np.ones((2, 2), np.float32))  # 14x10
    y0, x0 = 3 + rng.integers(0, 3), 3 + rng.integers(0, 3)
    out, mask = img.copy(), np.zeros(img.shape[:2], bool)
    h, w = g.shape
    region = out[y0:y0 + h, x0:x0 + w]
    out[y0:y0 + h, x0:x0 + w] = np.where(g[..., None] > 0, 0.95, region)
    mask[y0:y0 + h, x0:x0 + w] = g > 0
    return out, mask


def art_ruler(img, rng):
    """A ruler along the bottom edge: a thin bar with tick marks (dermatology rulers)."""
    out, mask = img.copy(), np.zeros(img.shape[:2], bool)
    y = SIZE - 6 - rng.integers(0, 3)
    color = np.array([0.92, 0.92, 0.85], np.float32)
    mask[y, 4:SIZE - 4] = True
    for x in range(4 + rng.integers(0, 4), SIZE - 4, 4):
        mask[y - 3:y, x] = True
    out[mask] = color
    return out, mask


def art_ink(img, rng):
    """A dark purple ellipse drawn around the centre (surgical skin marking)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cy, cx = 32 + rng.uniform(-2, 2), 32 + rng.uniform(-2, 2)
    ry, rx = 25 + rng.uniform(-2, 2), 27 + rng.uniform(-2, 2)
    r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    mask = np.abs(r - 1.0) < 0.045
    out = img.copy()
    out[mask] = 0.55 * out[mask] + 0.45 * np.array([0.35, 0.15, 0.45], np.float32)
    return out, mask


def art_tint(img, rng):
    """A global colour and contrast shift, like a different scanner or camera."""
    out = img.copy()
    out = np.clip(0.9 * out + 0.06, 0, 1) * np.array([1.0, 0.96, 0.9], np.float32)
    return out.astype(np.float32), np.ones(img.shape[:2], bool)


_rng_pat = np.random.default_rng(12345)
_PATTERN = np.sign(_rng_pat.standard_normal((8, 8))).astype(np.float32)
_PATTERN = np.tile(_PATTERN, (SIZE // 8, SIZE // 8))


def art_noise(img, rng):
    """An invisible fixed +-3/255 pixel pattern, like a scanner's processing signature."""
    out = np.clip(img + (3.0 / 255.0) * _PATTERN[..., None], 0, 1).astype(np.float32)
    return out, np.ones(img.shape[:2], bool)


ARTIFACTS = {"tag": art_tag, "ruler": art_ruler, "ink": art_ink, "tint": art_tint, "noise": art_noise}
ART_NAMES = {"tag": "Corner tag", "ruler": "Ruler", "ink": "Ink marking",
             "tint": "Scanner tint", "noise": "Invisible pattern"}


def assign(labels, p, rng):
    """Artifact presence: P(a=1 | y=1) = p, P(a=1 | y=0) = 1 - p. p = 0.5 means no correlation."""
    u = rng.random(len(labels))
    return np.where(labels == 1, u < p, u < 1 - p).astype(np.int64)


def apply(images, art, artifact, seed=0):
    """Draw `artifact` on images where art == 1. Returns images and masks."""
    fn = ARTIFACTS[artifact]
    rng = np.random.default_rng(seed)
    out = images.copy()
    masks = np.zeros(images.shape[:3], bool)
    for i in np.flatnonzero(art):
        out[i], masks[i] = fn(images[i], rng)
    return out, masks

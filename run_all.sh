#!/bin/sh
# Runs every experiment in order. Each script skips work that is already saved.
PY=${PY:-python}
set -e
$PY experiments/phase.py        # train the model zoo
$PY experiments/forensics.py    # recover triggers
$PY experiments/fix.py          # trigger scrubbing
$PY experiments/ground.py       # find the images that carry the shortcut
$PY experiments/reweight.py     # group-based fixes
$PY experiments/saliency.py     # Grad-CAM baseline
$PY experiments/real_audit.py   # the unmodified datasets
$PY experiments/figures.py

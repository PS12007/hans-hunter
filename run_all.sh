#!/bin/sh
# Runs every experiment in order. Each script skips work that is already saved.
PY=${PY:-python}
EXTRA="--ps 0.5,0.9,1.0 --seeds 2,3"   # two extra seeds at the key values of p
set -e
$PY experiments/phase.py        # train the model zoo
$PY experiments/phase.py $EXTRA
$PY experiments/forensics.py    # recover triggers
$PY experiments/forensics.py $EXTRA
$PY experiments/fix.py          # trigger scrubbing
$PY experiments/ground.py       # find the images that carry the shortcut
$PY experiments/reweight.py     # group-based fixes
$PY experiments/swap.py         # region-swap repair for local markers
$PY experiments/swap_iter.py    # ... iterated with re-audits
$PY experiments/context.py      # does the trigger need its context?
$PY experiments/context.py $EXTRA
$PY experiments/saliency.py     # Grad-CAM baseline
$PY experiments/real_audit.py   # the unmodified datasets
$PY experiments/figures.py

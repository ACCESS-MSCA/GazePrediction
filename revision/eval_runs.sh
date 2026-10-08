#!/bin/bash
# Evaluate every finished retrained run on all targets, then summarise.
cd "$(dirname "$0")"
EXTRA=""; MODELS=""
for d in runs/*/; do n=$(basename $d); [ -f $d/conv_$n.json ] && [ -f $d/best.pth ] || continue
  [ -f results/per_window/dsav_test__$n.npz ] && continue
  EXTRA="$EXTRA $n=$PWD/$d/best.pth"; MODELS="$MODELS $n"; done
if [ -n "$MODELS" ]; then
  conda run --no-capture-output -n access_env python eval_unified.py --models $MODELS --extra $EXTRA \
    --targets eht:test eht:val eht:isvc_holdout dsav:test 2>&1 | grep "^\["
fi
conda run --no-capture-output -n access_env python seeds_summary.py

#!/bin/bash
# Re-synchronised DSAV360 test data (pipelines/create_dsav360_dataset.py now maps every sample to the
# video frame recorded in the gaze log): rebuild the saliency cache, swap it in (old cache kept as
# dsav_unsynced), archive the old DSAV360 per-window outputs and re-evaluate every run and baseline.
set -e
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n access_env python"
DATA=/mnt/DataLake/AlienBackup/access/processed_dataset
NEW_PROC=$DATA/dsav360_processed_vid_sync

(cd .. && $PY revision/build_sal_cache.py --dataset dsav --processed_dir $NEW_PROC \
    --cache_root $DATA/sal_cache25_sync --sal_path ./models/unisal.pth)

mv $DATA/sal_cache25/dsav $DATA/sal_cache25/dsav_unsynced
mv $DATA/sal_cache25_sync/dsav $DATA/sal_cache25/dsav

mkdir -p results/per_window/superseded_dsav_unsynced
mv results/per_window/dsav_test__*.npz results/per_window/superseded_dsav_unsynced/

EXTRA=""; MODELS=""
for d in runs/*/; do n=$(basename $d); [ -f $d/conv_$n.json ] && [ -f $d/best.pth ] || continue
  EXTRA="$EXTRA $n=$PWD/$d/best.pth"; MODELS="$MODELS $n"; done
$PY eval_unified.py --models $MODELS --extra $EXTRA --targets dsav:test 2>&1 | grep "^\[" || true
$PY baselines.py --targets dsav:test 2>&1 | grep "^\[\|Nakashima" || true
echo "resync done"

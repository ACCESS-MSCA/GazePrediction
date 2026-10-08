#!/bin/bash
# Usage: ./run_queue.sh queue_file [parallel]
Q=$1; P=${2:-3}
cd "$(dirname "$0")"; mkdir -p logs runs
grep -v '^\s*$' "$Q" | xargs -P "$P" -I{} bash -c 'args="{}"; name=$(echo $args | sed -E "s/.*--name ([^ ]+).*/\1/"); if [ -f runs/$name/conv_$name.json ]; then echo "skip $name"; exit 0; fi; conda run --no-capture-output -n access_env python train_cached.py $args > logs/train_$name.log 2>&1 && echo "done $name" || echo "FAIL $name"'

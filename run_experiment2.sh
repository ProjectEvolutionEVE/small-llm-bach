#!/bin/sh
# Trains every model in experiment 2 (see PREDICTIONS.md), one after another.
# Run: sh run_experiment2.sh      (about 75 minutes on the Intel iMac)
# Output: models/<name>/ and models/<name>.log
cd "$(dirname "$0")"
mkdir -p models
train() {
  name=$1; shift
  [ -f "models/$name/model.safetensors" ] && { echo "skip $name"; return; }
  echo "$(date '+%H:%M') start $name"
  .venv/bin/python train_music.py "models/$name" "$@" > "models/$name.log" 2>&1
  echo "$(date '+%H:%M') done  $name: $(grep 'final val' models/$name.log)"
}
train m0_baseline
train m2_physics --physics
train m1_position --position
train m3_both --position --physics
train m0_half --half
train m2_half --physics --half

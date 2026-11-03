#!/bin/sh
# Scores the first model and every experiment 2 model on all 20 held-out
# chorales (evaluate.py), and probes what each learned about pitch.
# Run after run_experiment2.sh. Output: results/<name>.json, results/<name>_probe.txt
cd "$(dirname "$0")"
mkdir -p results
for dir in bachLLM models/m0_baseline models/m1_position models/m2_physics models/m3_both models/m0_half models/m2_half; do
  name=$(basename "$dir")
  echo "$(date '+%H:%M') scoring $name"
  .venv/bin/python evaluate.py "$dir" > "results/$name.log" 2>&1
  .venv/bin/python probe_pitch.py "$dir" > "results/${name}_probe.txt" 2>/dev/null
done
echo "$(date '+%H:%M') all scored"

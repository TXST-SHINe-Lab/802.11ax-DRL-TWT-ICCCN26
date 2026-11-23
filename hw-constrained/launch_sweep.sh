#!/usr/bin/env bash
# Learning-rate sweep (proposal, Section 3.3): 3 learning rates x 3 seeds x 2 agents, 3000 steps.
# Seeds 101-103 are reserved for the sweep and never reused in the final runs.
#
#   ./launch_sweep.sh                      # sequential
#   JOBS=2 ./launch_sweep.sh               # two runs at a time
#   LRS="1e-4 3e-4 1e-3" STEPS=3000 ./launch_sweep.sh
#   EXTRA="--mock" STEPS=200 ./launch_sweep.sh   # smoke test without ns-3
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
JOBS=${JOBS:-1}
STEPS=${STEPS:-3000}
LRS=${LRS:-"1e-4 3e-4 1e-3"}
SEEDS=${SEEDS:-"101 102 103"}
OBS_STATS=${OBS_STATS:-obs_stats.json}
EXTRA=${EXTRA:-}
mkdir -p runs/logs

STATS_ARG=""
[[ -f "$OBS_STATS" ]] && STATS_ARG="--obs-stats $OBS_STATS"

jobs_file=$(mktemp)
for algo in mlp lstm; do
  for lr in $LRS; do
    for s in $SEEDS; do
      echo "$PY train_hw.py --algo $algo --seed $s --purpose sweep --learning-rate $lr --total-timesteps $STEPS --eval-every 1000 $STATS_ARG $EXTRA > runs/logs/sweep_${algo}_lr${lr}_seed${s}.log 2>&1 && echo done $algo lr=$lr seed=$s || echo FAILED $algo lr=$lr seed=$s" >> "$jobs_file"
    done
  done
done

echo "Running $(wc -l < "$jobs_file") sweep runs, $JOBS at a time"
xargs -P "$JOBS" -I{} bash -c "{}" < "$jobs_file"
rm -f "$jobs_file"
$PY summarize_sweep.py

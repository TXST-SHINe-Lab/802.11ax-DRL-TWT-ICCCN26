#!/usr/bin/env bash
# Final training (proposal, Section 4): 10 seeds x 2 agents x 15k steps = 20 runs.
# Learning rates come from sweep_choice.json (summarize_sweep.py) if present, else 3e-4.
# Runs that already have model_final.zip are skipped, so the script can be re-run to resume.
#
#   ./launch_final.sh                       # sequential, both agents
#   JOBS=6 ./launch_final.sh                # six runs at a time (one per physical core)
#   ALGOS=lstm SEEDS="1 2 3" ./launch_final.sh
#   STEPS=15000 EXTRA="--step-timeout 1800" ./launch_final.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
JOBS=${JOBS:-1}
STEPS=${STEPS:-15000}
ALGOS=${ALGOS:-"mlp lstm"}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
OBS_STATS=${OBS_STATS:-obs_stats.json}
EXTRA=${EXTRA:-}
mkdir -p runs/logs

STATS_ARG=""
[[ -f "$OBS_STATS" ]] && STATS_ARG="--obs-stats $OBS_STATS" || echo "WARNING: $OBS_STATS not found; run collect_obs_stats.py first"

lr_for() {
  $PY - "$1" <<'EOF'
import json, os, sys
try:
    print(json.load(open("sweep_choice.json"))[sys.argv[1]]["learning_rate"])
except Exception:
    print(3e-4)
EOF
}

jobs_file=$(mktemp)
for algo in $ALGOS; do
  lr=$(lr_for "$algo")
  echo "$algo: learning rate $lr"
  for s in $SEEDS; do
    echo "$PY train_hw.py --algo $algo --seed $s --learning-rate $lr --total-timesteps $STEPS $STATS_ARG $EXTRA > runs/logs/final_${algo}_seed${s}.log 2>&1 && echo done $algo seed=$s || echo FAILED $algo seed=$s" >> "$jobs_file"
  done
done

echo "Running $(wc -l < "$jobs_file") runs, $JOBS at a time"
xargs -P "$JOBS" -I{} bash -c "{}" < "$jobs_file"
rm -f "$jobs_file"

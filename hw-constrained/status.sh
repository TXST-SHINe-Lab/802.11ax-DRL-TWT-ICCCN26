#!/usr/bin/env bash
# status.sh - one-screen status of everything this project runs on the workstation.
#   hw-constrained/status.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$HERE/runs"
NS3="$(cd "$HERE/../../../../.." && pwd)"          # …/ns-3.44
COPY="$(dirname "$NS3")/$(basename "$NS3")-orig"   # …/ns-3.44-orig, next to it
REPRO="$COPY/contrib/ai/examples/twt/ppo-sb3-scripts/checkpoints/repro"

echo "== Machine   up since $(uptime -s)   free disk $(df -h "$HERE" | awk 'NR==2 {print $4}')"
echo "   running: $(pgrep -fc twt-main-simulation-default) ns-3 simulations," \
     "$(pgrep -fc '^[^ ]*python[^ ]* train_hw.py') bench training runs, $(pgrep -fc '^[^ ]*python[^ ]* [^ ]*repro_train.py') reproduction"
for s in run_eda.sh launch_sweep.sh launch_final.sh reproduce_published.sh; do
    pgrep -f "$s" >/dev/null && echo "   active launcher: $s"
done

count() { local n=0; for d in $1; do [ -f "$d/model_final.zip" ] && n=$((n + 1)); done; echo $n; }
progress() {  # training steps logged so far by a run that has no final model yet
    local d=$1 n
    n=$(wc -l < "$d/train_episodes.jsonl" 2>/dev/null || echo 0)
    echo "$(basename "$d"): $n episodes"
}

echo "== Sweep     $(count "$RUNS/*_sweep") of 18 finished"
for d in "$RUNS"/*_sweep; do [ -d "$d" ] && [ ! -f "$d/model_final.zip" ] && echo "   unfinished $(progress "$d")"; done
[ -f "$HERE/sweep_choice.json" ] && echo "   sweep_choice.json: $(tr -d '\n ' < "$HERE/sweep_choice.json" | cut -c1-160)"

finals=$(ls -d "$RUNS"/*_seed[0-9] "$RUNS"/*_seed10 2>/dev/null)
echo "== Finals    $(count "$finals") of 20 finished"
for d in $finals; do [ ! -f "$d/model_final.zip" ] && echo "   unfinished $(progress "$d")"; done

echo "== Reproduction of the published MLP-PPO"
if [ -f "$HERE/published/mlp/model_final.zip" ]; then
    echo "   finished: $HERE/published/mlp/model_final.zip"
elif [ -d "$REPRO" ]; then
    for d in "$REPRO"/*_twt_energy_*; do
        [ -d "$d" ] || continue
        n=$(($(cat "$d"/training_log_*.jsonl 2>/dev/null | wc -l) - 1))
        echo "   $(basename "$d"): $((n < 0 ? 0 : n)) of ~422 episodes$( ls "$d"/*final.zip >/dev/null 2>&1 && echo ', finished')"
    done
else
    echo "   not started"
fi

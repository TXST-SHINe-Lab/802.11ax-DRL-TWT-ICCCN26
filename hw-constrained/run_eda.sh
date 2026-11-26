#!/usr/bin/env bash
# run_eda.sh - Ahmed's EDA pipeline (exploration-scripts 1 -> 3 -> 4 -> 5) for the bench config,
# with the simulations run in parallel instead of one after another.
#
# Why: the reward z-scores in derived_constants.json must be dialed for the 5-STA bench mix and
# for the legal-only action space (proposal, Section 3), not reused from the 16-STA run.
# 1-collect-data.sh runs its spawns serially (1275 x ~70 s = ~1 day); this runs JOBS at a time.
#
# Nothing of Ahmed's is modified. It calls his run-single-spawn.py, 3-prepare-data.sh,
# 4-eda-analysis.py and 5-dial-constants.py unchanged, with TWT_HW_LEGAL_ONLY=1 so the random
# actions are drawn only from schedules the bench can execute (that switch is added by
# ns3-patch/apply_hw_patch.py).
#
# Usage (from anywhere, EHRL venv active, bench patch applied and ns-3 rebuilt):
#     NUM_SPAWNS=20 JOBS=6 hw-constrained/run_eda.sh       # quick test (~5 min)
#     JOBS=6 hw-constrained/run_eda.sh                     # full run (1275 spawns)
#
# JOBS: one per PHYSICAL core. On a 6-core / 12-thread CPU, 10 jobs made every spawn 2x slower
# (136 s vs 68 s), so hyper-threads add almost nothing.
#
# Run it ALONE: like 1-collect-data.sh it must not share the machine with training, and it
# refuses to start if any twt-main-simulation is running.
#
# Two things it handles that a naive parallel run gets wrong:
#   * Log names. ns-3 and the Python wrapper name their CSVs by the wall-clock SECOND, so two
#     spawns starting in the same second would write into the same file. Starts are spaced
#     STAGGER_S apart (default 3 s) with a lock.
#   * Stale logs. 4-eda-analysis.py reads every py-wrapper-env-*.csv in data-log/, including
#     ones from earlier tests. data-log/ is archived first and the EDA's own logs are moved
#     into the run directory afterwards.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWT_DIR="$(dirname "$HERE")"
EXP="$TWT_DIR/exploration-scripts"
DATA_LOG="$TWT_DIR/data-log"

NUM_SPAWNS=${NUM_SPAWNS:-1275}
JOBS=${JOBS:-6}
BASE_SEED=${BASE_SEED:-42}
STAGGER_S=${STAGGER_S:-3}
SPAWN_TIMEOUT=${SPAWN_TIMEOUT:-900}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-run_${STAMP}_hw}
RUN_DIR="$EXP/eda-data/$RUN_ID"
PY=${PYTHON:-python}

die() { echo "ERROR: $*" >&2; exit 1; }

grep -q "EE7331-HW-PATCH" "$TWT_DIR/twt-constants.h" ||
    die "bench patch not applied (run ns3-patch/apply_hw_patch.py and rebuild ns-3 first)"
grep -q "TWT_HW_LEGAL_ONLY" "$EXP/run-single-spawn.py" ||
    die "run-single-spawn.py has no TWT_HW_LEGAL_ONLY switch (re-run apply_hw_patch.py)"
if pgrep -f twt-main-simulation >/dev/null; then
    die "a twt-main-simulation is already running; the EDA must run alone (pgrep -af twt-main-simulation)"
fi
"$PY" -c "import pb_twt_interface_py" 2>/dev/null ||
    (cd "$TWT_DIR" && "$PY" -c "import pb_twt_interface_py") ||
    die "cannot import pb_twt_interface_py with $PY (activate the EHRL venv)"

mkdir -p "$RUN_DIR/transitions" "$RUN_DIR/logs"
if [ -d "$DATA_LOG" ] && [ -n "$(ls -A "$DATA_LOG" 2>/dev/null)" ]; then
    mkdir -p "$TWT_DIR/data-log-archive"
    mv "$DATA_LOG" "$TWT_DIR/data-log-archive/before_${RUN_ID}"
    echo "Archived old data-log/ to data-log-archive/before_${RUN_ID}"
fi
mkdir -p "$DATA_LOG"

cat >"$RUN_DIR/run_config.json" <<EOF
{
    "timestamp": "${STAMP}",
    "num_spawns": ${NUM_SPAWNS},
    "jobs": ${JOBS},
    "base_seed": ${BASE_SEED},
    "hw_legal_only": true,
    "launcher": "hw-constrained/run_eda.sh",
    "schedule_table": "${EXP}/table_schedule.json",
    "assignment_table": "${EXP}/table_assignment.json"
}
EOF

echo "=============================================="
echo "EDA for the bench config: $NUM_SPAWNS spawns, $JOBS in parallel"
echo "Run dir: $RUN_DIR"
echo "=============================================="

export TWT_HW_LEGAL_ONLY=1
export EXP RUN_DIR BASE_SEED STAGGER_S SPAWN_TIMEOUT PY
LOCK="$RUN_DIR/.start.lock"
export LOCK

one_spawn() {
    i=$1
    seed=$((BASE_SEED + i))
    # Space out starts so no two spawns open their CSV logs in the same second.
    flock "$LOCK" sleep "$STAGGER_S"
    t0=$(date +%s)
    if timeout --signal=KILL "$SPAWN_TIMEOUT" "$PY" "$EXP/run-single-spawn.py" \
        --seed "$seed" --spawn-id "$i" --output "$RUN_DIR/transitions/spawn_${i}.jsonl" \
        >"$RUN_DIR/logs/spawn_${i}.log" 2>&1; then
        echo "ok   spawn $i (seed $seed) $(( $(date +%s) - t0 )) s"
    else
        echo "FAIL spawn $i (seed $seed), see logs/spawn_${i}.log"
        pkill -f "randSeed=${seed}" 2>/dev/null
    fi
    rm -f "/dev/shm/MySeg_${seed}" "/dev/shm/MyCpp2PyMsg_${seed}" \
        "/dev/shm/MyPy2CppMsg_${seed}" "/dev/shm/MyLockable_${seed}" 2>/dev/null
}
export -f one_spawn

START=$(date +%s)
seq 1 "$NUM_SPAWNS" | xargs -P "$JOBS" -I{} bash -c 'one_spawn {}' | tee "$RUN_DIR/collect.log"
ELAPSED=$(( $(date +%s) - START ))

OK=$(grep -c "^ok" "$RUN_DIR/collect.log")
FAILED=$(grep -c "^FAIL" "$RUN_DIR/collect.log")
echo "=============================================="
echo "Collected: $OK ok, $FAILED failed, in ${ELAPSED} s"
grep "^ok" "$RUN_DIR/collect.log" | awk -v j="$JOBS" '{s+=$(NF-1); n++} END {if (n) printf "Average spawn: %.0f s wall (with %d in parallel)\n", s/n, j}'
echo "Logs from this EDA: $(du -sh "$DATA_LOG" | cut -f1) in data-log/ ($(ls "$DATA_LOG" | wc -l) files)"
echo "Free disk: $(df -h "$TWT_DIR" | awk 'NR==2 {print $4}')"
echo "=============================================="
[ "$OK" -gt 0 ] || die "no spawn succeeded; read $RUN_DIR/logs/spawn_1.log"

# Keep the EDA's logs with the run, so 4-eda-analysis.py reads exactly these.
mv "$DATA_LOG" "$RUN_DIR/data-log"
mkdir -p "$DATA_LOG"

set -e
bash "$EXP/3-prepare-data.sh" "$RUN_DIR"
(cd "$EXP" && "$PY" 4-eda-analysis.py --npz-file "$RUN_DIR/stacked_transitions.npz" \
    --data-log-dir "$RUN_DIR/data-log" --output-dir "$RUN_DIR" \
    --schedule-table "$EXP/table_schedule.json" --no-plots)
(cd "$EXP" && "$PY" 5-dial-constants.py --run-dir "$RUN_DIR")

echo ""
echo "Done. Use these constants for obs_stats, the sweep and the final runs:"
echo "    export TWT_DERIVED_CONSTANTS=$RUN_DIR/derived_constants.json"

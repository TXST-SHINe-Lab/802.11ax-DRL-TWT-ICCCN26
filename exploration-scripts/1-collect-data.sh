#!/bin/bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>
#
# Batch data collection for the EDA: runs many NS-3 simulations with random actions to cover the action space.
# Writes JSONL transitions under eda-data/run_<timestamp>/, which 3-prepare-data.sh then stacks into NPZ.
#
# NUM_SPAWNS and BASE_SEED can be overridden from the environment; the other values are recorded
# in run_config.json for reference only (the spawn count and step length come from twt-constants.h).
#
# Usage:
#     ./1-collect-data.sh
#     NUM_SPAWNS=50 ./1-collect-data.sh                        # short run
#     BASE_SEED=99 ./1-collect-data.sh                        # different seed sequence

# set -e is deliberately off: one failed spawn must not abandon the whole collection.

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${HOME}/NS3-project/EHRL/bin/activate"

# Extract ACTIVE_NUM_STA from twt-constants.h
ACTIVE_NUM_STA=$(grep "^#define ACTIVE_NUM_STA" "${TWT_DIR}/twt-constants.h" | awk '{print $3}')

# Data collection parameters
NUM_SPAWNS=${NUM_SPAWNS:-1275} # 1316
NUM_STAS=${NUM_STAS:-${ACTIVE_NUM_STA}} # Default from twt-constants.h
STEPS_PER_SPAWN=${STEPS_PER_SPAWN:-38}
BASE_SEED=${BASE_SEED:-42}     # Starting random seed
CONFIG_FILE=${CONFIG_FILE:-""} # Optional config file path

# Output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE_DIR="${SCRIPT_DIR}/eda-data"
# run_all.sh exports TWT_RUN_ID so the EDA sweep shares the pipeline run id and every stage of
# one invocation lands under the same run_*/ name. Standalone runs keep their own timestamp.
RUN_DIR="${OUTPUT_BASE_DIR}/${TWT_RUN_ID:-run_${TIMESTAMP}}"

# Create output directories
mkdir -p "${RUN_DIR}/transitions"
mkdir -p "${RUN_DIR}/logs"

echo "=============================================="
echo "EDA Data Collection - TWT RL Environment"
echo "=============================================="
echo "Timestamp:       ${TIMESTAMP}"
echo "Output Dir:      ${RUN_DIR}"
echo "Num Spawns:      ${NUM_SPAWNS}"
echo "Num STAs:        ${NUM_STAS}"
echo "Steps/Spawn:     ${STEPS_PER_SPAWN}"
echo "Base Seed:       ${BASE_SEED}"
echo "Config File:     ${CONFIG_FILE:-'Using defaults'}"
echo "=============================================="

# Activate virtual environment
if [ -f "${VENV_PATH}" ]; then
    echo "Activating Python venv: ${VENV_PATH}"
    source "${VENV_PATH}"
else
    echo "WARNING: Virtual environment not found at ${VENV_PATH}"
    echo "Make sure Python environment is properly configured"
fi

# Kill any orphaned NS3 simulation processes from previous runs
echo "Cleaning up stale processes and shared memory..."
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1

# Check if action tables exist
SCHEDULE_TABLE="${SCRIPT_DIR}/table_schedule.json"
ASSIGNMENT_TABLE="${SCRIPT_DIR}/table_assignment.json"

if [ ! -f "${SCHEDULE_TABLE}" ] || [ ! -f "${ASSIGNMENT_TABLE}" ]; then
    echo "Generating action tables..."
    python3 "${SCRIPT_DIR}/generate_action_tables.py" \
        --output-dir "${SCRIPT_DIR}" \
        --num-schedules 20 \
        --num-assignments 20
    echo "Action tables generated."
fi

# Save run configuration
cat >"${RUN_DIR}/run_config.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "num_spawns": ${NUM_SPAWNS},
    "num_stas": ${NUM_STAS},
    "steps_per_spawn": ${STEPS_PER_SPAWN},
    "base_seed": ${BASE_SEED},
    "config_file": "${CONFIG_FILE}",
    "schedule_table": "${SCHEDULE_TABLE}",
    "assignment_table": "${ASSIGNMENT_TABLE}"
}
EOF

# Progress tracking
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Function to run a single spawn
run_spawn() {
    local spawn_idx=$1
    local seed=$((BASE_SEED + spawn_idx))
    local output_file="${RUN_DIR}/transitions/spawn_${spawn_idx}.jsonl"
    local log_file="${RUN_DIR}/logs/spawn_${spawn_idx}.log"

    echo "[${spawn_idx}/${NUM_SPAWNS}] Running spawn with seed=${seed}..."

    # Build command
    CMD="python3 ${SCRIPT_DIR}/run-single-spawn.py"
    CMD="${CMD} --seed ${seed}"
    CMD="${CMD} --spawn-id ${spawn_idx}"
    CMD="${CMD} --output ${output_file}"

    if [ -n "${CONFIG_FILE}" ] && [ -f "${CONFIG_FILE}" ]; then
        CMD="${CMD} --config ${CONFIG_FILE}"
    fi

    # Run with timeout (10 minutes per spawn)
    # --signal=KILL ensures child processes (NS3) are also terminated
    if timeout --signal=KILL 600 ${CMD} >"${log_file}" 2>&1; then
        echo "[${spawn_idx}/${NUM_SPAWNS}] Completed successfully"
        # Clean up any leftover shared memory for this seed
        rm -f "/dev/shm/MySeg_${seed}" "/dev/shm/MyCpp2PyMsg_${seed}" \
            "/dev/shm/MyPy2CppMsg_${seed}" "/dev/shm/MyLockable_${seed}" 2>/dev/null
        return 0
    else
        echo "[${spawn_idx}/${NUM_SPAWNS}] FAILED - check ${log_file}"
        # Kill any orphaned NS3 processes for this seed
        pkill -f "randSeed=${seed}" 2>/dev/null
        # Clean up shared memory for this seed
        rm -f "/dev/shm/MySeg_${seed}" "/dev/shm/MyCpp2PyMsg_${seed}" \
            "/dev/shm/MyPy2CppMsg_${seed}" "/dev/shm/MyLockable_${seed}" 2>/dev/null
        return 1
    fi
}

# Main collection loop
echo ""
echo "Starting data collection..."
echo ""

for ((i = 1; i <= NUM_SPAWNS; i++)); do
    echo "DEBUG: About to run spawn $i"
    if run_spawn $i; then
        ((COMPLETED++))
        echo "DEBUG: Spawn $i succeeded (COMPLETED=$COMPLETED)"
    else
        ((FAILED++))
        echo "DEBUG: Spawn $i failed (FAILED=$FAILED)"
    fi
    echo "DEBUG: Loop iteration $i complete, continuing..."
    echo ""

    # Progress update every 10 spawns
    if [ $((i % 10)) -eq 0 ]; then
        ELAPSED=$(($(date +%s) - START_TIME))
        AVG_TIME=$((ELAPSED / i))
        REMAINING=$(((NUM_SPAWNS - i) * AVG_TIME))
        echo ""
        echo "Progress: ${i}/${NUM_SPAWNS} (${COMPLETED} success, ${FAILED} failed)"
        echo "Elapsed: ${ELAPSED}s, Est. remaining: ${REMAINING}s"
        echo ""
    fi
done
echo "DEBUG: Loop completed"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "Data Collection Complete"
echo "=============================================="
echo "Total Time:      ${TOTAL_TIME}s"
echo "Successful:      ${COMPLETED}/${NUM_SPAWNS}"
echo "Failed:          ${FAILED}/${NUM_SPAWNS}"
echo "Output Dir:      ${RUN_DIR}"
echo "=============================================="

# Count total transitions
TOTAL_TRANSITIONS=$(cat "${RUN_DIR}/transitions"/*.jsonl 2>/dev/null | wc -l || echo "0")
echo "Total Transitions: ${TOTAL_TRANSITIONS}"

# Create summary file
cat >"${RUN_DIR}/collection_summary.json" <<EOF
{
    "completed_spawns": ${COMPLETED},
    "failed_spawns": ${FAILED},
    "total_time_seconds": ${TOTAL_TIME},
    "total_transitions": ${TOTAL_TRANSITIONS},
    "avg_time_per_spawn": $(echo "scale=2; ${TOTAL_TIME} / ${NUM_SPAWNS}" | bc),
    "avg_transitions_per_spawn": $(echo "scale=2; ${TOTAL_TRANSITIONS} / ${COMPLETED}" | bc 2>/dev/null || echo "0")
}
EOF

# echo ""
# echo "Cleaning up logs and transitions directories..."
# rm -rf "${RUN_DIR}/logs" "${RUN_DIR}/transitions"
# echo "Cleaned up temporary directories"
# echo ""
# echo "Next step: Run 3-prepare-data.sh to stack transitions into NPZ format"
# echo ""

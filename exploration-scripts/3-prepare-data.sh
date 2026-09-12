#!/bin/bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>
#
# Convert the JSONL transitions collected by 1-collect-data.sh into stacked NPZ for the EDA and training.
# Defaults to the newest eda-data/run_* directory; pass one explicitly to re-prepare an older run.
#
# Usage:
#     ./3-prepare-data.sh
#     ./3-prepare-data.sh eda-data/run_20260718_195843

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${HOME}/NS3-project/EHRL/bin/activate"

# Find the most recent run directory
DATA_DIR="${SCRIPT_DIR}/eda-data"

if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    echo "Run 1-collect-data.sh first to collect data"
    exit 1
fi

# Get run directory from argument or find latest (sort by timestamp in filename)
if [ -n "$1" ]; then
    RUN_DIR="$1"
else
    RUN_DIR=$(ls -d "${DATA_DIR}"/run_* 2>/dev/null | sort -r | head -1)
fi

if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "Error: No run directory found in ${DATA_DIR}"
    echo "Usage: $0 [run_directory]"
    exit 1
fi

echo "=============================================="
echo "Prepare EDA Data"
echo "=============================================="
echo "Run Directory: ${RUN_DIR}"
echo "=============================================="

# Activate virtual environment
if [ -f "${VENV_PATH}" ]; then
    echo "Activating Python venv..."
    source "${VENV_PATH}"
fi

# Check for transitions
TRANSITIONS_DIR="${RUN_DIR}/transitions"
TRANSITION_COUNT=$(ls -1 "${TRANSITIONS_DIR}"/*.jsonl 2>/dev/null | wc -l)

if [ "${TRANSITION_COUNT}" -eq 0 ]; then
    echo "Error: No transition files found in ${TRANSITIONS_DIR}"
    exit 1
fi

echo "Found ${TRANSITION_COUNT} transition files"

# Run stacking script
echo ""
echo "Stacking transitions..."
python3 "${SCRIPT_DIR}/stack-data.py" \
    --input-dir "${RUN_DIR}" \
    --max-stas 16 \
    --verbose

# Check output
NPZ_FILE="${RUN_DIR}/stacked_transitions.npz"
STATS_FILE="${RUN_DIR}/stacked_transitions.stats.json"

if [ -f "${NPZ_FILE}" ]; then
    SIZE=$(du -h "${NPZ_FILE}" | cut -f1)
    echo ""
    echo "=============================================="
    echo "Data Preparation Complete"
    echo "=============================================="
    echo "NPZ File:   ${NPZ_FILE} (${SIZE})"
    echo "Stats File: ${STATS_FILE}"
    echo "=============================================="

    # Print key statistics
    echo ""
    echo "Dataset Statistics:"
    python3 -c "
import json
with open('${STATS_FILE}') as f:
    stats = json.load(f)
print(f\"  Transitions: {stats['n_transitions']}\")
print(f\"  Episodes: {stats['n_episodes']}\")
print(f\"  State dim: {stats['state_dim']}\")
print(f\"  Schedule actions: {stats['action_schedule_unique']}\")
print(f\"  Assignment actions: {stats['action_assignment_unique']}\")
print(f\"  Reward mean: {stats['reward_mean']:.4f}\")
print(f\"  Reward std: {stats['reward_std']:.4f}\")
"
    echo ""
    echo "Next step: Run EDA analysis notebooks or scripts"
else
    echo "Error: NPZ file was not created!"
    exit 1
fi

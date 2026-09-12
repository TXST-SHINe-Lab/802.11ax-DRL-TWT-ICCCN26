#!/bin/bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>
#
# Train a PPO agent for TWT WiFi scheduling, once per reward preset, and print a suite summary.
# SHINE Lab, Texas State University.
#
# Activates the Python environment, optionally cleans old artifacts, trains each preset in turn, and saves models and logs under checkpoints/ and tb_logs/ organised by preset, nested under run_*/ when run_all.sh sets TWT_RUN_ID.
# A failing preset is recorded and the suite continues; the script exits 1 if any preset failed.
#
# Usage:
#     ./run_training.sh                                     # all presets, the full suite
#     ./run_training.sh throughput                          # a single preset
#     ./run_training.sh energy queue                        # selected presets
#     ./run_training.sh --training-script train_ppo_V1.py   # choose the model, default train_lstm_ppo_V1.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration, edit these to customise training ---

# Model choice: train_ppo_V1.py is the standard MLP, train_lstm_ppo_V1.py is the recurrent one.
TRAINING_SCRIPT="train_lstm_ppo_V1.py"

# Checkpoint pattern, derived from the script name: drop the train_ prefix and the .py suffix, then append _twt_.
# train_lstm_ppo_V1.py becomes lstm_ppo_V1_twt_, train_ppo_V1.py becomes ppo_V1_twt_.
CHECKPOINT_PATTERN="${TRAINING_SCRIPT#train_}"      # Drop the train_ prefix
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN%.py}_twt_" # Drop .py, append _twt_

TIMESTEPS=15000 # Timesteps per preset, roughly 3500 steps/hr, so about 4 hours each
SEED=1          # Fixed for reproducibility
CLEAN_OLD=false # true wipes this run's checkpoints/ and tb_logs/ before training

# run_all.sh exports TWT_RUN_ID as run_YYYYMMDD_HHMMSS so one pipeline invocation keeps its
# artifacts together, matching the run_*/ layout exploration-scripts/eda-data already uses.
# Unset, both paths collapse to the flat layout this script used before.
RUN_SUBDIR="${TWT_RUN_ID:+/$TWT_RUN_ID}"
CHECKPOINT_DIR="checkpoints$RUN_SUBDIR"
TB_DIR="tb_logs$RUN_SUBDIR"

# Presets trained when none are named on the command line.
ALL_PRESETS=("throughput" "energy" "queue")
# ALL_PRESETS=("throughput")

# Parse the command line: --training-script <script.py>, plus any number of preset names.
PRESETS_TO_TRAIN=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --training-script)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TRAINING_SCRIPT="$2"
                shift 2
            else
                echo "Error: --training-script requires a value"
                exit 1
            fi
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Positional argument = preset name
            PRESETS_TO_TRAIN+=("$1")
            shift
            ;;
    esac
done

# No presets named, so train them all.
if [ ${#PRESETS_TO_TRAIN[@]} -eq 0 ]; then
    PRESETS_TO_TRAIN=("${ALL_PRESETS[@]}")
fi

# Recompute in case --training-script changed TRAINING_SCRIPT after the first derivation above.
CHECKPOINT_PATTERN="${TRAINING_SCRIPT#train_}"      # Drop the train_ prefix
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN%.py}_twt_" # Drop .py, append _twt_

# Optional PPO hyperparameters, uncomment to override the training script's defaults.
# Each is passed through only when set; see the flag wiring further down.
# LEARNING_RATE=0.0001    # Lower LR for more stable learning, default 3e-4
# N_STEPS=256             # More steps per rollout for better gradient estimates
# BATCH_SIZE=128          # Larger batches for stability
# ENT_COEF=0.02           # Higher entropy for more exploration

# --- Terminal colours for the log helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate virtual environment
log_info "Activating Python environment..."
source ~/NS3-project/EHRL/bin/activate

# Clean old data if configured
if [ "$CLEAN_OLD" = true ]; then
    log_info "Cleaning old training artifacts..."
    rm -rf "${CHECKPOINT_DIR:?}"/*
    rm -rf "${TB_DIR:?}"/*
    rm -rf __pycache__
    log_success "Cleanup complete"
fi

# Print configuration
echo ""
echo "============================================================"
echo "PPO Training Suite for TWT WiFi Scheduling"
echo "============================================================"
echo "Training script:  $TRAINING_SCRIPT"
echo "Checkpoint pattern: $CHECKPOINT_PATTERN"
echo "Timesteps:        $TIMESTEPS per preset"
echo "Seed:             $SEED"
echo "Presets:          ${PRESETS_TO_TRAIN[*]}"
echo "Total presets:    ${#PRESETS_TO_TRAIN[@]}"
echo "Script dir:       $SCRIPT_DIR"
echo "============================================================"
echo ""

# Track overall results
SUITE_START_TIME=$(date +%s)
declare -A TRAINING_RESULTS
FAILED_PRESETS=()
SUCCESSFUL_PRESETS=()

# Train each preset
for PRESET_IDX in "${!PRESETS_TO_TRAIN[@]}"; do
    REWARD_PRESET="${PRESETS_TO_TRAIN[$PRESET_IDX]}"
    PRESET_NUM=$((PRESET_IDX + 1))

    echo ""
    echo "############################################################"
    echo "# Training Preset $PRESET_NUM/${#PRESETS_TO_TRAIN[@]}: $REWARD_PRESET"
    echo "############################################################"
    echo ""

    log_info "Starting training for preset: $REWARD_PRESET"
    START_TIME=$(date +%s)

    # Build training command with optional hyperparameters
    CMD="python3.11 $TRAINING_SCRIPT"
    CMD="$CMD --total-timesteps $TIMESTEPS"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --reward-preset $REWARD_PRESET"
    CMD="$CMD --output-dir $CHECKPOINT_DIR"
    CMD="$CMD --tensorboard-log $TB_DIR"

    # Add optional hyperparameters if set
    [ -n "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
    [ -n "$N_STEPS" ] && CMD="$CMD --n-steps $N_STEPS"
    [ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"
    [ -n "$ENT_COEF" ] && CMD="$CMD --ent-coef $ENT_COEF"

    # Clear the previous preset's logs and any ns-3 process it left running.
    rm -rf ../data-log/* 2>/dev/null || true
    pkill -f "twt-main[-]simulation" 2>/dev/null || true

    # set -e off so a failed preset is recorded rather than aborting the remaining ones.
    set +e
    $CMD
    EXIT_CODE=$?
    set -e

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESSFUL_PRESETS+=("$REWARD_PRESET")
        TRAINING_RESULTS[$REWARD_PRESET]="SUCCESS (${DURATION}s)"
        log_success "Preset $REWARD_PRESET completed successfully in ${DURATION}s"
    else
        FAILED_PRESETS+=("$REWARD_PRESET")
        TRAINING_RESULTS[$REWARD_PRESET]="FAILED (exit code: $EXIT_CODE)"
        log_error "Preset $REWARD_PRESET failed with exit code: $EXIT_CODE"
    fi
done

SUITE_END_TIME=$(date +%s)

# Calculate total duration
TOTAL_DURATION=$((SUITE_END_TIME - SUITE_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "############################################################"
echo "# Training Suite Complete"
echo "############################################################"
echo ""
echo "Results Summary:"
echo "------------------------------------------------------------"
for PRESET in "${PRESETS_TO_TRAIN[@]}"; do
    echo "  $PRESET: ${TRAINING_RESULTS[$PRESET]}"
done
echo "------------------------------------------------------------"
echo ""
echo "Successful: ${#SUCCESSFUL_PRESETS[@]}/${#PRESETS_TO_TRAIN[@]} presets"
echo "Failed:     ${#FAILED_PRESETS[@]}/${#PRESETS_TO_TRAIN[@]} presets"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Show trained models
if [ ${#SUCCESSFUL_PRESETS[@]} -gt 0 ]; then
    echo "Trained models:"
    for PRESET in "${SUCCESSFUL_PRESETS[@]}"; do
        # Find the model for this preset
        MODEL_DIR=$(ls -td "$CHECKPOINT_DIR"/${CHECKPOINT_PATTERN}${PRESET}_*/ 2>/dev/null | head -1)
        if [ -n "$MODEL_DIR" ]; then
            FINAL_MODEL=$(ls "${MODEL_DIR}"*_final.zip 2>/dev/null | head -1)
            if [ -n "$FINAL_MODEL" ]; then
                echo "  $PRESET: $FINAL_MODEL"
            fi
        fi
    done
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate all: ./run_eval.sh"
    echo "  2. Evaluate one: ./run_eval.sh throughput"
    echo "  3. Plot training: python3.11 plot_training.py --training-script $TRAINING_SCRIPT"
    echo "  4. Analyze rewards: python3.11 analyze_reward_signal.py --training-script $TRAINING_SCRIPT"
    echo "  5. TensorBoard:  tensorboard --logdir $TB_DIR/"
fi

# Exit with error if any preset failed
if [ ${#FAILED_PRESETS[@]} -gt 0 ]; then
    exit 1
fi

exit 0

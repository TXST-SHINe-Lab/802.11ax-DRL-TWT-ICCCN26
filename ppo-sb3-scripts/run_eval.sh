#!/bin/bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>
#
# Benchmark a trained PPO policy for TWT WiFi scheduling against baselines, one preset at a time.
# SHINE Lab, Texas State University.
#
# Every policy within a preset is evaluated under that preset's reward function, so the scores are comparable.
# Defaults run PPO and the analytical policy; the random and heuristic baselines are off, so their --skip flags only matter after INCLUDE_RANDOM or INCLUDE_HEURISTIC is turned on below.
#
# Usage:
#     ./run_eval.sh                                        # all presets
#     ./run_eval.sh throughput                             # a single preset
#     ./run_eval.sh --skip-analytical                      # PPO only
#     ./run_eval.sh --training-script train_ppo_V1.py      # choose the model, default train_lstm_ppo_V1.py
#     ./run_eval.sh --training-script=train_ppo_V1.py throughput

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration, edit these to customise evaluation ---
N_EPISODES=50
SEED=1000000000         # Fixed for reproducibility, well clear of the training seed
INCLUDE_RANDOM=true     # Random baseline, off by default
INCLUDE_HEURISTIC=true  # Preset-matched heuristic policies, off by default
INCLUDE_ANALYTICAL=true # Analytical model-based policies
CLEAN_OLD=false         # true wipes this run's eval_results/ before running

# run_all.sh exports TWT_RUN_ID as run_YYYYMMDD_HHMMSS so one pipeline invocation keeps its
# artifacts together, matching the run_*/ layout exploration-scripts/eda-data already uses.
# Unset, both paths collapse to the flat layout this script used before.
# eval_policy.py reads TWT_RUN_ID itself, so its output lands in EVAL_DIR without a flag.
RUN_SUBDIR="${TWT_RUN_ID:+/$TWT_RUN_ID}"
CHECKPOINT_DIR="checkpoints$RUN_SUBDIR"
EVAL_DIR="eval_results$RUN_SUBDIR"

# Model choice; the checkpoint pattern is derived from it, train_X.py becoming X_twt_.
TRAINING_SCRIPT='train_lstm_ppo_V1.py'

# Presets evaluated when none are named on the command line.
ALL_PRESETS=("throughput" "energy" "queue")
# ALL_PRESETS=("queue")

# Parse the command line: the --skip flags, --training-script in either form, plus any number of preset names.
PRESETS_TO_EVAL=()
SCRIPT_NEXT=false
for arg in "$@"; do
    if [ "$arg" = "--skip-random" ]; then
        INCLUDE_RANDOM=false
    elif [ "$arg" = "--skip-heuristic" ]; then
        INCLUDE_HEURISTIC=false
    elif [ "$arg" = "--skip-analytical" ]; then
        INCLUDE_ANALYTICAL=false
    elif [[ "$arg" == --training-script=* ]]; then
        TRAINING_SCRIPT="${arg#--training-script=}"
    elif [ "$arg" = "--training-script" ]; then
        SCRIPT_NEXT=true
    elif [ "$SCRIPT_NEXT" = true ]; then
        TRAINING_SCRIPT="$arg"
        SCRIPT_NEXT=false
    else
        PRESETS_TO_EVAL+=("$arg")
    fi
done

# Derive checkpoint pattern from training script name
# "train_lstm_ppo_V1.py" -> "lstm_ppo_V1_twt_"
CHECKPOINT_PATTERN="${TRAINING_SCRIPT#train_}"      # Remove "train_" prefix
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN%.py}_twt_" # Remove ".py", add "_twt_"

# Default to all presets if none specified
if [ ${#PRESETS_TO_EVAL[@]} -eq 0 ]; then
    PRESETS_TO_EVAL=("${ALL_PRESETS[@]}")
fi

# Pairs each reward preset with the heuristic it is compared against; currently the identity map.
declare -A HEURISTIC_MAP=(
    ["throughput"]="throughput"
    ["energy"]="energy"
    ["queue"]="queue"
)

VERBOSE=true # Show step-by-step progress

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

# Clean old evaluation results if configured
if [ "$CLEAN_OLD" = true ]; then
    log_info "Cleaning old evaluation results..."
    rm -rf "${EVAL_DIR:?}"/*
    log_success "Cleanup complete"
fi

# Echo the newest *_final.zip for a preset, or return 1 if no checkpoint directory matches.
find_model_for_preset() {
    local preset=$1
    local dir="$CHECKPOINT_DIR/${CHECKPOINT_PATTERN}${preset}_"*
    dir=$(ls -td $dir/ 2>/dev/null | head -1)

    if [ -n "$dir" ]; then
        model=$(ls "${dir}"*_final.zip 2>/dev/null | head -1)
        if [ -n "$model" ]; then
            echo "$model"
            return 0
        fi
    fi
    return 1
}

# Print configuration
echo ""
echo "============================================================"
echo "PPO Evaluation Suite for TWT WiFi Scheduling"
echo "============================================================"
echo "Checkpoint pattern: $CHECKPOINT_PATTERN"
echo "Episodes:         $N_EPISODES per policy"
echo "Seed:             $SEED"
echo "Presets to eval:  ${PRESETS_TO_EVAL[*]}"
echo "Include random:   $INCLUDE_RANDOM"
echo "Include heuristic: $INCLUDE_HEURISTIC"
echo "Include analytical: $INCLUDE_ANALYTICAL"
echo ""

# Count the policies per preset so the progress display can show test N of M.
POLICIES_PER_PRESET=1 # PPO is always included
[ "$INCLUDE_RANDOM" = true ] && POLICIES_PER_PRESET=$((POLICIES_PER_PRESET + 1))
[ "$INCLUDE_HEURISTIC" = true ] && POLICIES_PER_PRESET=$((POLICIES_PER_PRESET + 1))
[ "$INCLUDE_ANALYTICAL" = true ] && POLICIES_PER_PRESET=$((POLICIES_PER_PRESET + 1))

# Build policy list string
POLICY_LIST="PPO"
[ "$INCLUDE_RANDOM" = true ] && POLICY_LIST="$POLICY_LIST + Random"
[ "$INCLUDE_HEURISTIC" = true ] && POLICY_LIST="$POLICY_LIST + Heuristic"
[ "$INCLUDE_ANALYTICAL" = true ] && POLICY_LIST="$POLICY_LIST + Analytical"
echo "Policies per preset: $POLICY_LIST"
TOTAL_TESTS=$((${#PRESETS_TO_EVAL[@]} * POLICIES_PER_PRESET))
echo "Total tests: $TOTAL_TESTS"
echo "============================================================"
echo ""

# Track overall results
SUITE_START_TIME=$(date +%s)
declare -A EVAL_RESULTS
declare -A PPO_REWARDS
declare -A RANDOM_REWARDS
declare -A HEURISTIC_REWARDS
declare -A ANALYTICAL_REWARDS
FAILED_PRESETS=()
SUCCESSFUL_PRESETS=()
SKIPPED_PRESETS=()

# --- Evaluate each preset, every enabled policy scored under that preset's reward function ---
TEST_NUM=0

for PRESET_IDX in "${!PRESETS_TO_EVAL[@]}"; do
    REWARD_PRESET="${PRESETS_TO_EVAL[$PRESET_IDX]}"
    COMPARE_ONE_HEURISTIC="${HEURISTIC_MAP[$REWARD_PRESET]:-$REWARD_PRESET}"
    TEST_NUM=$((TEST_NUM + POLICIES_PER_PRESET))

    echo ""
    echo "############################################################"
    echo "# Preset: $REWARD_PRESET ($POLICY_LIST)"
    echo "############################################################"
    echo ""

    # No checkpoint means the preset is skipped, not failed, so the suite still exits 0.
    MODEL_PATH=$(find_model_for_preset "$REWARD_PRESET")

    if [ -z "$MODEL_PATH" ]; then
        log_error "No trained model found for preset: $REWARD_PRESET"
        log_info "Run './run_training.sh $REWARD_PRESET' first"
        SKIPPED_PRESETS+=("$REWARD_PRESET")
        EVAL_RESULTS[$REWARD_PRESET]="SKIPPED (no model)"
        continue
    fi

    log_info "Found model: $MODEL_PATH"
    if [ "$INCLUDE_HEURISTIC" = true ]; then
        log_info "Using heuristic: $COMPARE_ONE_HEURISTIC"
    fi
    if [ "$INCLUDE_RANDOM" = true ]; then
        log_info "Including random baseline"
    fi
    if [ "$INCLUDE_ANALYTICAL" = true ]; then
        log_info "Including analytical policy: $COMPARE_ONE_HEURISTIC"
    fi

    # Build the eval_policy.py command: the PPO model always, each enabled baseline appended as a --compare flag.
    CMD="python3.11 eval_policy.py $MODEL_PATH"
    CMD="$CMD --n-episodes $N_EPISODES"
    CMD="$CMD --seed $SEED"
    CMD="$CMD --reward-type $REWARD_PRESET"
    CMD="$CMD --output-prefix ${CHECKPOINT_PATTERN}"

    if [ "$INCLUDE_HEURISTIC" = true ]; then
        CMD="$CMD --compare-heuristic $COMPARE_ONE_HEURISTIC"
    fi

    if [ "$INCLUDE_RANDOM" = true ]; then
        CMD="$CMD --compare-random"
    fi

    if [ "$INCLUDE_ANALYTICAL" = true ]; then
        CMD="$CMD --compare-analytical $COMPARE_ONE_HEURISTIC"
    fi

    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi

    log_info "Starting evaluation for preset: $REWARD_PRESET"
    START_TIME=$(date +%s)

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
        EVAL_RESULTS[$REWARD_PRESET]="SUCCESS (${DURATION}s)"
        log_success "Preset $REWARD_PRESET evaluation completed in ${DURATION}s"

        # Pull each policy's mean reward out of the newest result JSON for the final summary table.
        # Any policy that did not run, or whose value is not a float, reports N/A.
        LATEST_RESULT=$(ls -t "$EVAL_DIR"/eval_${CHECKPOINT_PATTERN}${REWARD_PRESET}_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_RESULT" ]; then
            PPO_REWARD=$(python3.11 -c "import json; d=json.load(open('$LATEST_RESULT')); print(f\"{d.get('PPO',{}).get('mean_reward','N/A'):.4f}\" if isinstance(d.get('PPO',{}).get('mean_reward'),float) else 'N/A')" 2>/dev/null || echo "N/A")
            RAND_REWARD=$(python3.11 -c "import json; d=json.load(open('$LATEST_RESULT')); print(f\"{d.get('Random',{}).get('mean_reward','N/A'):.4f}\" if isinstance(d.get('Random',{}).get('mean_reward'),float) else 'N/A')" 2>/dev/null || echo "N/A")
            HEUR_REWARD=$(python3.11 -c "import json; d=json.load(open('$LATEST_RESULT')); k=[k for k in d.keys() if 'heuristic' in k.lower()]; print(f\"{d[k[0]]['mean_reward']:.4f}\" if k and isinstance(d[k[0]].get('mean_reward'),float) else 'N/A')" 2>/dev/null || echo "N/A")
            ANAL_REWARD=$(python3.11 -c "import json; d=json.load(open('$LATEST_RESULT')); k=[k for k in d.keys() if 'analytical' in k.lower()]; print(f\"{d[k[0]]['mean_reward']:.4f}\" if k and isinstance(d[k[0]].get('mean_reward'),float) else 'N/A')" 2>/dev/null || echo "N/A")
            PPO_REWARDS[$REWARD_PRESET]="$PPO_REWARD"
            RANDOM_REWARDS[$REWARD_PRESET]="$RAND_REWARD"
            HEURISTIC_REWARDS[$REWARD_PRESET]="$HEUR_REWARD"
            ANALYTICAL_REWARDS[$REWARD_PRESET]="$ANAL_REWARD"
        fi
    else
        FAILED_PRESETS+=("$REWARD_PRESET")
        EVAL_RESULTS[$REWARD_PRESET]="FAILED (exit code: $EXIT_CODE)"
        log_error "Preset $REWARD_PRESET evaluation failed with exit code: $EXIT_CODE"
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
echo "# Evaluation Suite Complete"
echo "############################################################"
echo ""
echo "Results Summary:"
echo "------------------------------------------------------------"
printf "%-12s %-12s %-12s %-12s %-12s\n" "Preset" "PPO" "Random" "Heuristic" "Analytical"
echo "------------------------------------------------------------"
for PRESET in "${PRESETS_TO_EVAL[@]}"; do
    STATUS="${EVAL_RESULTS[$PRESET]:-UNKNOWN}"
    PPO_R="${PPO_REWARDS[$PRESET]:-N/A}"
    RAND_R="${RANDOM_REWARDS[$PRESET]:-N/A}"
    HEUR_R="${HEURISTIC_REWARDS[$PRESET]:-N/A}"
    ANAL_R="${ANALYTICAL_REWARDS[$PRESET]:-N/A}"
    if [ "$INCLUDE_RANDOM" = false ]; then
        RAND_R="-"
    fi
    if [ "$INCLUDE_HEURISTIC" = false ]; then
        HEUR_R="-"
    fi
    if [ "$INCLUDE_ANALYTICAL" = false ]; then
        ANAL_R="-"
    fi
    printf "%-12s %-12s %-12s %-12s %-12s\n" "$PRESET" "$PPO_R" "$RAND_R" "$HEUR_R" "$ANAL_R"
done
echo "------------------------------------------------------------"
echo ""
echo "Presets: ${#SUCCESSFUL_PRESETS[@]} success, ${#FAILED_PRESETS[@]} failed, ${#SKIPPED_PRESETS[@]} skipped"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Show detailed results if any successful
if [ ${#SUCCESSFUL_PRESETS[@]} -gt 0 ]; then
    echo "Result files saved in $EVAL_DIR/"
    for PRESET in "${SUCCESSFUL_PRESETS[@]}"; do
        RESULT_FILE=$(ls -t "$EVAL_DIR"/eval_${PRESET}_*.json 2>/dev/null | head -1)
        if [ -n "$RESULT_FILE" ]; then
            echo "  $PRESET: $RESULT_FILE"
        fi
    done
fi

# Exit with error if any preset failed
if [ ${#FAILED_PRESETS[@]} -gt 0 ]; then
    exit 1
fi

exit 0

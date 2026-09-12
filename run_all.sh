#!/bin/bash
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>
#
# End-to-end TWT pipeline: build ns-3, run the sanity checks, run the EDA and constant dialing, then train and evaluate both PPO variants.
# Runs LSTM PPO first and MLP PPO second, each followed by evaluation and plotting.
#
# Artifacts are grouped per run: each invocation picks one run id, run_YYYYMMDD_HHMMSS, and every
# stage nests its output under that name in eda-data/, checkpoints/, eval_results/, plots/ and
# tb_logs/, so earlier runs are preserved rather than overwritten.
# Destructive only for scratch data: data-log/, the __pycache__ directories and the action-table
# JSONs are deleted before the run starts. derived_constants.json is never touched here -- it
# lives inside eda-data/$TWT_RUN_ID/, owned by that run, and is never overwritten by a later one.
# Requires the EHRL virtualenv at ~/NS3-project/EHRL and takes many hours end to end.
#
# Usage:
#     ./run_all.sh

# Resolve paths from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWT_DIR="$SCRIPT_DIR" # Script lives in the project root
NS3_ROOT="$(cd "$TWT_DIR/../../../.." && pwd)"

cd "$NS3_ROOT"

# --- Cleanup stale processes and shared memory from previous runs ---
# A killed ns-3 run leaves its /dev/shm segments behind and the next one fails to attach, so every stage boundary repeats this.
echo "Cleaning up stale NS3 processes and shared memory..."
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1

# One run id for the whole pipeline, exported so every stage nests its output under it.
# run_training.sh and run_eval.sh read it directly; the Python scripts go through run_paths.py;
# 1-collect-data.sh uses it for the EDA sweep directory.
export TWT_RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIRS=(
    "$TWT_DIR/exploration-scripts/eda-data/$TWT_RUN_ID"
    "$TWT_DIR/ppo-sb3-scripts/checkpoints/$TWT_RUN_ID"
    "$TWT_DIR/ppo-sb3-scripts/eval_results/$TWT_RUN_ID"
    "$TWT_DIR/ppo-sb3-scripts/plots/$TWT_RUN_ID"
    "$TWT_DIR/ppo-sb3-scripts/tb_logs/$TWT_RUN_ID"
)

echo "Run id: $TWT_RUN_ID"
echo "Removing transient artifacts from previous runs"
# Artifact directories are no longer wiped. Each run owns a run_*/ subdirectory, so previous
# results neither collide with this run nor hide it.
for artifact in \
    "$TWT_DIR/data-log" \
    "$TWT_DIR/exploration-scripts/table_schedule.json" \
    "$TWT_DIR/exploration-scripts/table_assignment.json" \
    "$TWT_DIR/__pycache__" \
    "$TWT_DIR/exploration-scripts/__pycache__" \
    "$TWT_DIR/ppo-sb3-scripts/__pycache__" \
    "$TWT_DIR/test-scripts/__pycache__"; do
    if [ -e "$artifact" ]; then
        echo "  removing ${artifact#$TWT_DIR/}"
        rm -rf "$artifact"
    fi
done
mkdir -p "$TWT_DIR/data-log" "${RUN_DIRS[@]}"

echo "Cleanup complete."

source ~/NS3-project/EHRL/bin/activate
./ns3 clean
pip install stable-baselines3 sb3-contrib

./ns3 configure --enable-examples --enable-tests -- \
    -DNS3_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="../../EHRL/bin/python" \
    -DNS3_BINDINGS_INSTALL_DIR="../../EHRL/lib/python3.11/site-packages" \
    -DPython3_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.11.so" \
    -DProtobuf_LIBRARY="/usr/lib/x86_64-linux-gnu/libprotobuf.so" \
    -DProtobuf_INCLUDE_DIRS="/usr/include" \
    -DProtobuf_DIR="/usr/lib/x86_64-linux-gnu/cmake/protobuf"

./ns3 build
cd "$TWT_DIR"

# Clean old logs (create directory if needed)
mkdir -p data-log
rm -rf data-log/*

# Sanity checks: drive one episode with a fixed seed, then confirm the ns-3 logs and the Python-side metrics agree
source ~/NS3-project/EHRL/bin/activate
cd "$TWT_DIR/test-scripts"
python3.11 simple-controller.py --seed 1234
python3.11 summary-metrics.py
python3.11 verify-call-level-metrics.py
python3.11 verify-ns3-to-python.py
python3.11 plot-bi-metrics.py
cd "$TWT_DIR"
rm -rf data-log/*

# EDA sweep, then dial the observation and reward constants from it.
# 5-dial-constants.py writes derived_constants.json into the EDA run directory; the training
# scripts locate it there via run_paths.py. A failure here aborts the pipeline rather than
# training on stale constants.
cd exploration-scripts
mkdir -p eda-data
source ~/NS3-project/EHRL/bin/activate
./1-collect-data.sh
python3.11 2-validate-logvstap.py
./3-prepare-data.sh
python3.11 4-eda-analysis.py
python3.11 5-dial-constants.py || {
    echo "CONSTANT DIALING FAILED - ABORTING"
    exit 1
}
cd ../test-scripts
python3.11 verify-call-level-metrics.py
python3.11 verify-ns3-to-python.py
cd ../
rm -rf data-log/*

# --- Cleanup between EDA and RL training ---
echo "Cleaning up stale NS3 processes and shared memory..."
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1

# --- Model 1 of 2: recurrent PPO ---
cd ppo-sb3-scripts

source ~/NS3-project/EHRL/bin/activate
MODL="train_lstm_ppo_V1.py"
./run_training.sh --training-script "$MODL"

# --- Cleanup between training and evaluation ---
echo "Cleaning up stale NS3 processes and shared memory..."
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1
rm -rf ../data-log/*

./run_eval.sh --training-script "$MODL"

# --- Cleanup after evaluation ---
echo "Cleaning up stale NS3 processes and shared memory..."
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1
rm -rf ../data-log/*
python3.11 plot_training.py --training-script "$MODL"
python3.11 plot_evaluation.py --training-script "$MODL"
python3.11 analyze_reward_signal.py --training-script "$MODL"

echo "Cleaning up stale NS3 processes and shared memory..."
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1
rm -rf ../data-log/*

# --- Model 2 of 2: feed-forward PPO, same stages as above ---
MODL="train_ppo_V1.py"
./run_training.sh --training-script "$MODL"
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1
rm -rf ../data-log/*
./run_eval.sh --training-script "$MODL"
pkill -f "twt-main-simulation" 2>/dev/null
rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_* 2>/dev/null
sleep 1
rm -rf ../data-log/*
python3.11 plot_training.py --training-script "$MODL"
python3.11 plot_evaluation.py --training-script "$MODL"
python3.11 analyze_reward_signal.py --training-script "$MODL"

rm -rf ../data-log/*

echo ""
echo "Pipeline complete. Artifacts for this run:"
for d in "${RUN_DIRS[@]}"; do
    [ -d "$d" ] && echo "  ${d#"$TWT_DIR/"}"
done

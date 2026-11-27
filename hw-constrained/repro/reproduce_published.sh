#!/usr/bin/env bash
# reproduce_published.sh - Re-create the published ICCCN'26 energy-preset policy (the H1 baseline).
#
# The public repository has the published runs' hyperparams.json, training logs and observation
# statistics, but not the trained weights. This re-trains them with Ahmed's own training script,
# the published hyperparameters (seed 1, 15,000 steps, energy preset) and his reward constants,
# on his ORIGINAL simulator (16 STAs, 20-BI decisions, no bench patch, no packet-counter fix),
# then compares the result with the published logs.
#
# The bench build in the main ns-3 tree is left alone: the original simulator is built in a
# separate copy, <ns-3.44>-orig, from the same sources with the bench patch reverted. That copy
# keeps Ahmed's Wi-Fi patches and the ns3-ai GCC 15 fixes, which it needs to build.
#
# Safe to run while the bench training runs (repro_train.py only removes its own shared memory),
# but it takes one core from it. Re-running resumes: the copy, the build and a finished training
# are each skipped when already done.
#
#   hw-constrained/repro/reproduce_published.sh                 # MLP-PPO (the H1 baseline)
#   ALGOS="mlp lstm" hw-constrained/repro/reproduce_published.sh
#
# Output: hw-constrained/published/<algo>/{model_final.zip,vecnormalize.pkl,compare.json} in the
# MAIN tree, which evaluate.py's "published:" policy spec reads.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWT="$(cd "$HERE/../.." && pwd)"                       # main tree: …/ns-3.44/contrib/ai/examples/twt
NS3="$(cd "$TWT/../../../.." && pwd)"                  # main tree: …/ns-3.44
COPY="${COPY:-${NS3}-orig}"
CTWT="$COPY/contrib/ai/examples/twt"
ALGOS=${ALGOS:-mlp}
PY=${PYTHON:-$(command -v python)}
BUILD_JOBS=${BUILD_JOBS:-4}
PUB_RUNS="$TWT/ppo-sb3-scripts/checkpoints/run_20260131_010432"
declare -A PUB=([mlp]="ppo_V1_twt_energy_20260202_062417" [lstm]="lstm_ppo_V1_twt_energy_20260131_072222")
declare -A FINAL=([mlp]="ppo_V1_twt_final.zip" [lstm]="lstm_ppo_V1_twt_final.zip")

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

"$PY" -c "import stable_baselines3, sb3_contrib, psutil" 2>/dev/null || die "activate the EHRL venv first"
# Ahmed's trainers log to TensorBoard and show SB3's progress bar. Without these packages
# model.learn() fails at its first step, the trainer catches the error and still saves an
# UNTRAINED model, so install them up front.
if ! "$PY" -c "import tensorboard, tqdm, rich" 2>/dev/null; then
    log "Installing tensorboard, tqdm and rich (needed by Ahmed's training scripts)"
    "$PY" -m pip install -q tensorboard tqdm rich || die "pip install tensorboard tqdm rich failed"
fi
[ -d "$PUB_RUNS" ] || die "published run logs not found at $PUB_RUNS"

# ---- 1. separate copy of ns-3 with the original TWT code ---------------------------------
if [ ! -d "$COPY" ]; then
    free_gb=$(df -BG --output=avail "$(dirname "$NS3")" | tail -1 | tr -dc 0-9)
    [ "$free_gb" -ge 8 ] || die "only ${free_gb} GB free; the copy and its build need about 6-8 GB"
    log "Copying ns-3 sources to $COPY (no build output, logs or bench results)"
    mkdir -p "$COPY"
    (cd "$NS3" && tar cf - --anchored \
        --exclude=./build --exclude=./cmake-cache \
        --exclude=./contrib/ai/examples/twt/data-log \
        --exclude=./contrib/ai/examples/twt/data-log-archive \
        --exclude="./contrib/ai/examples/twt/exploration-scripts/eda-data/run_*_hw" \
        --exclude=./contrib/ai/examples/twt/hw-constrained/runs \
        --exclude=./contrib/ai/examples/twt/hw-constrained/results \
        --exclude=./contrib/ai/examples/twt/hw-constrained/published \
        --exclude=./contrib/ai/examples/twt/ppo-sb3-scripts/checkpoints/repro \
        --no-anchored --exclude="*.so" .) | (cd "$COPY" && tar xf -) ||
        { rm -rf "$COPY"; die "copying the sources failed"; }
    # The bench patch is uncommitted, so restoring the tracked files gives Ahmed's originals.
    git -C "$CTWT" checkout -- .
    find "$CTWT" -name "*.orig" -delete
fi
grep -q "EE7331-HW-PATCH" "$CTWT/twt-constants.h" && die "the copy still has the bench patch ($CTWT)"
grep -q "define ACTIVE_NUM_STA 16" "$CTWT/twt-constants.h" || die "the copy is not the 16-STA original"
log "Copy OK: original 16-STA simulator in $COPY"

# ---- 2. build the copy -------------------------------------------------------------------
# A FULL build, not just the simulator target: ns3-ai starts every episode with `./ns3 run`,
# which first brings the whole build up to date (with all cores). Anything left unbuilt would be
# compiled at the first episode, taking longer than the runner's 120 s start-up window.
SO="$CTWT/pb_twt_interface_py.so"
BIN="$COPY/build/contrib/ai/examples/twt/ns3.44-twt-main-simulation-default"
DONE_MARK="$COPY/.repro-full-build-done"
if [ ! -f "$DONE_MARK" ] || [ ! -f "$SO" ] || [ ! -x "$BIN" ]; then
    cd "$COPY"
    if [ ! -f "$COPY/cmake-cache/CMakeCache.txt" ]; then
        log "Configuring the copy"
        ./ns3 configure --enable-examples -- -DPython3_EXECUTABLE="$PY" > "$COPY/configure.log" 2>&1 ||
            die "configure failed; see $COPY/configure.log"
    fi
    log "Full build of the copy ($BUILD_JOBS jobs; about 1-2 h next to the training runs)"
    ./ns3 build -j "$BUILD_JOBS" > "$COPY/build.log" 2>&1 || die "build failed; see $COPY/build.log"
    cmake --build cmake-cache -j "$BUILD_JOBS" >> "$COPY/build.log" 2>&1 ||
        die "build of the remaining targets (Python bridge) failed; see $COPY/build.log"
    touch "$DONE_MARK"
fi
[ -f "$SO" ] || die "pb_twt_interface_py.so missing in $CTWT"
[ -x "$BIN" ] || die "twt-main-simulation binary missing: $BIN"
(cd "$CTWT" && "$PY" -c "import pb_twt_interface_py") || die "the copy's Python bridge does not import"
log "Build OK"

# The copy must carry these scripts (it does when they were pulled before the copy was made);
# refresh them from the main tree so fixes made after the copy reach it.
mkdir -p "$CTWT/hw-constrained/repro"
cp "$HERE"/repro_train.py "$HERE"/compare_repro.py "$CTWT/hw-constrained/repro/"

# ---- 3. train ---------------------------------------------------------------------------
# FileCommEnv spawns its runner as "python3.11"; point that name at this venv's interpreter.
# A launcher script, not a symlink: Python finds its venv from the path it is started by, so a
# symlink in another folder would start the bare system Python, without numpy, torch or ns3-ai.
SHIM="$COPY/.py311-shim"
mkdir -p "$SHIM"
rm -f "$SHIM/python3.11"
printf '#!/bin/sh\nexec "%s" "$@"\n' "$PY" > "$SHIM/python3.11"
chmod +x "$SHIM/python3.11"
export PATH="$SHIM:$PATH"
python3.11 -c "import numpy, torch, stable_baselines3" 2>/dev/null ||
    die "python3.11 on PATH ($SHIM/python3.11) cannot import numpy/torch/stable_baselines3"
(cd "$CTWT" && python3.11 -c "import pb_twt_interface_py") ||
    die "python3.11 cannot import the copy's pb_twt_interface_py"
# Ahmed's reward constants (16 STAs), the ones the published runs used.
export TWT_DERIVED_CONSTANTS="$CTWT/exploration-scripts/eda-data/run_20260131_010432/derived_constants.json"
[ -f "$TWT_DERIVED_CONSTANTS" ] || die "missing $TWT_DERIVED_CONSTANTS"
unset TWT_RUN_ID

# The original simulator writes ~tens of MB of trace CSVs per episode into data-log/; nothing
# reads them during training, so files older than 15 minutes are removed while it runs.
DL="$CTWT/data-log"
mkdir -p "$DL"
( while sleep 300; do find "$DL" -type f -mmin +15 -delete 2>/dev/null; done ) &
CLEANER=$!
trap 'kill $CLEANER 2>/dev/null' EXIT

OUTROOT="$CTWT/ppo-sb3-scripts/checkpoints/repro"
mkdir -p "$OUTROOT"
for algo in $ALGOS; do
    pub="$PUB_RUNS/${PUB[$algo]}"
    hp="$pub/hyperparams.json"
    run=$(ls -td "$OUTROOT"/*_twt_energy_* 2>/dev/null | while read -r d; do
        [ -f "$d/${FINAL[$algo]}" ] && echo "$d" && break; done || true)
    if [ -n "$run" ]; then
        log "$algo: finished reproduction found ($run), not re-training"
    else
        args=$("$PY" - "$hp" "$algo" <<'EOF'
import json, sys
h, algo = json.load(open(sys.argv[1])), sys.argv[2]
a = ["--total-timesteps", h["total_timesteps"], "--seed", h["seed"], "--reward-type", h["reward_type"],
     "--reward-preset", h["reward_preset"], "--reward-scale", h["reward_scale"],
     "--learning-rate", h["learning_rate"], "--n-steps", h["n_steps"], "--batch-size", h["batch_size"],
     "--n-epochs", h["n_epochs"], "--gamma", h["gamma"], "--gae-lambda", h["gae_lambda"],
     "--clip-range", h["clip_range"], "--ent-coef", h["ent_coef"], "--vf-coef", h["vf_coef"],
     "--max-grad-norm", h["max_grad_norm"], "--clip-obs", h["clip_obs"], "--clip-reward", h["clip_reward"]]
a += ["--net-arch"] + h["net_arch_pi"]
if algo == "mlp":
    a += ["--features-dim", h["features_dim"]]
    if not h.get("use_custom_extractor", True):
        a += ["--no-custom-extractor"]
print(" ".join(str(x) for x in a))
EOF
)
        log "$algo: training with the published hyperparameters ($(basename "$pub"))"
        log "  $args"
        # The COPY's repro_train.py: it locates the simulator next to itself.
        "$PY" "$CTWT/hw-constrained/repro/repro_train.py" "$algo" -- $args --output-dir "$OUTROOT" \
            --tensorboard-log "$CTWT/ppo-sb3-scripts/tb_logs/repro" \
            > "$OUTROOT/train_${algo}.log" 2>&1 || die "$algo training failed; see $OUTROOT/train_${algo}.log"
        prefix=$([ "$algo" = mlp ] && echo ppo_V1 || echo lstm_ppo_V1)
        run=$(ls -td "$OUTROOT"/${prefix}_twt_energy_* | head -1)
    fi
    if grep -q "Training error" "$OUTROOT/train_${algo}.log" 2>/dev/null; then
        die "$algo: the trainer caught an error and saved a partial model; see $OUTROOT/train_${algo}.log"
    fi

    dest="$TWT/hw-constrained/published/$algo"
    mkdir -p "$dest"
    cp "$run/${FINAL[$algo]}" "$dest/model_final.zip"
    cp "$run/vecnormalize.pkl" "$dest/vecnormalize.pkl"
    cp "$run/hyperparams.json" "$run/normalization_stats.json" "$dest/" 2>/dev/null || true
    cp "$run"/training_log_*.jsonl "$dest/" 2>/dev/null || true
    echo "$run" > "$dest/SOURCE"
    log "$algo: reproduction vs published run"
    "$PY" "$HERE/compare_repro.py" --published "$pub" --repro "$run" --out "$dest/compare.json"
    log "$algo: model for evaluate.py -> $dest/model_final.zip"
done
log "Done."

#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Per-run artifact directories shared by the training, evaluation and plotting scripts.

run_all.sh exports TWT_RUN_ID as run_YYYYMMDD_HHMMSS so that one pipeline invocation keeps its
checkpoints, evaluation results and plots together, the way exploration-scripts/eda-data already
groups each EDA sweep under run_*/.

With TWT_RUN_ID unset every path collapses to the flat layout, so running any of these scripts by
hand behaves exactly as it did before.
"""

import glob
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 5-dial-constants.py writes derived_constants.json into the EDA run directory it derived the
# numbers from, so each run keeps the constants it was actually trained against.
EDA_DATA_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "exploration-scripts", "eda-data")
)
DERIVED_CONSTANTS_NAME = "derived_constants.json"

# Artifact roots that gain a run_*/ level. tb_logs is included so TensorBoard runs stay grouped.
ARTIFACT_ROOTS = ("checkpoints", "eval_results", "plots", "tb_logs")


def run_id():
    """Current run identifier, or an empty string when not running under run_all.sh.

    Returns:
        The value of TWT_RUN_ID with surrounding whitespace stripped, or "".
    """
    return os.environ.get("TWT_RUN_ID", "").strip()


def artifact_dir(name, create=False):
    """Absolute path to an artifact directory, nested under the run id when one is set.

    Args:
        name: Artifact root, one of ARTIFACT_ROOTS, e.g. "checkpoints" or "plots".
        create: Create the directory, and any missing parents, when it does not exist.

    Returns:
        "<script_dir>/<name>/<run_id>" when TWT_RUN_ID is set, "<script_dir>/<name>" otherwise.
    """
    path = os.path.join(_SCRIPT_DIR, name)
    rid = run_id()
    if rid:
        path = os.path.join(path, rid)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def resolve(explicit, name, create=False):
    """Honour a command-line directory when given, otherwise fall back to the run-aware default.

    An explicit relative path is resolved against this script's directory, matching how the
    --checkpoints-dir and --eval-dir flags behaved before run_*/ nesting existed.

    Args:
        explicit: Directory from a command-line flag, or None when the flag was not given.
        name: Artifact root to fall back to, one of ARTIFACT_ROOTS.
        create: Create the directory when it does not exist.

    Returns:
        Absolute path to the directory to use.
    """
    if not explicit:
        return artifact_dir(name, create=create)
    path = explicit if os.path.isabs(explicit) else os.path.join(_SCRIPT_DIR, explicit)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def derived_constants_path():
    """Locate derived_constants.json, written by 5-dial-constants.py into its EDA run directory.

    Resolution order:
    1. TWT_DERIVED_CONSTANTS, used verbatim when set.
    2. eda-data/$TWT_RUN_ID/, so a pipeline run reads the constants it just dialed.
    3. The newest eda-data/run_*/ holding the file, matching 5-dial-constants.py's own
       "latest run" convention of a reverse lexicographic sort.

    No fallback below eda-data/: every EDA run now keeps its own derived_constants.json,
    and there is deliberately no default when none exists yet -- callers raise their own
    FileNotFoundError naming the path, and the fix is to run the EDA dial step, not to
    train against silently-stale constants.

    The path is returned whether or not it exists, so each caller raises its own error naming
    the path it looked for. Callers print the resolved path, which is what makes a stale
    constants file visible in the training log.

    Returns:
        Absolute path to derived_constants.json. When no eda-data/run_*/ holds one yet, the
        path is inside the newest run_*/ directory if one exists, otherwise inside
        eda-data/ itself -- neither exists on disk, so the caller's own missing-file error
        fires and names that path.
    """
    override = os.environ.get("TWT_DERIVED_CONSTANTS", "").strip()
    if override:
        return override

    rid = run_id()
    if rid:
        candidate = os.path.join(EDA_DATA_DIR, rid, DERIVED_CONSTANTS_NAME)
        if os.path.exists(candidate):
            return candidate

    runs = sorted(
        glob.glob(os.path.join(EDA_DATA_DIR, "run_*", DERIVED_CONSTANTS_NAME)),
        reverse=True,
    )
    if runs:
        return runs[0]

    # Nothing exists yet. Point at the run this invocation belongs to, or the newest
    # run directory on disk, so the caller's FileNotFoundError names something useful
    # rather than a bare eda-data/ path when a run_*/ dir is sitting right there.
    if rid:
        return os.path.join(EDA_DATA_DIR, rid, DERIVED_CONSTANTS_NAME)
    run_dirs = sorted(glob.glob(os.path.join(EDA_DATA_DIR, "run_*")), reverse=True)
    if run_dirs:
        return os.path.join(run_dirs[0], DERIVED_CONSTANTS_NAME)
    return os.path.join(EDA_DATA_DIR, DERIVED_CONSTANTS_NAME)

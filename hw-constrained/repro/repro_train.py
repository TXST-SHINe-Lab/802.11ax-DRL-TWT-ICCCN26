#!/usr/bin/env python3
"""repro_train.py - Run Ahmed's train_ppo_V1.py / train_lstm_ppo_V1.py unchanged, made safe to
run next to other simulations.

The published model weights are not in the repository (only hyperparams.json, the training log
and normalization_stats.json), so the published policy is reproduced by re-running its training
script with the published hyperparameters on the original 16-STA simulator (reproduce_published.sh
builds that in a separate ns-3 copy).

Two process-level changes, nothing in the learning:

  * FileCommEnv._kill_process ends every episode with `rm -f /dev/shm/My*`, which deletes the
    shared-memory segments of EVERY ns3-ai simulation on the machine, including the bench
    training running in parallel. Here it deletes only its own episode's four segments, and
    also stops ns-3 itself (ns3-ai starts it in its own process group, so killing the runner's
    group misses it; same fix as hw_env.py).
  * FileCommEnv spawns the runner with "python3.11"; reproduce_published.sh puts a python3.11
    on PATH that points to the active venv's interpreter.

Usage (from reproduce_published.sh):
    python repro_train.py mlp  -- --total-timesteps 15000 --seed 1 --reward-preset energy ...
    python repro_train.py lstm -- ...
"""

from __future__ import annotations

import os
import signal
import sys


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("mlp", "lstm") or sys.argv[2] != "--":
        raise SystemExit(__doc__)
    algo, train_args = sys.argv[1], sys.argv[3:]

    # Ahmed's scripts, from the ns-3 copy this file lives in (…/twt/hw-constrained/repro/).
    twt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ppo_dir = os.path.join(twt_dir, "ppo-sb3-scripts")
    # Refuse to train on anything but Ahmed's original simulator. (This file must be run from
    # the ns-3 COPY: run from the main tree it would drive the bench build instead.)
    constants = open(os.path.join(twt_dir, "twt-constants.h")).read()
    if "EE7331-HW-PATCH" in constants or "define ACTIVE_NUM_STA 16" not in constants:
        raise SystemExit(f"{twt_dir} is not the original 16-STA simulator (bench patch present?). "
                         "Run the repro_train.py inside the ns-3 copy.")
    print(f"[repro_train] original 16-STA simulator: {twt_dir}", flush=True)
    sys.path.insert(0, ppo_dir)
    os.chdir(ppo_dir)

    import file_comm_env

    def _kill_process(self):
        """FileCommEnv._kill_process, minus the machine-wide /dev/shm sweep."""
        if self.current_process is not None:
            descendants = []
            try:
                import psutil

                descendants = psutil.Process(self.current_process.pid).children(recursive=True)
            except Exception:  # noqa: BLE001
                pass
            try:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                self.current_process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            for p in descendants:
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
            if descendants:
                try:
                    _, alive = psutil.wait_procs(descendants, timeout=5)
                    for p in alive:
                        p.kill()
                except Exception:  # noqa: BLE001
                    pass
            self.current_process = None
        # The original sweep exists so a SIGKILLed episode cannot block the next reset(); the
        # leftover segments are this env's own, named by the episode seed (pb_twt_wrapper_py).
        seed = getattr(self, "current_seed", None)
        if seed is not None:
            for name in ("MySeg", "MyCpp2PyMsg", "MyPy2CppMsg", "MyLockable"):
                try:
                    os.remove(f"/dev/shm/{name}_{seed}")
                except OSError:
                    pass

    file_comm_env.FileCommEnv._kill_process = _kill_process

    module = "train_ppo_V1" if algo == "mlp" else "train_lstm_ppo_V1"
    trainer = __import__(module)
    sys.argv = [f"{module}.py"] + train_args
    trainer.main()


if __name__ == "__main__":
    main()

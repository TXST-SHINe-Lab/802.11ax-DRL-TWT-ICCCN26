#!/usr/bin/env python3
"""mock_ns3_runner.py - TEST-ONLY stand-in for ns-3, speaking the episode runner's file protocol.

This is NOT a Wi-Fi model and its numbers mean nothing. It exists so that the training,
masking, observation and evaluation code can be exercised end to end on a machine without
ns-3: it writes response.json and consumes action.json exactly like episode_file_runner.py,
and fills the same env-dict keys (realistic and oracle per STA) with a toy queue/energy model
in which longer wake windows cut drops and cost energy, so a policy has something to learn.

Never report results produced with this runner.

Env vars (optional):
  MOCK_NUM_STA      default 5     (4 sensors + 1 stream, like the bench)
  MOCK_EPOCH_S      default 2.0   (simulated seconds per decision epoch)
  MOCK_ACTION_LOG   path; every schedule index received is appended, one per line
  MOCK_FAIL_AT      int; write an error response at this step (failure-path tests)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "exploration-scripts"))
from generate_action_tables import apply_assignment_pattern  # noqa: E402

BI_MS = 102.4
V = 3.0
I_IDLE, I_RX, I_TX, I_SLEEP = 50.0, 66.0, 232.0, 0.12  # mA, same as twt-constants.h
PHY_PKTS_PER_S = 900.0  # Toy channel capacity, packets/s while awake


def load_tables():
    d = os.path.join(_ROOT, "exploration-scripts")
    with open(os.path.join(d, "table_schedule.json")) as f:
        s = json.load(f)
    with open(os.path.join(d, "table_assignment.json")) as f:
        a = json.load(f)
    return s, a


def write_response(comm_dir, payload):
    path = os.path.join(comm_dir, "response.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.rename(tmp, path)


def read_action(comm_dir, timeout):
    path = os.path.join(comm_dir, "action.json")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    a = json.load(f)
                os.remove(path)
                return a
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.005)
    return None


class ToyNetwork:
    def __init__(self, seed, num_sta, epoch_s):
        self.rng = np.random.default_rng(seed)
        self.n = num_sta
        self.epoch_s = epoch_s
        self.t = 0.0
        # 4 sensors + 1 stream, rates re-randomized per episode (packets/s)
        self.rate = np.array([self.rng.uniform(2, 8) for _ in range(num_sta - 1)] + [self.rng.uniform(80, 200)])
        self.pkt_bytes = np.array([200] * (num_sta - 1) + [1400], dtype=float)
        self.q = np.zeros(num_sta)
        self.q_cap = np.array([400] * (num_sta - 1) + [2000], dtype=float)
        z = lambda: np.zeros(num_sta)
        self.cum = {k: z() for k in ("energy", "awake", "sleep", "tx", "txb", "drops", "enq", "air", "fcs", "frag")}
        self.last_rx_us = z()
        self.duty = z()

    def step(self, durations_ms, group_of):
        e = self.epoch_s
        duty = np.clip(durations_ms / BI_MS, 0, 1)
        arrivals = self.rng.poisson(self.rate * e).astype(float)
        self.q += arrivals
        # STAs sharing a group share its airtime
        share = np.array([max(1, int(np.sum(group_of == g))) for g in group_of], dtype=float)
        cap = PHY_PKTS_PER_S * duty * e / share
        served = np.minimum(self.q, cap)
        self.q -= served
        drops = np.maximum(0.0, self.q - self.q_cap)
        self.q -= drops

        tx_s = served / PHY_PKTS_PER_S
        awake_s = duty * e
        sleep_s = e - awake_s
        energy = (I_TX * tx_s + I_IDLE * np.maximum(0, awake_s - tx_s) + I_SLEEP * sleep_s) * V  # mJ

        self.t += e
        self.duty = duty
        c = self.cum
        c["energy"] += energy
        c["awake"] += awake_s * 1000
        c["sleep"] += sleep_s * 1000
        c["tx"] += served
        c["txb"] += served * self.pkt_bytes
        c["drops"] += drops
        c["enq"] += arrivals
        c["air"] += tx_s * 1e6
        c["fcs"] += self.rng.poisson(0.02 * served)
        c["frag"] += self.rng.poisson(0.01 * served)
        self.last_rx_us = np.where(served > 0, self.t * 1e6, self.last_rx_us)

    def env(self):
        c = self.cum
        stas = []
        for i in range(self.n):
            qb = self.q[i] * self.pkt_bytes[i]
            stas.append({
                "realistic": {
                    "sta_id": i,
                    "bsr_queue_ac_be": float(min(254, qb / 256.0)),
                    "airtime_used_us": float(c["air"][i]),
                    "fcs_error_count": float(c["fcs"][i]),
                    "rx_fragment_count": float(c["frag"][i]),
                    "last_rx_timestamp_us": float(self.last_rx_us[i]),
                    "packets_received_at_ap": float(c["tx"][i]),
                    "bytes_received_at_ap": float(c["txb"][i]),
                },
                "oracle": {
                    "total_energy_consumed_mj": float(c["energy"][i]),
                    "awake_time_ms": float(c["awake"][i]),
                    "sleep_time_ms": float(c["sleep"][i]),
                    "duty_cycle": float(self.duty[i]),
                    "packets_transmitted": float(c["tx"][i]),
                    "bytes_transmitted": float(c["txb"][i]),
                    "mpdu_drops_expired": float(c["drops"][i]),
                    "packets_enqueued": float(c["enq"][i]),
                    "queue_size_bytes": float(qb),
                    "queue_size_packets": float(self.q[i]),
                },
            })
        return {"num_sta": self.n, "simulation_time_sec": self.t, "observation_count": 0, "sta_observations": stas}


def durations_for(action, sched, assign, n):
    s = sched["schedules"][action[0]]
    groups = s["groups"]
    mapping = apply_assignment_pattern(assign["assignments"][action[1]], n, len(groups))
    dur = {g["group_id"]: g["wake_duration_ms"] for g in groups}
    group_of = np.zeros(n, dtype=int)
    d = np.zeros(n)
    for sta, g in mapping:
        group_of[sta] = g
        d[sta] = dur.get(g, 0.0)
    return d, group_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--comm-dir", required=True)
    p.add_argument("--action-timeout", type=float, default=600)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    n = int(os.environ.get("MOCK_NUM_STA", 5))
    epoch_s = float(os.environ.get("MOCK_EPOCH_S", 2.0))
    log_path = os.environ.get("MOCK_ACTION_LOG")
    fail_at = int(os.environ.get("MOCK_FAIL_AT", -1))
    sched, assign = load_tables()
    net = ToyNetwork(a.seed, n, epoch_s)

    # Warmup on the action the hardware runner would send: (0,0) clipped -> schedule 2.
    sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
    import hw_constraints as hwc
    warm = hwc.clip_action((0, 0))
    for _ in range(a.warmup_steps):
        net.step(*durations_for(warm, sched, assign, n))
    write_response(a.comm_dir, {"type": "obs", "obs": [], "env": net.env()})

    step = 0
    while True:
        act = read_action(a.comm_dir, a.action_timeout)
        if act is None:
            break
        s_idx, a_idx = int(act["schedule_idx"]), int(act["assignment_idx"])
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"{s_idx}\n")
        step += 1
        if step == fail_at:
            write_response(a.comm_dir, {"type": "error", "error": "mock failure"})
            break
        net.step(*durations_for((s_idx, a_idx), sched, assign, n))
        write_response(a.comm_dir, {"type": "obs", "obs": [], "env": net.env(), "done": False})


if __name__ == "__main__":
    main()

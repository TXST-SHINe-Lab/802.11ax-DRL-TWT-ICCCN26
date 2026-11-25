#!/usr/bin/env python3
"""apply_hw_patch.py - Configure the ICCCN'26 ns-3 scenario to match the HaLow bench.

EE 7331 proposal, Section 2. Run once from anywhere, then rebuild ns-3:

    python3 hw-constrained/ns3-patch/apply_hw_patch.py --stream-rate-kbps <iperf result>
    cd <ns-3.44> && ./ns3 build twt-main-simulation

What it changes (every file is backed up to <file>.orig the first time; --revert restores):

  twt-constants.h
    ACTIVE_NUM_STA          16 -> --num-sta (5)
    TWT_UPDATE_INTERVAL_BI  20 -> round(--epoch-s / 102.4 ms)   (25 s -> 244 BI = 24.99 s)
    DURATION_IN_UPDATE      38 -> --num-epochs + --warmup + 2   (spare cycles; the env truncates)
    + HW_BENCH_MIX, HW_STREAM_RATE_KBPS, HW_RATE_JITTER defines

  twt-simulation-config.cc  InitializeHeterogeneousNetwork()
    With HW_BENCH_MIX defined: STAs 0..N-2 are IoT sensors, STA N-1 is one steady CBR stream
    (the UdpClient "video camera" app, constant packet interval). Rates are re-drawn every
    episode from the ns-3 RNG, so they change with the episode seed:
      sensor rate  = 256 kbps x U[1 - j, 1 + j]
      stream rate  = HW_STREAM_RATE_KBPS x U[1 - j, 1]   (never above the measured cap)
    Positions were already re-randomized per episode by the original code.

  twt-trace-callbacks.cc  ApMonitorSnifferRxTrace()
    Bug fix: strip the 4-byte A-MPDU subframe header before reading the MAC header. Without
    it every HE data frame fails the STA lookup and packets_received_at_ap is always 0.

  exploration-scripts/run-single-spawn.py
    With TWT_HW_LEGAL_ONLY=1 the EDA's random actions skip illegal schedules, so the reward
    constants are dialed under the same action space the agents train in.
  exploration-scripts/5-dial-constants.py   BI_PER_DECISION_STEP follows the new epoch
  exploration-scripts/1-collect-data.sh     STEPS_PER_SPAWN default follows DURATION_IN_UPDATE

The C++ edit is anchored on exact lines of the published code; if an anchor is missing (the
upstream code changed) the script stops without writing anything.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TWT_DIR = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
import hw_constraints as hwc  # noqa: E402

MARK = "EE7331-HW-PATCH"

CPP_ANCHOR = "    sta_app_configs.clear();\n"
CPP_BLOCK = """    sta_app_configs.clear();

#ifdef HW_BENCH_MIX
    // {mark}: HaLow bench mix. STAs 0..N-2 are IoT sensors, STA N-1 is one steady CBR stream.
    // Rates are drawn from the ns-3 RNG, so they change with --randSeed (one per episode).
    {{
        Ptr<UniformRandomVariable> jitter = CreateObject<UniformRandomVariable>();
        uint32_t nSensors = 0;
        for (uint32_t i = 0; i < nStations; i++)
        {{
            StaApplicationConfig config;
            if (i + 1 < nStations)
            {{
                config = IOT_SENSOR_CONFIG;
                double scale = jitter->GetValue(1.0 - HW_RATE_JITTER, 1.0 + HW_RATE_JITTER);
                config.traffic_rate_bps = static_cast<uint32_t>(IOT_SENSOR_CONFIG.traffic_rate_bps * scale);
                config.device_name = "Sensor-" + std::to_string(nSensors++);
            }}
            else
            {{
                config = VIDEO_CAMERA_CONFIG; // UdpClient: constant packet interval, i.e. a steady stream
                double rate_bps = HW_STREAM_RATE_KBPS * 1000.0 * jitter->GetValue(1.0 - HW_RATE_JITTER, 1.0);
                config.traffic_rate_bps = static_cast<uint32_t>(rate_bps);
                config.packet_interval =
                    MicroSeconds(static_cast<uint64_t>(config.packet_size_bytes * 8.0 * 1e6 / rate_bps));
                config.device_name = "Stream-0";
            }}
            config.sta_id = i;
            sta_app_configs.push_back(config);
            std::cout << "  STA " << i << ": " << config.device_name << " ("
                      << GetDeviceClassName(config.device_class) << ", "
                      << config.traffic_rate_bps / 1000.0 << " kbps)" << std::endl;
        }}
        std::cout << "HaLow bench mix: " << nSensors << " sensors + 1 stream" << std::endl;
        return;
    }}
#endif
"""


# Sniffer fix. In ns-3.44, WifiPhy::NotifyMonitorSniffRx hands each MPDU of an A-MPDU (and HE
# sends every data frame as an A-MPDU or S-MPDU) to MonitorSnifferRx as
# WifiPsdu::GetAmpduSubframe(i), which starts with the 4-byte A-MPDU subframe header. The
# published ApMonitorSnifferRxTrace peeks a WifiMacHeader straight from byte 0, reads a wrong
# transmitter address, fails the STA lookup and returns. So no data frame is ever counted and
# packets_received_at_ap / bytes_received_at_ap stay 0. Strip the subframe header first.
TRACE_ANCHOR = "        Ptr<Packet> pktCopy = packet->Copy();\n        pktCopy->PeekHeader(hdr);\n"
TRACE_BLOCK = """        Ptr<Packet> pktCopy = packet->Copy();
        if (aMpdu.type != NORMAL_MPDU)
        {{
            // {mark}: A-MPDU subframes start with a 4-byte subframe header, not the MAC header
            AmpduSubframeHeader subframeHdr;
            pktCopy->RemoveHeader(subframeHdr);
        }}
        pktCopy->PeekHeader(hdr);
"""


def read(path):
    with open(path) as f:
        return f.read()


def backup_and_write(path, text, dry):
    if dry:
        print(f"  (dry run) would write {path}")
        return
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {path}")


def set_define(text, name, value, comment):
    pat = re.compile(rf"^#define {name}\b.*$", re.M)
    if not pat.search(text):
        raise SystemExit(f"twt-constants.h: no '#define {name}' found")
    return pat.sub(f"#define {name} {value} // {MARK}: {comment}", text, count=1)


def patch_constants(text, a):
    bi = int(round(a.epoch_s * 1000.0 / hwc.BEACON_INTERVAL_MS))
    cycles = a.num_epochs + a.warmup + 2
    text = set_define(text, "ACTIVE_NUM_STA", a.num_sta, "bench has 5 STAs")
    text = set_define(text, "TWT_UPDATE_INTERVAL_BI", bi, f"{bi} BI = {bi * hwc.BEACON_INTERVAL_MS / 1000:.2f} s epoch")
    text = set_define(text, "DURATION_IN_UPDATE", cycles, f"{a.warmup} warmup + {a.num_epochs} epochs + 2 spare")
    extra = (
        f"\n// {MARK}: HaLow bench traffic mix (hw-constrained/ns3-patch/apply_hw_patch.py)\n"
        f"#define HW_BENCH_MIX 1\n"
        f"#define HW_STREAM_RATE_KBPS {a.stream_rate_kbps:.1f} // steady stream cap, from iperf on the bench\n"
        f"#define HW_RATE_JITTER {a.rate_jitter} // per-episode rate re-draw, fraction\n"
    )
    if "#define HW_BENCH_MIX" in text:
        text = re.sub(r"\n// " + MARK + r": HaLow bench traffic mix.*?#define HW_RATE_JITTER[^\n]*\n", "\n", text, flags=re.S)
    idx = text.rfind("#endif")
    if idx < 0:
        raise SystemExit("twt-constants.h: no closing #endif for the include guard")
    return text[:idx] + extra + "\n" + text[idx:], bi, cycles


def patch_config_cc(text):
    if MARK in text:
        return text  # Already patched; the block reads its values from the defines
    if text.count(CPP_ANCHOR) != 1:
        raise SystemExit("twt-simulation-config.cc: anchor 'sta_app_configs.clear();' not found exactly once")
    return text.replace(CPP_ANCHOR, CPP_BLOCK.format(mark=MARK), 1)


def patch_trace_cc(text):
    if MARK in text:
        return text
    if text.count(TRACE_ANCHOR) != 1:
        raise SystemExit("twt-trace-callbacks.cc: sniffer anchor (pktCopy->PeekHeader(hdr)) not found exactly once")
    return text.replace(TRACE_ANCHOR, TRACE_BLOCK.format(mark=MARK), 1)


def patch_spawn(text):
    old = "            schedule_idx = random.randint(0, num_schedules - 1)\n"
    if MARK in text:
        return text
    if text.count(old) != 1:
        raise SystemExit("run-single-spawn.py: random schedule line not found exactly once")
    new = (
        f"            # {MARK}: TWT_HW_LEGAL_ONLY=1 samples only schedules the HaLow bench can execute\n"
        "            if os.environ.get(\"TWT_HW_LEGAL_ONLY\") == \"1\":\n"
        f"                schedule_idx = random.choice({list(hwc.legal_schedules())})\n"
        "            else:\n"
        "                schedule_idx = random.randint(0, num_schedules - 1)\n"
    )
    text = text.replace(old, new, 1)
    if not re.search(r"^import os\b", text, re.M):
        text = "import os\n" + text if not text.startswith("#!") else text.replace("\n", "\nimport os\n", 1)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--twt-dir", default=DEFAULT_TWT_DIR, help="The twt example directory (this repo)")
    p.add_argument("--num-sta", type=int, default=5)
    p.add_argument("--epoch-s", type=float, default=25.0)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--stream-rate-kbps", type=float, default=None,
                   help="Steady-stream cap measured with iperf on the bench (required)")
    p.add_argument("--rate-jitter", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--revert", action="store_true", help="Restore every .orig backup")
    a = p.parse_args(argv)

    files = {
        "const": os.path.join(a.twt_dir, "twt-constants.h"),
        "cc": os.path.join(a.twt_dir, "twt-simulation-config.cc"),
        "trace": os.path.join(a.twt_dir, "twt-trace-callbacks.cc"),
        "spawn": os.path.join(a.twt_dir, "exploration-scripts", "run-single-spawn.py"),
        "dial": os.path.join(a.twt_dir, "exploration-scripts", "5-dial-constants.py"),
        "collect": os.path.join(a.twt_dir, "exploration-scripts", "1-collect-data.sh"),
    }

    if a.revert:
        for path in files.values():
            if os.path.exists(path + ".orig"):
                shutil.move(path + ".orig", path)
                print(f"  restored {path}")
        return

    if a.stream_rate_kbps is None:
        p.error("--stream-rate-kbps is required: measure what the bench carries with iperf first "
                "(proposal, Section 2). Use --dry-run to preview with a placeholder.")
    if not 0 <= a.rate_jitter < 1:
        p.error("--rate-jitter must be in [0, 1)")

    # Compute every edit first; write only if all anchors were found.
    const_txt, bi, cycles = patch_constants(read(files["const"]), a)
    cc_txt = patch_config_cc(read(files["cc"]))
    trace_txt = patch_trace_cc(read(files["trace"]))
    spawn_txt = patch_spawn(read(files["spawn"]))
    dial_txt, n = re.subn(r"^BI_PER_DECISION_STEP = \d+.*$",
                          f"BI_PER_DECISION_STEP = {bi}  # {MARK}: follows TWT_UPDATE_INTERVAL_BI",
                          read(files["dial"]), count=1, flags=re.M)
    if n != 1:
        raise SystemExit("5-dial-constants.py: BI_PER_DECISION_STEP not found")
    dial_txt = re.sub(r"^DEFAULT_STEPS_PER_SPAWN = \d+.*$", f"DEFAULT_STEPS_PER_SPAWN = {cycles}  # {MARK}",
                      dial_txt, count=1, flags=re.M)
    collect_txt, n = re.subn(r"STEPS_PER_SPAWN=\$\{STEPS_PER_SPAWN:-\d+\}",
                             f"STEPS_PER_SPAWN=${{STEPS_PER_SPAWN:-{cycles}}}", read(files["collect"]), count=1)
    if n != 1:
        raise SystemExit("1-collect-data.sh: STEPS_PER_SPAWN default not found")

    print(f"Bench config: {a.num_sta} STAs, epoch {bi} BI ({bi * hwc.BEACON_INTERVAL_MS / 1000:.2f} s), "
          f"{cycles} update cycles, stream cap {a.stream_rate_kbps} kbps, jitter {a.rate_jitter}")
    backup_and_write(files["const"], const_txt, a.dry_run)
    backup_and_write(files["cc"], cc_txt, a.dry_run)
    backup_and_write(files["trace"], trace_txt, a.dry_run)
    backup_and_write(files["spawn"], spawn_txt, a.dry_run)
    backup_and_write(files["dial"], dial_txt, a.dry_run)
    backup_and_write(files["collect"], collect_txt, a.dry_run)
    if not a.dry_run:
        print("\nNext: rebuild ns-3 (./ns3 build twt-main-simulation) before collecting stats or training.")


if __name__ == "__main__":
    main()

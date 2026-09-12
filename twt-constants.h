// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-constants.h
 * @brief TWT scheduler constants and simulation configuration parameters
 */

#ifndef TWT_CONSTANTS_H
#define TWT_CONSTANTS_H

// --- Core configuration constants ---
// ACTIVE_NUM_STA must not exceed MAX_NUM_STA; the observation is sized from MAX_NUM_STA and zero-padded.
#define MAX_NUM_STA 16       // Maximum number of STAs
#define ACTIVE_NUM_STA 16    // Default number of active STAs in simulation
#define MAX_NUM_TWT_GROUPS 8 // Maximum number of TWT groups
#define MAC_ADDR_LEN 6       // Bytes

// --- Network topology constants ---
#define DEFAULT_ROOM_LENGTH 30.0 // m, side of the square room the mobility model places STAs in

// --- Beacon interval ---
#define BEACON_INTERVAL_MS 102.4 // ms, the standard beacon interval

// --- Energy model, mA at BATTERY_VOLTAGE_V ---
// PHY state draw for a typical 802.11ax device; energy is current times voltage times time in state.
#define PHY_STATE_IDLE_MA 50.0     // mA
#define PHY_STATE_CCA_BUSY_MA 50.0 // mA, same draw as idle since the radio is still listening
#define PHY_STATE_RX_MA 66.0       // mA
#define PHY_STATE_TX_MA 232.0      // mA
#define PHY_STATE_SLEEP_MA 0.12    // mA
#define BATTERY_VOLTAGE_V 3.0      // V, a standard Li-ion cell

// --- Traffic & queue constants ---
#define MAX_QUEUE_SIZE_BYTES 65536 // Maximum BSR reportable, matches 802.11ax BSR encoding
#define PAYLOAD_SIZE_BYTES 1400    // Bytes, the generated UDP payload

// --- TWT update timing constants (in beacon intervals) ---
// One agent step spans TWT_UPDATE_INTERVAL_BI beacons, so DURATION_IN_UPDATE is the episode length in steps.
#define TWT_UPDATE_INTERVAL_BI 20 // BI between successive TWT updates
#define DURATION_IN_UPDATE 38     // Number of update cycles, so 38 steps per episode
#define TWT_UPDATE_START_BI 90    // BI at which TWT updates begin
#define TWT_SETUP_TIME_BI 75      // BI at which the initial TWT agreement is set up

// --- Simulation timing constants (in beacon intervals) ---
// Long enough for every update cycle plus 5 BI of tail, so the last step is fully measured.
#define SIMULATION_DURATION_BI \
    (TWT_UPDATE_START_BI + (DURATION_IN_UPDATE * TWT_UPDATE_INTERVAL_BI) + 5)
// Traffic starts at a uniform random BI in [25, 45], staggering the STAs.
#define APP_START_TIME_MIN_BI 25 // Lower bound of that window
#define APP_START_TIME_MAX_BI 45 // Upper bound of that window
#define METRICS_START_TIME_BI 75 // Metrics start here, after the apps have settled

// --- WiFi PHY configuration constants ---
// HeMcs4 at 20 MHz, 1SS, 800 ns GI gives roughly 51.62 Mbps; analytical_policies.py hardcodes that rate.
#define DEFAULT_MCS 4                 // HE MCS index
#define DEFAULT_GUARD_INTERVAL_NS 800 // ns, the standard 0.8 us guard interval
#define DEFAULT_CHANNEL_WIDTH_MHZ 20  // MHz
#define RTS_CTS_THRESHOLD 2347        // Bytes, the standard WiFi RTS/CTS threshold
// Frames larger than this use an RTS/CTS handshake for collision avoidance.
// Payloads here top out near 1500 bytes, so the handshake is rarely triggered.

// --- WiFi MAC configuration constants ---
#define MAX_MISSED_BEACONS 0xFFFFFFFF // Effectively infinite, so a sleeping STA never deassociates
#define BLOCK_ACK_THRESHOLD 1         // Block Ack from the first packet, a WiFi 6 feature
#define WIFI_STARTUP_WINDOW_BI 20     // BI, the PHY startup window

// --- Random number generation constants ---
// RNG stream management for reproducible multi-run studies.
//
// NS-3 uses independent RNG streams to give three properties:
//   1. Deterministic behaviour: the same seed yields identical random sequences
//   2. Independence: each device draws from its own stream, so devices stay uncorrelated
//   3. Parameter sweeps: different seeds across runs support statistical analysis
//
// Usage for multi-run studies:
//   - Set RNG_INITIAL_SEED to a base seed, for example 42
//   - Loop: for(int run=1; run<=NUM_RUNS; run++) { seed = RNG_INITIAL_SEED + run; }
//   - Each run gets a different topology and traffic pattern while the streams stay independent
//
// Example multi-run execution:
//   ./ns3 run "twt-main-simulation --randSeed=42 --simulationTime=60"    // Run 1
//   ./ns3 run "twt-main-simulation --randSeed=43 --simulationTime=60"    // Run 2
//   ./ns3 run "twt-main-simulation --randSeed=44 --simulationTime=60"    // Run 3
// Analysis across those runs then gives confidence intervals.
//
#define RNG_INITIAL_STREAM_ID 100 // First stream ID assigned to WiFi devices
#define RNG_DEFAULT_RUN_NUMBER 1  // Default run number in a parameter study
#define RNG_INITIAL_SEED 10       // Base seed, overridden by --randSeed on the command line

#endif // TWT_CONSTANTS_H

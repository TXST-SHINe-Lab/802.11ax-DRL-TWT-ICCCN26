// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file pb-twt-core.h
 * @brief Shared data structures for C++/Python TWT communication via NS3-AI
 */

#ifndef PB_TWT_CORE_H
#define PB_TWT_CORE_H

#include "twt-constants.h"

#include <cstdint>

// Forward declaration for ns3-ai message interface
namespace ns3
{
template <typename EnvType, typename ActType>
class Ns3AiMsgInterfaceImpl;
}

// --- Per-STA protocol-compliant observations (802.11ax/k) ---
/**
 * @brief Per-STA metrics observable via the 802.11ax and 802.11k protocols.
 *
 * A real access point can obtain all of these through standard protocol mechanisms.
 * Train on this struct alone for an agent intended to deploy on real hardware.
 *
 * Data sources:
 * - 802.11ax: BSR (Buffer Status Report) per Access Category
 * - 802.11k: STA Statistics Report (Section 7.3.2.22)
 * - 802.11k: Link Measurement Report
 * - MAC headers: MCS, NSS observed from received frames
 * - 802.11e TSPEC: Device characteristics declared during association
 */
struct StaRealisticMetrics
{
    // --- Identification ---
    uint32_t sta_id;               // STA identifier (0 to NUM_STA-1)
    uint8_t sta_mac[MAC_ADDR_LEN]; // MAC address (6 bytes)
    uint8_t is_active;             // Is this STA currently active (0/1)

    // --- 802.11ax BSR (buffer status report) - per access category ---
    // IEEE 802.11ax-2021 Section 9.2.4.6.6
    // AP receives BSR in Trigger Frame responses or QoS Data frames
    uint8_t bsr_queue_ac_be;     // Best Effort queue (AC_BE, TID 0,3) - quantized 0-255
    uint8_t bsr_queue_ac_bk;     // Background queue (AC_BK, TID 1,2) - quantized 0-255
    uint8_t bsr_queue_ac_vi;     // Video queue (AC_VI, TID 4,5) - quantized 0-255
    uint8_t bsr_queue_ac_vo;     // Voice queue (AC_VO, TID 6,7) - quantized 0-255
    uint16_t bsr_scaling_factor; // Bytes per BSR unit (typically 256)

    // --- AP-observable RX counters (truly realistic - AP directly observes) ---
    uint64_t rx_fragment_count; // AP counts frames it receives from STA
    uint64_t fcs_error_count;   // AP counts RX decode failures (FCS errors)

    // --- 802.11k link measurement - raw + converted values ---
    // IEEE 802.11k-2008 Section 7.3.2.18
    // AP requests via Link Measurement Request frame
    // INT8_MIN (-128) stands in for no-measurement because int8_t cannot hold NaN.
    // Check for it before using any value in this block.
    int8_t rcpi;           // Raw RCPI (0.5 dBm units, range 0-220), INT8_MIN = no data
    int8_t rsni;           // Raw RSNI (0.5 dB units, range 0-255), INT8_MIN = no data
    int8_t rssi_dbm;       // Converted from RCPI: (rcpi/2) - 110, INT8_MIN = no data
    int8_t snr_db;         // Converted from RSNI: rsni/2, INT8_MIN = no data
    int8_t link_margin_db; // dB above RX sensitivity threshold, INT8_MIN = no data
    uint8_t tx_power_dbm;  // STA's current TX power (0 = no observation)

    // --- MAC layer observations - AP observes from received frame headers ---
    // For these uint8_t fields, 0 means no observation yet rather than a measured zero.
    uint8_t last_rx_frame_type;    // Frame type: 0=mgmt, 1=ctrl, 2=data
    uint8_t last_rx_frame_subtype; // Frame subtype (0-15)
    uint8_t last_rx_mcs;           // MCS index from last received frame (0-11 HE)
    uint8_t last_rx_nss;           // NSS from last received frame
    uint8_t channel_width_mhz;     // Channel width (20/40/80/160)
    uint16_t guard_interval_ns;    // Guard interval: 800, 1600, or 3200 ns
    uint8_t power_mgmt_bit;        // Power Management bit from Frame Control
    uint64_t last_rx_timestamp_us; // TSF timestamp of last received frame

    // --- AP-derived metrics - computed from received frames at AP ---
    uint64_t bytes_received_at_ap;   // Total bytes AP received from this STA
    uint64_t packets_received_at_ap; // Total packets AP received from this STA

    // Per-AC transmission tracking (from QoS Control field TID)
    uint64_t bytes_received_ac_vo;   // Voice (TID 6,7)
    uint64_t bytes_received_ac_vi;   // Video (TID 4,5)
    uint64_t bytes_received_ac_be;   // Best Effort (TID 0,3)
    uint64_t bytes_received_ac_bk;   // Background (TID 1,2)
    uint64_t packets_received_ac_vo; // Voice packet count
    uint64_t packets_received_ac_vi; // Video packet count
    uint64_t packets_received_ac_be; // Best Effort packet count
    uint64_t packets_received_ac_bk; // Background packet count

    // Airtime tracking (from frame duration field)
    uint64_t airtime_used_us; // Total airtime used by this STA (microseconds)

    // --- Device characteristics - from 802.11e TSPEC during association ---
    // STA declares these; AP cannot independently verify
    uint8_t device_class;       // 0=IoT, 1=Camera, 2=Voice, 3=Video
    double nominal_msdu_size;   // Expected MSDU size (from TSPEC)
    double mean_data_rate_kbps; // Mean data rate (from TSPEC)
    double delay_bound_ms;      // Maximum service interval (from TSPEC)
    uint8_t user_priority;      // QoS UP (0-7, from TSPEC)

    // --- Observation metadata ---
    double observation_time_ms;        // Simulation time of this observation
    uint32_t observation_sequence_num; // Monotonic sequence number for ordering
};

// --- Per-STA oracle/simulation metrics (not protocol compliant) ---
/**
 * @brief Per-STA simulation-only metrics, for validation rather than deployment.
 *
 * Available only in simulation: a real AP cannot observe these without STA cooperation
 * or firmware modifications. Use them for:
 * - Upper bound comparison, oracle agent against realistic agent
 * - Debugging and validation
 * - Ground truth for reward computation, during training only
 *
 * Never use these as RL state in a deployment-ready agent.
 * The reward may read them, since the reward is computed offline; the observation may not.
 */
struct StaOracleMetrics
{
    uint32_t sta_id; // STA identifier (must match realistic)

    // --- STA TX counters (would need 802.11k STA statistics request in real deployment) ---
    // These are STA-side local counters, NOT directly observable by AP
    uint64_t tx_fragment_count; // dot11TransmittedFragmentCount (STA-local)
    uint64_t tx_failed_count;   // dot11FailedCount (STA-local)
    uint64_t tx_retry_count;    // dot11RetryCount (STA-local)
    uint64_t ack_failure_count; // dot11ACKFailureCount (STA-local)

    // --- Energy (STA-private, not reportable via 802.11) ---
    double total_energy_consumed_mj; // Cumulative energy consumption
    double current_times_time_ma_ms; // Integral of current*time (for avg power)
    double awake_time_ms;            // Time in active states (TX/RX/IDLE/CCA)
    double sleep_time_ms;            // Time in SLEEP state
    double duty_cycle;               // awake_time / total_time

    // --- Application layer (above MAC, not visible to AP) ---
    uint64_t packets_generated; // Packets created by application
    uint64_t packets_enqueued;  // Packets successfully enqueued at MAC
    uint64_t bytes_generated;   // Bytes created by application

    // --- STA queue drops (internal to STA, not reportable) ---
    uint64_t mpdu_drops_expired;     // Drops due to lifetime expiry
    uint64_t mpdu_drops_queue_full;  // Drops due to queue overflow
    uint64_t psdu_response_timeouts; // PSDU timeouts (no ACK received)

    // --- TX side metrics (STA-local, not reported to AP) ---
    uint64_t packets_transmitted; // Packets sent by PHY
    uint64_t bytes_transmitted;   // Bytes sent by PHY
    uint64_t ampdu_count;         // Number of A-MPDU transmissions
    uint64_t ampdu_mpdus_total;   // Total MPDUs in A-MPDUs
    uint64_t ampdu_bytes_total;   // Total bytes in A-MPDUs

    // --- Queue state (STA-internal snapshot) ---
    uint32_t queue_size_packets; // Current queue size (packets)
    uint32_t queue_size_bytes;   // Current queue size (bytes)
    uint32_t queue_max_size;     // Maximum queue capacity

    // --- Latency (requires app-layer timestamp, not in 802.11) ---
    double avg_latency_ms;      // Average packet latency
    double queue_delay_sum_ms;  // Sum of queue delays
    uint64_t queue_delay_count; // Number of delay samples

    // --- STA position (simulation-only, not reportable via 802.11) ---
    double position_x_m; // STA x-coordinate in meters
    double position_y_m; // STA y-coordinate in meters
    double position_z_m; // STA z-coordinate in meters (height/altitude)
};

// --- Combined per-STA structure ---
/**
 * @brief Complete per-STA observations, realistic and oracle metrics side by side.
 *
 * Keeps the protocol-compliant and simulation-only metrics in separate sub-structs.
 * Python should read only `realistic` when training a deployable agent.
 */
struct StaEnvStruct
{
    StaRealisticMetrics realistic; // Protocol-compliant (802.11ax/k)
    StaOracleMetrics oracle;       // Simulation-only (for validation)
};

// --- Environment structure ---
/**
 * @brief Complete system observations, sent from NS-3 C++ to the Python controller.
 *
 * Carries metadata plus one entry per STA, and deliberately carries no aggregates:
 * - TWT group info: the controller already knows it from the action it just sent
 * - Channel and collision stats: derivable from the per-STA metrics
 * Every aggregation is computed in Python from the per-STA data.
 */
struct EnvStruct
{
    // --- Metadata ---
    uint32_t num_sta;                  // Number of active STAs
    double simulation_time_sec;        // Current simulation time
    uint64_t observation_timestamp_ms; // When observation was captured
    uint32_t observation_count;        // Sequential observation counter
    double beacon_interval_ms;         // Beacon interval (TWT timing reference)

    // --- Per-STA observations ---
    StaEnvStruct sta_observations[MAX_NUM_STA];
};

// --- TWT group configuration structure ---
/**
 * @brief TWT timing parameters for one group.
 *
 * Every STA assigned to the group inherits these parameters.
 */
struct TwtGroupConfig
{
    uint8_t group_id; // TWT group identifier (0 to MAX_NUM_TWT_GROUPS-1)

    // --- TWT timing parameters ---
    double twt_wake_interval_ms; // Wake interval (how often STAs wake up)
    double twt_wake_duration_ms; // Wake duration (how long STAs stay awake)
    double twt_sp_offset_ms;     // Service Period offset from beacon

    // --- Group metadata ---
    uint8_t num_stas_assigned; // Number of STAs currently assigned to this group
};

// --- STA group assignment structure ---
/**
 * @brief Assigns one STA to a TWT group.
 *
 * The STA inherits every TWT parameter from the group named here.
 */
struct StaGroupAssignment
{
    uint32_t sta_id;            // STA identifier (must match observation)
    uint8_t assigned_twt_group; // TWT group assignment (0 to MAX_NUM_TWT_GROUPS-1)
    uint8_t enable_twt;         // Enable TWT for this STA (0=disable, 1=enable)
};

// --- Aggregate action structure ---
/**
 * @brief Complete TWT scheduling decision, sent from the Python controller to NS-3 C++.
 *
 * Carries the TWT group configurations and the STA-to-group assignments.
 *
 * The design is group-centric:
 * 1. TWT groups define the timing parameters: wake interval, duration and offset
 * 2. STAs are assigned to groups and inherit those parameters
 * 3. Several STAs may share one group, contending for the medium within its service period
 */
struct ActionStruct
{
    // --- Metadata ---
    uint32_t num_sta;               // Number of STAs being controlled
    uint32_t num_active_twt_groups; // Number of active TWT groups (0 to MAX_NUM_TWT_GROUPS)
    uint64_t action_timestamp_ms;   // When action was generated

    // --- TWT group configurations ---
    TwtGroupConfig twt_group_configs[MAX_NUM_TWT_GROUPS]; // Array of TWT group parameters

    // --- STA-to-group assignments ---
    StaGroupAssignment sta_group_assignments[MAX_NUM_STA]; // Array of STA group assignments
};

// --- C++ interface functions ---

/**
 * GetNs3AiInterface: Initialize NS3-AI message interface for TWT controller
 * Creates and configures the message interface for EnvStruct/ActionStruct communication
 *
 * Returns:
 *   msgInterface: Configured message interface
 */
ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>* GetNs3AiInterface();

/**
 * Send2Python: Send environment observations to Python and receive TWT schedule
 *
 * @param msgInterface: NS-3 AI message interface
 * @param env_struct: Environment observations (all per-STA metrics)
 * @return action_struct: TWT scheduling decisions from Python controller
 */
ActionStruct Send2Python(ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>* msgInterface,
                         const EnvStruct& env_struct);

#endif // PB_TWT_CORE_H

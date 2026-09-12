// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file pb-twt-interface.cc
 * @brief pybind11 Python bindings for TWT simulation data structures
 */

#include "pb-twt-core.h"
#include "twt-simulation-config.h"

#include <ns3/ai-module.h>

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pb_twt_interface_py, m)
{
    m.doc() = "TWT Python Binding Interface for NS3-AI Communication - Author: Ahmed Maksud (SHINE "
              "Lab, Texas State University)";

    // --- StaRealisticMetrics: protocol-compliant per-STA observations (802.11ax/k) ---
    py::class_<StaRealisticMetrics>(m, "StaRealisticMetrics")
        .def(py::init<>(), "802.11ax/k protocol-compliant per-STA metrics")

        // --- Identification ---
        .def_readwrite("sta_id", &StaRealisticMetrics::sta_id, "STA identifier")
        .def_property(
            "sta_mac",
            [](const StaRealisticMetrics& s) -> py::bytes
            { return py::bytes(reinterpret_cast<const char*>(s.sta_mac), MAC_ADDR_LEN); },
            [](StaRealisticMetrics& s, py::bytes b) -> void
            {
                std::string str = b;
                if (str.size() == MAC_ADDR_LEN)
                {
                    std::memcpy(s.sta_mac, str.data(), MAC_ADDR_LEN);
                }
            },
            "MAC address (6 bytes)")
        .def_readwrite("is_active", &StaRealisticMetrics::is_active, "Is STA active (0/1)")

        // --- 802.11ax BSR (buffer status report) per access category - quantized 0-255 ---
        .def_readwrite("bsr_queue_ac_be",
                       &StaRealisticMetrics::bsr_queue_ac_be,
                       "Best Effort queue (quantized 0-255)")
        .def_readwrite("bsr_queue_ac_bk",
                       &StaRealisticMetrics::bsr_queue_ac_bk,
                       "Background queue (quantized 0-255)")
        .def_readwrite("bsr_queue_ac_vi",
                       &StaRealisticMetrics::bsr_queue_ac_vi,
                       "Video queue (quantized 0-255)")
        .def_readwrite("bsr_queue_ac_vo",
                       &StaRealisticMetrics::bsr_queue_ac_vo,
                       "Voice queue (quantized 0-255)")
        .def_readwrite("bsr_scaling_factor",
                       &StaRealisticMetrics::bsr_scaling_factor,
                       "Bytes per BSR unit (typically 256)")

        // --- AP-Observable RX Counters (Truly realistic) ---
        // Note: TX counters moved to StaOracleMetrics (would need 802.11k request)
        .def_readwrite("rx_fragment_count",
                       &StaRealisticMetrics::rx_fragment_count,
                       "Frames received by AP from this STA")
        .def_readwrite("fcs_error_count",
                       &StaRealisticMetrics::fcs_error_count,
                       "RX decode failures (FCS errors)")

        // --- 802.11k Link Measurement - Raw + converted ---
        .def_readwrite("rcpi", &StaRealisticMetrics::rcpi, "Raw RCPI (0.5 dBm units)")
        .def_readwrite("rsni", &StaRealisticMetrics::rsni, "Raw RSNI (0.5 dB units)")
        .def_readwrite("rssi_dbm", &StaRealisticMetrics::rssi_dbm, "Converted RCPI in dBm")
        .def_readwrite("snr_db", &StaRealisticMetrics::snr_db, "Converted RSNI in dB")
        .def_readwrite("link_margin_db", &StaRealisticMetrics::link_margin_db, "Link margin in dB")
        .def_readwrite("tx_power_dbm", &StaRealisticMetrics::tx_power_dbm, "TX power in dBm")

        // --- MAC layer observations (from frame headers) ---
        .def_readwrite("last_rx_frame_type",
                       &StaRealisticMetrics::last_rx_frame_type,
                       "Frame type: 0=mgmt, 1=ctrl, 2=data")
        .def_readwrite("last_rx_frame_subtype",
                       &StaRealisticMetrics::last_rx_frame_subtype,
                       "Frame subtype (0-15)")
        .def_readwrite("last_rx_mcs", &StaRealisticMetrics::last_rx_mcs, "Last RX MCS index")
        .def_readwrite("last_rx_nss", &StaRealisticMetrics::last_rx_nss, "Last RX NSS")
        .def_readwrite(
            "channel_width_mhz", &StaRealisticMetrics::channel_width_mhz, "Channel width in MHz")
        .def_readwrite(
            "guard_interval_ns", &StaRealisticMetrics::guard_interval_ns, "Guard interval in ns")
        .def_readwrite("power_mgmt_bit",
                       &StaRealisticMetrics::power_mgmt_bit,
                       "Power Management bit from Frame Control")
        .def_readwrite("last_rx_timestamp_us",
                       &StaRealisticMetrics::last_rx_timestamp_us,
                       "TSF timestamp of last received frame")

        // --- AP-Derived Metrics (from received frames) ---
        .def_readwrite("bytes_received_at_ap",
                       &StaRealisticMetrics::bytes_received_at_ap,
                       "Bytes received at AP from this STA")
        .def_readwrite("packets_received_at_ap",
                       &StaRealisticMetrics::packets_received_at_ap,
                       "Packets received at AP from this STA")

        // --- Per-AC TX tracking (from QoS control field TID) ---
        .def_readwrite("bytes_received_ac_vo",
                       &StaRealisticMetrics::bytes_received_ac_vo,
                       "Voice bytes (TID 6,7)")
        .def_readwrite("bytes_received_ac_vi",
                       &StaRealisticMetrics::bytes_received_ac_vi,
                       "Video bytes (TID 4,5)")
        .def_readwrite("bytes_received_ac_be",
                       &StaRealisticMetrics::bytes_received_ac_be,
                       "Best Effort bytes (TID 0,3)")
        .def_readwrite("bytes_received_ac_bk",
                       &StaRealisticMetrics::bytes_received_ac_bk,
                       "Background bytes (TID 1,2)")
        .def_readwrite("packets_received_ac_vo",
                       &StaRealisticMetrics::packets_received_ac_vo,
                       "Voice packet count")
        .def_readwrite("packets_received_ac_vi",
                       &StaRealisticMetrics::packets_received_ac_vi,
                       "Video packet count")
        .def_readwrite("packets_received_ac_be",
                       &StaRealisticMetrics::packets_received_ac_be,
                       "Best Effort packet count")
        .def_readwrite("packets_received_ac_bk",
                       &StaRealisticMetrics::packets_received_ac_bk,
                       "Background packet count")

        // --- Airtime tracking ---
        .def_readwrite("airtime_used_us",
                       &StaRealisticMetrics::airtime_used_us,
                       "Total airtime used (microseconds)")

        // --- Device characteristics (from 802.11e TSPEC) ---
        .def_readwrite("device_class",
                       &StaRealisticMetrics::device_class,
                       "Device type: 0=IoT, 1=Camera, 2=Voice, 3=Video")
        .def_readwrite(
            "nominal_msdu_size", &StaRealisticMetrics::nominal_msdu_size, "MSDU size from TSPEC")
        .def_readwrite("mean_data_rate_kbps",
                       &StaRealisticMetrics::mean_data_rate_kbps,
                       "Mean data rate from TSPEC")
        .def_readwrite(
            "delay_bound_ms", &StaRealisticMetrics::delay_bound_ms, "Delay bound from TSPEC")
        .def_readwrite(
            "user_priority", &StaRealisticMetrics::user_priority, "QoS User Priority (0-7)")

        // --- Observation Metadata ---
        .def_readwrite("observation_time_ms",
                       &StaRealisticMetrics::observation_time_ms,
                       "Observation time in ms")
        .def_readwrite("observation_sequence_num",
                       &StaRealisticMetrics::observation_sequence_num,
                       "Monotonic sequence number for ordering")

        .def("__repr__",
             [](const StaRealisticMetrics& s)
             {
                 return "<StaRealisticMetrics: STA " + std::to_string(s.sta_id) +
                        ", RSSI=" + std::to_string(s.rssi_dbm) + " dBm, " +
                        "MCS=" + std::to_string(s.last_rx_mcs) + ">";
             });

    // --- StaOracleMetrics: simulation-only per-STA metrics (for validation) ---
    py::class_<StaOracleMetrics>(m, "StaOracleMetrics")
        .def(py::init<>(), "Simulation-only per-STA metrics for validation")

        .def_readwrite("sta_id", &StaOracleMetrics::sta_id, "STA identifier")

        // --- STA TX counters (would need 802.11k request in real deployment) ---
        .def_readwrite("tx_fragment_count",
                       &StaOracleMetrics::tx_fragment_count,
                       "dot11TransmittedFragmentCount (STA-local)")
        .def_readwrite(
            "tx_failed_count", &StaOracleMetrics::tx_failed_count, "dot11FailedCount (STA-local)")
        .def_readwrite(
            "tx_retry_count", &StaOracleMetrics::tx_retry_count, "dot11RetryCount (STA-local)")
        .def_readwrite("ack_failure_count",
                       &StaOracleMetrics::ack_failure_count,
                       "dot11ACKFailureCount (STA-local)")

        // --- Energy (STA-private) ---
        .def_readwrite("total_energy_consumed_mj",
                       &StaOracleMetrics::total_energy_consumed_mj,
                       "Cumulative energy consumption")
        .def_readwrite("current_times_time_ma_ms",
                       &StaOracleMetrics::current_times_time_ma_ms,
                       "Integral of current*time")
        .def_readwrite("awake_time_ms", &StaOracleMetrics::awake_time_ms, "Time in active states")
        .def_readwrite("sleep_time_ms", &StaOracleMetrics::sleep_time_ms, "Time in SLEEP state")
        .def_readwrite("duty_cycle", &StaOracleMetrics::duty_cycle, "awake/(awake+sleep)")

        // --- Application layer (above MAC) ---
        .def_readwrite(
            "packets_generated", &StaOracleMetrics::packets_generated, "Packets created by app")
        .def_readwrite(
            "packets_enqueued", &StaOracleMetrics::packets_enqueued, "Packets enqueued at MAC")
        .def_readwrite(
            "bytes_generated", &StaOracleMetrics::bytes_generated, "Bytes created by app")

        // --- STA queue drops (internal) ---
        .def_readwrite("mpdu_drops_expired",
                       &StaOracleMetrics::mpdu_drops_expired,
                       "Drops due to lifetime expiry")
        .def_readwrite("mpdu_drops_queue_full",
                       &StaOracleMetrics::mpdu_drops_queue_full,
                       "Drops due to queue overflow")
        .def_readwrite("psdu_response_timeouts",
                       &StaOracleMetrics::psdu_response_timeouts,
                       "PSDU response timeouts")

        // --- TX side metrics (STA-local) ---
        .def_readwrite(
            "packets_transmitted", &StaOracleMetrics::packets_transmitted, "Packets sent by PHY")
        .def_readwrite(
            "bytes_transmitted", &StaOracleMetrics::bytes_transmitted, "Bytes sent by PHY")
        .def_readwrite("ampdu_count", &StaOracleMetrics::ampdu_count, "A-MPDU transmissions")
        .def_readwrite(
            "ampdu_mpdus_total", &StaOracleMetrics::ampdu_mpdus_total, "Total MPDUs in A-MPDUs")
        .def_readwrite(
            "ampdu_bytes_total", &StaOracleMetrics::ampdu_bytes_total, "Total bytes in A-MPDUs")

        // --- Queue State ---
        .def_readwrite("queue_size_packets",
                       &StaOracleMetrics::queue_size_packets,
                       "Current queue size (packets)")
        .def_readwrite(
            "queue_size_bytes", &StaOracleMetrics::queue_size_bytes, "Current queue size (bytes)")
        .def_readwrite(
            "queue_max_size", &StaOracleMetrics::queue_max_size, "Maximum queue capacity")

        // --- Latency (app-layer) ---
        .def_readwrite("avg_latency_ms", &StaOracleMetrics::avg_latency_ms, "Average latency")
        .def_readwrite(
            "queue_delay_sum_ms", &StaOracleMetrics::queue_delay_sum_ms, "Sum of queue delays")
        .def_readwrite(
            "queue_delay_count", &StaOracleMetrics::queue_delay_count, "Number of delay samples")

        // --- STA position (simulation-only) ---
        .def_readwrite(
            "position_x_m", &StaOracleMetrics::position_x_m, "STA x-coordinate in meters")
        .def_readwrite(
            "position_y_m", &StaOracleMetrics::position_y_m, "STA y-coordinate in meters")
        .def_readwrite("position_z_m",
                       &StaOracleMetrics::position_z_m,
                       "STA z-coordinate in meters (altitude)")

        .def("__repr__",
             [](const StaOracleMetrics& s)
             {
                 return "<StaOracleMetrics: STA " + std::to_string(s.sta_id) +
                        ", energy=" + std::to_string(s.total_energy_consumed_mj) + " mJ, " +
                        "duty=" + std::to_string(s.duty_cycle) + ">";
             });

    // --- StaEnvStruct: combined per-STA observations (realistic + oracle) ---
    py::class_<StaEnvStruct>(m, "StaEnvStruct")
        .def(py::init<>(), "Combined per-STA observations")
        .def_readwrite(
            "realistic", &StaEnvStruct::realistic, "Protocol-compliant metrics (802.11ax/k)")
        .def_readwrite("oracle", &StaEnvStruct::oracle, "Simulation-only metrics (for validation)")

        .def("__repr__",
             [](const StaEnvStruct& s)
             {
                 return "<StaEnvStruct: STA " + std::to_string(s.realistic.sta_id) +
                        ", realistic + oracle metrics>";
             });

    // --- Note: AggregateRealisticMetrics and AggregateOracleMetrics have been removed ---
    //
    // The Python controller already knows the TWT group assignments and configurations from the ActionStruct it sends, so echoing them back is redundant.
    // Every aggregate statistic, channel utilisation, collision counts, total throughput and fairness indices among them, is computed in Python from sta_observations[].

    // --- EnvStruct: system observations (per-STA only, no aggregates) ---
    py::class_<EnvStruct>(m, "EnvStruct")
        .def(py::init<>(), "Environment structure with per-STA observations")

        // --- Metadata ---
        .def_readwrite("num_sta", &EnvStruct::num_sta, "Number of active STAs")
        .def_readwrite("simulation_time_sec",
                       &EnvStruct::simulation_time_sec,
                       "Current simulation time in seconds")
        .def_readwrite("observation_timestamp_ms",
                       &EnvStruct::observation_timestamp_ms,
                       "Observation timestamp in ms")
        .def_readwrite("observation_count", &EnvStruct::observation_count, "Observation counter")
        .def_readwrite(
            "beacon_interval_ms", &EnvStruct::beacon_interval_ms, "Beacon interval in milliseconds")

        // --- Per-STA observations array ---
        .def_property(
            "sta_observations",
            [](EnvStruct& e) -> py::list
            {
                py::list sta_list;
                for (uint32_t i = 0; i < MAX_NUM_STA; ++i)
                {
                    sta_list.append(
                        py::cast(&e.sta_observations[i], py::return_value_policy::reference));
                }
                return sta_list;
            },
            [](EnvStruct& e, py::list sta_list) -> void
            {
                for (uint32_t i = 0; i < MAX_NUM_STA && i < py::len(sta_list); ++i)
                {
                    e.sta_observations[i] = sta_list[i].cast<StaEnvStruct>();
                }
            },
            "Array of per-STA observations")

        .def("__repr__",
             [](const EnvStruct& e)
             {
                 return "<EnvStruct: " + std::to_string(e.num_sta) +
                        " STAs, time=" + std::to_string(e.simulation_time_sec) + "s>";
             });

    // --- TwtGroupConfig: TWT group parameters ---
    py::class_<TwtGroupConfig>(m, "TwtGroupConfig")
        .def(py::init<>(), "Initialize TWT group configuration - Ahmed Maksud (SHINE Lab)")

        .def_readwrite("group_id", &TwtGroupConfig::group_id, "TWT group identifier (0-31)")

        // --- TWT timing parameters ---
        .def_readwrite("twt_wake_interval_ms",
                       &TwtGroupConfig::twt_wake_interval_ms,
                       "TWT wake interval in ms")
        .def_readwrite("twt_wake_duration_ms",
                       &TwtGroupConfig::twt_wake_duration_ms,
                       "TWT wake duration in ms")
        .def_readwrite("twt_sp_offset_ms",
                       &TwtGroupConfig::twt_sp_offset_ms,
                       "TWT service period offset in ms")

        // --- Group metadata ---
        .def_readwrite(
            "num_stas_assigned", &TwtGroupConfig::num_stas_assigned, "Number of STAs in this group")

        .def("__repr__",
             [](const TwtGroupConfig& g)
             {
                 return "<TwtGroupConfig: Group " + std::to_string(g.group_id) +
                        ", interval=" + std::to_string(g.twt_wake_interval_ms) +
                        "ms, duration=" + std::to_string(g.twt_wake_duration_ms) + "ms>";
             });

    // --- StaGroupAssignment: STA-to-group mapping ---
    py::class_<StaGroupAssignment>(m, "StaGroupAssignment")
        .def(py::init<>(), "Initialize STA group assignment - Ahmed Maksud (SHINE Lab)")

        .def_readwrite("sta_id", &StaGroupAssignment::sta_id, "STA identifier")
        .def_readwrite("assigned_twt_group",
                       &StaGroupAssignment::assigned_twt_group,
                       "Assigned TWT group (0-31)")
        .def_readwrite("enable_twt", &StaGroupAssignment::enable_twt, "Enable TWT for STA (0/1)")

        .def("__repr__",
             [](const StaGroupAssignment& a)
             {
                 return "<StaGroupAssignment by Ahmed Maksud (SHINE Lab): STA " +
                        std::to_string(a.sta_id) + " → Group " +
                        std::to_string(a.assigned_twt_group) +
                        ", enabled=" + std::to_string(a.enable_twt) + ">";
             });

    // --- ActionStruct: complete TWT scheduling decision ---
    py::class_<ActionStruct>(m, "ActionStruct")
        .def(py::init<>(), "Initialize group-based action structure - Ahmed Maksud (SHINE Lab)")

        // --- Metadata ---
        .def_readwrite("num_sta", &ActionStruct::num_sta, "Number of STAs")
        .def_readwrite("num_active_twt_groups",
                       &ActionStruct::num_active_twt_groups,
                       "Number of active TWT groups")
        .def_readwrite(
            "action_timestamp_ms", &ActionStruct::action_timestamp_ms, "Action timestamp in ms")

        // --- TWT group configurations array ---
        .def_property(
            "twt_group_configs",
            [](ActionStruct& a) -> py::list
            {
                py::list config_list;
                for (uint32_t i = 0; i < MAX_NUM_TWT_GROUPS; ++i)
                {
                    config_list.append(
                        py::cast(&a.twt_group_configs[i], py::return_value_policy::reference));
                }
                return config_list;
            },
            [](ActionStruct& a, py::list config_list) -> void
            {
                for (uint32_t i = 0; i < MAX_NUM_TWT_GROUPS && i < py::len(config_list); ++i)
                {
                    a.twt_group_configs[i] = config_list[i].cast<TwtGroupConfig>();
                }
            },
            "Array of TWT group configurations")

        // --- STA group assignments array ---
        .def_property(
            "sta_group_assignments",
            [](ActionStruct& a) -> py::list
            {
                py::list assignment_list;
                for (uint32_t i = 0; i < MAX_NUM_STA; ++i)
                {
                    assignment_list.append(
                        py::cast(&a.sta_group_assignments[i], py::return_value_policy::reference));
                }
                return assignment_list;
            },
            [](ActionStruct& a, py::list assignment_list) -> void
            {
                for (uint32_t i = 0; i < MAX_NUM_STA && i < py::len(assignment_list); ++i)
                {
                    a.sta_group_assignments[i] = assignment_list[i].cast<StaGroupAssignment>();
                }
            },
            "Array of STA-to-group assignments")

        .def("__repr__",
             [](const ActionStruct& a)
             {
                 return "<ActionStruct: " + std::to_string(a.num_sta) + " STAs, " +
                        std::to_string(a.num_active_twt_groups) + " active groups>";
             });

    // --- NS3-AI Message Interface ---
    py::class_<ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>(),
             "Initialize message interface - Ahmed Maksud (SHINE Lab)")
        // Communication control methods
        .def("PyRecvBegin",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::PyRecvBegin,
             "Begin receiving data from C++ - Ahmed Maksud (SHINE Lab)")
        .def("PyRecvEnd",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::PyRecvEnd,
             "End receiving data from C++ - Ahmed Maksud (SHINE Lab)")
        .def("PySendBegin",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::PySendBegin,
             "Begin sending data to C++ - Ahmed Maksud (SHINE Lab)")
        .def("PySendEnd",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::PySendEnd,
             "End sending data to C++ - Ahmed Maksud (SHINE Lab)")
        .def("PyGetFinished",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::PyGetFinished,
             "Check if simulation finished - Ahmed Maksud (SHINE Lab)")
        // Data access methods
        .def("GetCpp2PyStruct",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::GetCpp2PyStruct,
             py::return_value_policy::reference,
             "Get environment data from C++ - Ahmed Maksud (SHINE Lab)")
        .def("GetPy2CppStruct",
             &ns3::Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>::GetPy2CppStruct,
             py::return_value_policy::reference,
             "Get action structure to send to C++ - Ahmed Maksud (SHINE Lab)");

    // Module metadata
    m.attr("__author__") = "Ahmed Maksud (SHINE Lab, Texas State University)";
    m.attr("__email__") = "ahmed.maksud@email.ucr.edu";
    m.attr("__lab__") = "SHINE Lab, Texas State University";
    m.attr("__pi__") = "Marcelo Menezes De Carvalho";
    m.attr("__version__") = "2.0.0";
    m.attr("__description__") = "NS3-AI TWT Controller Python Binding Interface";

    // Constants
    m.attr("MAX_NUM_STA") = MAX_NUM_STA;
    m.attr("MAX_NUM_TWT_GROUPS") = MAX_NUM_TWT_GROUPS;
    m.attr("MAC_ADDR_LEN") = MAC_ADDR_LEN;
}

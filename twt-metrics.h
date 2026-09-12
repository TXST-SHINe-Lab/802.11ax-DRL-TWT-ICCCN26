// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-metrics.h
 * @brief Per-BI and per-step metric collection and logging for TWT simulation
 */

#ifndef TWT_METRICS_H
#define TWT_METRICS_H

#include "pb-twt-core.h"
#include "twt-simulation-config.h"

#include "ns3/bsr-manager.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace ns3
{

/**
 * @brief Collects and logs TWT simulation metrics at beacon-interval and controller-call granularity.
 *
 * Emits raw cumulative counters straight from NS-3 to Python and CSV.
 * Every derived quantity, rates, deltas and normalisation alike, is computed on the Python side,
 * so a counter here must never be pre-divided or reset between steps.
 */
class TwtMetrics
{
  public:
    TwtMetrics(const TwtSimulationConfig& config, NodeContainer staNodes);
    ~TwtMetrics();

    /** @brief Allocate the per-STA counter arrays; must run before any logging call. */
    void InitializeArrays();
    /** @brief Print end-of-run flow statistics gathered by the flow monitor. */
    void CalculateAndPrintResults(FlowMonitorHelper* flowmon, Ptr<FlowMonitor> monitor);
    void PrintPacketFlowStatistics();
    void PrintEnergyStatistics();

    /** @brief Append one CSV row per beacon interval and accumulate the BSR occupancy window. */
    void LogBiLevelMetrics();
    /** @brief Open the BI-level CSV under logdir and write its header. */
    void InitializeBiLevelLogging(std::string logdir);

    /** @brief Open the call-level CSV under logdir and write its header. */
    void InitializeCallLevelLogging(std::string logdir);

    /**
     * @brief Build the observation for the Python controller, one call per agent step.
     *
     * Also writes the call-level CSV row and resets the BSR window accumulators.
     */
    EnvStruct LogAndSendCallLevelMetrics();

    /** @brief Open the fallback CSV, used when a trace array is null and values read as 0. */
    void InitializeFallbackLogging(std::string logdir);
    void LogFallback(double time_ms, uint32_t staId, const std::string& variable_name);

    void SetNetworkSetup(TwtNetworkSetup* setup)
    {
        m_networkSetup = setup;
    }

  private:
    const TwtSimulationConfig& m_config;
    NodeContainer m_staNodes;
    TwtNetworkSetup* m_networkSetup;

    /** @brief Fill one per-STA observation with raw cumulative counters, no derived values. */
    void PopulateStaObservationRaw(StaEnvStruct& sta, uint32_t staId);

    // BI-level logging members
    std::ofstream m_biLevelCsv;
    bool m_biLevelLoggingEnabled;
    uint64_t m_biLevelObservationCount;
    double m_lastBiLogTime_ms;
    bool m_biLevelFirstCall;

    // Call-level logging members
    std::ofstream m_callLevelCsv;
    bool m_callLevelLoggingEnabled;
    uint64_t m_callLevelObservationCount;
    double m_lastCallLogTime_ms;
    bool m_callLevelFirstCall;

    // Fallback logging, recording every trace array that was null and therefore read as 0.
    std::ofstream m_fallbackCsv;
    bool m_fallbackLoggingEnabled;

    // BSR window accumulators, summed during BI-level logging and reset at each call-level log.
    std::vector<double> m_bsrOccupancySum;   // Sum of occupancy values, each 0.0 to 1.0, per STA
    std::vector<uint32_t> m_bsrSampleCount;  // BI samples per STA, the divisor for the mean
    std::vector<uint32_t> m_bsrAbove50Count; // BIs with occupancy above 50%
    std::vector<uint32_t> m_bsrAbove75Count; // BIs with occupancy above 75%
    std::vector<uint32_t> m_bsrAbove95Count; // BIs with occupancy above 95%

    // Server apps and throughput tracking
    uint64_t* m_totalRxBytes;
    double* m_throughput;
};

} // namespace ns3

#endif // TWT_METRICS_H

// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-main-simulation.cc
 * @brief Main NS-3 simulation entry point for the TWT scheduler RL environment
 */

#include "pb-twt-wrapper-ns3.h"
#include "twt-metrics.h"
#include "twt-simulation-config.h"
#include "twt-trace-callbacks.h"

#include "ns3/applications-module.h"
#include "ns3/bsr-manager.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/log.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UnilateralTwtDemo");

// Global pointers for periodic TWT update
static Ptr<TWTWrapper> g_twtWrapper = nullptr;
static TwtNetworkSetup* g_networkSetup = nullptr;
static TwtMetrics* g_metrics = nullptr;
static Time g_updateInterval;
static Time g_beaconInterval;
static uint32_t g_updateCount = 0;

// BI-level metrics logging callback - runs every beacon interval
void
PeriodicBiLevelLogging()
{
    if (!g_metrics)
    {
        return;
    }

    g_metrics->LogBiLevelMetrics();

    // Schedule next logging event
    Simulator::Schedule(g_beaconInterval, &PeriodicBiLevelLogging);
}

// Periodic TWT update callback - runs during simulation
void
PeriodicTWTUpdate()
{
    if (!g_twtWrapper || !g_networkSetup || !g_metrics)
    {
        std::cout << "[TWT ERROR: Null pointers detected!" << std::endl;
        return;
    }

    g_updateCount++;
    std::cout << "\n\033[34m========== Dynamic TWT Update #" << g_updateCount
              << " (t=" << Simulator::Now().GetSeconds() << "s) ==========\033[0m" << std::endl;

    // 1. Collect current environment metrics (also logs call-level metrics internally)
    EnvStruct env = g_metrics->LogAndSendCallLevelMetrics();

    std::cout << "\033[36m[Metrics] Collected environment data:\033[0m" << std::endl;
    std::cout << "  • STAs: " << env.num_sta << std::endl;
    std::cout << "  • Simulation Time: " << env.simulation_time_sec << "s" << std::endl;

    // 2. Request new TWT schedule from Python
    std::cout << "\033[32m[Controller] Requesting TWT schedule from Python...\033[0m" << std::endl;
    ActionStruct action = g_twtWrapper->RequestTWTSchedule(env);

    // 3. Apply new TWT schedule
    std::cout << "\033[32m[Controller] Received action with " << action.num_active_twt_groups
              << " active TWT groups\033[0m" << std::endl;
    g_networkSetup->ApplyTWTSchedule(action);

    // 5. Schedule next update only if we haven't reached the limit
    if (g_updateCount < DURATION_IN_UPDATE)
    {
        Simulator::Schedule(g_updateInterval, &PeriodicTWTUpdate);
    }
    else
    {
        std::cout << "\033[32m[TWT] All " << DURATION_IN_UPDATE
                  << " updates completed - no more updates scheduled\033[0m" << std::endl;
    }
}

int
main(int argc, char* argv[])
{
    // --- Configuration ---
    TwtSimulationConfig config;

    // Import beacon interval for timing calculations
    Time beaconInterval_s = config.beaconInterval_s;

    bool enableDynamicTWT = true;
    // In what interval we will update TWT settings (in BI)
    double twtUpdateInterval_s = (TWT_UPDATE_INTERVAL_BI * beaconInterval_s).GetSeconds();
    // When would be the first update from python coming in (in BI)
    double twtUpdateStart_s = (TWT_UPDATE_START_BI * beaconInterval_s).GetSeconds();

    // Custom shared memory segment names (for parallel/subprocess execution)
    std::string segmentName = "My Seg";
    std::string cpp2pyMsgName = "My Cpp to Python Msg";
    std::string py2cppMsgName = "My Python to Cpp Msg";
    std::string lockableName = "My Lockable";

    // --- Single source of truth: number of STAs ---
    // Number of active STAs is initialized in TwtSimulationConfig constructor with DEFAULT_NUM_STA
    // Can be overridden by --nStations command line argument
    // This value propagates to:
    //   1. Node creation: TwtNetworkSetup::CreateNodes() uses m_config.nStations
    //   2. Environment data: TwtMetrics::LogAndSendCallLevelMetrics() sets env.num_sta from node
    //   count
    //   3. Python controller: Reads env_dict["num_sta"] and echoes it back in
    //   action_dict["num_sta"]
    //   4. TWT application: Uses action.num_sta to iterate over STAs when applying schedules
    // Note: nStations is now initialized in TwtSimulationConfig constructor (pb-twt-core.h:
    // DEFAULT_NUM_STA = 5)

    std::cout << "setting up command line" << std::endl;
    CommandLine cmd(__FILE__);
    cmd.AddValue("simId", "Simulation ID", config.simId);
    cmd.AddValue("randSeed", "Random seed", config.randSeed);
    cmd.AddValue("parallelSim", "Parallel simulation mode", config.parallelSim);
    cmd.AddValue("scenario", "Scenario name", config.scenario);
    cmd.AddValue("nStations",
                 "Number of stations (configurable, default from MAX_NUM_STA)",
                 config.nStations);
    cmd.AddValue("simulationTime", "Simulation time in milliseconds", config.simulationTime_ms);
    cmd.AddValue("p2pLinkDelay", "P2P link delay in ms", config.p2pLinkDelay_ms);
    cmd.AddValue("enableStateLogs", "Enable state logs", config.enableStateLogs);
    cmd.AddValue("enablePcap", "Enable PCAP", config.enablePcap);
    cmd.AddValue("enableDynamicTWT", "Enable dynamic TWT reconfiguration", enableDynamicTWT);
    cmd.AddValue("twtUpdateInterval", "TWT update interval in seconds", twtUpdateInterval_s);
    cmd.AddValue("twtUpdateStart", "When to start dynamic TWT updates (seconds)", twtUpdateStart_s);
    cmd.AddValue("segmentName", "Shared memory segment name (for parallel episodes)", segmentName);
    cmd.AddValue("cpp2pyMsgName", "Cpp to Python message name", cpp2pyMsgName);
    cmd.AddValue("py2cppMsgName", "Python to Cpp message name", py2cppMsgName);
    cmd.AddValue("lockableName", "Lockable name for synchronization", lockableName);
    cmd.Parse(argc, argv);
    std::cout << "finished setting up command line" << std::endl;

    std::cout << "\n\n========== UNILATERAL TWT SIMULATION ==========" << std::endl;
    std::cout << "Simulation ID: " << config.simId << std::endl;
    std::cout << "Number of STAs: " << config.nStations << std::endl;
    std::cout << "TWT Mode: UNILATERAL (AP announces, no STA negotiation)" << std::endl;
    if (enableDynamicTWT)
    {
        std::cout << "Dynamic TWT: ENABLED (updates every " << twtUpdateInterval_s
                  << "s starting at t=" << twtUpdateStart_s << "s)" << std::endl;
    }
    std::cout << "===============================================\n\n" << std::endl;

    // Generate simulation ID string
    config.GenerateSimIdString();

    // Write device class assignments to file
    config.WriteDeviceClassAssignments();

    // Export config parameters to trace callbacks module (in milliseconds)
    keepTrackOfMetricsFrom_ms = config.keepTrackOfMetricsFrom_ms;
    TI_currentModel_mA = config.TI_currentModel_mA;

    // --- NS-3 global configuration ---
    std::cout << "setting up ns-3 global configuration" << std::endl;
    Config::SetDefault("ns3::ArpCache::MaxRetries", UintegerValue(20));
    Config::SetDefault("ns3::ArpCache::WaitReplyTimeout", TimeValue(MilliSeconds(1000)));
    Config::SetDefault("ns3::WifiMacQueue::MaxDelay", TimeValue(MilliSeconds(1000)));
    Config::SetDefault("ns3::WifiMacQueue::MaxSize", QueueSizeValue(QueueSize("1000p")));
    Config::SetDefault("ns3::QosFrameExchangeManager::SetQueueSize", BooleanValue(true));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(config.payloadSize));

    // --- Network setup ---
    std::cout << "setting up network and nodes" << std::endl;
    TwtNetworkSetup networkSetup(config);
    networkSetup.CreateNodes();
    networkSetup.ConfigureWifi();
    networkSetup.SetupMobility();
    networkSetup.ConfigureInternet();
    networkSetup.SetupApplications();
    std::cout << "setting up initial TWT schedule without python input" << std::endl;
    networkSetup.SetupTwtSchedule();
    networkSetup.EnablePcap();

    // --- Metrics initialization ---
    std::cout << "setting up metrics collection" << std::endl;
    TwtMetrics metrics(config, networkSetup.GetStaNodes());
    metrics.InitializeArrays();
    metrics.SetNetworkSetup(&networkSetup);

    // Initialize BI-level logging (per beacon interval metrics)
    std::string biLevelLogDir = "contrib/ai/examples/twt/data-log";
    metrics.InitializeBiLevelLogging(biLevelLogDir);

    // Initialize Call-level logging (per Python controller call metrics)
    metrics.InitializeCallLevelLogging(biLevelLogDir);

    // --- Dynamic TWT setup ---
    std::cout << "setting up dynamic TWT wrapper" << std::endl;
    Ptr<TWTWrapper> twtWrapper = nullptr;
    if (enableDynamicTWT)
    {
        std::cout << "\n\033[34m========== DYNAMIC TWT INITIALIZATION ==========\033[0m"
                  << std::endl;

        // Configure shared memory segment names BEFORE creating TWTWrapper
        // This is critical for subprocess-based training where each episode needs unique names
        auto msgInterface = Ns3AiMsgInterface::Get();
        msgInterface->SetNames(segmentName, cpp2pyMsgName, py2cppMsgName, lockableName);
        std::cout << "\033[36m• Shared memory segment: " << segmentName << "\033[0m" << std::endl;

        twtWrapper = CreateObject<TWTWrapper>();

        std::string twtLogFile = "contrib/ai/examples/twt/data-log/ns3-twt-wrapper-" +
                                 config.currentsimId_string + ".csv";
        twtWrapper->EnableLogging(true, twtLogFile);

        if (!twtWrapper->Initialize())
        {
            std::cout
                << "\033[31m[ERROR] Failed to initialize TWTWrapper! Disabling dynamic TWT.\033[0m"
                << std::endl;
            enableDynamicTWT = false;
        }
        else
        {
            std::cout << "\033[32m✓ TWTWrapper initialized successfully\033[0m" << std::endl;
            std::cout << "\033[32m✓ Python controller ready\033[0m" << std::endl;
            std::cout << "\033[36m• Update interval: " << twtUpdateInterval_s << " seconds\033[0m"
                      << std::endl;
            std::cout << "\033[36m• First update at: t=" << twtUpdateStart_s << "s\033[0m"
                      << std::endl;
            std::cout << "\033[36m• Log file: " << twtLogFile << "\033[0m" << std::endl;

            // Set global pointers for periodic update
            g_twtWrapper = twtWrapper;
            g_networkSetup = &networkSetup;
            g_metrics = &metrics;
            g_updateInterval = Seconds(twtUpdateInterval_s);
            g_beaconInterval = config.beaconInterval_s;
        }

        std::cout << "\033[34m===============================================\033[0m\n"
                  << std::endl;
    }

    // --- Trace files ---
    OpenTraceFiles(config.currentsimId_string);

    std::cout << "\n===== TRACING ENABLED =====" << std::endl;
    // std::cout << "E2E Trace file: data-log/e2e_trace_" << config.currentsimId_string
    //           << ".csv (from 10s to end)" << std::endl;
    // std::cout
    //     << "Packet journey: STA App → STA IP → STA MAC → STA PHY → AP PHY → Server IP → Server
    //     App"
    //     << std::endl;
    // std::cout << "Shows: Complete packet path with timestamps and delays at each stage\n"
    //           << std::endl;

    // std::cout << "===== QoS METRICS TRACING ENABLED =====" << std::endl;
    // std::cout << "Queue Size Trace: data-log/queue_size_trace_" << config.currentsimId_string
    //           << ".csv" << std::endl;
    // std::cout << "  Tracks: MAC queue depth at each STA over time (ground truth)" << std::endl;
    // std::cout << "  Compare with BSR to see reporting accuracy\n" << std::endl;
    // std::cout << "A-MPDU Trace: data-log/ampdu_trace_" << config.currentsimId_string << ".csv"
    //           << std::endl;
    // std::cout << "  Tracks: Frame aggregation events (MPDUs per A-MPDU)" << std::endl;
    // std::cout << "  Shows: Aggregation efficiency during TWT wake windows\n" << std::endl;
    // std::cout << "BSR Tracking: Real-time via BsrManager (no CSV file)" << std::endl;
    // std::cout << "  Runtime access: BsrManager::GetInstance()->GetCurrentQueueSize()" <<
    // std::endl; std::cout << "  Immediate callbacks available for adaptive control\n" <<
    // std::endl;

    // --- Connect traces ---
    std::cout << "setting up trace connections" << std::endl;
    ConnectSummaryTraces(networkSetup.GetStaNodes());
    ConnectE2ETraces(networkSetup.GetStaNodes(),
                     networkSetup.GetApNodes(),
                     networkSetup.GetServerNode(),
                     networkSetup.GetServerApps());
    ConnectQosMetricTraces(networkSetup.GetStaNodes());
    ConnectPhyStateTraces(networkSetup.GetStaNodes());
    ConnectTimeoutAndDropTraces(networkSetup.GetStaNodes(), networkSetup.GetApNodes());
    Connect802dot11kTraces(networkSetup.GetStaNodes(), networkSetup.GetApNodes());

    // --- BSR manager setup ---
    std::cout << "\n===== BSR MANAGER DEMO ENABLED =====" << std::endl;

    Ptr<BsrManager> bsrManager = BsrManager::GetInstance();

    // DEMO 1: Connect callback for immediate BSR notifications
    bool connectCallback = true;
    if (connectCallback)
    {
        bsrManager->TraceConnectWithoutContext("BsrReceived", MakeCallback(&BsrReceivedCallback));
        std::cout << "✓ BSR callback connected - will log BSR events to CSV" << std::endl;
    }

    // DEMO 2: Schedule periodic BSR queries (polling approach)
    // bool enablePeriodicCheck = false;
    // if (enablePeriodicCheck)
    // {
    //     Simulator::Schedule(Seconds(12.0), &PeriodicBsrCheck);
    //     std::cout << "✓ Periodic BSR check scheduled (every 5 seconds starting at t=12s)"
    //               << std::endl;
    // }

    std::cout << "======================================\n" << std::endl;

    // --- Flow monitor ---
    // std::cout << "setting up flow monitor" << std::endl;
    // FlowMonitorHelper flowmon;
    // Ptr<FlowMonitor> monitor;
    // if (config.enableFlowMon)
    // {
    //     flowmon.SetMonitorAttribute("StartTime",
    //                                 TimeValue(MilliSeconds(config.keepTrackOfMetricsFrom_ms)));
    //     flowmon.SetMonitorAttribute(
    //         "DelayBinWidth",
    //         DoubleValue(config.delayBinWidth_ms / 1000.0)); // Convert ms to seconds
    //     monitor = flowmon.InstallAll();
    // }

    // --- Schedule dynamic TWT updates ---
    std::cout << "first cycle starts here, then it is recursively called thru PeriodicTWTUpdate in "
                 "the scheduler. Only schedules for now, does not run yet."
              << std::endl;
    if (enableDynamicTWT && twtWrapper)
    {
        std::cout << "\n\033[34m[Dynamic TWT] Scheduling periodic updates...\033[0m" << std::endl;
        Simulator::Schedule(Seconds(twtUpdateStart_s), &PeriodicTWTUpdate);
        std::cout << "\033[32m✓ First update scheduled at t=" << twtUpdateStart_s << "s\033[0m\n"
                  << std::endl;
    }

    // --- Schedule BI-level logging ---
    // Schedule per-beacon-interval metrics logging starting at TWT setup time
    double twtSetupTime_s = TWT_SETUP_TIME_BI * config.beaconInterval_s.GetSeconds();
    std::cout << "\n\033[35m[BI-Level Logging] Scheduling per-beacon-interval metrics...\033[0m"
              << std::endl;
    Simulator::Schedule(Seconds(twtSetupTime_s), &PeriodicBiLevelLogging);
    std::cout << "\033[35m✓ BI-level logging scheduled at t=" << twtSetupTime_s << "s (every "
              << config.beaconInterval_s.GetMilliSeconds() << "ms)\033[0m\n"
              << std::endl;

    // --- Run simulation ---
    std::cout << "running simulation... going back to python" << std::endl;
    Simulator::Stop(MilliSeconds(config.simulationTime_ms));
    Simulator::Run();

    std::cout << "\n\n====== SIMULATION COMPLETE ======\n" << std::endl;

    // --- Signal python that simulation is done ---
    // Must happen before we exit, while Python may still be waiting.
    // ns3-ai's CppSetFinished() signals through shared memory to unblock Python's PyRecvBegin().
    if (enableDynamicTWT && twtWrapper)
    {
        std::cout << "[TWT] Sending finish signal to Python..." << std::endl;
        auto msgInterface = Ns3AiMsgInterface::Get();
        // Get the typed interface and call finish
        auto typedInterface = msgInterface->GetInterface<EnvStruct, ActionStruct>();
        typedInterface->CppSetFinished();
        std::cout << "[TWT] Finish signal sent" << std::endl;
    }

    // --- Results ---
    CloseTraceFiles();

    std::cout << "\n===== TRACE FILES SAVED =====" << std::endl;
    std::cout << "End-to-end trace: data-log/e2e_trace_" << config.currentsimId_string << ".csv"
              << std::endl;
    std::cout << "  Complete packet journey with timestamps at each stage" << std::endl;
    std::cout << "  Each packet tracked by UID: STA App → IP → MAC → PHY → AP → Server\n"
              << std::endl;

    std::cout << "QoS Metrics traces:" << std::endl;
    std::cout << "  Queue Size: data-log/queue_size_trace_" << config.currentsimId_string << ".csv"
              << std::endl;
    std::cout << "  A-MPDU Aggregation: data-log/ampdu_trace_" << config.currentsimId_string
              << ".csv" << std::endl;
    std::cout << "  BSR (Buffer Status Report): data-log/bsr_trace_" << config.currentsimId_string
              << ".csv" << std::endl;

    std::cout << "\n\nSimulation with ID " << config.simId << " completed." << std::endl;
    std::cout << "=========================================\n\n" << std::endl;

    Simulator::Destroy();

    return 0;
}

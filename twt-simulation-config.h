// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-simulation-config.h
 * @brief Network topology, device configuration, and TWT schedule application
 */

#ifndef TWT_SIMULATION_CONFIG_H
#define TWT_SIMULATION_CONFIG_H

#include "pb-twt-core.h"
#include "twt-constants.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-helper.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/wifi-module.h"

#include <unordered_map>

namespace ns3
{

// --- Device class definitions for heterogeneous network ---

/**
 * DeviceClass: Enumeration for device types in heterogeneous network
 * Each class has distinct traffic patterns, energy constraints, and QoS requirements
 */
enum DeviceClass
{
    DEVICE_IOT_SENSOR = 0,      // Low rate, periodic, battery-powered
    DEVICE_VIDEO_CAMERA = 1,    // Medium rate, constant, plugged-in
    DEVICE_VOICE_ASSISTANT = 2, // Low rate, bursty, latency-critical
    DEVICE_VIDEO_STREAMING = 3  // High rate, elastic, interactive
};

/**
 * StaApplicationConfig: Per-STA application configuration structure
 * Contains all parameters needed to configure traffic generation and QoS for one STA
 */
struct StaApplicationConfig
{
    uint32_t sta_id;          // STA identifier
    DeviceClass device_class; // Device type

    // Traffic characteristics
    uint32_t traffic_rate_bps;  // Bits per second
    Time packet_interval;       // Interval between packets (for periodic)
    uint32_t packet_size_bytes; // Packet size

    // Temporal characteristics
    double burstiness;  // Coefficient of variation (0=CBR, >1=bursty)
    Time on_time_mean;  // Mean ON time (for ON/OFF)
    Time off_time_mean; // Mean OFF time (for ON/OFF)

    // Energy characteristics
    double battery_capacity_mj; // Battery capacity in millijoules
    bool is_battery_powered;    // True if battery, false if plugged-in

    // QoS characteristics
    Time latency_requirement; // Maximum acceptable latency
    double priority_level;    // 0.0 (low) to 1.0 (critical)

    // Descriptive
    std::string device_name; // Descriptive name for logging
};

// --- Predefined device configurations ---

static const StaApplicationConfig IOT_SENSOR_CONFIG = {
    0,                 // sta_id (will be overwritten)
    DEVICE_IOT_SENSOR, // device_class
    256000,            // traffic_rate_bps: 256 Kbps
    MilliSeconds(100), // packet_interval: 100 ms
    200,               // packet_size_bytes: 200 bytes
    0.1,               // burstiness: Almost perfect CBR
    MilliSeconds(20),  // on_time_mean
    MilliSeconds(80),  // off_time_mean
    10.0,              // battery_capacity_mj: Small battery
    true,              // is_battery_powered: Yes
    Seconds(5),        // latency_requirement: 5 second tolerance
    0.3,               // priority_level: Low priority
    "IoT-Sensor"       // device_name
};

static const StaApplicationConfig VIDEO_CAMERA_CONFIG = {
    0,                   // sta_id
    DEVICE_VIDEO_CAMERA, // device_class
    2000000,             // traffic_rate_bps: 2 Mbps
    MilliSeconds(6),
    1500,              // packet_size_bytes: 1500 bytes
    0.3,               // burstiness: Low variation (CBR-like)
    MilliSeconds(100), // on_time_mean: Continuous
    MilliSeconds(1),   // off_time_mean: Minimal sleep
    1000.0,            // battery_capacity_mj: Large (AC-powered)
    false,             // is_battery_powered: No (AC)
    MilliSeconds(500), // latency_requirement: 500 ms
    0.6,               // priority_level: Medium
    "Video-Camera"     // device_name
};

static const StaApplicationConfig VOICE_ASSISTANT_CONFIG = {
    0,                      // sta_id
    DEVICE_VOICE_ASSISTANT, // device_class
    64000,                  // traffic_rate_bps: 64 Kbps (G.711)
    MilliSeconds(20),       // packet_interval: 20 ms
    160,                    // packet_size_bytes: 160 bytes
    0.2,                    // burstiness: Low variation
    Seconds(2),             // on_time_mean: 2 second turns
    Seconds(3),             // off_time_mean: 3 second silence
    1000.0,                 // battery_capacity_mj: Large (AC)
    false,                  // is_battery_powered: No (AC)
    MilliSeconds(50),       // latency_requirement: 50 ms ultra-low
    0.9,                    // priority_level: High (real-time)
    "Voice-Assistant"       // device_name
};

static const StaApplicationConfig VIDEO_STREAMING_CONFIG = {
    0,                      // sta_id
    DEVICE_VIDEO_STREAMING, // device_class
    5000000,                // traffic_rate_bps: 5 Mbps
    MilliSeconds(5),        // packet_interval: 5 ms
    1400,                   // packet_size_bytes: 1400 bytes
    2.0,                    // burstiness: High variation (bursty)
    Seconds(5),             // on_time_mean: 5 second watching
    Seconds(1),             // off_time_mean: 1 second pause
    1000.0,                 // battery_capacity_mj: Large (AC)
    false,                  // is_battery_powered: No (AC)
    MilliSeconds(100),      // latency_requirement: 100 ms
    0.7,                    // priority_level: Medium-high
    "Video-Streaming"       // device_name
};

// Configuration structure to hold all simulation parameters
struct TwtSimulationConfig
{
    // Simulation parameters
    uint32_t simId = 10001;
    uint32_t randSeed = 9000;
    bool parallelSim = false;
    std::string scenario = "ns3UnilateralTwt";
    std::string currentsimId_string;

    // Timing
    // Default value; overwritten in constructor to (METRICS_START_TIME_BI * beaconInterval_s)
    double keepTrackOfMetricsFrom_ms = 0.0;
    double delayBinWidth_ms = 0.1; // Histogram bin width in milliseconds for FlowMonitor

    // Network
    std::size_t nStations; // active STAs (NOT max capacity)
    double simulationTime_ms;
    double roomLength = DEFAULT_ROOM_LENGTH;
    uint32_t p2pLinkDelay_ms = 0;
    // Use Seconds() instead of MilliSeconds() to preserve fractional precision (102.4ms)
    // MilliSeconds(102.4) truncates to 102ms due to uint64_t conversion
    Time beaconInterval_s = Seconds(BEACON_INTERVAL_MS / 1000.0);
    std::string dlAckSeqType;
    uint32_t bsrLife_ms = 10.0; // BSR validity lifetime in milliseconds (10ms)
    uint32_t ampduLimitBytes = 20000;

    // Transport
    uint32_t payloadSize = 1500;

    // Logging
    bool enablePcap = false;
    bool enableStateLogs = false;
    bool enableFlowMon = false;
    bool recordApPhyState = false;
    bool linkStatusLogging = false;

    // TWT Configuration
    // Implicit TWT Operation:
    // - twtTriggerBased = false: STAs autonomously know wake times without explicit trigger frames
    // - All STAs in same TWT group wake simultaneously at twtNominalWakeDuration intervals
    // - Within each window: AP sends DL first, then STAs send UL (dynamic UL/DL split)
    // - Collision Behavior: With AP downlink present, first collision is nearly certain on UL
    //   (both STAs see clear channel simultaneously after AP finishes), resolved via exponential
    //   backoff
    // - maxMuSta = 1: No multi-user aggregation (sequential transmission per STA, not parallel MU)
    double twtSetupTimeBeaconIntervals = TWT_SETUP_TIME_BI;
    Time firstTwtSpStart;
    Time firstTwtSpOffsetFromBeacon = MilliSeconds(2.0); // 2 milliseconds
    bool twtTriggerBased = false;
    uint64_t maxMuSta = 1;
    Time twtWakeInterval;
    Time twtNominalWakeDuration = MilliSeconds(2.0); // 1 millisecond

    // App timing
    Time AppStartTimeMin;
    Time AppStartTimeMax;

    // Energy model
    std::unordered_map<std::string, double> TI_currentModel_mA = {
        {"IDLE", PHY_STATE_IDLE_MA},
        {"CCA_BUSY", PHY_STATE_CCA_BUSY_MA},
        {"RX", PHY_STATE_RX_MA},
        {"TX", PHY_STATE_TX_MA},
        {"SLEEP", PHY_STATE_SLEEP_MA}};

    // Constructor to initialize derived values
    TwtSimulationConfig()

    {
        // Initialize number of stations with default value
        nStations = ACTIVE_NUM_STA;

        twtWakeInterval = beaconInterval_s;
        firstTwtSpStart = twtSetupTimeBeaconIntervals * beaconInterval_s;
        keepTrackOfMetricsFrom_ms = (METRICS_START_TIME_BI * beaconInterval_s).GetMilliSeconds();
        // Total simulation duration in milliseconds
        simulationTime_ms = (SIMULATION_DURATION_BI * beaconInterval_s).GetMilliSeconds();
        // 3ms offset is to avoid collision with beacon.
        AppStartTimeMin = (APP_START_TIME_MIN_BI * beaconInterval_s) + MilliSeconds(3.0);
        AppStartTimeMax = (APP_START_TIME_MAX_BI * beaconInterval_s) + MilliSeconds(3.0);
    }

    void GenerateSimIdString();
    void WriteDeviceClassAssignments();

    // --- Heterogeneous network configuration ---

    // Device class assignments for each STA (size = nStations)
    std::vector<StaApplicationConfig> sta_app_configs;

    // Enable heterogeneous traffic generation
    bool enable_heterogeneous_traffic = true;

    /**
     * Initialize heterogeneous network with 4-class distribution
     * Divides STAs into 4 equal groups: IoT, Camera, Voice, Video
     */
    void InitializeHeterogeneousNetwork();

    /**
     * Get device class name for logging
     */
    std::string GetDeviceClassName(DeviceClass dc) const;
};

// Helper class to encapsulate network setup
class TwtNetworkSetup
{
  public:
    TwtNetworkSetup(const TwtSimulationConfig& config);

    void CreateNodes();
    void ConfigureWifi();
    void SetupMobility();
    void ConfigureInternet();
    void SetupApplications();
    void SetupTwtSchedule();
    void EnablePcap();

    // Getters
    NodeContainer GetStaNodes() const
    {
        return wifiStaNodes;
    }

    NodeContainer GetApNodes() const
    {
        return wifiApNodes;
    }

    Ptr<Node> GetServerNode() const
    {
        return p2pServerNode;
    }

    ApplicationContainer GetServerApps() const
    {
        return serverApp;
    }

    Ipv4InterfaceContainer GetStaInterfaces() const
    {
        return staNodeInterfaces;
    }

    Ptr<Node> GetApNode() const
    {
        return ApNode;
    }

    // Dynamic TWT reconfiguration
    void ApplyTWTSchedule(const ActionStruct& action);
    Ptr<WifiMac> GetStaMac(uint32_t staId) const;
    Ptr<WifiMac> GetApMac() const;

    // Get last applied action info
    uint32_t GetLastNumActiveTwtGroups() const
    {
        return m_lastNumActiveTwtGroups;
    }

    // Get STA's assigned TWT group from last applied action
    uint8_t GetStaTwtGroup(uint32_t staId) const
    {
        if (staId < MAX_NUM_STA)
        {
            return m_lastStaGroupAssignments[staId];
        }
        return 0;
    }

    // Get TWT group config from last applied action
    const TwtGroupConfig& GetTwtGroupConfig(uint8_t groupId) const
    {
        return m_lastTwtGroupConfigs[groupId < MAX_NUM_TWT_GROUPS ? groupId : 0];
    }

    // Check if STA has TWT enabled
    bool IsStaTwtEnabled(uint32_t staId) const
    {
        if (staId < MAX_NUM_STA)
        {
            return m_lastStaTwtEnabled[staId];
        }
        return false;
    }

  private:
    const TwtSimulationConfig& m_config;

    // Network containers
    NodeContainer wifiApNodes;
    NodeContainer wifiStaNodes;
    NodeContainer p2pServerNodes;
    Ptr<Node> p2pServerNode;
    Ptr<Node> ApNode;

    // Device containers
    NetDeviceContainer apDevice;
    NetDeviceContainer staDevices;
    NetDeviceContainer p2pdevices;

    // Interface containers
    Ipv4InterfaceContainer staNodeInterfaces;
    Ipv4InterfaceContainer apNodeInterface;
    Ipv4InterfaceContainer p2pNodeInterfaces;

    // Application container
    ApplicationContainer serverApp;

    // Track last applied TWT action
    uint32_t m_lastNumActiveTwtGroups = 0;
    uint8_t m_lastStaGroupAssignments[MAX_NUM_STA] = {0};          // STA -> Group mapping
    bool m_lastStaTwtEnabled[MAX_NUM_STA] = {false};               // STA TWT enabled flag
    TwtGroupConfig m_lastTwtGroupConfigs[MAX_NUM_TWT_GROUPS] = {}; // Group configs

    // Helpers
    SpectrumWifiPhyHelper phy;
    WifiMacHelper mac;
    WifiHelper wifi;

    // Random variable for app start times (heterogeneous traffic)
    Ptr<UniformRandomVariable> appStartTimeRand;

    void RandomWifiStart(Ptr<WifiPhy> phy);
    void initiateUnicastTwtAtAp(Ptr<WifiMac> apMac,
                                Mac48Address staMacAddress,
                                uint8_t flowId,
                                Time twtWakeInterval,
                                Time twtNominalWakeDuration,
                                Time nextTwtOffsetFromNextBeacon);
    void initiateTwtAtSta(Ptr<WifiMac> staMac,
                          Ptr<WifiMac> apMac,
                          uint8_t flowId,
                          Time twtWakeInterval,
                          Time twtNominalWakeDuration,
                          Time nextTwtOffsetFromNextBeacon);

    // --- Heterogeneous application creation helpers ---
    void CreateIoTSensorApplication(uint32_t sta_index,
                                    const StaApplicationConfig& config,
                                    const Address& server_addr);
    void CreateVideoCameraApplication(uint32_t sta_index,
                                      const StaApplicationConfig& config,
                                      const Address& server_addr);
    void CreateVoiceAssistantApplication(uint32_t sta_index,
                                         const StaApplicationConfig& config,
                                         const Address& server_addr);
    void CreateVideoStreamingApplication(uint32_t sta_index,
                                         const StaApplicationConfig& config,
                                         const Address& server_addr);
};

} // namespace ns3

#endif // TWT_SIMULATION_CONFIG_H

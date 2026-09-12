// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-simulation-config.cc
 * @brief Network setup implementation with heterogeneous WiFi 6 device classes
 */

#include "twt-simulation-config.h"

#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/on-off-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/udp-client-server-helper.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ns3
{

// Generate simulation ID string with timestamp
void
TwtSimulationConfig::GenerateSimIdString()
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_time_t);

    // Format: YYYYMMDD_HHMMSS (matching Python controller format)
    std::stringstream timestampStream;
    timestampStream << std::setfill('0') << std::setw(4) << (now_tm.tm_year + 1900) << std::setw(2)
                    << (now_tm.tm_mon + 1) << std::setw(2) << now_tm.tm_mday << "_" << std::setw(2)
                    << now_tm.tm_hour << std::setw(2) << now_tm.tm_min << std::setw(2)
                    << now_tm.tm_sec;

    currentsimId_string = timestampStream.str();
}

// Write device class assignments to file
void
TwtSimulationConfig::WriteDeviceClassAssignments()
{
    // Create filename with simulation ID
    std::string filename =
        "contrib/ai/examples/twt/data-log/device_class_assignments_" + currentsimId_string + ".txt";
    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Warning: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    outFile << "Device Class Assignments - Simulation: " << currentsimId_string << std::endl;
    outFile << "======================================" << std::endl;
    outFile << std::endl;

    for (const auto& config : sta_app_configs)
    {
        outFile << "STA " << config.sta_id << ": " << config.device_name << std::endl;
        outFile << "  Class: " << GetDeviceClassName(config.device_class) << std::endl;
        outFile << "  Rate: " << (config.traffic_rate_bps / 1e6) << " Mbps" << std::endl;
        outFile << "  Packet Size: " << config.packet_size_bytes << " bytes" << std::endl;
        outFile << "  Battery Capacity: " << config.battery_capacity_mj << " mJ" << std::endl;
        outFile << "  Battery Powered: " << (config.is_battery_powered ? "Yes" : "No") << std::endl;
        outFile << "  Priority Level: " << config.priority_level << std::endl;
        outFile << std::endl;
    }

    outFile.close();
    std::cout << "Device class assignments written to " << filename << std::endl;
}

void
TwtSimulationConfig::InitializeHeterogeneousNetwork()
{
    sta_app_configs.clear();

    // Random device class assignment
    // Each STA gets a random class from {0, 1, 2, 3}
    Ptr<UniformRandomVariable> classRand = CreateObject<UniformRandomVariable>();
    classRand->SetAttribute("Min", DoubleValue(0.0));
    classRand->SetAttribute("Max", DoubleValue(3.999)); // 0-3 range

    // Track count per class for naming
    uint32_t classCount[4] = {0, 0, 0, 0};

    for (uint32_t i = 0; i < nStations; i++)
    {
        StaApplicationConfig config;

        // Random class assignment
        uint32_t classIndex = static_cast<uint32_t>(classRand->GetValue());

        // Copy base configuration based on device class
        switch (classIndex)
        {
        case 0:
            config = IOT_SENSOR_CONFIG;
            config.device_name = "Sensor-" + std::to_string(classCount[0]++);
            break;
        case 1:
            config = VIDEO_CAMERA_CONFIG;
            config.device_name = "Camera-" + std::to_string(classCount[1]++);
            break;
        case 2:
            config = VOICE_ASSISTANT_CONFIG;
            config.device_name = "Voice-" + std::to_string(classCount[2]++);
            break;
        case 3:
        default:
            config = VIDEO_STREAMING_CONFIG;
            config.device_name = "Video-" + std::to_string(classCount[3]++);
            break;
        }

        config.sta_id = i;
        sta_app_configs.push_back(config);
        std::cout << "  STA " << i << ": " << config.device_name << " ("
                  << GetDeviceClassName(config.device_class) << ")" << std::endl;
    }

    std::cout << "Heterogeneous network initialized: " << nStations
              << " STAs with random class assignment" << std::endl;
    std::cout << "  IoT Sensors: " << classCount[0] << ", Video Cameras: " << classCount[1]
              << ", Voice Assistants: " << classCount[2] << ", Video Streaming: " << classCount[3]
              << std::endl;
}

// Get device class name for logging
std::string
TwtSimulationConfig::GetDeviceClassName(DeviceClass dc) const
{
    switch (dc)
    {
    case DEVICE_IOT_SENSOR:
        return "IoT-Sensor";
    case DEVICE_VIDEO_CAMERA:
        return "Video-Camera";
    case DEVICE_VOICE_ASSISTANT:
        return "Voice-Assistant";
    case DEVICE_VIDEO_STREAMING:
        return "Video-Streaming";
    default:
        return "Unknown";
    }
}

// TwtNetworkSetup Constructor
TwtNetworkSetup::TwtNetworkSetup(const TwtSimulationConfig& config)
      : m_config(config)
{
}

// Helper function to start WiFi PHY at random time
void
TwtNetworkSetup::RandomWifiStart(Ptr<WifiPhy> phyPtr)
{
    phyPtr->ResumeFromOff();
}

// Create network nodes
void
TwtNetworkSetup::CreateNodes()
{
    wifiApNodes.Create(1);
    ApNode = wifiApNodes.Get(0);

    wifiStaNodes.Create(m_config.nStations);

    p2pServerNodes.Create(1);
    p2pServerNode = p2pServerNodes.Get(0);

    // Initialize heterogeneous network configuration
    if (m_config.enable_heterogeneous_traffic)
    {
        const_cast<TwtSimulationConfig&>(m_config).InitializeHeterogeneousNetwork();

        std::cout << "\n================================================================"
                  << std::endl;
        std::cout << "HETEROGENEOUS NETWORK CONFIGURATION" << std::endl;
        std::cout << "================================================================"
                  << std::endl;

        for (uint32_t i = 0; i < m_config.nStations; i++)
        {
            const auto& cfg = m_config.sta_app_configs[i];
            std::cout << "STA " << i << ": " << cfg.device_name << " ("
                      << m_config.GetDeviceClassName(cfg.device_class) << ") "
                      << "Rate=" << (cfg.traffic_rate_bps / 1e6) << " Mbps "
                      << "Battery=" << cfg.battery_capacity_mj << " mJ "
                      << "Priority=" << cfg.priority_level << std::endl;
        }
        std::cout << "================================================================\n"
                  << std::endl;
    }
}

// Configure WiFi network
void
TwtNetworkSetup::ConfigureWifi()
{
    // P2P setup: Wired backbone link between AP and Server
    // This link simulates the backhaul connection where AP forwards STA traffic to/from backend
    // Topology: [WiFi STAs] <--WiFi--> [AP] <--P2P (1Gbps)--> [Server]
    NodeContainer p2pConnectedNodes;
    p2pConnectedNodes.Add(ApNode);
    p2pConnectedNodes.Add(p2pServerNode);

    PointToPointHelper pointToPoint;
    std::stringstream delayString;
    delayString << m_config.p2pLinkDelay_ms << "ms";
    pointToPoint.SetDeviceAttribute("DataRate", StringValue("1000Mbps"));
    pointToPoint.SetChannelAttribute("Delay", StringValue(delayString.str()));
    p2pdevices = pointToPoint.Install(p2pConnectedNodes);

    // WiFi configuration (constants defined in twt-simulation-config.h)
    int channelWidth = DEFAULT_CHANNEL_WIDTH_MHZ;
    int mcs = DEFAULT_MCS;

    std::ostringstream ossDataMode;
    ossDataMode << "HeMcs" << mcs;

    std::string channelStr = "{0, " + std::to_string(channelWidth) + ", BAND_5GHZ, 0}";

    wifi.SetStandard(WIFI_STANDARD_80211ax);
    std::ostringstream ossControlMode;
    auto nonHtRefRateMbps = HePhy::GetNonHtReferenceRate(mcs) / 1e6;
    ossControlMode << "OfdmRate" << nonHtRefRateMbps << "Mbps";

    Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold",
                       UintegerValue(RTS_CTS_THRESHOLD));

    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue(ossDataMode.str()),
                                 "ControlMode",
                                 StringValue("HeMcs0"));

    Ssid ssid = Ssid("unilateral-twt-network");

    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    Ptr<NakagamiPropagationLossModel> nakagamiLoss = CreateObject<NakagamiPropagationLossModel>();
    nakagamiLoss->SetAttribute("Distance1", DoubleValue(80.0));
    nakagamiLoss->SetAttribute("Distance2", DoubleValue(200.0));
    nakagamiLoss->SetAttribute("m0", DoubleValue(3.0)); // near zone: stronger LOS
    nakagamiLoss->SetAttribute("m1", DoubleValue(1.5)); // far zone: weaker LOS
    nakagamiLoss->SetAttribute("m2", DoubleValue(1.5)); // same as m1; no 3rd region in paper
    spectrumChannel->AddPropagationLossModel(nakagamiLoss);

    Ptr<LogDistancePropagationLossModel> logDistanceLoss =
        CreateObject<LogDistancePropagationLossModel>();
    logDistanceLoss->SetAttribute("Exponent", DoubleValue(3.0)); // path loss exponent α
    logDistanceLoss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    spectrumChannel->AddPropagationLossModel(logDistanceLoss);

    Ptr<ConstantSpeedPropagationDelayModel> propagationDelay =
        CreateObject<ConstantSpeedPropagationDelayModel>();
    spectrumChannel->SetPropagationDelayModel(propagationDelay);
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    phy.SetChannel(spectrumChannel);

    // Configure STA MAC
    // MaxMissedBeacons: STA never disconnects from AP (UINT32_MAX)
    // BE_BlockAckThreshold: Enable Block Ack after 1 packet for efficient ACK aggregation
    //   - Individual ACKs: Each frame gets its own ACK (high overhead)
    //   - Block ACK: Aggregates ACKs into single bitmap frame (efficient)
    //   - Threshold=1: Enables Block Ack immediately, critical for TWT Service Periods
    //     where STAs wake for short windows and may send only 1-2 packets
    mac.SetType("ns3::StaWifiMac",
                "MaxMissedBeacons",
                UintegerValue(MAX_MISSED_BEACONS),
                "BE_BlockAckThreshold",
                UintegerValue(BLOCK_ACK_THRESHOLD),
                "Ssid",
                SsidValue(ssid));

    phy.Set("ChannelSettings", StringValue(channelStr));
    phy.Set("TxGain", DoubleValue(2.0));
    phy.Set("RxGain", DoubleValue(2.0));

    staDevices = wifi.Install(phy, mac, wifiStaNodes);

    // Configure AP MAC
    // EnableBeaconJitter: Disabled (false) for precise, predictable beacon timing
    //   - Critical for TWT: STAs rely on beacon intervals to anchor wake schedules
    //   - Consistent timing ensures reliable Service Period synchronization
    // BE_BlockAckThreshold: Same aggressive threshold (BLOCK_ACK_THRESHOLD) as STA for symmetry
    //   - Reduces per-packet ACK overhead on downlink during TWT SPs
    // BE_MaxAmpduSize: A-MPDU aggregation limit (ampduLimitBytes = 20KB)
    //   - Balances throughput gains vs retry latency on lossy channels
    // BsrLifetime: Buffer Status Report validity window (bsrLife_ms = 10ms)
    //   - AP considers BSR valid for duration before requesting fresh reports
    //   - Allows AP to maintain accurate knowledge of STA queue states
    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter",
                BooleanValue(false),
                "BE_BlockAckThreshold",
                UintegerValue(BLOCK_ACK_THRESHOLD),
                "BE_MaxAmpduSize",
                UintegerValue(m_config.ampduLimitBytes),
                "BsrLifetime",
                TimeValue(MilliSeconds(m_config.bsrLife_ms)),
                "Ssid",
                SsidValue(ssid));

    phy.Set("TxPowerStart", DoubleValue(30.0));
    phy.Set("TxPowerEnd", DoubleValue(30.0));
    phy.Set("TxGain", DoubleValue(6.0));
    phy.Set("RxGain", DoubleValue(6.0));

    apDevice = wifi.Install(phy, mac, wifiApNodes);

    // Random Number Generator (RNG) Stream Assignment
    // Sets up deterministic randomness for reproducible simulations:
    //   - SetSeed(randSeed): Controls global RNG seed (from config, overridable via cmdline)
    //   - SetRun(N): Marks this as run N of a multi-run parameter study
    //   - AssignStreams(): Assigns independent RNG streams to each device to avoid correlation
    //
    // For single-run reproducibility:
    //   Set same seed → get identical random positions, traffic patterns, backoffs
    //
    // For multi-run statistical analysis (recommended for TWT controller evaluation):
    //   Run with increasing seeds (seed+0, seed+1, seed+2, ...) to generate different
    //   topologies while maintaining reproducibility per-run. Analyze aggregate statistics
    //   across runs to get confidence intervals on controller performance metrics.
    //
    // Stream ID allocation:
    //   - Start at RNG_INITIAL_STREAM_ID (100)
    //   - AP device: streams 100-119
    //   - STA devices: streams 120+ (auto-incremented by AssignStreams)
    RngSeedManager::SetSeed(m_config.randSeed);
    RngSeedManager::SetRun(RNG_DEFAULT_RUN_NUMBER);
    int64_t streamNumber = RNG_INITIAL_STREAM_ID;
    streamNumber += wifi.AssignStreams(apDevice, streamNumber);
    streamNumber += wifi.AssignStreams(staDevices, streamNumber);

    // WiFi 6 Guard Interval Configuration (802.11ax)
    // This is single-user mode, not OFDMA:
    //   - ConstantRateWifiManager: fixed MCS per device, one at a time
    //   - TWT implicit: non-overlapping service periods keep transmissions apart
    //   - Result: one STA transmits during its SP window
    //
    // Guard Interval (GI) in single-user WiFi 6:
    //   - Not for multicarrier separation (that's OFDMA's job)
    //   - Purpose: Protect against multipath reflections from walls/objects
    //   - 800ns GI: Safe margin for indoor room (~67ns/meter × 12m = ~800ns max delay)
    //   - Allows receiver to complete channel estimation before symbol sampling
    //
    // Why not use 0.4µs GI here?
    //   - Saves time per symbol (3.2µs total symbol instead of 4.0µs)
    //   - But requires very clean channel with minimal reflections
    //   - Indoor room with furniture has significant multipath → need 0.8µs margin
    //
    Config::Set(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
        TimeValue(NanoSeconds(DEFAULT_GUARD_INTERVAL_NS))); // Convert microseconds to nanoseconds

    // Randomized WiFi start times
    // Stagger WiFi PHY startup across STAs to avoid synchronized boot behavior:
    //   - Each STA randomly delays before turning on WiFi (uniform 0-2.5s)
    //   - Prevents "thundering herd" of simultaneous association attempts
    //   - Helps AP handle association requests more gracefully
    //   - More realistic boot scenario (devices power on at different times in practice)
    for (uint32_t i = 0; i < wifiStaNodes.GetN(); i++)
    {
        Ptr<Node> n = wifiStaNodes.Get(i);
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(n->GetDevice(0));
        Ptr<WifiPhy> phyPtr = wifi_dev->GetPhy();
        phyPtr->SetOffMode();

        Ptr<UniformRandomVariable> random = CreateObject<UniformRandomVariable>();
        uint64_t start_time_bi = random->GetInteger(0, WIFI_STARTUP_WINDOW_BI);
        uint64_t start_time_ms = static_cast<uint64_t>(start_time_bi * BEACON_INTERVAL_MS);

        Simulator::Schedule(
            MilliSeconds(start_time_ms), &TwtNetworkSetup::RandomWifiStart, this, phyPtr);
    }
}

// Setup mobility
// The STAs are stationary
void
TwtNetworkSetup::SetupMobility()
{
    double minX = -1.0 * m_config.roomLength / 2;
    double maxX = 1.0 * m_config.roomLength / 2;
    double minY = -1.0 * m_config.roomLength / 2;
    double maxY = 1.0 * m_config.roomLength / 2;

    Ptr<UniformRandomVariable> xCoordinateRand = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> yCoordinateRand = CreateObject<UniformRandomVariable>();
    xCoordinateRand->SetAttribute("Min", DoubleValue(minX));
    xCoordinateRand->SetAttribute("Max", DoubleValue(maxX));
    yCoordinateRand->SetAttribute("Min", DoubleValue(minY));
    yCoordinateRand->SetAttribute("Max", DoubleValue(maxY));

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0)); // AP at origin

    for (uint32_t ii = 0; ii < m_config.nStations; ii++)
    {
        double currentX = xCoordinateRand->GetValue();
        double currentY = yCoordinateRand->GetValue();
        std::cout << "STA " << ii << " position: [" << currentX << ", " << currentY << ", 0.0]"
                  << std::endl;
        positionAlloc->Add(Vector(currentX, currentY, 0.0));
    }

    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNodes);
    mobility.Install(wifiStaNodes);
}

// Configure Internet stack
void
TwtNetworkSetup::ConfigureInternet()
{
    InternetStackHelper stack;
    stack.Install(wifiApNodes);
    stack.Install(wifiStaNodes);
    stack.Install(p2pServerNodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    apNodeInterface = address.Assign(apDevice);
    staNodeInterfaces = address.Assign(staDevices);
    address.SetBase("192.168.2.0", "255.255.255.0");
    p2pNodeInterfaces = address.Assign(p2pdevices);

    // Print IP to MAC mapping
    std::map<Ipv4Address, Mac48Address> ipToMac;
    for (uint32_t i = 0; i < wifiStaNodes.GetN(); i++)
    {
        Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice>(wifiStaNodes.Get(i)->GetDevice(0));
        Ptr<WifiMac> wifi_mac = wifi_dev->GetMac();
        Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(wifi_mac);
        ipToMac[staNodeInterfaces.GetAddress(i)] = sta_mac->GetAddress();
    }

    std::cout << "\n\nIP to MAC mapping:\n";
    for (auto it = ipToMac.begin(); it != ipToMac.end(); it++)
    {
        std::cout << it->first << " => " << it->second << std::endl;
    }

    Simulator::Schedule(Seconds(0), &Ipv4GlobalRoutingHelper::PopulateRoutingTables);
}

// Setup applications - HETEROGENEOUS NETWORK VERSION
void
TwtNetworkSetup::SetupApplications()
{
    std::cout << "twt-simulation-config-SetupApplications Starting heterogeneous app setup"
              << std::endl;

    uint16_t port = 50000;

    // Create random variable for app start times
    appStartTimeRand = CreateObject<UniformRandomVariable>();
    appStartTimeRand->SetAttribute("Min", DoubleValue(m_config.AppStartTimeMin.GetMilliSeconds()));
    appStartTimeRand->SetAttribute("Max", DoubleValue(m_config.AppStartTimeMax.GetMilliSeconds()));

    // Create sink applications at server (one per STA)
    for (std::size_t i = 0; i < m_config.nStations; i++)
    {
        Address sinkLocalAddress(InetSocketAddress(Ipv4Address::GetAny(), port + i));
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory", sinkLocalAddress);
        ApplicationContainer tempServerApp = sinkHelper.Install(p2pServerNode);
        tempServerApp.Start(Seconds(0.0));
        tempServerApp.Stop(MilliSeconds(m_config.simulationTime_ms + 1));
        serverApp.Add(tempServerApp);
    }

    // Create heterogeneous client applications based on device class
    if (!m_config.enable_heterogeneous_traffic || m_config.sta_app_configs.empty())
    {
        std::cerr << "\n[ERROR] Heterogeneous traffic disabled or no STA application configs found!"
                  << std::endl;
        std::cerr << "  enable_heterogeneous_traffic: " << m_config.enable_heterogeneous_traffic
                  << std::endl;
        std::cerr << "  sta_app_configs size: " << m_config.sta_app_configs.size() << std::endl;
        throw std::runtime_error("Application setup failed: heterogeneous traffic must be enabled "
                                 "with valid STA configs");
    }

    std::cout << "\n[HETEROGENEOUS TRAFFIC SETUP]" << std::endl;

    for (uint32_t i = 0; i < m_config.nStations; i++)
    {
        if (i >= m_config.sta_app_configs.size())
        {
            std::cerr << "Error: STA " << i << " has no app config" << std::endl;
            throw std::runtime_error("Missing application config for STA " + std::to_string(i));
        }

        const auto& config = m_config.sta_app_configs[i];
        Address serverAddr(InetSocketAddress(p2pNodeInterfaces.GetAddress(1), port + i));

        switch (config.device_class)
        {
        case DEVICE_IOT_SENSOR:
            CreateIoTSensorApplication(i, config, serverAddr);
            break;
        case DEVICE_VIDEO_CAMERA:
            CreateVideoCameraApplication(i, config, serverAddr);
            break;
        case DEVICE_VOICE_ASSISTANT:
            CreateVoiceAssistantApplication(i, config, serverAddr);
            break;
        case DEVICE_VIDEO_STREAMING:
            CreateVideoStreamingApplication(i, config, serverAddr);
            break;
        default:
            std::cerr << "Error: Unknown device class for STA " << i << std::endl;
            throw std::runtime_error("Unknown device class for STA " + std::to_string(i));
        }
    }
    std::cout << std::endl;
}

// IoT Sensor: Periodic traffic, very low rate, battery-powered
void
TwtNetworkSetup::CreateIoTSensorApplication(uint32_t sta_index,
                                            const StaApplicationConfig& config,
                                            const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    // ON time: use config value for burst duration (expects random variable string in seconds)
    double onTime_s = config.on_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(onTime_s) + "]"));

    // OFF time: sleep between bursts (expects random variable string in seconds)
    double offTime_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ConstantRandomVariable[Constant=" + std::to_string(offTime_s) + "]"));

    // Data rate during ON period
    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));

    double dutyCycle = onTime_s / (onTime_s + offTime_s) * 100.0;
    double effectiveRate_kbps =
        config.traffic_rate_bps * onTime_s / (onTime_s + offTime_s) / 1000.0;
    std::cout << "  IoT Sensor STA " << sta_index << ": ON=" << (onTime_s * 1000)
              << "ms, OFF=" << (offTime_s * 1000)
              << "ms, Rate=" << (config.traffic_rate_bps / 1000.0) << "Kbps"
              << ", Effective=" << effectiveRate_kbps << "Kbps (" << dutyCycle << "% duty)"
              << std::endl;
}

// Video Camera: Constant bitrate, medium rate, always-on
void
TwtNetworkSetup::CreateVideoCameraApplication(uint32_t sta_index,
                                              const StaApplicationConfig& config,
                                              const Address& server_addr)
{
    UdpClientHelper client(server_addr);

    client.SetAttribute("Interval", TimeValue(config.packet_interval));
    client.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));
    client.SetAttribute("MaxPackets", UintegerValue(1000000)); // Continuous streaming

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = client.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));

    std::cout << "  Video Camera STA " << sta_index << ": CBR=" << (config.traffic_rate_bps / 1e6)
              << "Mbps, Interval=" << config.packet_interval.GetMilliSeconds() << "ms" << std::endl;
}

// Voice Assistant: ON/OFF pattern, low rate, latency-critical
void
TwtNetworkSetup::CreateVoiceAssistantApplication(uint32_t sta_index,
                                                 const StaApplicationConfig& config,
                                                 const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    // ON time: exponential distribution for realistic conversation turns (expects seconds)
    double onMean_s = config.on_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(onMean_s) + "]"));

    // OFF time: exponential silence periods (expects seconds)
    double offMean_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(offMean_s) + "]"));

    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));

    std::cout << "  Voice Assistant STA " << sta_index << ": OnTime=" << onMean_s
              << "s, OffTime=" << offMean_s
              << "s, Latency=" << config.latency_requirement.GetMilliSeconds() << "ms" << std::endl;
}

// Video Streaming: Bursty VBR, high rate, interactive
void
TwtNetworkSetup::CreateVideoStreamingApplication(uint32_t sta_index,
                                                 const StaApplicationConfig& config,
                                                 const Address& server_addr)
{
    OnOffHelper onoff("ns3::UdpSocketFactory", server_addr);

    // ON time: exponential for realistic video watching/seeking (expects seconds)
    double onMean_s = config.on_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OnTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(onMean_s) + "]"));

    // OFF time: exponential for pause/seek/buffer (expects seconds)
    double offMean_s = config.off_time_mean.GetSeconds();
    onoff.SetAttribute(
        "OffTime",
        StringValue("ns3::ExponentialRandomVariable[Mean=" + std::to_string(offMean_s) + "]"));

    std::stringstream rateStr;
    rateStr << config.traffic_rate_bps << "bps";
    onoff.SetAttribute("DataRate", StringValue(rateStr.str()));
    onoff.SetAttribute("PacketSize", UintegerValue(config.packet_size_bytes));

    double appStartTimeMs = appStartTimeRand->GetValue();
    ApplicationContainer clientApp = onoff.Install(wifiStaNodes.Get(sta_index));
    clientApp.Start(MilliSeconds(appStartTimeMs));
    clientApp.Stop(MilliSeconds(m_config.simulationTime_ms - 1));

    std::cout << "  Video Streaming STA " << sta_index
              << ": Rate=" << (config.traffic_rate_bps / 1e6)
              << "Mbps, Burstiness=" << config.burstiness << std::endl;
}

// UNILATERAL TWT - AP assigns individual schedules
void
TwtNetworkSetup::initiateUnicastTwtAtAp(Ptr<WifiMac> apMac,
                                        Mac48Address staMacAddress,
                                        uint8_t flowId,
                                        Time twtWakeInterval,
                                        Time twtNominalWakeDuration,
                                        Time nextTwtOffsetFromNextBeacon)
{
    apMac->SetTwtSchedule(flowId,
                          staMacAddress,
                          false, // isRequestingNode = false (AP is responder)
                          true,  // isImplicitAgreement = true
                          true,  // flowType = true (unannounced)
                          false, // isTriggerBasedAgreement = false
                          true,  // isIndividualAgreement = true (per-STA)
                          0, // twtChannel - specifies WiFi channel (0 = primary/default channel)
                          twtWakeInterval,
                          twtNominalWakeDuration,
                          nextTwtOffsetFromNextBeacon);
}

void
TwtNetworkSetup::initiateTwtAtSta(Ptr<WifiMac> staMac,
                                  Ptr<WifiMac> apMac,
                                  uint8_t flowId,
                                  Time twtWakeInterval,
                                  Time twtNominalWakeDuration,
                                  Time nextTwtOffsetFromNextBeacon)
{
    Mac48Address apMacAddress = apMac->GetAddress();

    staMac->SetTwtSchedule(flowId,
                           apMacAddress,
                           true,  // isRequestingNode = true at STA
                           true,  // isImplicitAgreement = true
                           true,  // flowType = true (unannounced)
                           false, // isTriggerBasedAgreement = false
                           true,  // isIndividualAgreement = true
                           0, // twtChannel - specifies WiFi channel (0 = primary/default channel)
                           twtWakeInterval,
                           twtNominalWakeDuration,
                           nextTwtOffsetFromNextBeacon);
}

// Setup TWT schedule
void
TwtNetworkSetup::SetupTwtSchedule()
{
    std::cout << "\n===== UNILATERAL TWT ASSIGNMENT =====" << std::endl;
    std::cout << "AP assigns individual TWT slots to each STA" << std::endl;
    std::cout << "STAs automatically adopt the announced schedule" << std::endl;
    std::cout << "No negotiation required\n" << std::endl;

    for (uint32_t i = 0; i < wifiStaNodes.GetN(); i++)
    {
        Ptr<WifiNetDevice> apDevicePtr = DynamicCast<WifiNetDevice>(ApNode->GetDevice(1));
        Ptr<WifiMac> apMac = apDevicePtr->GetMac();

        Ptr<WifiNetDevice> staDevice =
            DynamicCast<WifiNetDevice>(wifiStaNodes.Get(i)->GetDevice(0));
        Ptr<WifiMac> staMac = staDevice->GetMac();
        Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(staMac);
        Mac48Address staMacAddress = sta_mac->GetAddress();

        Time nextTwtOffsetFromNextBeacon =
            m_config.firstTwtSpOffsetFromBeacon + (i * m_config.twtNominalWakeDuration);

        if (nextTwtOffsetFromNextBeacon >=
            m_config.beaconInterval_s - m_config.firstTwtSpOffsetFromBeacon)
        {
            std::cerr << "ERROR: TWT offset for STA " << i << " exceeds beacon interval! "
                      << "Offset: " << nextTwtOffsetFromNextBeacon.As(Time::MS)
                      << ", Beacon Interval: " << m_config.beaconInterval_s.As(Time::MS)
                      << std::endl;
            throw std::runtime_error("TWT schedule exceeds beacon interval - no wrapping allowed");
        }

        // Flow ID = 0 for all STAs (individual TWT uses single Flow ID per STA)
        // TWT group assignment is managed separately via wake interval/duration/offset
        uint8_t flowId = 0;

        Simulator::Schedule(m_config.firstTwtSpStart,
                            &TwtNetworkSetup::initiateUnicastTwtAtAp,
                            this,
                            apMac,
                            staMacAddress,
                            flowId,
                            m_config.twtWakeInterval,
                            m_config.twtNominalWakeDuration,
                            nextTwtOffsetFromNextBeacon);

        Simulator::Schedule(m_config.firstTwtSpStart,
                            &TwtNetworkSetup::initiateTwtAtSta,
                            this,
                            staMac,
                            apMac,
                            flowId,
                            m_config.twtWakeInterval,
                            m_config.twtNominalWakeDuration,
                            nextTwtOffsetFromNextBeacon);

        std::cout << "STA " << i << " (" << staMacAddress << "):" << std::endl;
        std::cout << "  TWT announced at t=" << m_config.firstTwtSpStart.As(Time::S) << std::endl;
        std::cout << "  SP offset: +" << nextTwtOffsetFromNextBeacon.As(Time::MS) << " from beacon"
                  << std::endl;
        std::cout << "  Wake interval: " << m_config.twtWakeInterval.As(Time::MS) << std::endl;
        std::cout << "  Wake duration: " << m_config.twtNominalWakeDuration.As(Time::MS)
                  << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=====================================\n" << std::endl;
}

// Get STA MAC by index
Ptr<WifiMac>
TwtNetworkSetup::GetStaMac(uint32_t staId) const
{
    if (staId >= wifiStaNodes.GetN())
    {
        return nullptr;
    }
    Ptr<WifiNetDevice> staDevice =
        DynamicCast<WifiNetDevice>(wifiStaNodes.Get(staId)->GetDevice(0));
    return staDevice->GetMac();
}

// Get AP MAC
Ptr<WifiMac>
TwtNetworkSetup::GetApMac() const
{
    Ptr<WifiNetDevice> apDevicePtr = DynamicCast<WifiNetDevice>(ApNode->GetDevice(1));
    return apDevicePtr->GetMac();
}

// Apply TWT schedule from Python controller (group-based)
// We will be using ms for time as it is related to python
void
TwtNetworkSetup::ApplyTWTSchedule(const ActionStruct& action)
{
    std::cout << "\n[TWT Update] Applying group-based TWT schedule at t="
              << Simulator::Now().GetSeconds() << "s" << std::endl;
    std::cout << "  Active TWT Groups: " << action.num_active_twt_groups << std::endl;

    // Store last action for later retrieval by metrics
    m_lastNumActiveTwtGroups = action.num_active_twt_groups;
    for (uint32_t g = 0; g < MAX_NUM_TWT_GROUPS; g++)
    {
        m_lastTwtGroupConfigs[g] = action.twt_group_configs[g];
    }
    for (uint32_t i = 0; i < MAX_NUM_STA; i++)
    {
        if (i < action.num_sta)
        {
            m_lastStaGroupAssignments[i] = action.sta_group_assignments[i].assigned_twt_group;
            m_lastStaTwtEnabled[i] = (action.sta_group_assignments[i].enable_twt != 0);
        }
        else
        {
            m_lastStaGroupAssignments[i] = 0;
            m_lastStaTwtEnabled[i] = false;
        }
    }

    Ptr<WifiMac> apMac = GetApMac();

    // Process each STA assignment
    for (uint32_t i = 0; i < action.num_sta && i < wifiStaNodes.GetN(); i++)
    {
        const StaGroupAssignment& sta_assign = action.sta_group_assignments[i];

        if (!sta_assign.enable_twt)
        {
            std::cout << "  STA " << i << ": TWT disabled by controller" << std::endl;
            continue;
        }

        // Get the TWT group configuration for this STA
        uint8_t groupId = sta_assign.assigned_twt_group;
        if (groupId >= action.num_active_twt_groups)
        {
            std::cout << "  STA " << i << ": Invalid group ID " << (int)groupId << std::endl;
            continue;
        }

        const TwtGroupConfig& group_cfg = action.twt_group_configs[groupId];

        Ptr<WifiMac> staMac = GetStaMac(i);
        if (!staMac)
        {
            continue;
        }

        Ptr<StaWifiMac> sta_mac = DynamicCast<StaWifiMac>(staMac);
        Mac48Address staMacAddress = sta_mac->GetAddress();
        Mac48Address apMacAddress = apMac->GetAddress();

        // Use group timing parameters
        Time newWakeInterval = MilliSeconds(group_cfg.twt_wake_interval_ms);
        Time newWakeDuration = MilliSeconds(group_cfg.twt_wake_duration_ms);
        Time groupOffset = MilliSeconds(group_cfg.twt_sp_offset_ms);
        Time effectiveGroupOffset = m_config.firstTwtSpOffsetFromBeacon + groupOffset;

        uint8_t flowId = 0; // Flow ID must be 0-7 per IEEE 802.11ax

        // Update AP side
        apMac->SetTwtSchedule(
            flowId,
            staMacAddress,
            false, // isRequestingNode = false (AP is responder)
            true,  // isImplicitAgreement
            true,  // flowType (unannounced)
            false, // isTriggerBasedAgreement
            true,  // isIndividualAgreement
            0,     // twtChannel - specifies WiFi channel (0 = primary/default channel)
            newWakeInterval,
            newWakeDuration,
            effectiveGroupOffset);

        // Update STA side
        staMac->SetTwtSchedule(
            flowId,
            apMacAddress,
            true,  // isRequestingNode = true at STA
            true,  // isImplicitAgreement
            true,  // flowType (unannounced)
            false, // isTriggerBasedAgreement
            true,  // isIndividualAgreement
            0,     // twtChannel - specifies WiFi channel (0 = primary/default channel)
            newWakeInterval,
            newWakeDuration,
            effectiveGroupOffset);

        std::cout << "  STA " << i << " → Group " << (int)groupId
                  << ": Interval=" << group_cfg.twt_wake_interval_ms << "ms"
                  << ", Duration=" << group_cfg.twt_wake_duration_ms << "ms"
                  << ", Offset=" << group_cfg.twt_sp_offset_ms << "ms" << std::endl;
    }

    std::cout << "[TWT Update] Group-based schedule applied successfully\n" << std::endl;
}

// Enable PCAP
void
TwtNetworkSetup::EnablePcap()
{
    if (m_config.enablePcap)
    {
        std::stringstream ss1;
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        ss1 << "contrib/ai/examples/twt/pcap_AP_unilateral";
        phy.EnablePcap(ss1.str(), apDevice);

        std::stringstream ss2;
        ss2 << "contrib/ai/examples/twt/pcap_STA_unilateral";
        phy.EnablePcap(ss2.str(), staDevices);
    }
}

} // namespace ns3

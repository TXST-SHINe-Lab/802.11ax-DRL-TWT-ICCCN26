// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file twt-trace-callbacks.h
 * @brief NS-3 trace callback declarations for TWT simulation monitoring
 */

#ifndef TWT_TRACE_CALLBACKS_H
#define TWT_TRACE_CALLBACKS_H

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <fstream>

namespace ns3
{

// Global trace files
extern std::ofstream e2eTraceFile;
extern std::ofstream macQueueSizeTraceFile;
extern std::ofstream ampduTraceFile;
extern std::ofstream bsrTraceFile;
extern std::ofstream phyStateTraceFile;
extern std::ofstream txRxStatsTraceFile;       // TX/RX success/fail/retry counters
extern std::ofstream linkMeasurementTraceFile; // RSSI, SNR, noise measurements
extern std::ofstream timeoutDropTraceFile;     // PSDU timeouts and MPDU drops

// Tracking arrays, each indexed by STA id and allocated by TwtMetrics::InitializeArrays.
// Elapsed time is measured from the STA's own perspective, so it freezes while the STA sleeps.
// The _TI suffix marks a value accumulated since the last trace interval.
extern double* timeElapsedForSta_ms_TI;
extern double* awakeTimeElapsedForSta_ms_TI;
extern double* sleepTimeElapsedForSta_ms_TI;
extern double* current_mA_TimesTime_ms_ForSta_TI; // mA times ms, divide by 3600 for mAh
extern double* uplinkTimeoutsForSta;
extern double downlinkTimeoutsAllSta; // Downlink, unused: this study is uplink only
extern double* uplinkExpiredMpduForSta;
extern double downlinkExpiredMpduAllSta; // Downlink, unused
extern double* uplinkFailedEnqueueMpduForSta;
extern double downlinkFailedEnqueueMpduAllSta; // Downlink, unused
extern uint64_t* packetsGeneratedByAppForSta;
extern uint64_t* packetsEnqueuedAtMacForSta; // Unused, the MAC enqueue trace is not connected
extern uint64_t* packetsTransmittedByPhyForSta;
extern uint64_t* bytesTransmittedByPhyForSta;

// --- 802.11k STA statistics - raw cumulative counters per STA ---
extern uint64_t* txSuccessCountForSta; // Successful TX (got ACK)
extern uint64_t* txRetryCountForSta;   // TX retries (cumulative)
extern uint64_t* txFailedCountForSta;  // TX failures (max retries exceeded)
extern uint64_t* rxSuccessCountForSta; // Successfully received frames
extern uint64_t* rxErrorCountForSta;   // RX errors, both FCS failures and decoding errors

// --- 802.11k link measurement - last known values per STA ---
extern double* lastRssiDbmForSta;         // Last RSSI measurement (dBm)
extern double* lastSnrDbForSta;           // Last SNR measurement (dB)
extern double* lastNoiseDbmForSta;        // Last noise floor (dBm)
extern uint8_t* lastTxMcsForSta;          // Last TX MCS index
extern uint8_t* lastTxNssForSta;          // Last TX NSS (spatial streams)
extern uint16_t* lastTxRate100KbpsForSta; // Last TX rate in 100 kbps units

// --- 802.11ax BSR (buffer status report) - per access category per STA ---
// Updated by BsrReceivedCallback when the AP receives a BSR from a STA.
// TID to AC mapping: TIDs 0 and 3 are BE, 1 and 2 are BK, 4 and 5 are VI, 6 and 7 are VO.
// Only AC_BE carries traffic in this study; the other three stay at zero.
extern uint32_t* bsrQueueBytesAcBeForSta; // Best Effort queue (TID 0,3)
extern uint32_t* bsrQueueBytesAcBkForSta; // Background queue (TID 1,2)
extern uint32_t* bsrQueueBytesAcViForSta; // Video queue (TID 4,5)
extern uint32_t* bsrQueueBytesAcVoForSta; // Voice queue (TID 6,7)

// --- AP-received traffic - per access category per STA (from ApPhyRxEndTrace) ---
// TID extracted from QoS Data frames and mapped to AC
extern uint64_t* bytesReceivedAcBeForSta;   // Bytes received from STA for AC_BE
extern uint64_t* bytesReceivedAcBkForSta;   // Bytes received from STA for AC_BK
extern uint64_t* bytesReceivedAcViForSta;   // Bytes received from STA for AC_VI
extern uint64_t* bytesReceivedAcVoForSta;   // Bytes received from STA for AC_VO
extern uint64_t* packetsReceivedAcBeForSta; // Packets received from STA for AC_BE
extern uint64_t* packetsReceivedAcBkForSta; // Packets received from STA for AC_BK
extern uint64_t* packetsReceivedAcViForSta; // Packets received from STA for AC_VI
extern uint64_t* packetsReceivedAcVoForSta; // Packets received from STA for AC_VO

// --- A-MPDU aggregation - cumulative counters per STA ---
extern uint64_t* ampduCountForSta;      // Number of A-MPDU transmissions
extern uint64_t* ampduMpdusTotalForSta; // Total MPDUs in A-MPDUs
extern uint64_t* ampduBytesTotalForSta; // Total bytes in A-MPDUs

// --- Airtime & PHY parameters - per STA tracking ---
extern uint64_t* airtimeUsedUsForSta;       // Cumulative airtime in microseconds
extern uint8_t* lastChannelWidthMhzForSta;  // Last channel width used (20/40/80/160)
extern uint16_t* lastGuardIntervalNsForSta; // Last guard interval (800/1600/3200)
extern uint8_t* lastTxPowerDbmForSta;       // Last TX power in dBm
extern uint8_t* lastFrameTypeForSta;        // Last RX frame type (0=mgmt, 1=ctrl, 2=data)
extern uint8_t* lastFrameSubtypeForSta;     // Last RX frame subtype (0-15)
extern uint8_t* lastPowerMgmtBitForSta;     // Last Power Management bit from Frame Control
extern uint64_t* lastRxTimestampUsForSta;   // Last RX time from this STA in us, 0 if none yet

// Configuration parameters needed by callbacks
extern double keepTrackOfMetricsFrom_ms; // ms, callbacks discard samples before this time
// PHY state name to mA, from twt-constants.h.
extern std::unordered_map<std::string, double> TI_currentModel_mA;

// Helper functions
uint32_t ContextToNodeId(std::string context);
double GetPercentileValue(Histogram hist, double percentile);

// Summary trace callbacks, recording from t=0 with no time filter.
void TxTraceAtApp(std::string context, Ptr<const Packet> packet);
void MacTxTrace(std::string context, Ptr<const Packet> packet);
// TX power arrives in W, not dBm, because that is the ns-3 trace signature.
void PhyTxBeginTrace(std::string context, Ptr<const Packet> packet, double txPowerW);

// End-to-end detailed trace callbacks, ignored before keepTrackOfMetricsFrom_ms.
void E2E_AppTx(std::string context, Ptr<const Packet> packet);
void E2E_IpTx(std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
void E2E_MacEnqueue(std::string context, Ptr<const Packet> packet);
// TX power arrives in W, not dBm, because that is the ns-3 trace signature.
void E2E_PhyTx(std::string context, Ptr<const Packet> packet, double txPowerW);
void E2E_PhyRxAp(std::string context, Ptr<const Packet> packet);
void E2E_IpRx(std::string context, Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface);
void E2E_AppRx(Ptr<const Packet> packet, const Address& address);

// QoS metric callbacks
void QueueSizeTrace(std::string context, uint32_t oldSize, uint32_t newSize);
void AmpduAggregationTrace(std::string context,
                           WifiConstPsduMap psduMap,
                           WifiTxVector txVector,
                           // W, not dBm, per the ns-3 trace signature
                           double txPowerW);

// PHY state and error callbacks
void PhyStateTrace_inPlace(std::string context, Time start, Time duration, WifiPhyState state);
void PsduResponseTimeoutTraceSta(std::string context,
                                 uint8_t reason,
                                 Ptr<const WifiPsdu> psdu,
                                 const WifiTxVector& txVector);
void PsduResponseTimeoutTraceAp(std::string context,
                                uint8_t reason,
                                Ptr<const WifiPsdu> psdu,
                                const WifiTxVector& txVector);
void MpduDropped_atSta(std::string context, WifiMacDropReason dropReason, Ptr<const WifiMpdu> mpdu);
void MpduDropped_atAp(std::string context, WifiMacDropReason dropReason, Ptr<const WifiMpdu> mpdu);

// 802.11k Link Measurement callbacks (STA-side, for oracle/downlink)
void PhyRxEndTrace(std::string context,
                   Ptr<const WifiPsdu> psdu,
                   RxSignalInfo rxSignalInfo,
                   const WifiTxVector& txVector,
                   const std::vector<bool>& perMpduStatus);
void PhyRxDropTrace(std::string context, Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason);

// AP-side RX callbacks (REALISTIC - AP receives from STAs)
void ApPhyRxEndTrace(std::string context,
                     Ptr<const WifiPsdu> psdu,
                     RxSignalInfo rxSignalInfo,
                     const WifiTxVector& txVector,
                     const std::vector<bool>& perMpduStatus);
void ApPhyRxDropTrace(std::string context, Ptr<const WifiPsdu> psdu, WifiPhyRxfailureReason reason);

// AP monitor sniffer RX callback, matching the MonitorSnifferRx signature in NS-3.44.
void ApMonitorSnifferRxTrace(std::string context,
                             Ptr<const Packet> packet,
                             uint16_t channelFreqMhz,
                             WifiTxVector txVector,
                             MpduInfo aMpdu,
                             SignalNoiseDbm signalNoise,
                             uint16_t staId);

// Simplified AP PHY RX drop callback, matching the PhyRxDrop signature in NS-3.44.
void ApPhyRxDropTraceSimple(std::string context,
                            Ptr<const Packet> packet,
                            WifiPhyRxfailureReason reason);

// 802.11k TX statistics callbacks - track successful TX and retries
void MacTxOkTrace(std::string context, Ptr<const WifiMpdu> mpdu);
void MacTxDropTrace(std::string context, WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu);

// BSR callbacks
void BsrReceivedCallback(Mac48Address staAddress,
                         uint8_t tid,
                         uint8_t queueSizeUnits,
                         uint32_t queueSizeBytes);
void PeriodicBsrCheck();

// Register MAC address to STA ID mapping for BSR callback
void RegisterStaMacAddress(Mac48Address macAddr, uint32_t staId);

// Trace file management
void OpenTraceFiles(std::string simIdString);
void CloseTraceFiles();

// Initialize 802.11k tracking arrays (call after InitializeArrays)
void Initialize802dot11kArrays(uint32_t numSta);

// Connect all traces
void ConnectSummaryTraces(NodeContainer staNodes);
void ConnectE2ETraces(NodeContainer staNodes,
                      NodeContainer apNodes,
                      Ptr<Node> serverNode,
                      ApplicationContainer serverApps);
void ConnectQosMetricTraces(NodeContainer staNodes);
void ConnectPhyStateTraces(NodeContainer staNodes);
void ConnectTimeoutAndDropTraces(NodeContainer staNodes, NodeContainer apNodes);
void Connect802dot11kTraces(NodeContainer staNodes,
                            NodeContainer apNodes); // Connect 802.11k traces (STA + AP)

} // namespace ns3

#endif // TWT_TRACE_CALLBACKS_H

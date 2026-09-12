// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

/**
 * @file pb-twt-wrapper-ns3.h
 * @brief NS-3 Object TWTWrapper declaration for C++/Python scheduling bridge
 */

#ifndef PB_TWT_WRAPPER_H
#define PB_TWT_WRAPPER_H

#include "pb-twt-core.h"

#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/object.h"

#include <fstream>
#include <string>

using namespace ns3;

/**
 * @brief TWT controller wrapper for the NS3-AI Python integration.
 *
 * Carries observations from the NS-3 TWT simulation to the Python RL controller and brings scheduling actions back.
 * Sends per-STA observations and receives group assignments, wake intervals and durations.
 * One RequestTWTSchedule call is one agent step; the call blocks until Python replies.
 */
class TWTWrapper : public Object
{
  public:
    static TypeId GetTypeId();

    TWTWrapper();
    virtual ~TWTWrapper();

    /**
     * @brief Initialize the NS3-AI message interface and attach its shared memory segment.
     * @return true if successfully initialized
     */
    bool Initialize();

    /**
     * @brief Send TWT observations to Python and block until the scheduling action returns.
     * @param env Complete environment state, per-STA metrics only, no aggregates
     * @return ActionStruct TWT scheduling decisions from the Python controller, or CreateErrorAction() on failure
     */
    ActionStruct RequestTWTSchedule(const EnvStruct& env);

    /**
     * @brief Enable detailed CSV logging of TWT controller interactions
     * @param enable true to enable logging
     * @param logFile path to CSV log file (default: "twt-controller-log.csv")
     */
    void EnableLogging(bool enable, const std::string& logFile = "twt-controller-log.csv");

    /**
     * @brief Get communication statistics
     * @param txCount Reference to store number of observations sent
     * @param rxCount Reference to store number of actions received
     */
    void GetStatistics(uint32_t& txCount, uint32_t& rxCount) const;

    /**
     * @brief Reset statistics counters to zero
     */
    void ResetStatistics();

    /**
     * @brief Check if wrapper is properly initialized
     * @return true if initialized and ready for communication
     */
    bool IsInitialized() const;

  private:
    Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>* m_msgInterface;
    bool m_initialized;
    bool m_loggingEnabled;
    std::string m_logFile;
    uint32_t m_txCount;
    uint32_t m_rxCount;
    std::ofstream m_csvLogFile;

    void LogInteraction(const EnvStruct& env, const ActionStruct& act, bool success);
    /** @brief Neutral action returned when the Python exchange fails, so the simulation continues. */
    ActionStruct CreateErrorAction() const;
};

#endif // PB_TWT_WRAPPER_H
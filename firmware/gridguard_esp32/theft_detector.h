/*
 * ============================================================================
 * GridGuard-AI: Baseline Rule-Based Theft Detector
 * ============================================================================
 * Implements debounced baseline discrepancy detection before/alongside TinyML.
 * Prevents false positives from single-sample noise spikes.
 */

#ifndef GRIDGUARD_THEFT_DETECTOR_H
#define GRIDGUARD_THEFT_DETECTOR_H

#include <Arduino.h>
#include "config.h"

struct RuleDetectorState {
    const char* current_status;
    float confidence;
    uint8_t consecutive_discrepancy_count;
    bool is_alert_active;
    uint32_t last_alert_time_ms;
};

static RuleDetectorState g_rule_detector = {
    "NORMAL",
    0.95f,
    0,
    false,
    0
};

/**
 * Evaluates dual-channel current measurements using rule-based logic.
 *
 * @param total_A Total current from feeder sensor (GPIO32)
 * @param meter_A Legitimate metered current (GPIO33)
 * @param diff_A Discrepancy (total_A - meter_A)
 * @return Struct with rule status and alert state
 */
static RuleDetectorState evaluate_discrepancy_rule(float total_A, float meter_A, float diff_A) {
    float abs_diff = fabsf(diff_A);

    // 1. Power Loss Condition: Total and metered both near zero
    if (total_A < POWER_LOSS_THRESHOLD_AMPS && meter_A < POWER_LOSS_THRESHOLD_AMPS) {
        g_rule_detector.current_status = "POWER_LOSS";
        g_rule_detector.confidence = 0.98f;
        g_rule_detector.consecutive_discrepancy_count = 0;
        g_rule_detector.is_alert_active = false;
        return g_rule_detector;
    }

    // 2. Discrepancy Check (Feeder exceeds metered branch)
    if (diff_A >= BYPASS_THRESHOLD_AMPS) {
        g_rule_detector.consecutive_discrepancy_count++;
        
        // Multi-sample debounce to reject momentary noise/switching transients
        if (g_rule_detector.consecutive_discrepancy_count >= ALERT_DEBOUNCE_SAMPLES) {
            g_rule_detector.current_status = "POSSIBLE_BYPASS";
            // Confidence scales with discrepancy magnitude above threshold
            float excess = diff_A - BYPASS_THRESHOLD_AMPS;
            float conf = 0.85f + (excess * 2.0f);
            if (conf > 0.99f) conf = 0.99f;
            g_rule_detector.confidence = conf;
            g_rule_detector.is_alert_active = true;
            g_rule_detector.last_alert_time_ms = millis();
        } else {
            // Pending confirmation
            g_rule_detector.current_status = "NORMAL";
            g_rule_detector.confidence = 0.70f;
            g_rule_detector.is_alert_active = false;
        }
    } else {
        // Discrepancy within normal bounds
        g_rule_detector.consecutive_discrepancy_count = 0;
        g_rule_detector.current_status = "NORMAL";
        g_rule_detector.confidence = 0.95f;
        g_rule_detector.is_alert_active = false;
    }

    return g_rule_detector;
}

#endif // GRIDGUARD_THEFT_DETECTOR_H

/*
 * ============================================================================
 * GridGuard-AI: Sensor Calibration & Signal Acquisition Module
 * ============================================================================
 * Handles:
 * - 120-sample oversampling on ESP32 12-bit ADC
 * - Zero-current tare baseline calibration for GPIO32 and GPIO33
 * - 10k/10k voltage divider scaling on GPIO33
 * - Exponential Moving Average (EMA) filtering (alpha = 0.30)
 * - Calculation of raw ADC, filtered ADC, delta from baseline, and Amperes
 */

#ifndef GRIDGUARD_CALIBRATION_H
#define GRIDGUARD_CALIBRATION_H

#include <Arduino.h>
#include "config.h"

struct SensorData {
    // ADC Counts (0 - 4095)
    float baseline_total_adc;     // Feeder tare baseline (GPIO32)
    float baseline_metered_adc;   // Metered tare baseline (GPIO33)
    float filtered_total_adc;     // Oversampled & EMA-filtered ADC count
    float filtered_metered_adc;   // Oversampled & EMA-filtered ADC count
    float delta_total_adc;        // total_adc - baseline_total_adc
    float delta_metered_adc;      // metered_adc - baseline_metered_adc

    // Converted physical values (Amperes)
    float total_current_A;
    float meter_current_A;
    float diff_current_A;

    bool is_calibrated;
};

static SensorData g_sensor_data = {
    3102.0f,  // Nominal default baseline
    1551.0f,  // Nominal default baseline (after 10k/10k divider)
    3102.0f,
    1551.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    false
};

/**
 * Reads analog pin with oversampling and returns average raw ADC counts (0.0 to 4095.0).
 */
static inline float read_pin_raw_adc(uint8_t pin, uint16_t samples = OVERSAMPLE_COUNT) {
    uint32_t adc_sum = 0;
    for (uint16_t i = 0; i < samples; i++) {
        adc_sum += analogRead(pin);
        delayMicroseconds(40);
    }
    return (float)adc_sum / (float)samples;
}

/**
 * Performs startup zero-current tare calibration for both channels.
 * IMPORTANT: No load or motor should be drawing current during tare.
 */
static void calibrate_sensors() {
    Serial.println("\n[CALIB] ===================================================");
    Serial.println("[CALIB] Starting Dual ACS712 Zero-Current Baseline Calibration");
    Serial.println("[CALIB] Ensure NO electrical load is drawing current now...");
    Serial.println("[CALIB] ===================================================");

    uint32_t sum_ch1 = 0;
    uint32_t sum_ch2 = 0;

    for (uint16_t i = 0; i < CALIB_SAMPLE_COUNT; i++) {
        sum_ch1 += analogRead(PIN_SENSOR_TOTAL);
        sum_ch2 += analogRead(PIN_SENSOR_METER);
        delay(2);
    }

    g_sensor_data.baseline_total_adc = (float)sum_ch1 / (float)CALIB_SAMPLE_COUNT;
    g_sensor_data.baseline_metered_adc = (float)sum_ch2 / (float)CALIB_SAMPLE_COUNT;

    g_sensor_data.filtered_total_adc = g_sensor_data.baseline_total_adc;
    g_sensor_data.filtered_metered_adc = g_sensor_data.baseline_metered_adc;
    g_sensor_data.delta_total_adc = 0.0f;
    g_sensor_data.delta_metered_adc = 0.0f;
    g_sensor_data.total_current_A = 0.0f;
    g_sensor_data.meter_current_A = 0.0f;
    g_sensor_data.diff_current_A = 0.0f;
    g_sensor_data.is_calibrated = true;

    float v_ch1 = (g_sensor_data.baseline_total_adc / ADC_MAX_VAL) * ESP32_VREF_VOLTS;
    float v_ch2_pin = (g_sensor_data.baseline_metered_adc / ADC_MAX_VAL) * ESP32_VREF_VOLTS;
    float v_ch2_sensor = v_ch2_pin * CH2_VOLTAGE_DIVIDER_RATIO;

    Serial.println("[CALIB] Baseline Tare Complete:");
    Serial.printf("  - GPIO32 Feeder Baseline   : %6.1f ADC counts (%.3f V)\n", g_sensor_data.baseline_total_adc, v_ch1);
    Serial.printf("  - GPIO33 Metered Baseline  : %6.1f ADC counts (%.3f V pin -> %.3f V sensor)\n", 
                  g_sensor_data.baseline_metered_adc, v_ch2_pin, v_ch2_sensor);
    Serial.println("[CALIB] ===================================================\n");
}

/**
 * Acquires filtered ADC readings from both sensors, calculates deltas from baseline,
 * and estimates Amperes.
 */
static void read_sensor_measurements(SensorData &out_data) {
    if (!g_sensor_data.is_calibrated) {
        calibrate_sensors();
    }

    // 1. Acquire oversampled raw ADC counts
    float raw_total_adc = read_pin_raw_adc(PIN_SENSOR_TOTAL, OVERSAMPLE_COUNT);
    float raw_metered_adc = read_pin_raw_adc(PIN_SENSOR_METER, OVERSAMPLE_COUNT);

    // 2. Exponential Moving Average (EMA) smoothing
    g_sensor_data.filtered_total_adc = (EMA_ALPHA * raw_total_adc) + ((1.0f - EMA_ALPHA) * g_sensor_data.filtered_total_adc);
    g_sensor_data.filtered_metered_adc = (EMA_ALPHA * raw_metered_adc) + ((1.0f - EMA_ALPHA) * g_sensor_data.filtered_metered_adc);

    // 3. Deltas from baseline tare
    g_sensor_data.delta_total_adc = g_sensor_data.filtered_total_adc - g_sensor_data.baseline_total_adc;
    g_sensor_data.delta_metered_adc = g_sensor_data.filtered_metered_adc - g_sensor_data.baseline_metered_adc;

    // 4. Physical Amperes conversion
    // GPIO32: Direct ADC
    float v_delta_total = (g_sensor_data.delta_total_adc / ADC_MAX_VAL) * ESP32_VREF_VOLTS;
    g_sensor_data.total_current_A = fabsf(v_delta_total / ACS712_SENSITIVITY_V_PER_A);

    // GPIO33: Voltage divider compensated (x2.0)
    float v_delta_metered_pin = (g_sensor_data.delta_metered_adc / ADC_MAX_VAL) * ESP32_VREF_VOLTS;
    float v_delta_metered_sensor = v_delta_metered_pin * CH2_VOLTAGE_DIVIDER_RATIO;
    g_sensor_data.meter_current_A = fabsf(v_delta_metered_sensor / ACS712_SENSITIVITY_V_PER_A);

    // Minor residual noise clamp
    if (g_sensor_data.total_current_A < 0.006f) g_sensor_data.total_current_A = 0.0f;
    if (g_sensor_data.meter_current_A < 0.006f) g_sensor_data.meter_current_A = 0.0f;

    g_sensor_data.diff_current_A = g_sensor_data.total_current_A - g_sensor_data.meter_current_A;

    out_data = g_sensor_data;
}

#endif // GRIDGUARD_CALIBRATION_H

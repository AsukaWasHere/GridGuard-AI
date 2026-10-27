/*
 * ============================================================================
 * GridGuard-AI: Firmware Configuration
 * ============================================================================
 * Laboratory setup configuration for dual ACS712 current sensors on ESP32:
 * - GPIO32: Feeder total current sensor
 * - GPIO33: Metered branch sensor (with 10k/10k voltage divider)
 * - Wi-Fi HTTP telemetry to Flask backend (/api/readings)
 * ============================================================================
 */

#ifndef GRIDGUARD_CONFIG_H
#define GRIDGUARD_CONFIG_H

#include <Arduino.h>

// ==========================================
// 1. Wi-Fi & Backend Configuration
// ==========================================
// Update with your local Wi-Fi network SSID and Password
#define WIFI_SSID               "Galaxy S25 Ultra"
#define WIFI_PASSWORD           "12345678"

// Backend Server API Endpoint
// NOTE: Set to your laptop's Wi-Fi LAN IPv4 address (e.g. 172.18.117.43)
// Never use localhost or 127.0.0.1 on the ESP32!
#define BACKEND_API_URL         "http://192.168.56.1:5000/api/readings"

// HTTP Network Timeout (ms)
#define HTTP_TIMEOUT_MS         2500

// Telemetry reporting interval in milliseconds (1 Hz default)
#define TELEMETRY_INTERVAL_MS   1000

// Maximum failed transmissions before applying temporary backoff
#define MAX_FAILED_RETRIES      5
#define RETRY_BACKOFF_MS        3000

// ==========================================
// 2. Hardware Pin Definitions
// ==========================================
// Channel 1: Total Feeder Current Sensor (Direct to GPIO32)
#define PIN_SENSOR_TOTAL        32

// Channel 2: Metered Branch Current Sensor (via 10k/10k voltage divider to GPIO33)
#define PIN_SENSOR_METER        33

// Indicator LEDs (Blue LEDs with 1k series resistors)
#define PIN_LED_ALERT           2    // Onboard / External alert LED (flashes on theft)
#define PIN_LED_STATUS          4    // Link status LED (solid ON when Wi-Fi connected)

// ==========================================
// 3. Sensor & ADC Specifications
// ==========================================
#define ADC_RESOLUTION_BITS     12                  // 12-bit ADC (0 - 4095)
#define ADC_MAX_VAL             4095.0f
#define ESP32_VREF_VOLTS        3.30f               // Nominal ADC full scale reference voltage

// ACS712-05B Nominal Sensitivity: 185 mV / A (0.185 V/A)
#define ACS712_SENSITIVITY_V_PER_A 0.185f

// Voltage Divider on Channel 2 (GPIO33):
// R1 = 10k, R2 = 10k -> Divider ratio = R2 / (R1 + R2) = 0.5
// Voltage at pin = Sensor Voltage * 0.5
// Compensation multiplier: 1.0 / 0.5 = 2.0
#define CH2_VOLTAGE_DIVIDER_RATIO  2.000f

// Number of ADC samples per measurement for oversampling noise mitigation
#define OVERSAMPLE_COUNT        120

// Exponential Moving Average filter alpha (0.0 < alpha <= 1.0)
// 0.30 provides responsive transient tracking while damping electrical noise
#define EMA_ALPHA               0.30f

// Zero-current tare sample count
#define CALIB_SAMPLE_COUNT      400

#endif // GRIDGUARD_CONFIG_H

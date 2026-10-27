/*
 * ============================================================================
 * GridGuard-AI: ESP32 Dual-Sensor Power Theft Detection Firmware
 * ============================================================================
 * Hardware: ESP32-WROOM DevKit
 * Sensor 1 (Feeder): ACS712-05B on GPIO32 (Direct ADC)
 * Sensor 2 (Metered): ACS712-05B on GPIO33 (via 10k/10k voltage divider)
 *
 * Operational Telemetry:
 * - Samples dual sensors with 120-sample oversampling and EMA smoothing (alpha=0.30)
 * - Transmits actual raw ADC counts and baseline deltas via HTTP POST to Flask
 * - Receives Random Forest classification from backend and services alert LEDs
 * - Selectable Serial Logging mode (CSV for dataset collection vs Human diagnostics)
 * ============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "config.h"
#include "calibration.h"

// Telemetry state
static uint32_t last_telemetry_time_ms = 0;
static uint32_t last_wifi_check_time_ms = 0;
static uint32_t last_led_toggle_time_ms = 0;
static bool alert_led_state = false;
static bool active_theft_alert = false;

// Statistics and error backoff tracking
static uint32_t total_transmissions = 0;
static uint32_t failed_transmissions = 0;
static uint8_t consecutive_http_failures = 0;
static uint32_t backoff_until_ms = 0;

// Serial output mode: 0 = Human Text, 1 = Machine CSV
static uint8_t serial_mode = 0; 
static char current_session_id[32] = "esp32_hardware";

void setup_wifi() {
    Serial.printf("\n[WiFi] Connecting to: %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint8_t retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) {
        delay(400);
        Serial.print(".");
        digitalWrite(PIN_LED_STATUS, !digitalRead(PIN_LED_STATUS));
        retries++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected successfully!");
        Serial.printf("[WiFi] IP Address    : %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("[WiFi] Target Backend: %s\n\n", BACKEND_API_URL);
        digitalWrite(PIN_LED_STATUS, HIGH);
    } else {
        Serial.println("\n[WiFi] Connection timeout. Operating in offline/serial mode; will reconnect in background.");
        digitalWrite(PIN_LED_STATUS, LOW);
    }
}

void check_wifi_connection() {
    static uint32_t lastReconnectAttemptMs = 0;
    static bool wasConnected = false;

    uint32_t now = millis();
    wl_status_t status = WiFi.status();

    // Already connected: no reconnect required.
    if (status == WL_CONNECTED) {
        if (!wasConnected && serial_mode == 0) {
            Serial.printf(
                "[WiFi] Connected. IP: %s\n",
                WiFi.localIP().toString().c_str()
            );
        }

        wasConnected = true;
        return;
    }

    wasConnected = false;

    // Retry at most once every 15 seconds.
    if (now - lastReconnectAttemptMs < 15000) {
        return;
    }

    lastReconnectAttemptMs = now;

    if (serial_mode == 0) {
        Serial.printf(
            "[WiFi] Not connected (status %d). Requesting reconnect...\n",
            static_cast<int>(status)
        );
    }

    // Do not call WiFi.disconnect() immediately before reconnect.
    // Avoid repeatedly issuing reconnect requests while connecting.
    if (status == WL_DISCONNECTED) {
        WiFi.reconnect();
    }

}

/**
 * Transmits real sensor readings to Flask backend via HTTP POST.
 * Returns true if accepted (HTTP 200/201), false otherwise.
 */
bool send_sensor_readings(const SensorData &data) {
    if (WiFi.status() != WL_CONNECTED) {
        return false;
    }

    // Apply backoff if consecutive failures occurred
    if (millis() < backoff_until_ms) {
        return false;
    }

    HTTPClient http;
    http.begin(BACKEND_API_URL);
    http.setTimeout(HTTP_TIMEOUT_MS);
    http.addHeader("Content-Type", "application/json");

    // Construct JSON payload with real sensor measurements
    char json_payload[384];
    snprintf(json_payload, sizeof(json_payload),
        "{"
        "\"timestamp_ms\":%lu,"
        "\"session_id\":\"%s\","
        "\"total_adc\":%.2f,"
        "\"metered_adc\":%.2f,"
        "\"delta_total\":%.2f,"
        "\"delta_metered\":%.2f,"
        "\"data_source\":\"hardware\""
        "}",
        millis(),
        current_session_id,
        data.filtered_total_adc,
        data.filtered_metered_adc,
        data.delta_total_adc,
        data.delta_metered_adc
    );

    int http_code = http.POST(json_payload);
    bool success = false;

    if (http_code > 0) {
        if (http_code == HTTP_CODE_OK || http_code == HTTP_CODE_CREATED) {
            success = true;
            total_transmissions++;
            consecutive_http_failures = 0;

            // Parse response for predicted status to update local LEDs
            String response = http.getString();
            if (response.indexOf("POSSIBLE_THEFT") >= 0 || response.indexOf("POSSIBLE_BYPASS") >= 0) {
                active_theft_alert = true;
            } else if (response.indexOf("NORMAL") >= 0) {
                active_theft_alert = false;
            }

            if (serial_mode == 0) {
                Serial.printf("  ↳ [HTTP %d] Telemetry ingested. Backend classified: %s\n", 
                              http_code, active_theft_alert ? "POSSIBLE_THEFT ⚠️" : "NORMAL ✅");
            }
        } else {
            consecutive_http_failures++;
            failed_transmissions++;
            if (serial_mode == 0) Serial.printf("[HTTP] Server returned error status: %d\n", http_code);
        }
    } else {
        consecutive_http_failures++;
        failed_transmissions++;
        if (serial_mode == 0) Serial.printf("[HTTP] POST failed, error: %s\n", http.errorToString(http_code).c_str());
    }

    // Apply exponential backoff if backend unreachable
    if (consecutive_http_failures >= MAX_FAILED_RETRIES) {
        backoff_until_ms = millis() + RETRY_BACKOFF_MS;
        if (serial_mode == 0) Serial.printf("[HTTP] Bounded backoff: Pausing transmissions for %d ms\n", RETRY_BACKOFF_MS);
    }

    http.end();
    return success;
}

void update_led_indicators(bool alert_active) {
    if (alert_active) {
        // Fast flashing warning pattern (100ms cycle)
        if (millis() - last_led_toggle_time_ms >= 100) {
            last_led_toggle_time_ms = millis();
            alert_led_state = !alert_led_state;
            digitalWrite(PIN_LED_ALERT, alert_led_state);
        }
    } else {
        digitalWrite(PIN_LED_ALERT, LOW);
    }

    // Status LED reflects Wi-Fi health
    digitalWrite(PIN_LED_STATUS, (WiFi.status() == WL_CONNECTED) ? HIGH : LOW);
}

void handle_serial_commands() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        if (cmd.equalsIgnoreCase("MODE_CSV") || cmd.equalsIgnoreCase("C")) {
            serial_mode = 1;
            Serial.println("[MODE] Switched to CSV format output.");
        } else if (cmd.equalsIgnoreCase("MODE_TEXT") || cmd.equalsIgnoreCase("T")) {
            serial_mode = 0;
            Serial.println("[MODE] Switched to Human Text diagnostic output.");
        } else if (cmd.equalsIgnoreCase("CALIBRATE") || cmd.equalsIgnoreCase("R")) {
            calibrate_sensors();
        } else if (cmd.startsWith("SESSION ")) {
            String new_sess = cmd.substring(8);
            new_sess.trim();
            if (new_sess.length() > 0) {
                strncpy(current_session_id, new_sess.c_str(), sizeof(current_session_id) - 1);
                Serial.printf("[SESSION] Updated session_id to: %s\n", current_session_id);
            }
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n=======================================================");
    Serial.println("           GRIDGUARD-AI: ESP32 FIRMWARE START          ");
    Serial.println("=======================================================");
    Serial.println("Commands available in Serial Monitor:");
    Serial.println("  'MODE_CSV'   or 'C' : Switch to machine CSV output mode");
    Serial.println("  'MODE_TEXT'  or 'T' : Switch to human diagnostic logs");
    Serial.println("  'CALIBRATE'  or 'R' : Re-run zero-current tare calibration");
    Serial.println("  'SESSION <id>'      : Set session identifier");
    Serial.println("=======================================================\n");

    // Pin configuration
    pinMode(PIN_SENSOR_TOTAL, INPUT);
    pinMode(PIN_SENSOR_METER, INPUT);
    pinMode(PIN_LED_ALERT, OUTPUT);
    pinMode(PIN_LED_STATUS, OUTPUT);

    digitalWrite(PIN_LED_ALERT, LOW);
    digitalWrite(PIN_LED_STATUS, LOW);

    // ADC Attenuation (11dB gives maximum input voltage range up to ~3.3V)
    analogSetAttenuation(ADC_11db);

    // Zero-Current Baseline Tare Calibration
    calibrate_sensors();

    // Wi-Fi Initialization
    setup_wifi();

    Serial.println("[SYSTEM] Setup complete. Entering continuous measurement loop...\n");
}

void loop() {
    uint32_t now = millis();

    // Check for incoming serial commands
    handle_serial_commands();

    // Periodic Wi-Fi connection heartbeat every 10 seconds
    if (now - last_wifi_check_time_ms >= 10000) {
        last_wifi_check_time_ms = now;
        check_wifi_connection();
    }

    // Main telemetry acquisition loop (default 1000ms / 1 Hz)
    if (now - last_telemetry_time_ms >= TELEMETRY_INTERVAL_MS) {
        last_telemetry_time_ms = now;

        // 1. Acquire oversampled and EMA-filtered ADC readings
        SensorData data;
        read_sensor_measurements(data);

        // 2. Serial Output Formatting
        if (serial_mode == 1) {
            // Machine CSV format for collection: CSV: ts,total_adc,metered_adc,delta_total,delta_metered
            Serial.printf("CSV: %lu,%.2f,%.2f,%.2f,%.2f\n",
                          now,
                          data.filtered_total_adc,
                          data.filtered_metered_adc,
                          data.delta_total_adc,
                          data.delta_metered_adc);
        } else {
            // Human diagnostic format
            Serial.printf("[SENSOR] Feeder: %6.1f ADC (Δ%+5.1f) | Metered: %6.1f ADC (Δ%+5.1f) | Est Current: Feeder: %.3fA, Meter: %.3fA (Δ%+.3fA)\n",
                          data.filtered_total_adc,
                          data.delta_total_adc,
                          data.filtered_metered_adc,
                          data.delta_metered_adc,
                          data.total_current_A,
                          data.meter_current_A,
                          data.diff_current_A);
        }

        // 3. Transmit real sensor readings to Flask backend via HTTP POST
        send_sensor_readings(data);
    }

    // Service alert indicator LED
    update_led_indicators(active_theft_alert);

    delay(10);
}

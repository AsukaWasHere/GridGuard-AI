"""
GridGuard-AI: USB Serial Bridge Unit Tests
==========================================
Validates:
1. Parsing of firmware CSV lines with 'CSV:' prefix
2. Parsing of raw comma-separated lines
3. Rejection of diagnostic/log lines ([WiFi], [CALIB], [MODE], etc.)
4. Rejection of malformed values, NaN/Inf, and incomplete fields
5. Schema parity between bridge payload and Flask /api/readings endpoint
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from serial_to_flask_bridge import parse_serial_line
from app import app, readings_buffer, alerts_buffer, telemetry_state, device_windows, load_ml_model


def test_parse_valid_csv_prefixed_line():
    line = "CSV: 2308881,1086.79,0.00,-1831.15,-1382.69"
    parsed = parse_serial_line(line)
    assert parsed is not None
    assert parsed["timestamp_ms"] == 2308881
    assert parsed["total_adc"] == 1086.79
    assert parsed["metered_adc"] == 0.0
    assert parsed["delta_total"] == -1831.15
    assert parsed["delta_metered"] == -1382.69


def test_parse_valid_raw_comma_line():
    line = "123456,3102.50,1551.25,5.50,2.10"
    parsed = parse_serial_line(line)
    assert parsed is not None
    assert parsed["timestamp_ms"] == 123456
    assert parsed["total_adc"] == 3102.50
    assert parsed["metered_adc"] == 1551.25
    assert parsed["delta_total"] == 5.50
    assert parsed["delta_metered"] == 2.10


def test_reject_diagnostic_and_status_lines():
    diagnostic_lines = [
        "[WiFi] Connecting to: Galaxy S25 Ultra",
        "[WiFi] Not connected (status 1). Requesting reconnect...",
        "[CALIB] Baseline Tare Complete:",
        "[MODE] Switched to CSV format output.",
        "[SYSTEM] Setup complete. Entering continuous measurement loop...",
        "[SENSOR] Feeder: 1214.2 ADC | Metered: 0.0 ADC",
        "=======================================================",
        "",
        "   ",
        "Commands available in Serial Monitor:"
    ]
    for line in diagnostic_lines:
        assert parse_serial_line(line) is None, f"Expected None for: {line}"


def test_reject_malformed_and_nan_lines():
    invalid_lines = [
        "CSV: 1000,not_a_number,1550,0,0",
        "CSV: 1000,nan,1550,0,0",
        "CSV: 1000,inf,1550,0,0",
        "CSV: 1000,3100,1550",           # Insufficient columns
        "1000,3100,1550,0",              # Only 4 columns
        "CSV: ,,,,",
    ]
    for line in invalid_lines:
        assert parse_serial_line(line) is None, f"Expected None for: {line}"


def test_bridge_payload_accepted_by_flask():
    """Confirms that the output of parse_serial_line matches Flask's /api/readings requirements."""
    load_ml_model()
    app.config["TESTING"] = True
    readings_buffer.clear()
    alerts_buffer.clear()
    device_windows.clear()

    line = "CSV: 50000,2842.0,1345.0,-75.0,-37.0"
    parsed = parse_serial_line(line)
    assert parsed is not None

    payload = {
        "timestamp_ms": parsed["timestamp_ms"],
        "session_id": "test_bridge_session",
        "total_adc": parsed["total_adc"],
        "metered_adc": parsed["metered_adc"],
        "delta_total": parsed["delta_total"],
        "delta_metered": parsed["delta_metered"],
        "data_source": "hardware"
    }

    with app.test_client() as client:
        resp = client.post("/api/readings", json=payload)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["data_source"] == "hardware"
        assert data["raw_measurements"]["total_adc"] == 2842.0
        assert data["raw_measurements"]["metered_adc"] == 1345.0

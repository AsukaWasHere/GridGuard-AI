"""
GridGuard-AI: Backend & Model Inference Test Suite
===================================================
Validates:
1. System health check and heartbeat status
2. Ingestion of real ADC sensor readings (total_adc, metered_adc, delta_total, delta_metered)
3. Startup warm-up handling (WARMING_UP status before full rolling window)
4. Random Forest model inference on complete feature window (NORMAL and POSSIBLE_THEFT)
5. Backward compatibility for legacy telemetry payloads
6. Rejection of missing fields, NaN/Inf, and invalid data sources
7. Debounced discrepancy alert firing
8. Latest and history retrieval endpoints (/api/readings/latest, /api/readings/history)
"""

import os
import sys
import math
import pytest
import numpy as np

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from app import app, readings_buffer, alerts_buffer, telemetry_state, device_windows, load_ml_model


@pytest.fixture
def client():
    app.config["TESTING"] = True
    readings_buffer.clear()
    alerts_buffer.clear()
    device_windows.clear()
    telemetry_state["total_readings_received"] = 0
    telemetry_state["hardware_readings_count"] = 0
    telemetry_state["simulated_readings_count"] = 0
    telemetry_state["last_hardware_seen"] = None
    telemetry_state["last_simulation_seen"] = None
    telemetry_state["last_reading"] = None
    telemetry_state["consecutive_theft_count"] = 0

    load_ml_model()

    with app.test_client() as client:
        yield client


def test_health_check_initial(client):
    """Verifies /api/health returns 200 OK and expected telemetry schema."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["total_readings"] == 0
    assert data["esp32_connection_status"] == "OFFLINE"
    assert data["model_loaded"] is True
    assert data["window_size"] == 5


def test_latest_reading_standby(client):
    """Verifies /api/readings/latest returns clean fallback when no readings ingested yet."""
    for endpoint in ["/api/readings/latest", "/api/latest"]:
        resp = client.get(endpoint)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["latest"]["prediction"] == "STANDBY"
        assert data["esp32_connection_status"] == "OFFLINE"


def test_post_adc_reading_warmup(client):
    """Verifies that initial samples return WARMING_UP status until window is full."""
    payload = {
        "timestamp_ms": 1000,
        "session_id": "warmup_test",
        "total_adc": 3108.0,
        "metered_adc": 1554.0,
        "delta_total": 6.0,
        "delta_metered": 3.0,
        "data_source": "hardware"
    }

    # Samples 1 to 4 should be WARMING_UP
    for i in range(1, 5):
        payload["timestamp_ms"] = i * 1000
        resp = client.post("/api/readings", json=payload)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["status"] == "WARMING_UP"
        assert data["predicted_status"] == "WARMING_UP"
        assert data["extracted_features"] is None
        assert data["confidence"] is None

    # Sample 5 should complete the window and trigger Random Forest prediction!
    payload["timestamp_ms"] = 5000
    resp5 = client.post("/api/readings", json=payload)
    assert resp5.status_code == 201
    data5 = resp5.get_json()
    assert data5["status"] == "success"
    assert data5["predicted_status"] in ["NORMAL", "POSSIBLE_THEFT"]
    assert data5["extracted_features"] is not None
    assert "mean_total_adc" in data5["extracted_features"]
    assert "delta_diff" in data5["extracted_features"]
    assert data5["confidence"] is not None
    assert 0.0 <= data5["confidence"] <= 1.0


def test_post_normal_operation_classification(client):
    """Simulates sustained NORMAL operation (LED ON, Motor OFF) and checks prediction."""
    session = "test_normal_session"
    # Send 6 normal samples using hardware-representative readings
    for i in range(6):
        payload = {
            "timestamp_ms": 10000 + i * 1000,
            "session_id": session,
            "total_adc": 2842.0 + (i % 2),
            "metered_adc": 1345.0 + (i % 2),
            "delta_total": -75.0 + (i % 2),
            "delta_metered": -37.0 + (i % 2),
            "data_source": "hardware"
        }
        resp = client.post("/api/readings", json=payload)
        assert resp.status_code == 201

    latest_resp = client.get("/api/readings/latest")
    data = latest_resp.get_json()["latest"]
    assert data["predicted_status"] == "NORMAL"
    assert data["prediction"] == "NORMAL"
    assert isinstance(data["confidence"], (int, float))
    assert math.isfinite(data["confidence"])
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["data_source"] == "hardware"


def test_post_possible_theft_classification(client):
    """Simulates POSSIBLE_THEFT (LED ON, Motor ON bypassing Sensor 2)."""
    session = "test_theft_session"
    # Send 7 theft samples using hardware-representative readings
    for i in range(7):
        payload = {
            "timestamp_ms": 20000 + i * 1000,
            "session_id": session,
            "total_adc": 3070.0 + (i % 3),
            "metered_adc": 1350.0,
            "delta_total": 155.0 + (i % 3),
            "delta_metered": -30.0,
            "data_source": "hardware"
        }
        resp = client.post("/api/readings", json=payload)
        assert resp.status_code == 201

    latest_resp = client.get("/api/readings/latest")
    data = latest_resp.get_json()["latest"]
    assert data["predicted_status"] == "POSSIBLE_THEFT"
    assert data["prediction"] == "POSSIBLE_THEFT"
    assert isinstance(data["confidence"], (int, float))
    assert math.isfinite(data["confidence"])
    assert 0.0 <= data["confidence"] <= 1.0

    # Check alert was triggered
    alerts_resp = client.get("/api/alerts")
    alerts_data = alerts_resp.get_json()
    assert alerts_data["count"] >= 1
    assert alerts_data["alerts"][0]["type"] == "POSSIBLE_THEFT"


def test_post_missing_fields(client):
    """Verifies rejection with 400 when required fields are missing."""
    incomplete_payload = {
        "timestamp_ms": 125000,
        "session_id": "test"
    }
    resp = client.post("/api/readings", json=incomplete_payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "Missing required" in data["message"]


def test_post_invalid_data_source(client):
    """Verifies rejection of invalid data_source value."""
    payload = {
        "timestamp_ms": 125000,
        "total_adc": 3102.0,
        "metered_adc": 1551.0,
        "delta_total": 0.0,
        "delta_metered": 0.0,
        "data_source": "invalid_source"
    }
    resp = client.post("/api/readings", json=payload)
    assert resp.status_code == 400
    assert "data_source must be 'hardware' or 'simulation'" in resp.get_json()["message"]


def test_post_nan_or_inf_rejected(client):
    """Verifies that NaN and Inf values are rejected."""
    payload = {
        "timestamp_ms": 125000,
        "total_adc": float("nan"),
        "metered_adc": 1551.0,
        "delta_total": 0.0,
        "delta_metered": 0.0,
        "data_source": "hardware"
    }
    resp = client.post("/api/readings", json=payload)
    assert resp.status_code == 400
    assert "cannot be NaN or Inf" in resp.get_json()["message"]


def test_backward_compatibility_legacy_schema(client):
    """Verifies backward compatibility with legacy schema (total_current_A, etc.)."""
    payload = {
        "timestamp_ms": 125000,
        "total_current_A": 0.25,
        "meter_current_A": 0.24,
        "difference_A": 0.01,
        "data_source": "hardware"
    }
    resp = client.post("/api/readings", json=payload)
    assert resp.status_code == 201


def test_clear_alerts(client):
    """Verifies clearing the alert buffer."""
    for i in range(5):
        client.post("/api/readings", json={
            "timestamp_ms": 1000 + i * 1000,
            "session_id": "alert_clear_test",
            "total_adc": 3220.0,
            "metered_adc": 1551.0,
            "delta_total": 118.0,
            "delta_metered": 0.0,
            "data_source": "hardware"
        })

    clear_resp = client.post("/api/alerts/clear")
    assert clear_resp.status_code == 200
    assert len(client.get("/api/alerts").get_json()["alerts"]) == 0


def test_history_endpoints(client):
    """Verifies /api/readings/history and alias /api/history."""
    for i in range(10):
        client.post("/api/readings", json={
            "timestamp_ms": i * 1000,
            "session_id": "hist_test",
            "total_adc": 3108.0,
            "metered_adc": 1554.0,
            "delta_total": 6.0,
            "delta_metered": 3.0,
            "data_source": "simulation"
        })

    for endpoint in ["/api/readings/history?limit=5", "/api/history?limit=5"]:
        resp = client.get(endpoint)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 5
        assert len(data["readings"]) == 5

"""
GridGuard-AI Backend API
=========================
Flask REST API for real-time power theft detection telemetry, Random Forest model
inference, window-based feature extraction, alert tracking, and sensor data logging.

Endpoints:
- POST /api/readings          : Ingest sensor measurements & execute Random Forest inference
- GET  /api/readings/latest   : Return latest telemetry, predicted status & ESP32 connection state
- GET  /api/latest            : Alias for /api/readings/latest (dashboard compatibility)
- GET  /api/readings/history  : Return recent measurements and event log
- GET  /api/history           : Alias for /api/readings/history (dashboard compatibility)
- GET  /api/health            : System health check & heartbeat telemetry
- GET  /api/alerts            : Retrieve detected discrepancy alerts
- POST /api/alerts/clear      : Clear alert history
- POST /api/threshold         : Read or update bypass detection threshold
- POST /api/simulate          : Trigger simulated sensor reading (normal/bypass/theft)
- GET  /                      : Serve the web dashboard
"""

import os
import sys
import time
import math
import json
from collections import deque, defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

# Ensure root is in sys.path to import ml modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from ml.feature_extractor import (
        FEATURE_VERSION,
        FEATURE_NAMES,
        DEFAULT_WINDOW_SIZE,
        CLASS_MAPPING,
        LABEL_NORMAL,
        LABEL_POSSIBLE_THEFT,
        compute_window_features
    )
except ImportError:
    # Fallback if running directly
    FEATURE_VERSION = "1.0.0"
    FEATURE_NAMES = [
        "mean_total_adc",
        "mean_metered_adc",
        "mean_delta_total",
        "mean_delta_metered",
        "delta_diff",
        "std_total_adc",
        "std_metered_adc"
    ]
    DEFAULT_WINDOW_SIZE = 5
    LABEL_NORMAL = "NORMAL"
    LABEL_POSSIBLE_THEFT = "POSSIBLE_THEFT"
    CLASS_MAPPING = {0: "NORMAL", 1: "POSSIBLE_THEFT"}

# Configure Flask app with static folder pointing to dashboard
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path="")
CORS(app)

# In-memory storage buffers
MAX_HISTORY = 1000
readings_buffer = deque(maxlen=MAX_HISTORY)
alerts_buffer = deque(maxlen=200)

# Per-device / per-session rolling windows for feature extraction
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
device_windows = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# Server and telemetry state
SERVER_START_TIME = time.time()
telemetry_state = {
    "total_readings_received": 0,
    "hardware_readings_count": 0,
    "simulated_readings_count": 0,
    "last_hardware_seen": None,
    "last_simulation_seen": None,
    "last_reading": None,
    "consecutive_theft_count": 0,
    "alert_threshold_A": 0.035,      # 35mA threshold
    "consecutive_debounce_n": 3,     # Require 3 consecutive samples before alert
    "model_name": "GridGuard-AI Random Forest Classifier",
    "model_version": "2.0.0",
    "model_loaded": False,
    "model_data_source": "unknown"
}

READINGS_LOG_FILE = os.path.join(DATA_DIR, "readings_log.jsonl")

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
rf_model = None
feature_config = None

def load_ml_model():
    global rf_model, feature_config
    candidate_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "models"))
    ]
    
    for base_dir in candidate_paths:
        model_file = os.path.join(base_dir, "random_forest.joblib")
        config_file = os.path.join(base_dir, "feature_config.json")
        if os.path.exists(model_file) and os.path.exists(config_file):
            try:
                rf_model = joblib.load(model_file)
                with open(config_file, "r", encoding="utf-8") as f:
                    feature_config = json.load(f)
                telemetry_state["model_loaded"] = True
                telemetry_state["model_version"] = feature_config.get("model_version", "2.0.0")
                telemetry_state["model_data_source"] = feature_config.get("data_source", "unknown")
                print(f"[*] Successfully loaded Random Forest model from {model_file}")
                return True
            except Exception as e:
                print(f"[!] Error loading model from {model_file}: {e}")
    
    print("[!] Warning: Random Forest model artifact not found.")
    telemetry_state["model_loaded"] = False
    return False

# Attempt initial model load
load_ml_model()


def persist_reading(reading_dict):
    """Appends reading to disk in JSONL format."""
    try:
        with open(READINGS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(reading_dict) + "\n")
    except Exception as e:
        app.logger.error(f"Error persisting reading: {e}")


def get_connection_status(last_seen_timestamp, stale_seconds=10.0, offline_seconds=30.0):
    """Calculates connection status based on heartbeat."""
    if last_seen_timestamp is None:
        return "OFFLINE"
    elapsed = time.time() - last_seen_timestamp
    if elapsed < stale_seconds:
        return "CONNECTED"
    elif elapsed < offline_seconds:
        return "STALE"
    else:
        return "OFFLINE"


# ---------------------------------------------------------------------------
# Routes: Static Dashboard
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(DASHBOARD_DIR, path)


# ---------------------------------------------------------------------------
# Route: GET /api/health
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    now = time.time()
    esp32_conn = get_connection_status(telemetry_state["last_hardware_seen"])
    sim_conn = get_connection_status(telemetry_state["last_simulation_seen"], stale_seconds=5.0, offline_seconds=15.0)

    active_source = "none"
    if esp32_conn == "CONNECTED":
        active_source = "hardware"
    elif sim_conn == "CONNECTED":
        active_source = "simulation"

    return jsonify({
        "status": "ok",
        "service": "GridGuard-AI Backend",
        "uptime_seconds": round(now - SERVER_START_TIME, 1),
        "total_readings": telemetry_state["total_readings_received"],
        "hardware_readings": telemetry_state["hardware_readings_count"],
        "simulated_readings": telemetry_state["simulated_readings_count"],
        "esp32_connection_status": esp32_conn,
        "simulation_status": sim_conn,
        "active_source": active_source,
        "alerts_count": len(alerts_buffer),
        "model_loaded": telemetry_state["model_loaded"],
        "model_name": telemetry_state["model_name"],
        "model_version": telemetry_state["model_version"],
        "model_data_source": telemetry_state["model_data_source"],
        "window_size": WINDOW_SIZE,
        "server_timestamp_ms": int(now * 1000)
    }), 200


# ---------------------------------------------------------------------------
# Route: POST /api/readings
# ---------------------------------------------------------------------------
@app.route("/api/readings", methods=["POST"])
def post_readings():
    """
    Ingests sensor readings from ESP32 or Simulator.
    Computes rolling window features and evaluates using loaded Random Forest model.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"status": "error", "message": "Malformed JSON payload"}), 400

    now = time.time()
    server_time_iso = datetime.utcnow().isoformat() + "Z"

    # Support both new ADC schema and legacy Amperes schema
    is_adc_schema = ("total_adc" in payload and "metered_adc" in payload)
    is_legacy_schema = ("total_current_A" in payload and "meter_current_A" in payload)

    if not is_adc_schema and not is_legacy_schema:
        return jsonify({
            "status": "error",
            "message": "Missing required sensor fields. Expected ('total_adc', 'metered_adc', 'delta_total', 'delta_metered') or legacy current fields."
        }), 400

    # Validate data_source
    data_source = payload.get("data_source")
    if data_source is None:
        return jsonify({"status": "error", "message": "Field 'data_source' is required ('hardware' or 'simulation')"}), 400
    data_source = str(data_source).strip().lower()
    if data_source not in ["hardware", "simulation"]:
        return jsonify({"status": "error", "message": "data_source must be 'hardware' or 'simulation'"}), 400

    # Validate timestamp
    timestamp_ms = payload.get("timestamp_ms")
    if timestamp_ms is None:
        return jsonify({"status": "error", "message": "Field 'timestamp_ms' is required"}), 400
    try:
        timestamp_ms = int(timestamp_ms)
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Field 'timestamp_ms' must be an integer"}), 400

    session_id = str(payload.get("session_id", "default")).strip()

    # Process ADC schema
    if is_adc_schema:
        try:
            total_adc = float(payload["total_adc"])
            metered_adc = float(payload["metered_adc"])
            delta_total = float(payload.get("delta_total", 0.0))
            delta_metered = float(payload.get("delta_metered", 0.0))
        except (ValueError, TypeError) as e:
            return jsonify({"status": "error", "message": f"Invalid numeric value in ADC fields: {e}"}), 400

        # Check for NaN / Inf
        for val, name in [(total_adc, "total_adc"), (metered_adc, "metered_adc"), 
                          (delta_total, "delta_total"), (delta_metered, "delta_metered")]:
            if math.isnan(val) or math.isinf(val):
                return jsonify({"status": "error", "message": f"Field '{name}' cannot be NaN or Inf"}), 400

        # Convert to estimated current (ACS712 185mV/A, ESP32 3.3V/4095) for dashboard display
        # GPIO32: Direct ADC
        total_current_A = round(max(0.0, (delta_total / 4095.0 * 3.30) / 0.185), 4)
        # GPIO33: Voltage divider x2.0
        meter_current_A = round(max(0.0, (delta_metered / 4095.0 * 3.30 * 2.00) / 0.185), 4)
        difference_A = round(total_current_A - meter_current_A, 4)

    else:
        # Legacy schema
        try:
            total_current_A = float(payload["total_current_A"])
            meter_current_A = float(payload["meter_current_A"])
            difference_A = float(payload.get("difference_A", total_current_A - meter_current_A))
        except (ValueError, TypeError) as e:
            return jsonify({"status": "error", "message": f"Invalid numeric value in current fields: {e}"}), 400

        for val, name in [(total_current_A, "total_current_A"), (meter_current_A, "meter_current_A"), (difference_A, "difference_A")]:
            if math.isnan(val) or math.isinf(val):
                return jsonify({"status": "error", "message": f"Field '{name}' cannot be NaN or Inf"}), 400

        # Approximate ADC counts from currents
        delta_total = (total_current_A * 0.185 / 3.30) * 4095.0
        delta_metered = (meter_current_A * 0.185 / 2.00 / 3.30) * 4095.0
        total_adc = 3102.0 + delta_total
        metered_adc = 1551.0 + delta_metered

    # Buffer management for rolling window features
    buffer_key = f"{data_source}_{session_id}"
    window_buf = device_windows[buffer_key]
    
    sample_item = {
        "total_adc": total_adc,
        "metered_adc": metered_adc,
        "delta_total": delta_total,
        "delta_metered": delta_metered
    }
    window_buf.append(sample_item)

    # Startup Warm-up handling
    is_warming_up = (len(window_buf) < WINDOW_SIZE)
    extracted_features = None
    predicted_status = "WARMING_UP"
    confidence = None

    if is_warming_up:
        predicted_status = "WARMING_UP"
    else:
        # Extract features using shared mathematical implementation
        extracted_features = compute_window_features(list(window_buf))

        # Check if model is loaded
        if not telemetry_state["model_loaded"] or rf_model is None:
            # Attempt reloading
            if not load_ml_model():
                return jsonify({
                    "status": "error",
                    "message": "Random Forest model is not loaded on backend. Inference unavailable."
                }), 503

        # Prepare ordered feature vector
        feature_vector = np.array([[extracted_features[f] for f in FEATURE_NAMES]])

        # Execute model prediction
        try:
            pred_idx = int(rf_model.predict(feature_vector)[0])
            predicted_status = CLASS_MAPPING.get(pred_idx, LABEL_NORMAL)

            if hasattr(rf_model, "predict_proba"):
                probas = rf_model.predict_proba(feature_vector)[0]
                confidence = round(float(np.max(probas)), 4)
            else:
                confidence = 1.0
        except Exception as e:
            app.logger.error(f"Inference error: {e}")
            return jsonify({
                "status": "error",
                "message": f"Model inference execution failed: {str(e)}"
            }), 500

    # Rule-based cross check
    rule_status = "NORMAL"
    if difference_A >= telemetry_state["alert_threshold_A"]:
        rule_status = "POSSIBLE_THEFT"

    # Debounce alert logic
    if predicted_status == LABEL_POSSIBLE_THEFT or (is_warming_up and rule_status == "POSSIBLE_THEFT"):
        telemetry_state["consecutive_theft_count"] += 1
        if telemetry_state["consecutive_theft_count"] >= telemetry_state["consecutive_debounce_n"]:
            alert = {
                "alert_id": len(alerts_buffer) + 1,
                "timestamp_iso": server_time_iso,
                "type": "POSSIBLE_THEFT",
                "severity": "HIGH" if difference_A > 0.080 else "MEDIUM",
                "message": f"Power discrepancy detected: {round(difference_A * 1000, 1)} mA unaccounted branch current.",
                "total_current_A": total_current_A,
                "meter_current_A": meter_current_A,
                "difference_A": difference_A,
                "total_adc": total_adc,
                "metered_adc": metered_adc,
                "prediction": predicted_status,
                "confidence": confidence,
                "data_source": data_source
            }
            if not alerts_buffer or (now - alerts_buffer[-1].get("_created_time", 0)) > 5.0:
                alert["_created_time"] = now
                alerts_buffer.append(alert)
    else:
        if predicted_status == LABEL_NORMAL:
            telemetry_state["consecutive_theft_count"] = 0

    # Build persistent and telemetry record
    reading_record = {
        "id": telemetry_state["total_readings_received"] + 1,
        "timestamp_ms": timestamp_ms,
        "server_time_iso": server_time_iso,
        "session_id": session_id,
        "data_source": data_source,
        "raw_measurements": {
            "total_adc": round(total_adc, 2),
            "metered_adc": round(metered_adc, 2),
            "delta_total": round(delta_total, 2),
            "delta_metered": round(delta_metered, 2)
        },
        # Display compatibility fields
        "total_adc": round(total_adc, 2),
        "metered_adc": round(metered_adc, 2),
        "delta_total": round(delta_total, 2),
        "delta_metered": round(delta_metered, 2),
        "total_current_A": total_current_A,
        "meter_current_A": meter_current_A,
        "difference_A": difference_A,
        "extracted_features": extracted_features,
        "prediction": predicted_status,
        "predicted_status": predicted_status,
        "confidence": confidence,
        "rule_based_status": rule_status,
        "model_name": telemetry_state["model_name"],
        "model_version": telemetry_state["model_version"],
        "window_status": f"{len(window_buf)}/{WINDOW_SIZE}",
        "esp32_ip": request.remote_addr
    }

    telemetry_state["total_readings_received"] += 1
    if data_source == "hardware":
        telemetry_state["hardware_readings_count"] += 1
        telemetry_state["last_hardware_seen"] = now
    else:
        telemetry_state["simulated_readings_count"] += 1
        telemetry_state["last_simulation_seen"] = now

    telemetry_state["last_reading"] = reading_record
    readings_buffer.append(reading_record)
    persist_reading(reading_record)

    response_status = "WARMING_UP" if is_warming_up else "success"
    return jsonify({
        "status": response_status,
        "reading_id": reading_record["id"],
        "timestamp_ms": timestamp_ms,
        "server_time_iso": server_time_iso,
        "session_id": session_id,
        "data_source": data_source,
        "raw_measurements": reading_record["raw_measurements"],
        "extracted_features": extracted_features,
        "predicted_status": predicted_status,
        "prediction": predicted_status,
        "confidence": confidence,
        "window_status": reading_record["window_status"],
        "model_name": telemetry_state["model_name"],
        "model_version": telemetry_state["model_version"]
    }), 201


# ---------------------------------------------------------------------------
# Routes: GET Latest & History
# ---------------------------------------------------------------------------
@app.route("/api/readings/latest", methods=["GET"])
@app.route("/api/latest", methods=["GET"])
def get_latest():
    now = time.time()
    esp32_conn = get_connection_status(telemetry_state["last_hardware_seen"])
    sim_conn = get_connection_status(telemetry_state["last_simulation_seen"], stale_seconds=5.0, offline_seconds=15.0)

    reading = telemetry_state["last_reading"]
    if reading is None:
        reading = {
            "timestamp_ms": int(now * 1000),
            "total_adc": 3102.0,
            "metered_adc": 1551.0,
            "delta_total": 0.0,
            "delta_metered": 0.0,
            "total_current_A": 0.0,
            "meter_current_A": 0.0,
            "difference_A": 0.0,
            "prediction": "STANDBY",
            "predicted_status": "STANDBY",
            "confidence": 1.0,
            "rule_based_status": "STANDBY",
            "data_source": "none",
            "extracted_features": None
        }

    return jsonify({
        "status": "success",
        "latest": reading,
        "esp32_connection_status": esp32_conn,
        "simulation_status": sim_conn,
        "last_hardware_seconds_ago": round(now - telemetry_state["last_hardware_seen"], 1) if telemetry_state["last_hardware_seen"] else None,
        "alert_count": len(alerts_buffer),
        "total_readings": telemetry_state["total_readings_received"],
        "model_loaded": telemetry_state["model_loaded"],
        "server_time_ms": int(now * 1000)
    }), 200


@app.route("/api/readings/history", methods=["GET"])
@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(1, min(limit, 500))
    except ValueError:
        limit = 50

    source = request.args.get("source", "all").strip().lower()

    filtered = list(readings_buffer)
    if source in ["hardware", "simulation"]:
        filtered = [r for r in filtered if r["data_source"] == source]

    return jsonify({
        "status": "success",
        "count": min(len(filtered), limit),
        "readings": filtered[-limit:],
        "recent_alerts": list(alerts_buffer)[-20:]
    }), 200


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    return jsonify({
        "status": "success",
        "count": len(alerts_buffer),
        "alerts": list(alerts_buffer)
    }), 200


@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    alerts_buffer.clear()
    telemetry_state["consecutive_theft_count"] = 0
    return jsonify({"status": "success", "message": "Alert buffer cleared"}), 200


@app.route("/api/threshold", methods=["GET", "POST"])
def manage_threshold():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "threshold_A" in data:
            try:
                new_t = float(data["threshold_A"])
                if 0.005 <= new_t <= 5.0:
                    telemetry_state["alert_threshold_A"] = new_t
                    return jsonify({"status": "success", "threshold_A": telemetry_state["alert_threshold_A"]})
                else:
                    return jsonify({"status": "error", "message": "Threshold must be between 0.005A and 5.0A"}), 400
            except ValueError:
                return jsonify({"status": "error", "message": "Invalid numeric threshold"}), 400
    return jsonify({"threshold_A": telemetry_state["alert_threshold_A"]})


@app.route("/api/simulate", methods=["POST"])
def simulate_reading():
    """
    Injects a realistic ADC sensor reading for testing dashboard, alerts, and model.
    Accepts scenario: 'normal', 'bypass' / 'theft', 'power_loss'
    """
    import random
    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario", "normal").lower()

    tare_feeder = 3102.0
    tare_metered = 1551.0
    led_feeder = 6.5
    led_metered = 3.2

    if scenario in ["bypass", "theft"]:
        # Motor running in bypass: high delta on feeder, normal delta on metered
        motor_delta = random.uniform(85.0, 115.0)
        noise_f = random.gauss(0, 3.2)
        noise_m = random.gauss(0, 0.8)

        total_adc = round(tare_feeder + led_feeder + motor_delta + noise_f, 2)
        metered_adc = round(tare_metered + led_metered + noise_m, 2)
        delta_total = round(total_adc - tare_feeder, 2)
        delta_metered = round(metered_adc - tare_metered, 2)
    elif scenario == "power_loss":
        total_adc = round(tare_feeder + random.gauss(0, 0.5), 2)
        metered_adc = round(tare_metered + random.gauss(0, 0.5), 2)
        delta_total = round(total_adc - tare_feeder, 2)
        delta_metered = round(metered_adc - tare_metered, 2)
    else:  # normal
        noise_f = random.gauss(0, 1.0)
        noise_m = random.gauss(0, 0.8)
        total_adc = round(tare_feeder + led_feeder + noise_f, 2)
        metered_adc = round(tare_metered + led_metered + noise_m, 2)
        delta_total = round(total_adc - tare_feeder, 2)
        delta_metered = round(metered_adc - tare_metered, 2)

    sim_payload = {
        "timestamp_ms": int(time.time() * 1000),
        "session_id": "sim_session",
        "total_adc": total_adc,
        "metered_adc": metered_adc,
        "delta_total": delta_total,
        "delta_metered": delta_metered,
        "data_source": "simulation"
    }

    with app.test_request_context("/api/readings", method="POST", json=sim_payload):
        return post_readings()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] GridGuard-AI Backend starting on http://0.0.0.0:{port}")
    print(f"[*] Dashboard available at http://localhost:{port}/")
    app.run(host="0.0.0.0", port=port, debug=True)

"""
GridGuard-AI: USB Serial-to-Flask Bridge
=========================================
Bridges live physical sensor readings from an ESP32 connected over USB Serial
directly to the Flask backend (/api/readings).

Pipeline:
ESP32 + ACS712 sensors -> USB COM port -> Bridge -> Flask /api/readings -> ML inference -> Dashboard
"""

import sys
import os
import time
import math
import argparse
import signal
from typing import Optional, Dict, Any

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 115200
DEFAULT_URL = "http://127.0.0.1:5000/api/readings"
DEFAULT_TIMEOUT_SEC = 2.5


def parse_serial_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parses a raw serial line from the ESP32 firmware.
    
    Expected format:
      CSV: <timestamp_ms>,<total_adc>,<metered_adc>,<delta_total>,<delta_metered>
    Or raw comma-separated values:
      <timestamp_ms>,<total_adc>,<metered_adc>,<delta_total>,<delta_metered>
      
    Returns a dict with parsed values or None if invalid / non-sensor log.
    """
    if not line:
        return None

    line = line.strip()
    if not line:
        return None

    if line.startswith("CSV:"):
        raw = line[4:].strip()
    elif line.count(",") >= 4 and not line.startswith("["):
        raw = line
    else:
        return None

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) < 5:
        return None

    try:
        ts_ms = int(float(parts[0]))
        total_adc = float(parts[1])
        metered_adc = float(parts[2])
        delta_total = float(parts[3])
        delta_metered = float(parts[4])
    except (ValueError, TypeError):
        return None

    for v in (total_adc, metered_adc, delta_total, delta_metered):
        if math.isnan(v) or math.isinf(v):
            return None

    return {
        "timestamp_ms": ts_ms,
        "total_adc": round(total_adc, 2),
        "metered_adc": round(metered_adc, 2),
        "delta_total": round(delta_total, 2),
        "delta_metered": round(delta_metered, 2),
    }


def find_default_com_port() -> str:
    """Detects available COM ports and prefers known USB-serial devices."""
    if not HAS_SERIAL:
        return DEFAULT_PORT

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return DEFAULT_PORT

    # Check for COM3 first or CH340 / CP210x / FTDI / USB
    for p in ports:
        if p.device.upper() == "COM3":
            return p.device
    for p in ports:
        desc = (p.description or "").lower()
        if "ch340" in desc or "cp210" in desc or "usb-serial" in desc or "esp32" in desc:
            return p.device

    return ports[0].device


def run_bridge(
    port: str,
    baudrate: int = DEFAULT_BAUD,
    target_url: str = DEFAULT_URL,
    session_id: Optional[str] = None,
    http_timeout: float = DEFAULT_TIMEOUT_SEC,
    max_readings: Optional[int] = None
) -> Dict[str, int]:
    """
    Runs the serial to Flask bridge. Reads from the ESP32 and posts to Flask.
    Returns stats dict upon completion.
    """
    if not HAS_SERIAL:
        print("[ERROR] pyserial is not installed. Please run: pip install pyserial", file=sys.stderr)
        return {"forwarded": 0, "errors": 1}

    if not HAS_REQUESTS:
        print("[ERROR] requests is not installed. Please run: pip install requests", file=sys.stderr)
        return {"forwarded": 0, "errors": 1}

    if not session_id:
        session_id = f"usb_bridge_{int(time.time())}"

    stats = {
        "lines_read": 0,
        "records_parsed": 0,
        "forwarded": 0,
        "http_errors": 0,
        "parse_errors": 0
    }

    print("=" * 72)
    print("        GRIDGUARD-AI: USB SERIAL TO FLASK TELEMETRY BRIDGE        ")
    print("=" * 72)
    print(f"  COM Port      : {port} @ {baudrate} baud")
    print(f"  Backend URL   : {target_url}")
    print(f"  Session ID    : {session_id}")
    print(f"  Data Source   : hardware")
    print(f"  HTTP Timeout  : {http_timeout}s")
    print("=" * 72)

    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=1.0)
    except Exception as e:
        print(f"[FATAL] Failed to open serial port '{port}': {e}", file=sys.stderr)
        return stats

    # Give ESP32 a moment if opening toggled DTR/RTS
    time.sleep(1.5)
    ser.reset_input_buffer()

    # Command firmware to output CSV format
    print("[*] Sending 'MODE_CSV' command to ESP32...")
    try:
        ser.write(b"MODE_CSV\n")
        ser.flush()
    except Exception as e:
        print(f"[WARN] Failed to write MODE_CSV to serial: {e}")

    session = requests.Session()
    running = True

    def sig_handler(sig, frame):
        nonlocal running
        print("\n[*] Stopping bridge (interrupt received)...")
        running = False

    try:
        signal.signal(signal.SIGINT, sig_handler)
    except (ValueError, AttributeError):
        pass

    try:
        while running:
            if max_readings is not None and stats["forwarded"] >= max_readings:
                print(f"[*] Reached target limit of {max_readings} forwarded readings.")
                break

            try:
                line_bytes = ser.readline()
            except serial.SerialException as e:
                print(f"[ERROR] Serial read error: {e}", file=sys.stderr)
                stats["parse_errors"] += 1
                time.sleep(0.5)
                continue

            if not line_bytes:
                continue

            stats["lines_read"] += 1
            line_str = line_bytes.decode("utf-8", errors="replace").strip()

            parsed = parse_serial_line(line_str)
            if parsed is None:
                # Informative logging for non-CSV diagnostic lines
                if line_str and not line_str.startswith("CSV:"):
                    # Print status messages from firmware cleanly
                    if line_str.startswith(("[WiFi]", "[CALIB]", "[MODE]", "[SYSTEM]", "[SENSOR]")):
                        clean_str = line_str.encode("ascii", errors="replace").decode("ascii")
                        print(f"  [ESP32] {clean_str}")
                continue

            stats["records_parsed"] += 1

            payload = {
                "timestamp_ms": parsed["timestamp_ms"],
                "session_id": session_id,
                "total_adc": parsed["total_adc"],
                "metered_adc": parsed["metered_adc"],
                "delta_total": parsed["delta_total"],
                "delta_metered": parsed["delta_metered"],
                "data_source": "hardware"
            }

            try:
                resp = session.post(target_url, json=payload, timeout=http_timeout)
                if resp.status_code in (200, 201):
                    stats["forwarded"] += 1
                    data = resp.json()
                    pred = data.get("predicted_status") or data.get("prediction", "UNKNOWN")
                    conf = data.get("confidence")
                    win = data.get("window_status", "")
                    conf_str = f"({conf * 100:.1f}%)" if conf is not None else ""
                    print(
                        f"  [{stats['forwarded']:04d}] Forwarded | "
                        f"Feeder: {parsed['total_adc']:6.1f} | "
                        f"Metered: {parsed['metered_adc']:6.1f} | "
                        f"dTotal: {parsed['delta_total']:+6.1f} | "
                        f"dMeter: {parsed['delta_metered']:+6.1f} | "
                        f"Pred: {pred:<14} {conf_str:<8} [{win}]"
                    )
                else:
                    stats["http_errors"] += 1
                    print(f"  [!] HTTP {resp.status_code} Error: {resp.text[:120]}")
            except requests.exceptions.RequestException as e:
                stats["http_errors"] += 1
                print(f"  [!] POST to Flask failed: {e}")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[*] Bridge interrupted by operator.")
    finally:
        try:
            ser.close()
            print("[*] Serial port closed.")
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("BRIDGE SESSION SUMMARY:")
    print(f"  Lines read      : {stats['lines_read']}")
    print(f"  Records parsed  : {stats['records_parsed']}")
    print(f"  Forwarded to API: {stats['forwarded']}")
    print(f"  HTTP errors     : {stats['http_errors']}")
    print(f"  Parse errors    : {stats['parse_errors']}")
    print("=" * 72)

    return stats


def main():
    parser = argparse.ArgumentParser(description="GridGuard-AI USB Serial-to-Flask Bridge")
    parser.add_argument("--port", type=str, default="", help="ESP32 Serial port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default: 115200)")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Flask API URL (default: http://127.0.0.1:5000/api/readings)")
    parser.add_argument("--session", type=str, default="", help="Session ID (default: usb_bridge_<timestamp>)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="HTTP POST timeout in seconds (default: 2.5)")
    parser.add_argument("--max-readings", type=int, default=None, help="Stop after forwarding N readings (optional)")
    args = parser.parse_args()

    port = args.port or find_default_com_port()
    run_bridge(
        port=port,
        baudrate=args.baud,
        target_url=args.url,
        session_id=args.session or None,
        http_timeout=args.timeout,
        max_readings=args.max_readings
    )


if __name__ == "__main__":
    main()

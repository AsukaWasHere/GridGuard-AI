"""
GridGuard-AI: Real Hardware Sensor Data Collection Utility
===========================================================
Connects to ESP32 over Serial, allows operator confirmation of physical circuit state
(NORMAL vs POSSIBLE_THEFT), captures live sensor readings, and logs to data/sensor_readings.csv.

CSV Schema:
timestamp_ms,session_id,total_adc,metered_adc,delta_total,delta_metered,label,data_source
"""

import os
import sys
import time
import argparse
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

OUTPUT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sensor_readings.csv"))
CSV_HEADER = "timestamp_ms,session_id,total_adc,metered_adc,delta_total,delta_metered,label,data_source\n"


def ensure_csv_header(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)


def list_serial_ports():
    if not HAS_SERIAL:
        print("[!] pyserial is not installed. Run: pip install pyserial")
        return []
    ports = list(serial.tools.list_ports.comports())
    return ports


def run_interactive_collector(port: str, baudrate: int = 115200, output_path: str = OUTPUT_CSV_PATH):
    ensure_csv_header(output_path)

    print("=" * 70)
    print("      GRIDGUARD-AI: HARDWARE SENSOR DATA COLLECTION SENTINEL      ")
    print("=" * 70)
    print(f"Output File: {output_path}")

    if not HAS_SERIAL:
        print("[ERROR] pyserial is required for hardware capture. Install with: pip install pyserial")
        return

    available_ports = list_serial_ports()
    if not port:
        if not available_ports:
            print("[!] No active COM / Serial ports detected.")
            print("[*] Please connect your ESP32 DevKit via USB and rerun with --port COMx")
            return
        print("Available Serial Ports:")
        for idx, p in enumerate(available_ports):
            print(f"  [{idx + 1}] {p.device} - {p.description}")
        choice = input(f"Select port (1-{len(available_ports)}): ").strip()
        try:
            port = available_ports[int(choice) - 1].device
        except (ValueError, IndexError):
            print("[!] Invalid selection.")
            return

    print(f"\n[*] Connecting to {port} @ {baudrate} baud...")
    try:
        ser = serial.Serial(port, baudrate, timeout=1.0)
        time.sleep(2.0)  # ESP32 reset settling time
        ser.reset_input_buffer()
        print("[OK] Serial connection established.")
    except Exception as e:
        print(f"[ERROR] Could not open serial port {port}: {e}")
        return

    while True:
        print("\n" + "-" * 70)
        print("SELECT OPERATING CONDITION FOR THIS RECORDING SESSION:")
        print("  [1] NORMAL         -> LED ON, Motor OFF (Baseline / Normal noise)")
        print("  [2] POSSIBLE_THEFT -> LED ON, Motor ON (Motor bypasses metered sensor)")
        print("  [Q] Quit / Finish Collection")
        print("-" * 70)
        sel = input("Enter choice (1, 2, or Q): ").strip().upper()

        if sel == "Q":
            print("[*] Closing serial port and exiting data collection.")
            ser.close()
            break

        if sel == "1":
            label = "NORMAL"
            default_session = f"normal_{int(time.time())}"
        elif sel == "2":
            label = "POSSIBLE_THEFT"
            default_session = f"theft_{int(time.time())}"
        else:
            print("[!] Invalid choice. Please select 1, 2, or Q.")
            continue

        session_id = input(f"Enter session_id [{default_session}]: ").strip() or default_session
        duration_s = input("Target duration in seconds [30]: ").strip()
        duration_s = int(duration_s) if duration_s.isdigit() else 30

        print(f"\n⚠️  CONFIRMATION REQUIRED: Please verify the physical hardware state:")
        if label == "NORMAL":
            print("   -> Confirm: LED is ON, Motor is OFF.")
        else:
            print("   -> Confirm: LED is ON, Motor is actively running and bypassing Sensor 2.")
        confirm = input("Are the physical connections verified? (y/n): ").strip().lower()
        if confirm != "y":
            print("[*] Session cancelled by operator.")
            continue

        # Send command to ESP32 to switch to CSV output mode if supported
        ser.write(b"MODE_CSV\n")
        ser.flush()

        print(f"\n[*] Recording session '{session_id}' [{label}] for {duration_s}s...")
        t_start = time.time()
        samples_recorded = 0

        with open(output_path, "a", encoding="utf-8") as csv_file:
            while time.time() - t_start < duration_s:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # Parse either CSV line or tagged line from ESP32
                # Expected format: CSV: <timestamp_ms>,<total_adc>,<metered_adc>,<delta_total>,<delta_metered>
                # Or raw comma separated numbers
                parts = None
                if line.startswith("CSV:"):
                    parts = line.replace("CSV:", "").strip().split(",")
                elif line.count(",") >= 4:
                    parts = line.split(",")

                if parts and len(parts) >= 5:
                    try:
                        ts_ms = int(float(parts[0]))
                        tot_adc = float(parts[1])
                        met_adc = float(parts[2])
                        d_tot = float(parts[3])
                        d_met = float(parts[4])

                        csv_record = f"{ts_ms},{session_id},{tot_adc:.2f},{met_adc:.2f},{d_tot:.2f},{d_met:.2f},{label},hardware\n"
                        csv_file.write(csv_record)
                        csv_file.flush()
                        samples_recorded += 1

                        print(f"  [{samples_recorded:03d}] Feeder ADC: {tot_adc:6.1f} | Metered ADC: {met_adc:6.1f} | ΔFeeder: {d_tot:+5.1f} | ΔMeter: {d_met:+5.1f} | {label}")
                    except ValueError:
                        continue
                else:
                    # Informative print for debugging
                    if not line.startswith("[DATA]"):
                        print(f"  [ESP32] {line}")

        print(f"\n[OK] Session '{session_id}' finished. Recorded {samples_recorded} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridGuard-AI Hardware Data Collector")
    parser.add_argument("--port", type=str, default="", help="Serial port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--out", type=str, default=OUTPUT_CSV_PATH, help="Output CSV path")
    args = parser.parse_args()

    run_interactive_collector(args.port, args.baud, args.out)

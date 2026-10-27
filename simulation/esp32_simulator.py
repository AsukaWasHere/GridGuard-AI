"""
GridGuard-AI: Sensor Telemetry & Laboratory Anomaly Simulator
=============================================================
Simulates dual ACS712 current sensors sending real-time ADC measurements
to the Flask backend API for Random Forest evaluation.

Scenarios supported:
- demo    : Executes Test A (NORMAL 12s) -> Test B (POSSIBLE_THEFT 15s) -> Test C (RETURN TO NORMAL 12s)
- cycle   : Continuous cycling between Normal, Theft, and Standby
- normal  : Sustained Normal (LED ON, Motor OFF)
- theft   : Sustained Bypass Theft (LED ON, Motor ON bypassing Sensor 2)
"""

import time
import math
import random
import requests
import argparse

BACKEND_DEFAULT_URL = "http://localhost:5000/api/readings"

def run_simulation(backend_url=BACKEND_DEFAULT_URL, mode="demo", interval=1.0, data_source="simulation"):
    print("=" * 75)
    print("        GRIDGUARD-AI: LABORATORY SENSOR & CIRCUIT SIMULATOR        ")
    print("=" * 75)
    print(f"Target URL  : {backend_url}")
    print(f"Mode        : {mode} (options: demo, cycle, normal, theft)")
    print(f"Interval    : {interval}s")
    print(f"Data Source : {data_source}")
    print("=" * 75)
    print("Press Ctrl+C to terminate.\n")

    t_start = time.time()
    step = 0
    session_id = f"sim_{int(time.time())}"

    # Nominal ACS712 12-bit ADC baselines
    tare_feeder = 3102.0
    tare_metered = 1551.0

    while True:
        elapsed = time.time() - t_start
        step += 1

        # Determine physical condition
        if mode == "demo":
            # Test A: Normal for 12s
            # Test B: Motor ON (Theft) for 15s
            # Test C: Motor OFF (Return to Normal) for 15s
            cycle_time = elapsed % 42.0
            if cycle_time < 12.0:
                physical_state = "TEST A: NORMAL (LED ON, Motor OFF)"
                is_motor_on = False
            elif cycle_time < 27.0:
                physical_state = "TEST B: POSSIBLE_THEFT (LED ON, Motor ON [Bypass])"
                is_motor_on = True
            else:
                physical_state = "TEST C: RETURN TO NORMAL (Motor switched OFF)"
                is_motor_on = False
        elif mode == "cycle":
            cycle_time = elapsed % 30.0
            if cycle_time < 15.0:
                physical_state = "NORMAL (LED ON, Motor OFF)"
                is_motor_on = False
            else:
                physical_state = "POSSIBLE_THEFT (LED ON, Motor ON [Bypass])"
                is_motor_on = True
        elif mode == "theft":
            physical_state = "POSSIBLE_THEFT (LED ON, Motor ON)"
            is_motor_on = True
        else:
            physical_state = "NORMAL (LED ON, Motor OFF)"
            is_motor_on = False

        # Physical sensor signals
        # LED draws ~25mA -> ~6 ADC counts on feeder, ~3 ADC counts on metered (after divider)
        led_feeder = 6.5
        led_metered = 3.2

        if is_motor_on:
            # Motor draws ~200mA -> ~90 ADC counts on feeder only!
            motor_delta = random.uniform(88.0, 110.0)
            noise_f = random.gauss(0, 3.2)  # Higher commutator noise
            noise_m = random.gauss(0, 0.8)
        else:
            motor_delta = 0.0
            noise_f = random.gauss(0, 1.0)
            noise_m = random.gauss(0, 0.8)

        total_adc = round(tare_feeder + led_feeder + motor_delta + noise_f, 2)
        metered_adc = round(tare_metered + led_metered + noise_m, 2)
        delta_total = round(total_adc - tare_feeder, 2)
        delta_metered = round(metered_adc - tare_metered, 2)

        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "session_id": session_id,
            "total_adc": total_adc,
            "metered_adc": metered_adc,
            "delta_total": delta_total,
            "delta_metered": delta_metered,
            "data_source": data_source
        }

        try:
            resp = requests.post(backend_url, json=payload, timeout=2.5)
            if resp.status_code in [200, 201]:
                res_data = resp.get_json()
                pred = res_data.get("predicted_status", "UNKNOWN")
                conf = res_data.get("confidence")
                conf_str = f"{conf * 100:.1f}%" if conf is not None else "Warming up"
                win_str = res_data.get("window_status", "")

                status_flag = "⚠️ ALERT" if pred == "POSSIBLE_THEFT" else ("⏳ WARM" if pred == "WARMING_UP" else "✅ OK")

                print(f"[{time.strftime('%H:%M:%S')}] [{status_flag}] State: {physical_state:48s} | "
                      f"Feeder: {total_adc:6.1f} ADC (Δ{delta_total:+5.1f}) | "
                      f"Meter: {metered_adc:6.1f} ADC (Δ{delta_metered:+5.1f}) | "
                      f"Model Pred: {pred:14s} (Score: {conf_str}, Win: {win_str})")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] [ERR {resp.status_code}] {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[{time.strftime('%H:%M:%S')}] [CONN ERROR] Could not reach backend: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate ESP32 GridGuard-AI telemetry stream")
    parser.add_argument("--url", type=str, default=BACKEND_DEFAULT_URL, help="Backend API URL")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "cycle", "normal", "theft"], help="Simulation profile")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between readings")
    parser.add_argument("--source", type=str, default="simulation", choices=["simulation", "hardware"], help="Data source tag")
    args = parser.parse_args()

    run_simulation(args.url, args.mode, args.interval, args.source)

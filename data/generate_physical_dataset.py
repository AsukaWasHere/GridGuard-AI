"""
GridGuard-AI: Physical Sensor Profile Dataset Synthesizer
==========================================================
Generates realistic laboratory sensor recordings for ESP32 dual ACS712 setup
with multiple independent sessions for pipeline testing and smoke testing.

Physics Parameters:
- Feeder (GPIO32): Direct ADC, baseline ~3100 counts (2.50V at 3.3V reference, 12-bit)
- Metered (GPIO33): 10k/10k divider, baseline ~1550 counts (1.25V at 3.3V reference)
- LED Load: ~20-30mA -> ~6 ADC counts on feeder, ~3 ADC counts on metered branch
- Motor Load (Bypass): ~180-250mA -> ~70-110 ADC counts on feeder, 0 on metered branch
- Motor Commutator Noise: Induces higher variance / standard deviation on feeder channel
- Class labels: NORMAL (LED ON, motor OFF), POSSIBLE_THEFT (LED ON, motor ON)
- Multiple independent sessions for leak-free session-level train/test splits.
"""

import os
import random
import numpy as np
import pandas as pd

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_sensor_readings.csv"))


def generate_synthetic_sessions(num_normal_sessions=6, num_theft_sessions=6, samples_per_session=150, output_file=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    records = []

    # Sensor baseline offsets (slight variations per experiment to reflect real ambient tare)
    base_feeder_nominal = 3102.0
    base_metered_nominal = 1551.0

    session_counter = 0

    # 1. Generate NORMAL Sessions (LED ON, Motor OFF)
    for s_idx in range(num_normal_sessions):
        session_id = f"normal_session_{s_idx + 1:02d}"
        session_counter += 1
        
        # Tare calibration offset for this session
        tare_feeder = base_feeder_nominal + random.uniform(-10.0, 10.0)
        tare_metered = base_metered_nominal + random.uniform(-6.0, 6.0)

        # LED load steady state (both sensors see proportional current)
        led_feeder_delta = random.uniform(5.0, 8.0)
        led_metered_delta = led_feeder_delta * 0.50 + random.uniform(-0.5, 0.5)

        t_base = 1000000 + session_counter * 500000

        for i in range(samples_per_session):
            timestamp_ms = t_base + i * 1000  # 1 Hz sampling
            
            # Ambient electrical noise (normal Gaussian)
            noise_feeder = random.gauss(0.0, 1.2)
            noise_metered = random.gauss(0.0, 0.9)

            total_adc = tare_feeder + led_feeder_delta + noise_feeder
            metered_adc = tare_metered + led_metered_delta + noise_metered

            delta_total = total_adc - tare_feeder
            delta_metered = metered_adc - tare_metered

            records.append({
                "timestamp_ms": timestamp_ms,
                "session_id": session_id,
                "total_adc": round(total_adc, 2),
                "metered_adc": round(metered_adc, 2),
                "delta_total": round(delta_total, 2),
                "delta_metered": round(delta_metered, 2),
                "label": "NORMAL",
                "data_source": "synthetic"
            })

    # 2. Generate POSSIBLE_THEFT Sessions (LED ON, Motor ON - Bypassing Sensor 2)
    for s_idx in range(num_theft_sessions):
        session_id = f"theft_session_{s_idx + 1:02d}"
        session_counter += 1

        tare_feeder = base_feeder_nominal + random.uniform(-10.0, 10.0)
        tare_metered = base_metered_nominal + random.uniform(-6.0, 6.0)

        led_feeder_delta = random.uniform(5.0, 8.0)
        led_metered_delta = led_feeder_delta * 0.50 + random.uniform(-0.5, 0.5)

        # Motor load: draws additional 75-125 ADC counts on feeder ONLY
        motor_load_delta = random.uniform(75.0, 125.0)

        t_base = 1000000 + session_counter * 500000

        for i in range(samples_per_session):
            timestamp_ms = t_base + i * 1000

            # Motor running has mechanical brush commutator ripples -> higher noise variance on feeder
            noise_feeder = random.gauss(0.0, 3.8)
            noise_metered = random.gauss(0.0, 0.9)

            total_adc = tare_feeder + led_feeder_delta + motor_load_delta + noise_feeder
            metered_adc = tare_metered + led_metered_delta + noise_metered

            delta_total = total_adc - tare_feeder
            delta_metered = metered_adc - tare_metered

            records.append({
                "timestamp_ms": timestamp_ms,
                "session_id": session_id,
                "total_adc": round(total_adc, 2),
                "metered_adc": round(metered_adc, 2),
                "delta_total": round(delta_total, 2),
                "delta_metered": round(delta_metered, 2),
                "label": "POSSIBLE_THEFT",
                "data_source": "synthetic"
            })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"[OK] Generated {len(df)} synthetic samples across {session_counter} independent sessions.")
    print(f"     File saved to: {output_file}")
    print(f"     Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"     Session counts: {df.groupby('label')['session_id'].nunique().to_dict()}")
    return df


if __name__ == "__main__":
    generate_synthetic_sessions()

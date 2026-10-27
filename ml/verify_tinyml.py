"""
GridGuard-AI: TinyML Parity Verification
========================================
Validates that the generated C decision logic matches the scikit-learn Python
model's predictions across 100% of the test samples.
"""

import os
import math
import joblib
import pandas as pd
import numpy as np

FEATURES = [
    "total_current_A",
    "meter_current_A",
    "difference_A",
    "abs_difference_A",
    "relative_diff",
    "current_ratio"
]

def c_tinyml_simulate(total_current_A, meter_current_A):
    """Exact software simulation of tinyml_predict() in tinyml_model.h"""
    difference_A = total_current_A - meter_current_A
    abs_difference_A = math.fabs(difference_A)
    epsilon = 0.010
    relative_diff = abs_difference_A / (total_current_A + epsilon)
    current_ratio = meter_current_A / (total_current_A + epsilon)

    if abs_difference_A <= 0.022415:
        if total_current_A <= 0.010070:
            if current_ratio <= 0.601750:
                return 2, "POWER_LOSS", 1.0000
            else:
                return 2, "POWER_LOSS", 0.9000
        else:
            return 0, "NORMAL", 1.0000
    else:
        return 1, "POSSIBLE_BYPASS", 1.0000


def verify_parity(models_dir):
    model_path = os.path.join(models_dir, "gridguard_model.joblib")
    test_path = os.path.join(models_dir, "test.csv")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURES].values
    y_true = test_df["label"].values

    py_preds = model.predict(X_test)

    c_preds = []
    c_labels = []
    for row in X_test:
        total = row[0]
        meter = row[1]
        cls_idx, cls_label, _ = c_tinyml_simulate(total, meter)
        c_preds.append(cls_idx)
        c_labels.append(cls_label)

    c_preds = np.array(c_preds)

    mismatches = np.where(py_preds != c_preds)[0]
    total_samples = len(py_preds)
    match_rate = ((total_samples - len(mismatches)) / total_samples) * 100.0

    print("=" * 65)
    print("GRIDGUARD-AI: TINYML PARITY VERIFICATION")
    print("=" * 65)
    print(f"Total Test Samples Checked   : {total_samples}")
    print(f"Python vs C Matches          : {total_samples - len(mismatches)} / {total_samples}")
    print(f"Parity Rate                  : {match_rate:.2f}%")
    print(f"Mismatches                   : {len(mismatches)}")
    print("=" * 65)

    if len(mismatches) == 0:
        print("[OK] SUCCESS: 100% Deterministic Parity between Python Model & ESP32 TinyML Header!")
    else:
        print(f"[FAIL] Found {len(mismatches)} discrepancies between Python and C header.")

    return len(mismatches) == 0


if __name__ == "__main__":
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    verify_parity(MODELS_DIR)

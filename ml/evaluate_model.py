"""
GridGuard-AI: Model Evaluation & Benchmarking
=============================================
Evaluates the trained TinyML model against the held-out test split,
computes precision, recall, F1-score, confusion matrix, and measures
inference latency.
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURES = [
    "total_current_A",
    "meter_current_A",
    "difference_A",
    "abs_difference_A",
    "relative_diff",
    "current_ratio"
]
CLASS_NAMES = ["NORMAL", "POSSIBLE_BYPASS", "POWER_LOSS"]

def run_evaluation(models_dir):
    model_path = os.path.join(models_dir, "gridguard_model.joblib")
    test_path = os.path.join(models_dir, "test.csv")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Train the model first.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing {test_path}.")

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURES].values
    y_test = test_df["label"].values

    # Benchmark single-sample inference latency
    n_benchmark = 1000
    sample = X_test[:1]
    t0 = time.perf_counter()
    for _ in range(n_benchmark):
        _ = model.predict(sample)
    t1 = time.perf_counter()
    avg_latency_us = ((t1 - t0) / n_benchmark) * 1e6

    # Full test evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print("=" * 65)
    print("GRIDGUARD-AI: TINYML MODEL TEST EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Test Samples Evaluated : {len(y_test)}")
    print(f"Overall Accuracy             : {acc * 100:.2f}%")
    print(f"Macro F1-Score               : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Python Inference Latency     : {avg_latency_us:.2f} us/sample")
    print(f"Estimated ESP32 Latency      : < 5 us (Direct C execution)")
    print("-" * 65)
    print(report_text)
    print("Confusion Matrix:")
    print(f"Rows = Actual, Columns = Predicted [{', '.join(CLASS_NAMES)}]")
    print(cm)
    print("=" * 65)

    eval_summary = {
        "test_samples": len(y_test),
        "accuracy": acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "latency_us_python": avg_latency_us
    }

    eval_out = os.path.join(models_dir, "evaluation_report.json")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"[OK] Saved evaluation report to: {eval_out}")
    return eval_summary

if __name__ == "__main__":
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    run_evaluation(MODELS_DIR)

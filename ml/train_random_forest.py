"""
GridGuard-AI: Random Forest Model Training Pipeline
====================================================
Trains a Random Forest classifier for detecting power theft anomalies
using dual-channel ADC telemetry.

Key Design Elements:
1. Shared, versioned feature extraction (ml/feature_extractor.py)
2. Strict session-level train/test split to prevent data leakage
3. Comprehensive evaluation reporting (Confusion matrix, Precision, Recall, F1)
4. Saves model to model/random_forest.joblib and config to model/feature_config.json
5. Supports both real hardware data and synthetic smoke test data with clear labeling
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

# Ensure ml package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.feature_extractor import (
    FEATURE_VERSION,
    FEATURE_NAMES,
    DEFAULT_WINDOW_SIZE,
    CLASS_MAPPING,
    INV_CLASS_MAPPING,
    LABEL_NORMAL,
    LABEL_POSSIBLE_THEFT,
    extract_features_from_dataframe
)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
REAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sensor_readings.csv"))
SYNTHETIC_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_sensor_readings.csv"))


def load_dataset():
    """
    Loads real hardware readings if available with sufficient sessions,
    otherwise loads synthetic data clearly labeled as smoke test.
    """
    use_real = False
    if os.path.exists(REAL_DATA_PATH) and os.path.getsize(REAL_DATA_PATH) > 100:
        try:
            df = pd.read_csv(REAL_DATA_PATH)
            if "session_id" in df.columns and "label" in df.columns:
                session_counts = df.groupby("label")["session_id"].nunique()
                if session_counts.get("NORMAL", 0) >= 2 and session_counts.get("POSSIBLE_THEFT", 0) >= 2:
                    use_real = True
                    print(f"[*] Found valid real hardware dataset at {REAL_DATA_PATH}")
                    print(f"    Sessions: {session_counts.to_dict()}")
                    return df, "hardware", False
                else:
                    print(f"[!] Real data exists but has insufficient sessions for held-out evaluation: {session_counts.to_dict()}")
        except Exception as e:
            print(f"[!] Error reading {REAL_DATA_PATH}: {e}")

    if not use_real:
        print("[!] Using synthetic physical sensor profiles for model training and smoke test.")
        print("    NOTE: Synthetic evaluation is for pipeline validation, NOT physical validation.")
        if not os.path.exists(SYNTHETIC_DATA_PATH):
            raise FileNotFoundError(f"Missing {SYNTHETIC_DATA_PATH}. Run data/generate_physical_dataset.py first.")
        df = pd.read_csv(SYNTHETIC_DATA_PATH)
        return df, "synthetic", True


def train_model(window_size=DEFAULT_WINDOW_SIZE, test_size=0.33, random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df, data_source, is_smoke_test = load_dataset()

    print("\n" + "=" * 70)
    print("           GRIDGUARD-AI: RANDOM FOREST TRAINING PIPELINE           ")
    print("=" * 70)
    print(f"Dataset Rows      : {len(df)}")
    print(f"Data Source       : {data_source.upper()} ({'SMOKE TEST' if is_smoke_test else 'HARDWARE'})")
    print(f"Rolling Window (W): {window_size} samples")
    print(f"Feature Set       : {FEATURE_NAMES}")

    # Inspect class and session counts
    class_counts = df["label"].value_counts().to_dict()
    session_counts = df.groupby("label")["session_id"].nunique().to_dict()
    print(f"Class Sample Counts : {class_counts}")
    print(f"Unique Sessions     : {session_counts}")

    if session_counts.get(LABEL_NORMAL, 0) < 2 or session_counts.get(LABEL_POSSIBLE_THEFT, 0) < 2:
        raise ValueError(
            f"Insufficient independent sessions for held-out evaluation. "
            f"Need at least 2 sessions per class, got: {session_counts}"
        )

    # 1. Feature Extraction using shared module
    X_df, meta_df = extract_features_from_dataframe(df, window_size=window_size, session_col="session_id")
    print(f"Extracted {len(X_df)} windowed samples.")

    # Convert labels to integer targets
    y = np.array([INV_CLASS_MAPPING[lbl] for lbl in meta_df["label"]])
    groups = meta_df["session_id"].values

    # 2. Prevent Data Leakage: Split by independent recording session
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X_df, y, groups=groups))

    X_train = X_df.iloc[train_idx].values
    y_train = y[train_idx]
    train_sessions = sorted(list(set(groups[train_idx])))

    X_test = X_df.iloc[test_idx].values
    y_test = y[test_idx]
    test_sessions = sorted(list(set(groups[test_idx])))

    print(f"\nTrain Set: {len(X_train)} samples across {len(train_sessions)} sessions: {train_sessions}")
    print(f"Test Set : {len(X_test)} samples across {len(test_sessions)} sessions: {test_sessions}")

    # Check that no session overlaps between train and test
    overlap = set(train_sessions).intersection(set(test_sessions))
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping sessions: {overlap}"

    # 3. Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    # 4. Evaluate on Held-Out Test Sessions
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    target_names = [CLASS_MAPPING[0], CLASS_MAPPING[1]]
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=target_names)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    print("\n" + "-" * 70)
    print("EVALUATION ON HELD-OUT INDEPENDENT SESSIONS:")
    print("-" * 70)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Macro F1  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("\nClassification Report:\n", report_str)
    print("Confusion Matrix (Rows: Actual [NORMAL, POSSIBLE_THEFT], Cols: Predicted):")
    print(cm)

    # 5. Save Artifacts
    model_save_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(rf, model_save_path)

    feature_config = {
        "model_name": "GridGuard-AI Random Forest Classifier",
        "model_version": "2.0.0",
        "feature_version": FEATURE_VERSION,
        "feature_order": FEATURE_NAMES,
        "window_size": window_size,
        "class_mapping": CLASS_MAPPING,
        "classes": target_names,
        "data_source": data_source,
        "is_smoke_test": is_smoke_test,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "train_sessions": train_sessions,
        "test_sessions": test_sessions,
        "feature_importances": {
            feat: float(imp) for feat, imp in zip(FEATURE_NAMES, rf.feature_importances_)
        }
    }

    config_save_path = os.path.join(MODEL_DIR, "feature_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    eval_report = {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "test_sample_count": int(len(y_test)),
        "test_class_counts": {
            CLASS_MAPPING[0]: int(np.sum(y_test == 0)),
            CLASS_MAPPING[1]: int(np.sum(y_test == 1))
        },
        "train_sessions_count": len(train_sessions),
        "test_sessions_count": len(test_sessions),
        "data_source": data_source
    }

    eval_save_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(eval_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    # Also keep a copy in ml/models for backwards compatibility
    ml_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    os.makedirs(ml_models_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(ml_models_dir, "random_forest.joblib"))
    with open(os.path.join(ml_models_dir, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Model saved to:  {model_save_path}")
    print(f"[OK] Config saved to: {config_save_path}")
    print(f"[OK] Report saved to: {eval_save_path}")
    print("=" * 70)

    return rf, feature_config, eval_report


if __name__ == "__main__":
    train_model()

"""
GridGuard-AI: Model Training & Evaluation
=========================================
Trains and compares lightweight classifiers for ESP32 TinyML deployment:
1. Rule-Based Baseline
2. Decision Tree Classifier (Ideal for TinyML: 100% deterministic, zero dependencies, <1KB flash)
3. Logistic Regression
4. Random Forest (Benchmark comparison)

Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
Saves model and metadata for C header export.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

FEATURES = [
    "total_current_A",
    "meter_current_A",
    "difference_A",
    "abs_difference_A",
    "relative_diff",
    "current_ratio"
]

TARGET = "label"
CLASS_NAMES = ["NORMAL", "POSSIBLE_BYPASS", "POWER_LOSS"]


class RuleBasedBaseline:
    """
    Hardware-grounded rule-based discrepancy classifier.
    Threshold calibrated for 5A ACS712 after filtering.
    """
    def __init__(self, bypass_threshold_A=0.035, power_loss_threshold_A=0.012):
        self.bypass_threshold_A = bypass_threshold_A
        self.power_loss_threshold_A = power_loss_threshold_A

    def predict(self, X):
        preds = []
        for row in X:
            total = row[0]
            meter = row[1]
            diff = row[2]

            if total < self.power_loss_threshold_A and meter < self.power_loss_threshold_A:
                preds.append(2)  # POWER_LOSS
            elif diff >= self.bypass_threshold_A:
                preds.append(1)  # POSSIBLE_BYPASS
            else:
                preds.append(0)  # NORMAL
        return np.array(preds)


def train_and_evaluate(models_dir):
    train_path = os.path.join(models_dir, "train.csv")
    val_path = os.path.join(models_dir, "val.csv")
    test_path = os.path.join(models_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}. Run generate_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values

    X_val = val_df[FEATURES].values
    y_val = val_df[TARGET].values

    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    print(f"[*] Training on {len(X_train)} samples, validating on {len(X_val)}, testing on {len(X_test)}")
    print(f"[*] Feature order: {FEATURES}\n")

    models = {
        "Rule-Based Baseline": RuleBasedBaseline(bypass_threshold_A=0.035),
        "Decision Tree (TinyML)": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest (Reference)": RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    }

    results = {}

    for name, clf in models.items():
        if hasattr(clf, "fit"):
            clf.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
            "report": report,
            "model_obj": clf
        }

        print("=" * 65)
        print(f"Model: {name}")
        print(f"Overall Accuracy: {acc * 100:.2f}% | Macro F1-Score: {f1_macro:.4f}")
        print("-" * 65)
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        print("Confusion Matrix:")
        print(cm)
        print()

    # Select Decision Tree for TinyML export due to:
    # 1. Zero runtime dependencies (no TFLite micro or BLAS needed)
    # 2. Sub-microsecond inference on ESP32 (single digits of microseconds)
    # 3. Transparent branch decisions for auditing
    # 4. Equivalent or near-optimal accuracy compared to random forest
    selected_model = results["Decision Tree (TinyML)"]["model_obj"]
    model_save_path = os.path.join(models_dir, "gridguard_model.joblib")
    joblib.dump(selected_model, model_save_path)

    metadata = {
        "model_type": "DecisionTreeClassifier",
        "max_depth": 4,
        "features": FEATURES,
        "classes": CLASS_NAMES,
        "class_mapping": {i: name for i, name in enumerate(CLASS_NAMES)},
        "test_accuracy": float(results["Decision Tree (TinyML)"]["accuracy"]),
        "test_f1_macro": float(results["Decision Tree (TinyML)"]["f1_macro"]),
        "confusion_matrix": results["Decision Tree (TinyML)"]["confusion_matrix"].tolist(),
        "baseline_rule_accuracy": float(results["Rule-Based Baseline"]["accuracy"]),
        "feature_importances": {f: float(imp) for f, imp in zip(FEATURES, selected_model.feature_importances_)},
        "limitations": [
            "Trained on synthetic profiles modeled after ACS712 5A characteristics with 100-sample oversampling.",
            "Sub-15mA discrepancies approach the noise floor of standard ACS712 without additional filtering.",
            "Requires accurate hardware zero-offset calibration before inference."
        ]
    }

    meta_save_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved model to {model_save_path}")
    print(f"[OK] Saved metadata to {meta_save_path}")
    print("\nFeature Importances for TinyML Model:")
    for f, imp in sorted(metadata["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {f:18s}: {imp * 100:5.2f}%")

    return results


if __name__ == "__main__":
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    train_and_evaluate(MODELS_DIR)

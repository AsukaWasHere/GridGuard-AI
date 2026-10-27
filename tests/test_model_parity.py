"""
GridGuard-AI: ML Model & TinyML Parity Tests
============================================
Runs automated verification to guarantee that:
1. The scikit-learn model reaches >98% accuracy on the test set.
2. The Decision Tree logic exported to tinyml_model.h produces identical predictions.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml")))

from verify_tinyml import verify_parity
from evaluate_model import run_evaluation

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "models"))

def test_model_accuracy_benchmark():
    """Verifies that the trained model meets performance criteria."""
    eval_res = run_evaluation(MODELS_DIR)
    assert eval_res["accuracy"] >= 0.98, f"Accuracy {eval_res['accuracy']} below 98%"
    assert eval_res["classification_report"]["macro avg"]["f1-score"] >= 0.98


def test_tinyml_c_logic_exact_parity():
    """Verifies 100% parity between Python scikit-learn model and ESP32 C logic."""
    has_parity = verify_parity(MODELS_DIR)
    assert has_parity is True, "Parity mismatch between Python and TinyML C logic!"

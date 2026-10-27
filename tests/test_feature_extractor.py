"""
GridGuard-AI: Feature Extractor Test Suite
===========================================
Validates:
1. compute_window_features mathematical correctness
2. extract_features_from_dataframe session isolation (no window leakage across sessions)
3. Output feature names and ordering parity
"""

import math
import pytest
import pandas as pd
from ml.feature_extractor import (
    FEATURE_NAMES,
    DEFAULT_WINDOW_SIZE,
    compute_window_features,
    extract_features_from_dataframe
)


def test_compute_window_features_exact():
    # 5 test samples with known numbers
    window = [
        {"total_adc": 3100.0, "metered_adc": 1550.0, "delta_total": 0.0, "delta_metered": 0.0},
        {"total_adc": 3110.0, "metered_adc": 1552.0, "delta_total": 10.0, "delta_metered": 2.0},
        {"total_adc": 3120.0, "metered_adc": 1554.0, "delta_total": 20.0, "delta_metered": 4.0},
        {"total_adc": 3130.0, "metered_adc": 1556.0, "delta_total": 30.0, "delta_metered": 6.0},
        {"total_adc": 3140.0, "metered_adc": 1558.0, "delta_total": 40.0, "delta_metered": 8.0},
    ]

    feats = compute_window_features(window)

    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert feats["mean_total_adc"] == pytest.approx(3120.0)
    assert feats["mean_metered_adc"] == pytest.approx(1554.0)
    assert feats["mean_delta_total"] == pytest.approx(20.0)
    assert feats["mean_delta_metered"] == pytest.approx(4.0)
    assert feats["delta_diff"] == pytest.approx(16.0)

    # Check std dev
    expected_std_tot = math.sqrt(((20**2 + 10**2 + 0**2 + 10**2 + 20**2) / 5.0))
    assert feats["std_total_adc"] == pytest.approx(expected_std_tot)


def test_dataframe_session_isolation():
    # Construct dataframe with 2 sessions of 4 samples each (window=5)
    # If sessions leak, sample 5 would be formed from session 1 + session 2.
    # With session isolation, each session with 4 samples (< window_size) should yield 0 samples!
    df = pd.DataFrame([
        {"session_id": "s1", "total_adc": 3100, "metered_adc": 1550, "delta_total": 0, "delta_metered": 0, "label": "NORMAL"},
        {"session_id": "s1", "total_adc": 3100, "metered_adc": 1550, "delta_total": 0, "delta_metered": 0, "label": "NORMAL"},
        {"session_id": "s1", "total_adc": 3100, "metered_adc": 1550, "delta_total": 0, "delta_metered": 0, "label": "NORMAL"},
        {"session_id": "s1", "total_adc": 3100, "metered_adc": 1550, "delta_total": 0, "delta_metered": 0, "label": "NORMAL"},
        {"session_id": "s2", "total_adc": 3200, "metered_adc": 1550, "delta_total": 100, "delta_metered": 0, "label": "POSSIBLE_THEFT"},
        {"session_id": "s2", "total_adc": 3200, "metered_adc": 1550, "delta_total": 100, "delta_metered": 0, "label": "POSSIBLE_THEFT"},
        {"session_id": "s2", "total_adc": 3200, "metered_adc": 1550, "delta_total": 100, "delta_metered": 0, "label": "POSSIBLE_THEFT"},
        {"session_id": "s2", "total_adc": 3200, "metered_adc": 1550, "delta_total": 100, "delta_metered": 0, "label": "POSSIBLE_THEFT"},
    ])

    X, meta = extract_features_from_dataframe(df, window_size=5, session_col="session_id")
    assert len(X) == 0  # Neither session reached window_size=5

    # Now add 5th sample to s1
    df5 = pd.concat([
        df.iloc[:4],
        pd.DataFrame([{"session_id": "s1", "total_adc": 3100, "metered_adc": 1550, "delta_total": 0, "delta_metered": 0, "label": "NORMAL"}]),
        df.iloc[4:]
    ], ignore_index=True)

    X5, meta5 = extract_features_from_dataframe(df5, window_size=5, session_col="session_id")
    assert len(X5) == 1
    assert meta5.iloc[0]["session_id"] == "s1"
    assert meta5.iloc[0]["label"] == "NORMAL"

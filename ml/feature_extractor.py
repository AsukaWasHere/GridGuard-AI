"""
GridGuard-AI: Shared Feature Extraction Module
===============================================
Versioned feature engineering shared between model training and real-time Flask inference.

Features extracted over a rolling window:
1. mean_total_adc: Rolling mean of GPIO32 ADC readings
2. mean_metered_adc: Rolling mean of GPIO33 ADC readings
3. mean_delta_total: Rolling mean of feeder baseline offset
4. mean_delta_metered: Rolling mean of metered branch baseline offset
5. delta_diff: mean_delta_total - mean_delta_metered (discrepancy indicator)
6. std_total_adc: Standard deviation of feeder readings in window
7. std_metered_adc: Standard deviation of metered readings in window
"""

import math
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

FEATURE_VERSION = "1.0.0"

FEATURE_NAMES = [
    "mean_total_adc",
    "mean_metered_adc",
    "mean_delta_total",
    "mean_delta_metered",
    "delta_diff",
    "std_total_adc",
    "std_metered_adc"
]

DEFAULT_WINDOW_SIZE = 5

LABEL_NORMAL = "NORMAL"
LABEL_POSSIBLE_THEFT = "POSSIBLE_THEFT"
CLASS_MAPPING = {
    0: LABEL_NORMAL,
    1: LABEL_POSSIBLE_THEFT
}
INV_CLASS_MAPPING = {
    LABEL_NORMAL: 0,
    LABEL_POSSIBLE_THEFT: 1
}


def compute_window_features(readings: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes exact statistical features over an in-memory window of readings.
    
    Args:
        readings: List of dicts, each containing:
                  total_adc, metered_adc, delta_total, delta_metered
    
    Returns:
        Dict mapping feature name to float value.
    """
    n = len(readings)
    if n == 0:
        raise ValueError("Window cannot be empty")

    totals = [float(r["total_adc"]) for r in readings]
    metereds = [float(r["metered_adc"]) for r in readings]
    delta_totals = [float(r["delta_total"]) for r in readings]
    delta_metereds = [float(r["delta_metered"]) for r in readings]

    mean_total = sum(totals) / n
    mean_metered = sum(metereds) / n
    mean_d_total = sum(delta_totals) / n
    mean_d_metered = sum(delta_metereds) / n
    delta_diff = mean_d_total - mean_d_metered

    # Population std dev (ddof=0) for consistency
    var_total = sum((x - mean_total) ** 2 for x in totals) / n
    var_metered = sum((x - mean_metered) ** 2 for x in metereds) / n

    std_total = math.sqrt(max(0.0, var_total))
    std_metered = math.sqrt(max(0.0, var_metered))

    return {
        "mean_total_adc": float(mean_total),
        "mean_metered_adc": float(mean_metered),
        "mean_delta_total": float(mean_d_total),
        "mean_delta_metered": float(mean_d_metered),
        "delta_diff": float(delta_diff),
        "std_total_adc": float(std_total),
        "std_metered_adc": float(std_metered)
    }


def extract_features_from_dataframe(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE, session_col: str = "session_id") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts rolling window features from a historical dataset grouped by session.
    Prevents window leakage across distinct recording sessions.
    
    Returns:
        X_df: DataFrame containing FEATURE_NAMES
        metadata_df: DataFrame containing session_id, timestamp_ms, label, etc.
    """
    required_cols = ["total_adc", "metered_adc", "delta_total", "delta_metered"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in input DataFrame")

    feature_rows = []
    meta_rows = []

    # If session_id not in df, treat as single session
    if session_col not in df.columns:
        df = df.copy()
        df[session_col] = "default_session"

    grouped = df.groupby(session_col, sort=False)

    for session_id, group in grouped:
        records = group.to_dict("records")
        for i in range(len(records)):
            # Only generate sample when full window is available
            if i + 1 < window_size:
                continue
            window = records[i + 1 - window_size : i + 1]
            feats = compute_window_features(window)
            feature_rows.append(feats)

            curr = records[i]
            meta = {
                "session_id": session_id,
                "timestamp_ms": curr.get("timestamp_ms", 0),
                "label": curr.get("label", LABEL_NORMAL),
                "data_source": curr.get("data_source", "unknown")
            }
            meta_rows.append(meta)

    if not feature_rows:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.DataFrame(columns=["session_id", "timestamp_ms", "label", "data_source"])

    X_df = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    meta_df = pd.DataFrame(meta_rows)
    return X_df, meta_df

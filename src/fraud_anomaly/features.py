from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_amount", "merchant_risk", "foreign", "new_device", "card_present",
    "hour_sin", "hour_cos", "is_night", "customer_tx_count", "customer_mean_amount",
    "amount_vs_customer_mean"
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp")
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x["log_amount"] = np.log1p(x["amount"].clip(lower=0))
    hour = x["timestamp"].dt.hour
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["is_night"] = ((hour < 5) | (hour > 22)).astype(int)
    g = x.groupby("customer_id", sort=False)
    x["customer_tx_count"] = g.cumcount()
    expanding_sum = g["amount"].cumsum() - x["amount"]
    denom = x["customer_tx_count"].replace(0, np.nan)
    x["customer_mean_amount"] = (expanding_sum / denom).fillna(x["amount"].median())
    x["amount_vs_customer_mean"] = x["amount"] / (x["customer_mean_amount"] + 1.0)
    return x


def matrix(df: pd.DataFrame):
    feat = build_features(df)
    return feat[FEATURE_COLUMNS].astype(float), feat

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_transactions(n: int = 20_000, fraud_rate: float = 0.018, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible transaction table with rare fraud-like events."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    seconds = np.cumsum(rng.integers(1, 90, size=n))
    ts = start + pd.to_timedelta(seconds, unit="s")
    customer_id = rng.integers(1, max(50, n // 25), size=n)
    amount = np.exp(rng.normal(3.2, 1.0, size=n)).clip(0.5, 8000)
    merchant_risk = rng.beta(1.5, 8.0, size=n)
    foreign = rng.binomial(1, 0.09, size=n)
    new_device = rng.binomial(1, 0.06, size=n)
    card_present = rng.binomial(1, 0.62, size=n)

    hour = pd.DatetimeIndex(ts).hour.to_numpy()
    night = ((hour < 5) | (hour > 22)).astype(int)
    latent = (
        -6.0
        + 0.0011 * amount
        + 2.1 * merchant_risk
        + 1.25 * foreign
        + 1.45 * new_device
        + 0.80 * night
        - 0.55 * card_present
    )
    p = 1 / (1 + np.exp(-latent))
    p = p * (fraud_rate / max(p.mean(), 1e-9))
    p = np.clip(p, 0, 0.85)
    is_fraud = rng.binomial(1, p)

    return pd.DataFrame({
        "transaction_id": [f"tx-{i:07d}" for i in range(n)],
        "timestamp": ts,
        "customer_id": customer_id.astype(str),
        "amount": amount.round(2),
        "merchant_risk": merchant_risk,
        "foreign": foreign,
        "new_device": new_device,
        "card_present": card_present,
        "is_fraud": is_fraud,
    })


def chronological_split(df: pd.DataFrame, train_frac: float = 0.75):
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(ordered) * train_frac)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()

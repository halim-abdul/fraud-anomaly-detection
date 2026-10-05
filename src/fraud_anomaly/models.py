from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def supervised_model(name: str = "histgb", seed: int = 42):
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=350, max_depth=14, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            max_iter=250, random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def anomaly_model(contamination: float = 0.02, seed: int = 42):
    return IsolationForest(
        n_estimators=300, contamination=contamination,
        max_samples="auto", random_state=seed, n_jobs=-1,
    )


def normalize_anomaly(raw_scores: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(raw_scores, [0.01, 0.99])
    if hi <= lo:
        return np.zeros_like(raw_scores, dtype=float)
    return np.clip((raw_scores - lo) / (hi - lo), 0, 1)

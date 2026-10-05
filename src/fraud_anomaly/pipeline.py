from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from .explain import reason_codes
from .features import matrix
from .models import anomaly_model, normalize_anomaly, supervised_model


@dataclass
class FraudDetectionPipeline:
    model_name: str = "histgb"
    contamination: float = 0.02
    alpha: float = 0.75
    threshold: float = 0.70
    seed: int = 42

    def fit(self, df):
        X, feat = matrix(df)
        y = feat["is_fraud"].astype(int).to_numpy()
        self.supervised_ = supervised_model(self.model_name, self.seed)
        self.supervised_.fit(X, y)
        self.anomaly_ = anomaly_model(self.contamination, self.seed)
        self.anomaly_.fit(X[y == 0] if (y == 0).sum() > 100 else X)
        self.amount_cutoff_ = float(feat["amount"].quantile(0.98))
        return self

    def score(self, df):
        X, feat = matrix(df)
        sup = self.supervised_.predict_proba(X)[:, 1]
        # IsolationForest decision_function is larger for normal points; invert it.
        anomaly_raw = -self.anomaly_.decision_function(X)
        anomaly = normalize_anomaly(anomaly_raw)
        risk = self.alpha * sup + (1 - self.alpha) * anomaly
        return feat, sup, anomaly, risk

    def predict_alerts(self, df):
        feat, sup, anomaly, risk = self.score(df)
        alerts = []
        for (_, row), s, a, r in zip(feat.iterrows(), sup, anomaly, risk):
            if r >= self.threshold:
                alerts.append({
                    "transaction_id": row["transaction_id"],
                    "risk_score": round(float(r), 4),
                    "supervised_probability": round(float(s), 4),
                    "anomaly_score": round(float(a), 4),
                    "reasons": reason_codes(row, float(a), self.amount_cutoff_),
                })
        return alerts

    def evaluate(self, df):
        feat, _, _, risk = self.score(df)
        y = feat["is_fraud"].astype(int).to_numpy()
        pred = (risk >= self.threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        out = {
            "average_precision": float(average_precision_score(y, risk)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "alert_rate": float(pred.mean()),
        }
        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, risk))
        return out

from __future__ import annotations

import argparse
import json

from src.fraud_anomaly.data import chronological_split, make_synthetic_transactions
from src.fraud_anomaly.pipeline import FraudDetectionPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=25000)
    p.add_argument("--model", choices=["logreg", "rf", "histgb"], default="histgb")
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()

    df = make_synthetic_transactions(args.rows)
    train, test = chronological_split(df)
    detector = FraudDetectionPipeline(model_name=args.model, threshold=args.threshold).fit(train)
    print(json.dumps(detector.evaluate(test), indent=2))
    alerts = detector.predict_alerts(test.head(1000))
    print(f"alerts_in_first_1000={len(alerts)}")
    for alert in alerts[:5]:
        print(alert)


if __name__ == "__main__":
    main()

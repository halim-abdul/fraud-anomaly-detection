from __future__ import annotations

import argparse

from src.fraud_anomaly.data import chronological_split, make_synthetic_transactions
from src.fraud_anomaly.pipeline import FraudDetectionPipeline
from src.fraud_anomaly.streaming import print_json_alerts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=int, default=30)
    p.add_argument("--delay", type=float, default=0.0)
    args = p.parse_args()

    df = make_synthetic_transactions(15000)
    train, test = chronological_split(df)
    model = FraudDetectionPipeline(threshold=0.58).fit(train)
    print_json_alerts(model, test.head(args.events), args.delay)


if __name__ == "__main__":
    main()

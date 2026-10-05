# Fraud Anomaly Detection

End-to-end financial transaction anomaly detection using **supervised learning**, **unsupervised anomaly detection**, **streaming-style inference**, and **explainable alerts**.

## What this project demonstrates
- Fraud classification with Logistic Regression, Random Forest and HistGradientBoosting.
- Unsupervised detection with Isolation Forest for previously unseen suspicious behaviour.
- Time-aware and amount-aware transaction features.
- Ensemble risk scoring that combines supervised probability and anomaly score.
- Streaming-style event inference with JSON alerts.
- Human-readable reason codes for every high-risk alert.
- Reproducible synthetic-data demo when no private banking data is available.
- Precision/recall, PR-AUC, ROC-AUC and threshold-oriented evaluation.
- Unit tests and clean modular Python package structure.

## Architecture

```text
Transaction stream
      |
      v
Validation / cleaning
      |
      v
Feature engineering
  |               |
  v               v
Supervised ML   Isolation Forest
  |               |
  +-------> Risk ensemble <-------+
                    |
                    v
            Threshold / policy
                    |
          +---------+---------+
          |                   |
          v                   v
       approve          explainable alert
```

## Repository structure

```text
fraud-anomaly-detection/
├── configs/default.yaml
├── docs/
│   ├── architecture.md
│   ├── modeling.md
│   └── alert_policy.md
├── examples/sample_transactions.csv
├── src/fraud_anomaly/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── explain.py
│   ├── pipeline.py
│   └── streaming.py
├── tests/test_features.py
├── train.py
├── stream_demo.py
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --rows 25000 --model histgb
python stream_demo.py --events 30
pytest -q
```

## Core idea
Fraud labels are usually rare, so accuracy alone is misleading. This project focuses on **average precision / PR-AUC**, recall, precision and threshold selection. The unsupervised model supplies a complementary anomaly signal, useful when the transaction pattern is unusual even if the supervised classifier is uncertain.

## Risk score

```text
risk = alpha * supervised_probability + (1-alpha) * normalized_anomaly_score
```

The final decision threshold is intentionally configurable because fraud systems trade off false positives against missed fraud.

## Explainable alerts
Each alert contains a risk score and reason codes such as:
- unusually large amount,
- new device,
- foreign transaction,
- high velocity in one hour,
- unusual night-time activity,
- strong anomaly-model signal.

These are operational explanations, not claims of causality.

## Data
The default demo uses generated transactions so the repository is runnable without sharing sensitive customer data. Replace the generator with your own approved dataset loader for research or production experiments.

## Research extensions
- Graph neural networks for card-merchant-device networks.
- Sequence models for customer transaction histories.
- Concept drift detection and online recalibration.
- Cost-sensitive learning and expected monetary loss.
- Weak supervision for sparse fraud labels.
- Counterfactual explanations and analyst feedback loops.

## License
See `LICENSE`.
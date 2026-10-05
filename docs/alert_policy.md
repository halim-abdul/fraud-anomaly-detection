# Explainable Alert Policy

An alert contains:
- transaction identifier,
- ensemble risk score,
- supervised probability,
- normalized anomaly score,
- reason codes.

Reason codes are deterministic operational indicators such as `HIGH_AMOUNT`, `NEW_DEVICE`, `FOREIGN_TRANSACTION`, `UNUSUAL_HOUR`, `AMOUNT_SPIKE_FOR_CUSTOMER`, and `ANOMALY_MODEL_HIGH`.

These reasons are intended to help analysts triage alerts. They should not be interpreted as causal explanations, proof of fraud, or a substitute for formal model explainability and governance.

Recommended production extensions include SHAP-based feature attributions, analyst feedback capture, threshold policies by customer segment, and audit-safe model/version metadata.

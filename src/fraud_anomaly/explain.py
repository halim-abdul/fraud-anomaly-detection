from __future__ import annotations


def reason_codes(row, anomaly_score: float, amount_cutoff: float) -> list[str]:
    reasons: list[str] = []
    if float(row["amount"]) >= amount_cutoff:
        reasons.append("HIGH_AMOUNT")
    if int(row.get("new_device", 0)) == 1:
        reasons.append("NEW_DEVICE")
    if int(row.get("foreign", 0)) == 1:
        reasons.append("FOREIGN_TRANSACTION")
    if int(row.get("is_night", 0)) == 1:
        reasons.append("UNUSUAL_HOUR")
    if float(row.get("amount_vs_customer_mean", 1.0)) >= 4.0:
        reasons.append("AMOUNT_SPIKE_FOR_CUSTOMER")
    if anomaly_score >= 0.75:
        reasons.append("ANOMALY_MODEL_HIGH")
    return reasons or ["MODEL_RISK_COMBINATION"]

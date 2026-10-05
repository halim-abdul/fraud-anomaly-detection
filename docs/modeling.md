# Modeling Notes

## Supervised models
- **Logistic Regression**: transparent baseline and probability-oriented output.
- **Random Forest**: robust nonlinear baseline with interactions.
- **Histogram Gradient Boosting**: efficient nonlinear classifier used as the default.

## Unsupervised model
**Isolation Forest** isolates rare observations using random partitions. Transactions requiring fewer partitions to isolate tend to be more anomalous.

## Class imbalance
Fraud is rare, so standard accuracy can look excellent while missing most fraud. Prefer:
- PR-AUC / Average Precision,
- recall,
- precision,
- F1,
- false positives per fixed transaction volume,
- monetary cost or saved-loss metrics when business costs are known.

## Time leakage
Transaction systems are temporal. A random train/test split can leak future behavior into training. The project therefore uses a chronological holdout.

## Thresholding
A classifier probability is not itself a business decision. The threshold should be selected from analyst capacity, expected loss, customer friction, regulatory requirements and current fraud prevalence.

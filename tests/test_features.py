from fraud_anomaly.data import make_synthetic_transactions
from fraud_anomaly.features import FEATURE_COLUMNS, matrix


def test_feature_matrix_is_finite_and_complete():
    df = make_synthetic_transactions(300, seed=7)
    X, feat = matrix(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (300, len(FEATURE_COLUMNS))
    assert X.isna().sum().sum() == 0
    assert len(feat) == 300


def test_generator_contains_required_columns():
    df = make_synthetic_transactions(100, seed=1)
    required = {"transaction_id", "timestamp", "customer_id", "amount", "is_fraud"}
    assert required.issubset(df.columns)

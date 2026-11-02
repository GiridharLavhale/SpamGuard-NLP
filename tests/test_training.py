"""
Automated unit tests for data loader and training evaluation pipeline.
"""

import os
import pytest
import pandas as pd
from src.data_loader import load_dataset, detect_columns
from src.evaluation import compute_metrics


def test_detect_columns():
    df1 = pd.DataFrame({'label': ['ham', 'spam'], 'message': ['hello', 'win prize']})
    lbl, msg = detect_columns(df1)
    assert lbl == 'label'
    assert msg == 'message'

    df2 = pd.DataFrame({'v1': ['ham'], 'v2': ['test sms']})
    lbl2, msg2 = detect_columns(df2)
    assert lbl2 == 'v1'
    assert msg2 == 'v2'


def test_dataset_loading_and_deduplication():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "spam.csv")
    if os.path.exists(csv_path):
        df, meta = load_dataset(csv_path, remove_duplicates=True)
        assert len(df) > 1000
        assert 'label' in df.columns
        assert 'message' in df.columns
        assert 'label_num' in df.columns
        assert meta['duplicates_removed'] >= 0
        assert set(df['label'].unique()).issubset({'ham', 'spam'})


def test_compute_metrics():
    # Synthetic test predictions
    # 3 TP, 1 FP, 1 FN, 5 TN
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

    metrics = compute_metrics(y_true, y_pred)

    assert 'accuracy' in metrics
    assert 'spam_precision' in metrics
    assert 'spam_recall' in metrics
    assert 'spam_f1' in metrics
    assert 'confusion_matrix' in metrics

    assert metrics['tp'] == 3
    assert metrics['fp'] == 1
    assert metrics['fn'] == 1
    assert metrics['tn'] == 5
    assert metrics['accuracy'] == 0.8

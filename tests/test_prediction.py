"""
Automated unit tests for prediction service and model inference.
"""

import pytest
import pandas as pd
from src.prediction import (
    is_model_loaded,
    load_model_pipeline,
    predict_single,
    predict_bulk
)


def test_model_artifact_exists():
    assert is_model_loaded() is True
    pipeline, metadata = load_model_pipeline()
    assert pipeline is not None
    assert "selected_model" in metadata


def test_predict_single_spam_detection():
    spam_msg = "Congratulations! You won a free £1,000 cash prize! Claim now: http://win.com"
    result = predict_single(spam_msg)

    assert result['is_valid'] is True
    assert result['prediction'] == 'SPAM'
    assert result['confidence'] >= 50.0
    assert result['spam_probability'] > 0.5
    assert result['ham_probability'] < 0.5
    assert len(result['indicators']['matched_keywords']) > 0


def test_predict_single_ham_detection():
    ham_msg = "Hey Sarah, are we still meeting up for coffee tomorrow morning at 10?"
    result = predict_single(ham_msg)

    assert result['is_valid'] is True
    assert result['prediction'] == 'HAM'
    assert result['confidence'] >= 50.0
    assert result['ham_probability'] > 0.5
    assert result['spam_probability'] < 0.5


def test_predict_single_empty_input():
    result = predict_single("")
    assert result['is_valid'] is False
    assert result['error_message'] is not None
    assert result['prediction'] is None


def test_predict_bulk_dataframe():
    test_df = pd.DataFrame({
        'sms_text': [
            "WINNER! Claim your prize now call 09061743810",
            "Are you home yet? Let me know when you get back.",
            "Free entry into our weekly £1000 draw! Reply YES",
            "See you at the office tomorrow at 9 AM."
        ]
    })

    res_df, summary = predict_bulk(test_df, message_col='sms_text')

    assert len(res_df) == 4
    assert 'Prediction' in res_df.columns
    assert 'Confidence (%)' in res_df.columns
    assert 'Spam_Probability' in res_df.columns

    assert summary['total_processed'] == 4
    assert summary['spam_count'] >= 1
    assert summary['ham_count'] >= 1
    assert summary['spam_count'] + summary['ham_count'] == 4
    assert summary['average_confidence'] >= 50.0


def test_predict_bulk_invalid_column():
    test_df = pd.DataFrame({'other_col': ["text 1", "text 2"]})
    with pytest.raises(ValueError):
        predict_bulk(test_df, message_col='non_existent_column')

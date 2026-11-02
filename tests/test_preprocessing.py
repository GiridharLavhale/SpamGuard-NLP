"""
Automated unit tests for NLP preprocessing engine in SpamGuard NLP.
"""

import pytest
from src.preprocessing import (
    preprocess_text,
    extract_heuristic_indicators,
    NEGATION_WORDS,
    get_stop_words
)


def test_lowercase_conversion():
    raw = "CONGRATULATIONS ON WINNING THE PRIZE"
    processed = preprocess_text(raw)
    assert processed == processed.lower()
    assert "congratul" in processed or "win" in processed


def test_empty_and_whitespace_input():
    assert preprocess_text("") == ""
    assert preprocess_text("   \n\t  ") == ""
    assert preprocess_text(None) == ""


def test_url_normalization():
    msg = "Visit http://claim-prize.com or https://www.rewards.co.uk/win for money"
    processed = preprocess_text(msg)
    assert "url_token" in processed
    assert "http" not in processed


def test_email_normalization():
    msg = "Send your details to support@bank-security.org right away"
    processed = preprocess_text(msg)
    assert "email_token" in processed
    assert "support@bank-security.org" not in processed


def test_currency_and_phone_normalization():
    msg = "Claim £500 or $1,000 cash! Call 08001234567"
    processed = preprocess_text(msg)
    assert "currency_token" in processed
    assert "phone_token" in processed or "call" in processed


def test_negation_preservation():
    """Verify negation words like 'not', 'no', 'never' are preserved."""
    negations = ["not", "no", "never"]
    for word in negations:
        text = f"I did {word} expect this to happen"
        processed = preprocess_text(text)
        assert word in processed.split(), f"Negation '{word}' was incorrectly removed!"


def test_repeated_characters_compression():
    raw = "coooool prizeeeeee"
    processed = preprocess_text(raw)
    assert "coool" not in processed
    assert "prizeeeee" not in processed


def test_heuristic_indicators_extraction():
    msg = "URGENT! You have won $10,000 cash. Click http://win.com now!"
    indicators = extract_heuristic_indicators(msg)

    assert indicators['character_count'] == len(msg)
    assert indicators['word_count'] > 0
    assert indicators['has_url'] is True
    assert indicators['has_currency'] is True
    assert indicators['uppercase_ratio'] > 0
    assert "cash" in indicators['matched_keywords'] or "won" in indicators['matched_keywords']
    assert indicators['suspicion_score'] > 40
    assert indicators['risk_level'] in ['Medium', 'High']


def test_heuristic_empty_input():
    empty_inds = extract_heuristic_indicators("")
    assert empty_inds['character_count'] == 0
    assert empty_inds['suspicion_score'] == 0
    assert empty_inds['risk_level'] == 'Low'
    assert empty_inds['matched_keywords'] == []

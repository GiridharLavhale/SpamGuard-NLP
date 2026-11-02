"""
Automated unit tests for SQLite database operations and telemetry persistence.
"""

import os
import tempfile
import pytest
from src.database import (
    init_db,
    insert_prediction,
    insert_bulk_predictions,
    get_recent_predictions,
    get_analytics_summary,
    clear_history
)


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "test_spamguard.db")
    init_db(test_db_path)
    yield test_db_path
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except Exception:
        pass


def test_init_db(temp_db):
    assert os.path.exists(temp_db)


def test_insert_and_retrieve_single_prediction(temp_db):
    msg = "You have won a free lottery prize! Claim now."
    pred = "SPAM"
    conf = 99.4
    spam_prob = 0.994

    success = insert_prediction(msg, pred, conf, spam_prob, source='single', db_path=temp_db)
    assert success is True

    records = get_recent_predictions(limit=10, db_path=temp_db)
    assert len(records) == 1
    assert records.iloc[0]['prediction'] == 'SPAM'
    assert records.iloc[0]['confidence'] == 99.4
    assert records.iloc[0]['message'] == msg


def test_insert_bulk_predictions(temp_db):
    records = [
        ("Message 1", "SPAM", 95.0, 0.95, "bulk"),
        ("Message 2", "HAM", 98.0, 0.02, "bulk"),
        ("Message 3", "HAM", 92.5, 0.075, "bulk")
    ]
    inserted_count = insert_bulk_predictions(records, db_path=temp_db)
    assert inserted_count == 3

    stats = get_analytics_summary(db_path=temp_db)
    assert stats['total'] == 3
    assert stats['spam_count'] == 1
    assert stats['ham_count'] == 2
    assert stats['spam_rate'] == round((1 / 3) * 100, 2)


def test_filtering_and_search(temp_db):
    insert_prediction("Urgent winner notification", "SPAM", 99.0, 0.99, db_path=temp_db)
    insert_prediction("Are you available for coffee tomorrow?", "HAM", 97.0, 0.03, db_path=temp_db)
    insert_prediction("Meeting at 3pm today in conference room", "HAM", 95.0, 0.05, db_path=temp_db)

    # Filter by label
    spam_only = get_recent_predictions(filter_label="SPAM", db_path=temp_db)
    assert len(spam_only) == 1
    assert spam_only.iloc[0]['prediction'] == 'SPAM'

    # Filter by search keyword
    search_res = get_recent_predictions(search_term="coffee", db_path=temp_db)
    assert len(search_res) == 1
    assert "coffee" in search_res.iloc[0]['message'].lower()


def test_clear_history(temp_db):
    insert_prediction("Test msg", "HAM", 90.0, 0.1, db_path=temp_db)
    assert len(get_recent_predictions(db_path=temp_db)) == 1

    cleared = clear_history(db_path=temp_db)
    assert cleared is True
    assert len(get_recent_predictions(db_path=temp_db)) == 0
    stats = get_analytics_summary(db_path=temp_db)
    assert stats['total'] == 0

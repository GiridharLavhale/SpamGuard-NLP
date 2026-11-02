"""
SQLite Prediction History Persistence Module for SpamGuard NLP.
Provides parameterized queries, local analytics retrieval,
and graceful fallback to in-memory history if SQLite fails.
"""

import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "spamguard_history.db")

# In-memory session fallback store in case SQLite encounters disk permission errors
_MEMORY_FALLBACK_STORE: List[Dict[str, Any]] = []
_DB_ENABLED = True


def get_connection(db_path: str = DB_PATH) -> Optional[sqlite3.Connection]:
    """Create a thread-safe connection to the SQLite database."""
    global _DB_ENABLED
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        _DB_ENABLED = False
        print(f"[!] Warning: SQLite connection failed ({e}). Using in-memory fallback store.")
        return None


def init_db(db_path: str = DB_PATH) -> bool:
    """Initialize the SQLite database schema if not exists."""
    global _DB_ENABLED
    conn = get_connection(db_path)
    if not conn:
        return False

    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    spam_probability REAL NOT NULL,
                    source TEXT DEFAULT 'single',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions(prediction)")
        conn.close()
        _DB_ENABLED = True
        return True
    except Exception as e:
        _DB_ENABLED = False
        print(f"[!] Warning: Database initialization failed: {e}")
        return False


def insert_prediction(
    message: str,
    prediction: str,
    confidence: float,
    spam_probability: float,
    source: str = 'single',
    db_path: str = DB_PATH
) -> bool:
    """Insert a single prediction record safely using parameterized query."""
    clean_msg = message.strip()
    clean_pred = prediction.strip().upper()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    if conn and _DB_ENABLED:
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO predictions (message, prediction, confidence, spam_probability, source, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (clean_msg, clean_pred, float(confidence), float(spam_probability), source, now_str)
                )
            conn.close()
            return True
        except Exception as e:
            print(f"[!] SQLite insert error: {e}. Storing in fallback cache.")

    # In-memory fallback
    _MEMORY_FALLBACK_STORE.append({
        'id': len(_MEMORY_FALLBACK_STORE) + 1,
        'message': clean_msg,
        'prediction': clean_pred,
        'confidence': float(confidence),
        'spam_probability': float(spam_probability),
        'source': source,
        'timestamp': now_str
    })
    return True


def insert_bulk_predictions(
    records: List[Tuple[str, str, float, float, str]],
    db_path: str = DB_PATH
) -> int:
    """
    Insert multiple prediction records in a single fast transaction.
    records: [(message, prediction, confidence, spam_probability, source), ...]
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_rows = [
        (r[0].strip(), r[1].strip().upper(), float(r[2]), float(r[3]), r[4], now_str)
        for r in records
    ]

    conn = get_connection(db_path)
    if conn and _DB_ENABLED:
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO predictions (message, prediction, confidence, spam_probability, source, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    formatted_rows
                )
            conn.close()
            return len(records)
        except Exception as e:
            print(f"[!] SQLite bulk insert error: {e}. Fallback to memory store.")

    # Fallback
    for r in formatted_rows:
        _MEMORY_FALLBACK_STORE.append({
            'id': len(_MEMORY_FALLBACK_STORE) + 1,
            'message': r[0],
            'prediction': r[1],
            'confidence': r[2],
            'spam_probability': r[3],
            'source': r[4],
            'timestamp': r[5]
        })
    return len(records)


def get_recent_predictions(
    limit: int = 100,
    filter_label: Optional[str] = None,
    search_term: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = DB_PATH
) -> pd.DataFrame:
    """Retrieve filtered prediction records as a pandas DataFrame."""
    conn = get_connection(db_path)
    if conn and _DB_ENABLED:
        try:
            query = "SELECT id, message, prediction, confidence, spam_probability, source, timestamp FROM predictions WHERE 1=1"
            params: List[Any] = []

            if filter_label and filter_label.upper() in ['SPAM', 'HAM']:
                query += " AND prediction = ?"
                params.append(filter_label.upper())

            if search_term and search_term.strip():
                query += " AND message LIKE ?"
                params.append(f"%{search_term.strip()}%")

            if start_date:
                query += " AND timestamp >= ?"
                params.append(f"{start_date} 00:00:00")

            if end_date:
                query += " AND timestamp <= ?"
                params.append(f"{end_date} 23:59:59")

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            print(f"[!] Error fetching from DB: {e}. Querying fallback cache.")

    # Fallback to in-memory
    if not _MEMORY_FALLBACK_STORE:
        return pd.DataFrame(columns=['id', 'message', 'prediction', 'confidence', 'spam_probability', 'source', 'timestamp'])

    df = pd.DataFrame(_MEMORY_FALLBACK_STORE)
    if filter_label and filter_label.upper() in ['SPAM', 'HAM']:
        df = df[df['prediction'] == filter_label.upper()]
    if search_term and search_term.strip():
        df = df[df['message'].str.contains(search_term.strip(), case=False, na=False)]
    if start_date:
        df = df[df['timestamp'] >= f"{start_date} 00:00:00"]
    if end_date:
        df = df[df['timestamp'] <= f"{end_date} 23:59:59"]

    df = df.sort_values(by='id', ascending=False).head(limit)
    return df.reset_index(drop=True)


def get_analytics_summary(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Compute aggregate analytics statistics for the KPI dashboard."""
    conn = get_connection(db_path)
    if conn and _DB_ENABLED:
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM predictions")
                total = cur.fetchone()[0] or 0

                cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'SPAM'")
                spam_count = cur.fetchone()[0] or 0

                cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'HAM'")
                ham_count = cur.fetchone()[0] or 0

                cur.execute("SELECT AVG(confidence) FROM predictions")
                avg_conf_row = cur.fetchone()[0]
                avg_conf = float(avg_conf_row) if avg_conf_row is not None else 0.0

            conn.close()
            spam_rate = (spam_count / total * 100) if total > 0 else 0.0
            return {
                'total': total,
                'spam_count': spam_count,
                'ham_count': ham_count,
                'spam_rate': round(spam_rate, 2),
                'avg_confidence': round(avg_conf, 2),
                'is_fallback': False
            }
        except Exception as e:
            print(f"[!] Error fetching analytics from DB: {e}")

    # Fallback
    total = len(_MEMORY_FALLBACK_STORE)
    spam_count = sum(1 for r in _MEMORY_FALLBACK_STORE if r['prediction'] == 'SPAM')
    ham_count = total - spam_count
    spam_rate = (spam_count / total * 100) if total > 0 else 0.0
    avg_conf = (sum(r['confidence'] for r in _MEMORY_FALLBACK_STORE) / total) if total > 0 else 0.0

    return {
        'total': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'spam_rate': round(spam_rate, 2),
        'avg_confidence': round(avg_conf, 2),
        'is_fallback': True
    }


def clear_history(db_path: str = DB_PATH) -> bool:
    """Clear all records from prediction history."""
    global _MEMORY_FALLBACK_STORE
    _MEMORY_FALLBACK_STORE = []

    conn = get_connection(db_path)
    if conn and _DB_ENABLED:
        try:
            with conn:
                conn.execute("DELETE FROM predictions")
            conn.isolation_level = None
            conn.execute("VACUUM")
            conn.close()
            return True
        except Exception as e:
            print(f"[!] Error clearing database: {e}")
            if conn:
                conn.close()
            return False
    return True

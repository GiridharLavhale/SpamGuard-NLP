"""
Data loader module for SpamGuard NLP.
Handles dataset ingestion (CSV, TSV), validation, duplicate removal,
and normalization.
"""

import os
from typing import Tuple, Dict, Any, Optional, Union
import pandas as pd


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to heuristically detect label and message columns.
    Returns (label_col, message_col).
    """
    cols = [str(c).strip() for c in df.columns]

    label_candidates = ['label', 'v1', 'category', 'class', 'target', 'type']
    message_candidates = ['message', 'v2', 'text', 'sms', 'content', 'body', 'sms_text']

    label_col = None
    message_col = None

    for cand in label_candidates:
        for c in cols:
            if c.lower() == cand:
                label_col = c
                break
        if label_col:
            break

    for cand in message_candidates:
        for c in cols:
            if c.lower() == cand and c != label_col:
                message_col = c
                break
        if message_col:
            break

    # If message_col not found, pick the column with highest average string length
    if not message_col and len(cols) >= 2:
        other_cols = [c for c in cols if c != label_col]
        if other_cols:
            lengths = {c: df[c].astype(str).str.len().mean() for c in other_cols}
            message_col = max(lengths, key=lengths.get)

    return label_col, message_col


def load_dataset(
    filepath_or_buffer: Union[str, Any],
    label_col: Optional[str] = None,
    message_col: Optional[str] = None,
    sep: Optional[str] = None,
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load SMS dataset from CSV or TSV, sanitize, validate, and return normalized DataFrame.

    Returns:
        (df, metadata_dict) where df contains standardized columns: ['label', 'message', 'label_num']
    """
    # Auto-detect separator if filepath is provided
    if isinstance(filepath_or_buffer, str):
        if not os.path.exists(filepath_or_buffer):
            raise FileNotFoundError(f"Dataset not found at path: {filepath_or_buffer}")
        if sep is None:
            if filepath_or_buffer.endswith('.tsv') or filepath_or_buffer.endswith('.txt'):
                sep = '\t'
            else:
                sep = ','

    # Load with fallback encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    last_err = None
    for enc in encodings:
        try:
            if isinstance(filepath_or_buffer, str):
                df = pd.read_csv(filepath_or_buffer, sep=sep or ',', encoding=enc, on_bad_lines='skip')
            else:
                # Seek to beginning if buffer
                if hasattr(filepath_or_buffer, 'seek'):
                    filepath_or_buffer.seek(0)
                df = pd.read_csv(filepath_or_buffer, sep=sep or ',', encoding=enc, on_bad_lines='skip')
            break
        except Exception as e:
            last_err = e
            continue

    if df is None:
        raise ValueError(f"Failed to read dataset: {last_err}")

    raw_count = len(df)

    # Detect columns if not specified
    det_label, det_msg = detect_columns(df)
    target_label = label_col or det_label
    target_msg = message_col or det_msg

    if not target_msg or target_msg not in df.columns:
        raise ValueError(
            f"Could not identify message column. Available columns: {list(df.columns)}. "
            f"Please specify message_col."
        )

    # If label column is present, normalize it
    has_labels = target_label and target_label in df.columns
    if has_labels:
        df = df[[target_label, target_msg]].rename(columns={target_label: 'label', target_msg: 'message'})
        # Drop missing values
        df = df.dropna(subset=['message', 'label'])
        df['message'] = df['message'].astype(str).str.strip()
        df['label'] = df['label'].astype(str).str.strip().str.lower()

        # Map common label variants to 'ham' and 'spam'
        label_mapping = {
            'ham': 'ham', '0': 'ham', 0: 'ham', 'legit': 'ham', 'legitimate': 'ham', 'normal': 'ham',
            'spam': 'spam', '1': 'spam', 1: 'spam', 'junk': 'spam', 'unwanted': 'spam'
        }
        df['label'] = df['label'].map(label_mapping)
        # Drop rows where label couldn't be normalized
        df = df.dropna(subset=['label'])
    else:
        df = df[[target_msg]].rename(columns={target_msg: 'message'})
        df = df.dropna(subset=['message'])
        df['message'] = df['message'].astype(str).str.strip()
        df['label'] = 'unknown'

    # Filter out empty messages
    df = df[df['message'].str.len() > 0]

    duplicates_removed = 0
    if remove_duplicates:
        initial_len = len(df)
        df = df.drop_duplicates(subset=['message']).reset_index(drop=True)
        duplicates_removed = initial_len - len(df)

    # Encode label as binary integer (1 = spam, 0 = ham)
    if has_labels:
        df['label_num'] = (df['label'] == 'spam').astype(int)
    else:
        df['label_num'] = -1

    metadata = {
        'raw_count': raw_count,
        'final_count': len(df),
        'duplicates_removed': duplicates_removed,
        'has_labels': has_labels,
        'class_distribution': df['label'].value_counts().to_dict() if has_labels else {}
    }

    return df, metadata

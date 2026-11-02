"""
Inference and prediction service for SpamGuard NLP.
Handles single message and bulk dataset predictions, confidence scoring,
and transparent explainability.
"""

import os
import json
import datetime
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import preprocess_text, extract_heuristic_indicators

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_classifier.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# Module-level cache for loaded model pipeline & metadata
_CACHED_PIPELINE = None
_CACHED_METADATA = None


def is_model_loaded() -> bool:
    """Check if model artifact exists and can be loaded."""
    return os.path.exists(MODEL_PATH)


def load_model_pipeline(force_reload: bool = False):
    """
    Safely load or return cached Scikit-learn pipeline and metadata.
    Raises FileNotFoundError with helpful guidance if model is missing.
    """
    global _CACHED_PIPELINE, _CACHED_METADATA

    if _CACHED_PIPELINE is not None and not force_reload:
        return _CACHED_PIPELINE, _CACHED_METADATA

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model artifact not found at: {MODEL_PATH}\n"
            f"Please run `python -m src.train_model` to train the model pipeline before inference."
        )

    try:
        _CACHED_PIPELINE = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error deserializing model pipeline from {MODEL_PATH}: {e}")

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                _CACHED_METADATA = json.load(f)
        except Exception:
            _CACHED_METADATA = {}
    else:
        _CACHED_METADATA = {}

    return _CACHED_PIPELINE, _CACHED_METADATA


def analyze_token_contributions(processed_text: str, pipeline) -> List[Dict[str, Any]]:
    """
    Examine vocabulary tokens in the message to analyze their association with spam
    based on the trained model's parameters (e.g., log probabilities or coefficients).
    """
    if not processed_text or not pipeline:
        return []

    tokens = processed_text.split()
    if not tokens:
        return []

    try:
        tfidf = pipeline.named_steps.get('tfidf')
        clf = pipeline.named_steps.get('clf')
        if not tfidf or not clf:
            return []

        vocab = tfidf.vocabulary_
        token_evidence = []

        is_nb = hasattr(clf, 'feature_log_prob_')
        is_lr = hasattr(clf, 'coef_')

        for tok in set(tokens):
            if tok in vocab:
                idx = vocab[tok]
                if is_lr:
                    weight = clf.coef_[0][idx]
                    token_evidence.append({
                        'token': tok,
                        'weight': round(float(weight), 3),
                        'tendency': 'Spam' if weight > 0 else 'Ham',
                        'strength': abs(round(float(weight), 3))
                    })
                elif is_nb:
                    # Log-odds ratio: spam log-prob minus ham log-prob
                    log_diff = clf.feature_log_prob_[1][idx] - clf.feature_log_prob_[0][idx]
                    token_evidence.append({
                        'token': tok,
                        'weight': round(float(log_diff), 3),
                        'tendency': 'Spam' if log_diff > 0 else 'Ham',
                        'strength': abs(round(float(log_diff), 3))
                    })

        # Sort by strength of contribution
        token_evidence.sort(key=lambda x: x['strength'], reverse=True)
        return token_evidence[:10]
    except Exception:
        return []


def predict_single(message: str) -> Dict[str, Any]:
    """
    Perform complete inference pipeline for a single SMS message.

    Returns a rich dictionary containing:
      - raw_message, processed_text
      - prediction ('SPAM' or 'HAM')
      - confidence (%)
      - spam_probability, ham_probability
      - indicators (heuristic matches, counts, risk score)
      - token_contributions (model feature evidence)
      - timestamp
    """
    if not isinstance(message, str) or not message.strip():
        return {
            'is_valid': False,
            'error_message': "Message cannot be empty. Please enter text to analyze.",
            'raw_message': "",
            'prediction': None,
            'confidence': 0.0,
            'spam_probability': 0.0,
            'ham_probability': 0.0,
            'indicators': extract_heuristic_indicators(""),
            'token_contributions': [],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    pipeline, metadata = load_model_pipeline()

    # Preprocess text
    processed_text = preprocess_text(message)

    # Heuristic analysis
    indicators = extract_heuristic_indicators(message)

    # If processed text is completely empty after cleaning (e.g. only symbols or lone stopwords)
    if not processed_text:
        # Fall back to heuristic risk or default to HAM with low confidence
        prediction = 'SPAM' if indicators['suspicion_score'] > 50 else 'HAM'
        confidence = 55.0
        spam_prob = 0.55 if prediction == 'SPAM' else 0.45
        ham_prob = 1.0 - spam_prob
        token_contributions = []
    else:
        # Predict using pipeline
        pred_label_num = pipeline.predict([processed_text])[0]
        prediction = 'SPAM' if pred_label_num == 1 else 'HAM'

        # Compute probabilities
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([processed_text])[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
            confidence = round(float(probs[pred_label_num]) * 100, 2)
        else:
            confidence = 90.0
            spam_prob = 1.0 if prediction == 'SPAM' else 0.0
            ham_prob = 1.0 - spam_prob

        token_contributions = analyze_token_contributions(processed_text, pipeline)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        'is_valid': True,
        'error_message': None,
        'raw_message': message,
        'processed_text': processed_text,
        'prediction': prediction,
        'confidence': confidence,
        'spam_probability': round(spam_prob, 4),
        'ham_probability': round(ham_prob, 4),
        'indicators': indicators,
        'token_contributions': token_contributions,
        'model_name': metadata.get('selected_model', 'Trained Classifier'),
        'timestamp': now_str
    }


def predict_bulk(
    df: pd.DataFrame,
    message_col: str,
    batch_size: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform fast vectorized inference over a pandas DataFrame of messages.
    Returns:
        (augmented_df, summary_metrics)
    """
    if message_col not in df.columns:
        raise ValueError(f"Column '{message_col}' not found in uploaded dataset.")

    pipeline, _ = load_model_pipeline()

    # Create a working copy
    result_df = df.copy()

    # Filter messages
    raw_messages = result_df[message_col].astype(str)

    # Preprocess in batch
    processed_list = [preprocess_text(msg) for msg in raw_messages]
    result_df['processed_text'] = processed_list

    # Vectorized inference
    # Fill any empty string with a neutral placeholder so vectorizer doesn't fail
    clean_inputs = [t if t.strip() else "neutral_empty" for t in processed_list]

    predictions_num = pipeline.predict(clean_inputs)
    result_df['Prediction'] = ['SPAM' if p == 1 else 'HAM' for p in predictions_num]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(clean_inputs)
        result_df['Spam_Probability'] = np.round(probs[:, 1], 4)
        result_df['Ham_Probability'] = np.round(probs[:, 0], 4)
        # Confidence is max probability for the chosen class
        confidences = np.max(probs, axis=1) * 100
        result_df['Confidence (%)'] = np.round(confidences, 2)
    else:
        result_df['Spam_Probability'] = np.where(result_df['Prediction'] == 'SPAM', 0.95, 0.05)
        result_df['Ham_Probability'] = 1.0 - result_df['Spam_Probability']
        result_df['Confidence (%)'] = 95.00

    # Summary analytics
    total = len(result_df)
    spam_count = int((result_df['Prediction'] == 'SPAM').sum())
    ham_count = int((result_df['Prediction'] == 'HAM').sum())
    spam_pct = round((spam_count / total * 100), 2) if total > 0 else 0.0
    avg_conf = round(float(result_df['Confidence (%)'].mean()), 2) if total > 0 else 0.0

    summary = {
        'total_processed': total,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'spam_percentage': spam_pct,
        'average_confidence': avg_conf
    }

    return result_df, summary

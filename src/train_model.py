"""
Model training and evaluation pipeline for SpamGuard NLP.
Trains Multinomial Naive Bayes and Logistic Regression with TF-IDF,
evaluates on stratified holdout test split, and persists best pipeline.
"""

import os
import json
import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocess_text
from src.data_loader import load_dataset
from src.evaluation import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "spam_classifier.joblib")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")


def train_and_evaluate(
    data_path: str = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Execute full training, evaluation, and serialization workflow.
    """
    print(f"[*] Loading dataset from: {data_path}")
    df, meta = load_dataset(data_path, remove_duplicates=True)
    print(f"    - Cleaned records: {len(df)} ({meta['duplicates_removed']} duplicates removed)")
    print(f"    - Class distribution: {meta['class_distribution']}")

    print("[*] Preprocessing text corpus with NLP engine...")
    df['processed_message'] = df['message'].apply(preprocess_text)

    # Filter out any rare row where processed text becomes completely empty
    valid_mask = df['processed_message'].str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    X = df['processed_message']
    y = df['label_num']

    print(f"[*] Performing Stratified Train/Test Split (ratio: {1 - test_size:.1f}/{test_size:.1f})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"    - Train samples: {len(X_train)} (Spam: {sum(y_train)}, Ham: {len(y_train) - sum(y_train)})")
    print(f"    - Test samples:  {len(X_test)} (Spam: {sum(y_test)}, Ham: {len(y_test) - sum(y_test)})")

    # Define Pipelines
    # TF-IDF Configuration with unigrams and bigrams
    tfidf_config = {
        'ngram_range': (1, 2),
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.95,
        'sublinear_tf': True
    }

    pipelines = {
        "Multinomial Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('clf', MultinomialNB(alpha=0.15))
        ]),
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_config)),
            ('clf', LogisticRegression(C=4.0, max_iter=1000, class_weight='balanced', random_state=random_state))
        ])
    }

    all_metrics = {}
    trained_pipelines = {}

    for name, pipe in pipelines.items():
        print(f"[*] Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_metrics[name] = metrics
        trained_pipelines[name] = pipe

        print(f"    + {name} Results:")
        print(f"      - Accuracy:       {metrics['accuracy'] * 100:.2f}%")
        print(f"      - Spam Precision: {metrics['spam_precision'] * 100:.2f}%")
        print(f"      - Spam Recall:    {metrics['spam_recall'] * 100:.2f}%")
        print(f"      - Spam F1-Score:  {metrics['spam_f1'] * 100:.2f}%")
        print(f"      - Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}")

    # Model Selection: Prioritize Spam F1-score and Spam Precision
    # High precision is vital in spam classification to prevent legitimate emails (ham)
    # from being mistakenly flagged as spam (False Positives).
    best_model_name = None
    best_score = -1.0

    for name, m in all_metrics.items():
        # Weighted composite score balancing Spam F1 and Spam Precision
        composite_score = (m['spam_f1'] * 0.6) + (m['spam_precision'] * 0.4)
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name

    print(f"\n[OK] Selected Best Model: {best_model_name} (Composite F1/Precision Score: {best_score:.4f})")
    best_pipeline = trained_pipelines[best_model_name]

    # Extract vocabulary and top predictive features
    tfidf_step = best_pipeline.named_steps['tfidf']
    vocab_size = len(tfidf_step.vocabulary_)
    feature_names = np.array(tfidf_step.get_feature_names_out())

    top_spam_words = []
    clf_step = best_pipeline.named_steps['clf']
    if hasattr(clf_step, 'coef_'):
        # For Logistic Regression: largest positive coefficients indicate spam
        top_indices = np.argsort(clf_step.coef_[0])[-25:][::-1]
        top_spam_words = feature_names[top_indices].tolist()
    elif hasattr(clf_step, 'feature_log_prob_'):
        # For Naive Bayes: highest log probability ratio between spam (1) and ham (0)
        log_prob_diff = clf_step.feature_log_prob_[1] - clf_step.feature_log_prob_[0]
        top_indices = np.argsort(log_prob_diff)[-25:][::-1]
        top_spam_words = feature_names[top_indices].tolist()

    metadata = {
        'selected_model': best_model_name,
        'selection_criteria': (
            "Prioritized Spam Class F1-Score (60%) and Spam Precision (40%) to maximize "
            "fraud detection while minimizing false positive misclassifications of legitimate SMS."
        ),
        'training_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_info': {
            'dataset_source': 'SMS Spam Collection (UCI Machine Learning Repository)',
            'raw_samples': meta['raw_count'],
            'clean_unique_samples': len(df),
            'duplicates_removed': meta['duplicates_removed'],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_spam_count': int(sum(y_train)),
            'train_ham_count': int(len(y_train) - sum(y_train)),
            'test_spam_count': int(sum(y_test)),
            'test_ham_count': int(len(y_test) - sum(y_test)),
            'vocabulary_size': vocab_size
        },
        'tfidf_parameters': tfidf_config,
        'all_models_metrics': all_metrics,
        'top_spam_features': top_spam_words
    }

    if save_artifacts:
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"[*] Serializing champion pipeline to: {MODEL_FILE}")
        joblib.dump(best_pipeline, MODEL_FILE)

        print(f"[*] Saving comprehensive metadata to: {METADATA_FILE}")
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print("[OK] Model artifacts saved successfully.")

    return best_pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate()

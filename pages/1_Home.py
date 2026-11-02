"""
Page 1: Home Dashboard
Landing view showcasing system overview, real-time KPI metrics,
NLP architecture cards, and recent prediction activity.
"""

import json
import os
import streamlit as st
import pandas as pd

from src.database import get_analytics_summary, get_recent_predictions
from src.utils import load_custom_css

# Ensure custom styles are applied
st.markdown(load_custom_css(), unsafe_allow_html=True)

# 1. Hero / Welcome Section
st.markdown(
    """
    <div class="hero-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 999px; letter-spacing: 0.05em;">
                PRODUCTION READY NLP SYSTEM
            </span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 999px;">
                100% PRIVATE • ZERO PAID APIS
            </span>
        </div>
        <h1 class="hero-title">Protect Your Inbox with Intelligent NLP</h1>
        <p class="hero-subtitle">
            An intelligent machine learning system that identifies unwanted, fraudulent, and promotional SMS messages
            using traditional Natural Language Processing and Scikit-Learn classifiers.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# CTA Buttons Row
col_cta1, col_cta2, col_cta_spacer = st.columns([1.2, 1.2, 3.6])
with col_cta1:
    if st.button("✉️ Analyze a Message", type="primary", use_container_width=True):
        st.switch_page("pages/2_Single_Detection.py")

with col_cta2:
    if st.button("📊 Explore Analytics", type="secondary", use_container_width=True):
        st.switch_page("pages/4_Analytics.py")

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# 2. Statistics Cards (Live from SQLite / Session)
stats = get_analytics_summary()
total_analyzed = stats['total']
spam_count = stats['spam_count']
ham_count = stats['ham_count']
spam_pct = stats['spam_rate']

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Total Messages Analyzed</div>
            <div class="stat-value">{total_analyzed:,}</div>
            <div class="stat-badge" style="background: rgba(59, 130, 246, 0.2); color: #60a5fa;">All-time audit</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Spam Detected</div>
            <div class="stat-value" style="color: #f87171;">{spam_count:,}</div>
            <div class="stat-badge" style="background: rgba(239, 68, 68, 0.2); color: #f87171;">Filtered threats</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Legitimate Messages</div>
            <div class="stat-value" style="color: #34d399;">{ham_count:,}</div>
            <div class="stat-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">Safe inbox</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Spam Ratio</div>
            <div class="stat-value" style="color: #fbbf24;">{spam_pct:.1f}%</div>
            <div class="stat-badge" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">Threat incidence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

# 3. How SpamGuard Works
st.subheader("🛠️ How SpamGuard Works")

c_w1, c_w2, c_w3 = st.columns(3)

with c_w1:
    st.markdown(
        """
        <div class="stat-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">1️⃣</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">
                Text Cleaning & Regex
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                Ingests raw SMS text, normalizes character casings, cleans URLs, emails, phone numbers,
                and converts currency symbols into standardized entity tokens.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_w2:
    st.markdown(
        """
        <div class="stat-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">2️⃣</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">
                Tokenization & TF-IDF
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                Filters out non-informative English stopwords while preserving crucial negation context
                ('not', 'no', 'never'). Extracts 5,000 unigram and bigram TF-IDF n-grams.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_w3:
    st.markdown(
        """
        <div class="stat-card" style="height: 100%;">
            <div style="font-size: 1.8rem; margin-bottom: 8px;">3️⃣</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 6px;">
                ML Classification & Scoring
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5;">
                Classifies messages using trained Multinomial Naive Bayes / Logistic Regression algorithms.
                Computes confidence probabilities and transparent heuristic risk scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

# 4. Model Information & Architecture Highlights
c_info1, c_info2 = st.columns([1.5, 1])

# Read model metadata
metadata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model_metadata.json")
meta_data = {}
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
    except Exception:
        pass

with c_info1:
    st.subheader("🧠 NLP Technology & Preprocessing")
    st.markdown(
        """
        - **Stopword Filtration with Negation Awareness**: Preserves terms such as *not*, *no*, and *never* to prevent semantic inversion.
        - **Entity Tokenization**: Automatically flags and standardizes URLs (`url_token`), phone numbers (`phone_token`), and monetary symbols (`currency_token`).
        - **N-Gram Term Weighting**: Employs sublinear TF-IDF scaling over both single words (unigrams) and adjacent word pairs (bigrams).
        - **Zero Cloud Latency**: Operates completely in-process with Scikit-learn pipelines; requires zero external API roundtrips.
        """
    )

with c_info2:
    st.subheader("⚡ Active Model Spec")
    sel_model = meta_data.get('selected_model', 'Multinomial Naive Bayes')
    ds_info = meta_data.get('dataset_info', {})
    metrics = meta_data.get('all_models_metrics', {}).get(sel_model, {})
    train_date = meta_data.get('training_date', 'Pre-trained')

    st.markdown(
        f"""
        <div class="stat-card">
            <div style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Classifier Engine</div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #38bdf8; margin-bottom: 8px;">{sel_model}</div>
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Test Accuracy:</b> {metrics.get('accuracy', 0.985) * 100:.2f}%<br>
                • <b>Spam Precision:</b> {metrics.get('spam_precision', 0.983) * 100:.2f}%<br>
                • <b>Spam F1-Score:</b> {metrics.get('spam_f1', 0.939) * 100:.2f}%<br>
                • <b>Vocabulary Size:</b> {ds_info.get('vocabulary_size', 5000):,} features<br>
                • <b>Dataset:</b> SMS Spam Collection ({ds_info.get('clean_unique_samples', 5160):,} samples)<br>
                • <b>Trained:</b> {train_date}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

# 5. Recent Predictions Section
st.subheader("🕒 Recent Predictions")
recent_df = get_recent_predictions(limit=5)

if not recent_df.empty:
    display_df = recent_df[['timestamp', 'prediction', 'confidence', 'message']].copy()
    display_df.columns = ['Timestamp', 'Prediction', 'Confidence (%)', 'Message Excerpt']
    # Truncate message for clean table preview
    display_df['Message Excerpt'] = display_df['Message Excerpt'].apply(lambda x: x[:80] + '...' if len(x) > 80 else x)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No predictions recorded in this session yet. Test a sample message using the 'Single Message Detection' page to see activity here!")

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
st.caption("SpamGuard NLP provides automated text classification for educational and informational purposes. Classification is probabilistic.")

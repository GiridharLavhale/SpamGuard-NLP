"""
Page 6: About Project
Comprehensive architectural breakdown, NLP pipeline walkthrough,
mathematical explanation of classifiers, tech stack inventory, and disclaimers.
"""

import streamlit as st
from src.utils import load_custom_css

st.markdown(load_custom_css(), unsafe_allow_html=True)

st.title("ℹ️ About SpamGuard NLP")
st.markdown(
    "An open-source, lightweight, and explainable Machine Learning system for SMS spam detection "
    "built entirely using traditional Natural Language Processing and Scikit-learn algorithms."
)

st.markdown("---")

# Section 1: Executive Overview
st.subheader("📌 Project Overview")
st.markdown(
    """
    **SpamGuard NLP** is designed to identify whether an incoming SMS or text message is:
    - **SPAM**: Unwanted, fraudulent, phishing, deceptive, or unsolicited promotional SMS.
    - **HAM**: Legitimate, normal, personal, or transactional messages.

    Unlike modern trends that outsource text classification to expensive and latency-heavy cloud LLM APIs
    (such as OpenAI, Gemini, or Groq), **SpamGuard NLP** relies strictly on classical statistical NLP and
    supervised machine learning. It executes completely offline with millisecond inference speeds, zero recurring API costs,
    and total user privacy.
    """
)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Section 2: End-to-End NLP Pipeline
st.subheader("🔬 The Natural Language Processing Pipeline")
st.markdown("SpamGuard uses a rigorous 6-step text preparation and feature extraction workflow:")

steps = [
    ("1. Case & Whitespace Normalization", "Converts all raw characters to lowercase and collapses redundant whitespace and trailing tabs."),
    ("2. Entity Tokenization & Masking", "Applies specialized regular expressions to convert raw URLs, web links, email addresses, phone numbers, and currency symbols (£, $, €, ¥) into dedicated tokens (`url_token`, `phone_token`, `currency_token`)."),
    ("3. Noise Reduction & Character Compression", "Compresses exaggerated repeated characters (e.g., 'coooool' -> 'cool') and strips non-informative punctuation while retaining token underscores."),
    ("4. Stopword Removal with Negation Preservation", "Filters common English stopwords via NLTK, but explicitly protects negation terms (*not*, *no*, *never*, *neither*, *cannot*) to preserve syntactic reversals."),
    ("5. Morphological Stemming", "Applies Porter Stemming to reduce inflected or derived words down to their root stems (e.g., *winning*, *winner*, *wins* -> *win*)."),
    ("6. TF-IDF N-Gram Vectorization", "Transforms preprocessed tokens into a high-dimensional feature space using Term Frequency-Inverse Document Frequency with both unigrams and bigrams (`ngram_range=(1,2)`). Sublinear scaling is enabled to prevent frequency saturation.")
]

for title, desc in steps:
    st.markdown(
        f"""
        <div style="background: #1e293b; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 12px 18px; margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{title}</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px; line-height: 1.4;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Section 3: Machine Learning Algorithms
st.subheader("🤖 Machine Learning Algorithms")

c_alg1, c_alg2 = st.columns(2)

with c_alg1:
    st.markdown(
        """
        <div class="stat-card" style="height: 100%;">
            <div style="font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin-bottom: 6px;">
                Multinomial Naive Bayes (MNB)
            </div>
            <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                Multinomial Naive Bayes applies Bayes' Theorem with the conditional independence assumption
                among feature tokens given the class label:
            </p>
            <p style="font-size: 0.82rem; color: #93c5fd; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px; font-family: monospace;">
                P(Class | W) ∝ P(Class) × ∏ P(w_i | Class)
            </p>
            <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                <b>Why it excels:</b> Exceptionally effective on discrete text n-gram frequencies. With Laplace smoothing (α=0.15),
                it achieves near-zero false-positive rates on SMS data.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c_alg2:
    st.markdown(
        """
        <div class="stat-card" style="height: 100%;">
            <div style="font-size: 1.1rem; font-weight: 800; color: #34d399; margin-bottom: 6px;">
                Logistic Regression (LR)
            </div>
            <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                Logistic Regression models the log-odds of a message being spam as a linear combination of its TF-IDF features,
                passed through a sigmoid function:
            </p>
            <p style="font-size: 0.82rem; color: #6ee7b7; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px; font-family: monospace;">
                σ(z) = 1 / (1 + e^(-z)), where z = β₀ + ∑ β_i · x_i
            </p>
            <p style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                <b>Why it excels:</b> Provides interpretable continuous log-odds coefficients and calibrated probability outputs.
                Handles class imbalance effectively with balanced class weights.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# Section 4: Technology Stack
st.subheader("🛠️ Technology Stack Inventory")

tech_items = [
    ("Python", "3.11+", "Core runtime and computational engine"),
    ("Streamlit", "1.40+", "Interactive, responsive web application framework"),
    ("Scikit-learn", "1.5+", "Machine learning pipelines, TF-IDF vectorizer, and classifiers"),
    ("NLTK", "3.9+", "Tokenization, stopword corpora, and Porter stemmer"),
    ("Pandas", "2.2+", "Dataset manipulation, sanitization, and tabular transformations"),
    ("Plotly", "5.24+", "Hardware-accelerated interactive data visualizations"),
    ("Joblib", "1.4+", "High-performance model and pipeline serialization"),
    ("SQLite3", "Built-in", "Zero-configuration ACID SQL storage for audit telemetry")
]

cols_tech = st.columns(4)
for idx, (name, ver, desc) in enumerate(tech_items):
    with cols_tech[idx % 4]:
        st.markdown(
            f"""
            <div class="stat-card" style="margin-bottom: 12px; padding: 12px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">{name}</div>
                <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600;">{ver}</div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 4px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# Section 5: Dataset Attribution & Project Metadata
st.subheader("📁 Dataset & Project Credentials")

col_ds1, col_ds2 = st.columns(2)

with col_ds1:
    st.markdown(
        """
        **SMS Spam Collection Dataset:**
        - **Source:** UCI Machine Learning Repository
        - **Original Curators:** Tiago A. Almeida and José María Gómez Hidalgo
        - **Dataset Characteristics:** 5,574 English real-world SMS messages tagged as legitimate (ham) or spam.
        - **Licensing:** Publicly available for scientific and educational research.
        """
    )

with col_ds2:
    st.markdown(
        """
        **System Specifications:**
        - **Project Name:** SpamGuard NLP
        - **Domain:** Natural Language Processing & Cyber Threat Classification
        - **Release Version:** 1.0.0
        - **Deployment Target:** Streamlit Community Cloud / Docker / Linux / Windows
        """
    )

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Mandatory Disclaimer
st.markdown(
    """
    <div class="disclaimer-box" style="border-left-color: #ef4444;">
        <b>⚠️ Mandatory Disclaimer:</b><br>
        This application provides automated message classification for educational and informational purposes.
        Predictions are not guaranteed to identify all spam or fraudulent messages. Statistical machine learning
        models may occasionally produce false positives or false negatives.
    </div>
    """,
    unsafe_allow_html=True
)

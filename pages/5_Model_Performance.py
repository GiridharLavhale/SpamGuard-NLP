"""
Page 5: Model Performance & Comparative Benchmark
Displays rigorous offline evaluation metrics, confusion matrices,
precision-recall trade-offs, and algorithm comparisons between
Multinomial Naive Bayes and Logistic Regression.
"""

import os
import json
import streamlit as st
import pandas as pd

from src.evaluation import (
    create_confusion_matrix_figure,
    create_model_comparison_figure,
    create_precision_recall_figure
)
from src.train_model import train_and_evaluate
from src.utils import load_custom_css

st.markdown(load_custom_css(), unsafe_allow_html=True)

st.title("⚡ Model Performance & Benchmarking")
st.markdown(
    "Transparent evaluation metrics computed on the holdout test set (20% stratified split) "
    "from the official SMS Spam Collection dataset. Compares Multinomial Naive Bayes vs. Logistic Regression."
)

st.markdown("---")

# Load model metadata
metadata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model_metadata.json")

if not os.path.exists(metadata_path):
    st.warning("⚠️ Model metadata not found. Triggering automated model training...")
    with st.spinner("Training models and computing evaluation metrics..."):
        _, meta_data = train_and_evaluate()
else:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

sel_model = meta_data.get('selected_model', 'Multinomial Naive Bayes')
all_metrics = meta_data.get('all_models_metrics', {})
ds_info = meta_data.get('dataset_info', {})
tfidf_params = meta_data.get('tfidf_parameters', {})
top_words = meta_data.get('top_spam_features', [])

# Champion Model Summary Header
champion_metrics = all_metrics.get(sel_model, {})

st.markdown(
    f"""
    <div class="stat-card" style="border-left: 5px solid #10b981; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;">
                    SELECTED CHAMPION MODEL
                </span>
                <h2 style="margin: 6px 0 2px 0; font-size: 1.5rem; color: #ffffff;">{sel_model}</h2>
                <div style="font-size: 0.88rem; color: #94a3b8;">
                    Selection Rationale: {meta_data.get('selection_criteria', '')}
                </div>
            </div>
            <div style="display: flex; gap: 24px; margin-top: 10px;">
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">ACCURACY</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #60a5fa;">{champion_metrics.get('accuracy', 0)*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">SPAM PRECISION</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #34d399;">{champion_metrics.get('spam_precision', 0)*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">SPAM RECALL</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #fbbf24;">{champion_metrics.get('spam_recall', 0)*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">SPAM F1-SCORE</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #f472b6;">{champion_metrics.get('spam_f1', 0)*100:.2f}%</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Visual Comparison Section
st.subheader("📊 Comparative Algorithm Benchmark")
c_bm1, c_bm2 = st.columns(2)

with c_bm1:
    comp_fig = create_model_comparison_figure(all_metrics)
    st.plotly_chart(comp_fig, use_container_width=True)

with c_bm2:
    pr_fig = create_precision_recall_figure(all_metrics)
    st.plotly_chart(pr_fig, use_container_width=True)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Confusion Matrices Section
st.subheader("🎯 Confusion Matrix Analysis (Test Split: 1,030 Samples)")
c_cm1, c_cm2 = st.columns(2)

model_names = list(all_metrics.keys())
for i, m_name in enumerate(model_names):
    col = c_cm1 if i == 0 else c_cm2
    m_info = all_metrics[m_name]
    with col:
        cm_fig = create_confusion_matrix_figure(m_info['confusion_matrix'], title=f"Confusion Matrix: {m_name}")
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5; padding: 0 10px;">
                • <b>True Negatives (Legit Ham Caught):</b> {m_info['tn']} / {m_info['tn'] + m_info['fp']}<br>
                • <b>False Positives (Ham Mistakenly Flagged as Spam):</b> <span style="color: {'#ef4444' if m_info['fp'] > 2 else '#10b981'}; font-weight: 700;">{m_info['fp']}</span><br>
                • <b>False Negatives (Missed Spam):</b> {m_info['fn']}<br>
                • <b>True Positives (Spam Correctly Blocked):</b> {m_info['tp']} / {m_info['tp'] + m_info['fn']}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# Detailed Classification Reports Tab
st.subheader("📋 Detailed Per-Class Classification Reports")

tab_nb, tab_lr = st.tabs(["Multinomial Naive Bayes", "Logistic Regression"])

def render_report_table(report_dict):
    rows = []
    for label, vals in report_dict.items():
        if isinstance(vals, dict):
            rows.append({
                'Class / Metric': label,
                'Precision': f"{vals.get('precision', 0)*100:.2f}%",
                'Recall': f"{vals.get('recall', 0)*100:.2f}%",
                'F1-Score': f"{vals.get('f1-score', 0)*100:.2f}%",
                'Support': f"{int(vals.get('support', 0)):,}"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_nb:
    nb_rep = all_metrics.get("Multinomial Naive Bayes", {}).get('classification_report', {})
    render_report_table(nb_rep)

with tab_lr:
    lr_rep = all_metrics.get("Logistic Regression", {}).get('classification_report', {})
    render_report_table(lr_rep)

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# Dataset & Feature Space Metadata
st.subheader("🔬 Training Pipeline & Feature Space Metadata")

col_m1, col_m2 = st.columns([1.5, 1])

with col_m1:
    st.markdown("**Dataset Architecture:**")
    st.markdown(
        f"""
        - **Source:** {ds_info.get('dataset_source', 'SMS Spam Collection')}
        - **Raw Dataset Rows:** `{ds_info.get('raw_samples', 5574):,}`
        - **Exact Deduplicated Messages:** `{ds_info.get('clean_unique_samples', 5160):,}` ({ds_info.get('duplicates_removed', 414)} duplicates purged to prevent train-test contamination)
        - **Training Set (80% Stratified):** `{ds_info.get('train_samples', 4118):,}` messages ({ds_info.get('train_spam_count', 513)} Spam, {ds_info.get('train_ham_count', 3605)} Ham)
        - **Holdout Test Set (20% Stratified):** `{ds_info.get('test_samples', 1030):,}` messages ({ds_info.get('test_spam_count', 128)} Spam, {ds_info.get('test_ham_count', 902)} Ham)
        - **TF-IDF Vocabulary Size:** `{ds_info.get('vocabulary_size', 5000):,}` extracted n-grams
        - **N-Gram Ranges:** `{tfidf_params.get('ngram_range', [1, 2])}` (Unigrams + Bigrams)
        - **Trained At:** `{meta_data.get('training_date', 'N/A')}`
        """
    )

with col_m2:
    st.markdown("**Top Predictive Spam N-Grams:**")
    if top_words:
        top_tags_html = " ".join([f"<span class='feature-tag'>{w}</span>" for w in top_words[:20]])
        st.markdown(top_tags_html, unsafe_allow_html=True)
    else:
        st.caption("No top feature weights available.")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Retraining Action
with st.expander("⚙️ Pipeline Retraining Controls"):
    st.markdown(
        "Click the button below to re-execute the offline training pipeline across the dataset. "
        "This recalculates train/test splits, retrains both models, updates metrics, and saves the new champion artifact."
    )
    if st.button("🔁 Retrain & Benchmark Models Now", type="secondary"):
        with st.spinner("Retraining NLP pipelines and evaluating models..."):
            _, new_meta = train_and_evaluate(save_artifacts=True)
            st.success("Successfully retrained and benchmarked models!")
            st.rerun()

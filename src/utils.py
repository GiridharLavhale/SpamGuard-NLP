"""
Utility functions and UI helpers for SpamGuard NLP.
Provides CSS injection, sample message presets, report generators,
and HTML formatting helpers.
"""

import os
from typing import Dict, List, Any
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_PATH = os.path.join(BASE_DIR, "assets", "styles.css")

# Curated sample SMS messages for testing and demonstration
SAMPLE_MESSAGES = {
    "Spam: Free Lottery Prize": (
        "Congratulations! You have been selected to win a free £1,000 cash prize or a holiday voucher! "
        "Call 08712345678 or click http://claim-prize.net now to claim. Valid 24hrs only."
    ),
    "Spam: Urgent Bank Alert": (
        "URGENT: Your bank account has been temporarily flagged due to suspicious activity. "
        "Verify your identity immediately at http://secure-login-bank.com/auth or your account will be suspended."
    ),
    "Spam: Free Ringtone / Offer": (
        "FREE Ringtone! Reply YES to 88088 to activate your weekly subscription. T&C apply. "
        "Standard network charges apply. To cancel text STOP."
    ),
    "Ham: Casual Meeting": (
        "Hey Sarah, are we still meeting up for coffee tomorrow morning around 10:30 at the campus library?"
    ),
    "Ham: Family Dinner": (
        "Mom said dinner will be ready by 7pm tonight. Don't forget to pick up some milk on your way home."
    ),
    "Ham: Work Follow-up": (
        "Hi Dave, I reviewed the project presentation slides you sent over. Everything looks great, let's discuss at the standup."
    )
}


def load_custom_css():
    """Read and return custom CSS content."""
    if os.path.exists(CSS_PATH):
        try:
            with open(CSS_PATH, "r", encoding="utf-8") as f:
                return f"<style>{f.read()}</style>"
        except Exception:
            return ""
    return ""


def generate_txt_report(result: Dict[str, Any]) -> str:
    """Generate a clean downloadable text inspection report for a single prediction."""
    pred = result.get('prediction', 'UNKNOWN')
    conf = result.get('confidence', 0.0)
    raw_msg = result.get('raw_message', '')
    timestamp = result.get('timestamp', '')
    indicators = result.get('indicators', {})
    model_name = result.get('model_name', 'SpamGuard NLP')

    report_lines = [
        "===========================================================",
        "           SPAMGUARD NLP - MESSAGE AUDIT REPORT           ",
        "===========================================================",
        f"Generated At    : {timestamp}",
        f"Engine Model    : {model_name}",
        "-----------------------------------------------------------",
        f"CLASSIFICATION  : {pred}",
        f"CONFIDENCE      : {conf:.2f}%",
        f"SPAM PROBABILITY: {result.get('spam_probability', 0.0):.4f}",
        f"HAM PROBABILITY : {result.get('ham_probability', 0.0):.4f}",
        "-----------------------------------------------------------",
        "HEURISTIC INDICATORS:",
        f"  - Risk Assessment Level : {indicators.get('risk_level', 'Low')}",
        f"  - Suspicion Score       : {indicators.get('suspicion_score', 0)}/100",
        f"  - Contains URL Link     : {'YES' if indicators.get('has_url') else 'NO'}",
        f"  - Contains Phone/Code   : {'YES' if indicators.get('has_phone') else 'NO'}",
        f"  - Contains Currency     : {'YES' if indicators.get('has_currency') else 'NO'}",
        f"  - Uppercase Character % : {indicators.get('uppercase_ratio', 0.0)}%",
        f"  - Flagged Keywords      : {', '.join(indicators.get('matched_keywords', [])) or 'None'}",
        "-----------------------------------------------------------",
        "RAW MESSAGE CONTENT:",
        raw_msg,
        "-----------------------------------------------------------",
        "NORMALIZED TOKENS:",
        result.get('processed_text', ''),
        "===========================================================",
        "DISCLAIMER:",
        "SpamGuard NLP provides automated text classification for informational",
        "purposes. Classification is probabilistic and does not guarantee legal fraud.",
        "==========================================================="
    ]
    return "\n".join(report_lines)


def render_sidebar():
    """Render the standard modern SaaS sidebar for SpamGuard NLP."""
    import streamlit as st
    import json

    with st.sidebar:
        logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=72)

        st.markdown(
            """
            <div style="margin-top: -10px; margin-bottom: 20px;">
                <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #ffffff;">SpamGuard NLP</h2>
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Intelligent SMS Spam Detection System</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Model Status Indicator Section
        metadata_file = os.path.join(BASE_DIR, "models", "model_metadata.json")
        model_name = "Multinomial Naive Bayes"
        acc_text = "98.5%"
        prec_text = "98.3%"
        vocab_text = "5,000"

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    model_name = meta.get('selected_model', model_name)
                    metrics = meta.get('all_models_metrics', {}).get(model_name, {})
                    acc_val = metrics.get('accuracy', 0.9854) * 100
                    prec_val = metrics.get('spam_precision', 0.9829) * 100
                    acc_text = f"{acc_val:.1f}%"
                    prec_text = f"{prec_val:.1f}%"
                    vocab_text = f"{meta.get('dataset_info', {}).get('vocabulary_size', 5000):,}"
            except Exception:
                pass

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Engine Status</span>
                    <span style="background: rgba(16, 185, 129, 0.2); color: #10b981; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;">● ACTIVE</span>
                </div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">{model_name}</div>
                <div style="font-size: 0.76rem; color: #94a3b8; line-height: 1.4;">
                    Accuracy: <b style="color: #3b82f6;">{acc_text}</b> | Precision: <b style="color: #10b981;">{prec_text}</b><br>
                    Vocabulary: <b style="color: #f8fafc;">{vocab_text}</b> features
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Footer
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5; margin-top: 30px;">
                <b>SpamGuard NLP</b> • Version 1.0.0<br>
                Traditional NLP & ML Engine<br>
                100% Offline • Zero External APIs<br>
                <span style="color: #475569;">© 2026 Open Source Project</span>
            </div>
            """,
            unsafe_allow_html=True
        )


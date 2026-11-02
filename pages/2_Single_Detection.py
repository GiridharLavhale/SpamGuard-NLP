"""
Page 2: Single Message Detection
Primary interactive SMS inspection page with character counter, preset examples,
real-time ML inference, confidence metrics, heuristic explainability, and report downloads.
"""

import streamlit as st
import pandas as pd

from src.prediction import predict_single
from src.database import insert_prediction
from src.utils import load_custom_css, SAMPLE_MESSAGES, generate_txt_report

# Ensure styling is injected
st.markdown(load_custom_css(), unsafe_allow_html=True)

st.title("✉️ Single Message Detection")
st.markdown(
    "Analyze any SMS or text message with SpamGuard's trained machine learning model. "
    "Get instant classification, confidence levels, and transparent feature explainability."
)

st.markdown("---")

# Session state initialization for text input
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def set_example_text(key: str):
    st.session_state.message_input = SAMPLE_MESSAGES[key]


# Sample Message Buttons
st.markdown("**💡 Quick Test Presets:**")
sample_cols = st.columns(3)
preset_keys = list(SAMPLE_MESSAGES.keys())

for idx, key in enumerate(preset_keys):
    col = sample_cols[idx % 3]
    with col:
        is_spam_preset = "Spam" in key
        btn_label = f"🔴 {key}" if is_spam_preset else f"🟢 {key}"
        if st.button(btn_label, key=f"preset_{idx}", use_container_width=True):
            set_example_text(key)
            st.rerun()

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Input Section with Text Area
message_text = st.text_area(
    "Enter SMS message text to inspect:",
    value=st.session_state.message_input,
    height=140,
    placeholder="Type or paste an SMS message here (e.g., 'URGENT! Call 0800 to claim your £1000 prize now...')"
)

# Keep state synchronized
st.session_state.message_input = message_text

# Character and Word Counter
char_len = len(message_text)
word_len = len(message_text.split()) if message_text else 0

col_counters, col_actions = st.columns([2, 2])
with col_counters:
    st.caption(f"📏 **Characters:** {char_len} | **Words:** {word_len} | **Segments:** {(char_len // 160) + 1} standard SMS")

with col_actions:
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        clear_clicked = st.button("🗑️ Clear Text", use_container_width=True)
        if clear_clicked:
            st.session_state.message_input = ""
            st.session_state.last_result = None
            st.rerun()
    with c_btn2:
        analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)

# Processing when Analyze is clicked
if analyze_clicked:
    if not message_text.strip():
        st.warning("⚠️ Please enter or select a message before analyzing.")
    else:
        with st.spinner("Analyzing message through NLP preprocessing & ML inference..."):
            result = predict_single(message_text)
            st.session_state.last_result = result
            # Persist to database history
            insert_prediction(
                message=message_text,
                prediction=result['prediction'],
                confidence=result['confidence'],
                spam_probability=result['spam_probability'],
                source='single'
            )

# Display Results Section
if st.session_state.last_result:
    res = st.session_state.last_result
    pred = res['prediction']
    conf = res['confidence']
    indicators = res['indicators']
    tokens_evidence = res['token_contributions']

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    if pred == 'SPAM':
        st.markdown(
            f"""
            <div class="result-card-spam">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="result-badge-spam">⚠️ SPAM DETECTED</span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 10px;">
                            Flagged as Suspicious or Unwanted Message
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 4px;">
                            This message displays characteristics commonly associated with promotional, unsolicited, or phishing SMS communications.
                        </div>
                    </div>
                    <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.4);">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Model Confidence</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #f87171;">{conf:.2f}%</div>
                        <div style="font-size: 0.75rem; color: #cbd5e1;">Spam Prob: {res['spam_probability']:.4f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-ham">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="result-badge-ham">✅ LEGITIMATE (HAM)</span>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 10px;">
                            Classified as Safe / Normal Message
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 4px;">
                            This message appears consistent with legitimate, conversational, or expected personal SMS communications.
                        </div>
                    </div>
                    <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.4);">
                        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Model Confidence</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #34d399;">{conf:.2f}%</div>
                        <div style="font-size: 0.75rem; color: #cbd5e1;">Ham Prob: {res['ham_probability']:.4f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Detailed Explainability Tabs
    tab_explain, tab_heuristic, tab_nlp, tab_export = st.tabs([
        "🔍 Feature Explainability",
        "🚩 Heuristic Risk Breakdown",
        "🧪 NLP Tokens & Pipeline",
        "📥 Export & Audit Report"
    ])

    with tab_explain:
        st.subheader("Model Feature Evidence")
        st.markdown(
            "Below are the influential vocabulary tokens detected in your message along with their "
            "correlation towards **Spam** or **Ham** based on the trained model parameters:"
        )

        if tokens_evidence:
            ev_df = pd.DataFrame(tokens_evidence)
            ev_df.columns = ['Token', 'Model Weight / Log-Odds', 'Tendency', 'Impact Strength']
            st.dataframe(ev_df, use_container_width=True, hide_index=True)
        else:
            st.info("No strong vocabulary tokens from the TF-IDF feature space were isolated in this input.")

        st.caption(
            "Note: Model weights represent statistical correlations extracted from the training corpus "
            "(SMS Spam Collection) and do not imply legal intent or fraudulent malice."
        )

    with tab_heuristic:
        st.subheader("Rule-Based Heuristic Indicators")
        c_h1, c_h2, c_h3, c_h4 = st.columns(4)

        with c_h1:
            st.metric("Risk Score", f"{indicators['suspicion_score']}/100", delta=indicators['risk_level'], delta_color="inverse")
        with c_h2:
            st.metric("Contains Links", "YES" if indicators['has_url'] else "NO")
        with c_h3:
            st.metric("Contains Phone / Code", "YES" if indicators['has_phone'] else "NO")
        with c_h4:
            st.metric("Monetary Symbols", "YES" if indicators['has_currency'] else "NO")

        st.markdown("**Detected Spam Trigger Keywords:**")
        if indicators['matched_keywords']:
            kw_html = " ".join([f"<span class='spam-tag'>⚠️ {kw.upper()}</span>" for kw in indicators['matched_keywords']])
            st.markdown(kw_html, unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #94a3b8;'>No common spam trigger keywords matched.</span>", unsafe_allow_html=True)

        st.markdown(f"**Capitalization Analysis:** `{indicators['uppercase_ratio']}%` of characters are uppercase.")

    with tab_nlp:
        st.subheader("Preprocessed Text Representation")
        st.markdown("Here is how the message was normalized before vectorization:")
        st.code(res.get('processed_text', '(empty)'), language="text")

        st.markdown(
            """
            **Transformation Details:**
            - Converted to lowercase.
            - URLs replaced with `url_token`.
            - Phone numbers replaced with `phone_token`.
            - Currency symbols replaced with `currency_token`.
            - Stopwords removed while preserving negations (*no*, *not*, *never*).
            - Porter stemming applied to normalize morphological word variants.
            """
        )

    with tab_export:
        st.subheader("Export Prediction Result")
        st.markdown(f"**Audit Timestamp:** `{res['timestamp']}`")
        st.markdown(f"**Classification:** `{res['prediction']}` ({res['confidence']:.2f}% confidence)")

        txt_report = generate_txt_report(res)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="📄 Download Inspection Report (.TXT)",
                data=txt_report,
                file_name=f"spamguard_report_{pred.lower()}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with c_exp2:
            csv_data = pd.DataFrame([{
                'Timestamp': res['timestamp'],
                'Message': res['raw_message'],
                'Prediction': res['prediction'],
                'Confidence': res['confidence'],
                'Spam_Probability': res['spam_probability'],
                'Risk_Level': indicators['risk_level'],
                'Keywords': ", ".join(indicators['matched_keywords'])
            }]).to_csv(index=False)

            st.download_button(
                label="📊 Download Result Record (.CSV)",
                data=csv_data,
                file_name=f"spamguard_record_{pred.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Disclaimer:</b> Predictions are generated using statistical Natural Language Processing and Scikit-Learn algorithms.
            A spam classification indicates probabilistic similarity to unsolicited or marketing messages in the training dataset,
            not that the message is conclusively fraudulent. Always verify unexpected communications independently.
        </div>
        """,
        unsafe_allow_html=True
    )

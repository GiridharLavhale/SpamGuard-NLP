"""
Page 3: Bulk Message Detection
Processes batches of SMS messages from uploaded CSV/TSV files.
Provides column selection, high-speed batch inference, interactive Plotly visualizations,
filtering, and CSV/TXT exports.
"""

import io
import streamlit as st
import pandas as pd

from src.data_loader import detect_columns
from src.prediction import predict_bulk
from src.database import insert_bulk_predictions
from src.analytics import (
    create_distribution_pie,
    create_confidence_histogram,
    create_length_comparison_box
)
from src.utils import load_custom_css, generate_bulk_summary_report

st.markdown(load_custom_css(), unsafe_allow_html=True)

st.title("📂 Bulk Message Detection")
st.markdown(
    "Upload a CSV or TSV file containing SMS messages to perform batch classification. "
    "Analyze hundreds or thousands of messages simultaneously and download detailed audit reports."
)

st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload SMS Dataset (.CSV or .TSV)",
    type=['csv', 'tsv', 'txt'],
    help="Upload a file containing SMS messages. Max file size: 50MB."
)

if uploaded_file is None:
    st.info("👆 Please upload a CSV or TSV file above to get started.")

    # Show a sample template / format guidance
    with st.expander("ℹ️ Supported Dataset Formats & Sample Preview"):
        st.markdown(
            """
            Your uploaded file can have one or more columns. The system will automatically detect the text column,
            or you can select it manually:
            """
        )
        sample_demo_df = pd.DataFrame({
            "message": [
                "Win a guaranteed £500 cash gift card! Click here: http://win.me/gift",
                "Are we still on for lunch today at the cafeteria?",
                "URGENT! Your account security needs attention. Reply STOP to cancel.",
                "Can you please send me the homework notes from yesterday?"
            ]
        })
        st.dataframe(sample_demo_df, use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_demo_df.to_csv(index=False),
            file_name="spamguard_sample_template.csv",
            mime="text/csv"
        )
else:
    # Read uploaded file
    try:
        # Determine separator
        fname = uploaded_file.name.lower()
        sep = '\t' if (fname.endswith('.tsv') or fname.endswith('.txt')) else ','

        raw_bytes = uploaded_file.read()
        try:
            df_upload = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df_upload = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding='latin-1', on_bad_lines='skip')

        if df_upload.empty:
            st.error("Uploaded file appears to be empty. Please upload a file with valid data rows.")
            st.stop()

        st.success(f"Successfully loaded `{uploaded_file.name}` ({len(df_upload):,} rows, {len(df_upload.columns)} columns).")

    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}")
        st.stop()

    # Column Selection
    _, detected_msg_col = detect_columns(df_upload)
    cols_list = list(df_upload.columns)
    default_idx = cols_list.index(detected_msg_col) if detected_msg_col in cols_list else 0

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        chosen_col = st.selectbox(
            "Select the column containing the SMS message text:",
            options=cols_list,
            index=default_idx
        )
    with c_sel2:
        persist_choice = st.checkbox("Save batch results to Local History", value=False)

    # Preview uploaded data
    with st.expander("👁️ Preview First 5 Rows of Uploaded File"):
        st.dataframe(df_upload.head(5), use_container_width=True)

    if st.button("🚀 Process Batch Classification", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {len(df_upload):,} messages with NLP preprocessing and ML inference..."):
            try:
                res_df, summary = predict_bulk(df_upload, message_col=chosen_col)
                st.session_state['bulk_results_df'] = res_df
                st.session_state['bulk_summary'] = summary
                st.session_state['bulk_filename'] = uploaded_file.name

                if persist_choice:
                    records_to_save = [
                        (str(row[chosen_col]), str(row['Prediction']), float(row['Confidence (%)']), float(row['Spam_Probability']), 'bulk')
                        for _, row in res_df.iterrows()
                    ]
                    insert_bulk_predictions(records_to_save)
                    st.toast("Saved bulk predictions to Analytics History!", icon="💾")

            except Exception as e:
                st.error(f"Inference processing error: {e}")
                st.stop()

# Display Bulk Results if available in session
if 'bulk_results_df' in st.session_state:
    res_df = st.session_state['bulk_results_df']
    summary = st.session_state['bulk_summary']
    filename = st.session_state.get('bulk_filename', 'Batch Results')

    st.markdown("---")
    st.subheader(f"📊 Batch Audit Summary: {filename}")

    # Summary KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    with col_kpi1:
        st.metric("Total Processed", f"{summary['total_processed']:,}")
    with col_kpi2:
        st.metric("Spam Detected", f"{summary['spam_count']:,}", delta=f"{summary['spam_percentage']:.1f}%")
    with col_kpi3:
        st.metric("Legitimate (Ham)", f"{summary['ham_count']:,}")
    with col_kpi4:
        st.metric("Spam Ratio", f"{summary['spam_percentage']:.1f}%")
    with col_kpi5:
        st.metric("Avg Confidence", f"{summary['average_confidence']:.2f}%")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Interactive Visualizations Row
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        pie_fig = create_distribution_pie(summary['spam_count'], summary['ham_count'])
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_chart2:
        conf_fig = create_confidence_histogram(res_df['Confidence (%)'].tolist())
        st.plotly_chart(conf_fig, use_container_width=True)

    # Length comparison chart
    length_fig = create_length_comparison_box(res_df, message_col=chosen_col, label_col='Prediction')
    if length_fig.data:
        st.plotly_chart(length_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📑 Classified Messages Table")

    # Filter Tabs
    tab_all, tab_spam, tab_ham = st.tabs([
        f"All Messages ({len(res_df):,})",
        f"🚨 Spam Only ({summary['spam_count']:,})",
        f"✅ Ham Only ({summary['ham_count']:,})"
    ])

    search_query = st.text_input("🔎 Search text in classified messages:", "")

    def filter_and_format(df_subset):
        if search_query.strip():
            df_subset = df_subset[df_subset[chosen_col].astype(str).str.contains(search_query.strip(), case=False, na=False)]
        cols_to_show = [chosen_col, 'Prediction', 'Confidence (%)', 'Spam_Probability']
        return df_subset[cols_to_show]

    with tab_all:
        st.dataframe(filter_and_format(res_df), use_container_width=True, hide_index=True)

    with tab_spam:
        spam_subset = res_df[res_df['Prediction'] == 'SPAM']
        st.dataframe(filter_and_format(spam_subset), use_container_width=True, hide_index=True)

    with tab_ham:
        ham_subset = res_df[res_df['Prediction'] == 'HAM']
        st.dataframe(filter_and_format(ham_subset), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Export Section
    st.subheader("📥 Export Batch Results")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        full_csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Complete Results (.CSV)",
            data=full_csv,
            file_name=f"spamguard_bulk_results_{filename}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_exp2:
        txt_summary = generate_bulk_summary_report(summary, filename)
        st.download_button(
            label="📄 Download Summary Audit Report (.TXT)",
            data=txt_summary,
            file_name=f"spamguard_bulk_summary_{filename}.txt",
            mime="text/plain",
            use_container_width=True
        )

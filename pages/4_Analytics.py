"""
Page 4: Analytics Dashboard
Live KPI metrics, interactive Plotly visualizations, timeline trends,
searchable audit history, and SQLite persistence controls.
"""

import streamlit as st
import pandas as pd

from src.database import (
    get_analytics_summary,
    get_recent_predictions,
    clear_history
)
from src.analytics import (
    create_distribution_pie,
    create_confidence_histogram,
    create_timeline_chart,
    create_length_comparison_box,
    create_top_keywords_bar
)
from src.preprocessing import extract_heuristic_indicators
from src.utils import load_custom_css

st.markdown(load_custom_css(), unsafe_allow_html=True)

st.title("📊 Analytics Dashboard")
st.markdown(
    "Explore historical prediction intelligence, audit trends, and classification distributions. "
    "All telemetry is persisted locally in SQLite with zero third-party telemetry."
)

st.markdown("---")

# Aggregate KPI Summary
stats = get_analytics_summary()
total = stats['total']
spam_count = stats['spam_count']
ham_count = stats['ham_count']
spam_rate = stats['spam_rate']
avg_conf = stats['avg_confidence']

# KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Audited", f"{total:,}")
with c2:
    st.metric("Spam Detected", f"{spam_count:,}", delta=f"{spam_rate:.1f}%")
with c3:
    st.metric("Legitimate (Ham)", f"{ham_count:,}")
with c4:
    st.metric("Spam Incidence", f"{spam_rate:.1f}%")
with c5:
    st.metric("Avg Confidence", f"{avg_conf:.1f}%")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# History Filter Controls
with st.expander("🔍 Filter History & Activity Table", expanded=True):
    col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.5, 2, 1])

    with col_f1:
        label_filter = st.selectbox(
            "Classification Filter",
            options=["ALL", "SPAM", "HAM"],
            index=0
        )
    with col_f2:
        limit_choice = st.selectbox(
            "Record Limit",
            options=[25, 50, 100, 250, 500],
            index=1
        )
    with col_f3:
        search_kw = st.text_input("Search Text Keyword", placeholder="e.g. winner, prize, lunch...")
    with col_f4:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)

# Fetch filtered records
filter_val = None if label_filter == "ALL" else label_filter
history_df = get_recent_predictions(
    limit=limit_choice,
    filter_label=filter_val,
    search_term=search_kw
)

# Visualizations Section
if total > 0 and not history_df.empty:
    st.subheader("📈 Classification Visualizations")

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        pie_chart = create_distribution_pie(spam_count, ham_count)
        st.plotly_chart(pie_chart, use_container_width=True)

    with v_col2:
        conf_chart = create_confidence_histogram(history_df['confidence'].tolist())
        st.plotly_chart(conf_chart, use_container_width=True)

    v_col3, v_col4 = st.columns(2)
    with v_col3:
        timeline_chart = create_timeline_chart(history_df)
        st.plotly_chart(timeline_chart, use_container_width=True)

    with v_col4:
        box_chart = create_length_comparison_box(history_df)
        if box_chart.data:
            st.plotly_chart(box_chart, use_container_width=True)

    # Keyword frequency among spam predictions
    spam_records = history_df[history_df['prediction'] == 'SPAM']
    if not spam_records.empty:
        keyword_counts = {}
        for msg in spam_records['message']:
            inds = extract_heuristic_indicators(str(msg))
            for kw in inds.get('matched_keywords', []):
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        if keyword_counts:
            kw_bar = create_top_keywords_bar(keyword_counts)
            st.plotly_chart(kw_bar, use_container_width=True)

else:
    st.info("ℹ️ No historical prediction records match the active filter. Analyze a message to populate charts.")

st.markdown("---")

# Recent Activity Table
st.subheader("📋 Prediction Audit History")

if not history_df.empty:
    table_display = history_df[['timestamp', 'prediction', 'confidence', 'source', 'message']].copy()
    table_display.columns = ['Timestamp', 'Prediction', 'Confidence (%)', 'Source', 'Message Content']

    st.dataframe(table_display, use_container_width=True, hide_index=True)

    # Action row for export and clear
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_act1, c_act2, c_spacer = st.columns([1.5, 1.5, 3])

    with c_act1:
        csv_bytes = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered History (.CSV)",
            data=csv_bytes,
            file_name="spamguard_history_export.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c_act2:
        # Confirm Clear
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = False

        if not st.session_state.confirm_delete:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning("Are you sure? This will delete all saved predictions.")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("⚠️ Yes, Delete", type="primary", use_container_width=True):
                    clear_history()
                    st.session_state.confirm_delete = False
                    st.toast("Audit history cleared!", icon="🧹")
                    st.rerun()
            with c_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()

else:
    st.caption("No records to display.")

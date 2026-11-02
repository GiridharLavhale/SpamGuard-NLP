"""
Interactive Plotly visualizations and analytics charts for SpamGuard NLP.
Generates responsive charts for the Analytics and Bulk Detection dashboards.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Consistent SaaS color palette
THEME_COLORS = {
    'spam': '#ef4444',
    'ham': '#10b981',
    'accent_blue': '#3b82f6',
    'accent_purple': '#8b5cf6',
    'accent_amber': '#f59e0b',
    'dark_bg': 'rgba(15, 23, 42, 0.6)',
    'paper_bg': 'rgba(0,0,0,0)',
    'text': '#f8fafc',
    'grid': '#334155'
}


def create_distribution_pie(
    spam_count: int,
    ham_count: int,
    title: str = "Message Classification Distribution"
) -> go.Figure:
    """Generate a clean donut chart showing proportion of Spam vs Ham messages."""
    labels = ['Spam Messages', 'Legitimate (Ham)']
    values = [spam_count, ham_count]
    colors = [THEME_COLORS['spam'], THEME_COLORS['ham']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#0f172a', width=2)),
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        textfont=dict(size=13, color='white')
    )])

    total = spam_count + ham_count
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        annotations=[dict(text=f'Total<br><b>{total:,}</b>', x=0.5, y=0.5, font_size=16, font_color='white', showarrow=False)],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=40),
        height=350,
        paper_bgcolor=THEME_COLORS['paper_bg'],
        plot_bgcolor=THEME_COLORS['paper_bg'],
        font=dict(color=THEME_COLORS['text'])
    )
    return fig


def create_confidence_histogram(
    confidences: List[float],
    title: str = "Prediction Confidence Distribution"
) -> go.Figure:
    """Generate a histogram showing the spread of prediction confidence percentages."""
    if not confidences:
        confidences = [95.0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=confidences,
        nbinsx=20,
        marker_color=THEME_COLORS['accent_blue'],
        opacity=0.85,
        name='Confidence %'
    ))

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Confidence Percentage (%)",
        yaxis_title="Message Count",
        xaxis=dict(range=[50, 101], gridcolor=THEME_COLORS['grid']),
        yaxis=dict(gridcolor=THEME_COLORS['grid']),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
        paper_bgcolor=THEME_COLORS['paper_bg'],
        plot_bgcolor=THEME_COLORS['paper_bg'],
        font=dict(color=THEME_COLORS['text'])
    )
    return fig


def create_timeline_chart(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    label_col: str = 'prediction',
    title: str = "Prediction Activity Over Time"
) -> go.Figure:
    """Generate a grouped time series chart of spam and ham activity."""
    if df.empty or timestamp_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "No timeline data available yet", 'x': 0.5},
            paper_bgcolor=THEME_COLORS['paper_bg'],
            font=dict(color=THEME_COLORS['text'])
        )
        return fig

    # Parse timestamps
    df_time = df.copy()
    df_time['datetime'] = pd.to_datetime(df_time[timestamp_col], errors='coerce')
    df_time = df_time.dropna(subset=['datetime'])

    if df_time.empty:
        fig = go.Figure()
        fig.update_layout(title={'text': "No valid timestamps found", 'x': 0.5})
        return fig

    # Format date bucket
    df_time['date_bucket'] = df_time['datetime'].dt.strftime('%Y-%m-%d %H:00')
    grouped = df_time.groupby(['date_bucket', label_col]).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    if 'SPAM' in grouped.columns:
        fig.add_trace(go.Bar(
            x=grouped['date_bucket'],
            y=grouped['SPAM'],
            name='Spam Detected',
            marker_color=THEME_COLORS['spam']
        ))
    if 'HAM' in grouped.columns:
        fig.add_trace(go.Bar(
            x=grouped['date_bucket'],
            y=grouped['HAM'],
            name='Legitimate (Ham)',
            marker_color=THEME_COLORS['ham']
        ))

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        barmode='stack',
        xaxis_title="Time Interval",
        yaxis_title="Message Count",
        xaxis=dict(gridcolor=THEME_COLORS['grid']),
        yaxis=dict(gridcolor=THEME_COLORS['grid']),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
        paper_bgcolor=THEME_COLORS['paper_bg'],
        plot_bgcolor=THEME_COLORS['paper_bg'],
        font=dict(color=THEME_COLORS['text']),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_length_comparison_box(
    df: pd.DataFrame,
    message_col: str = 'message',
    label_col: str = 'prediction',
    title: str = "Message Character Length Comparison"
) -> go.Figure:
    """Generate box plots comparing character lengths of Spam vs Ham messages."""
    if df.empty or message_col not in df.columns or label_col not in df.columns:
        fig = go.Figure()
        return fig

    df_len = df.copy()
    df_len['char_length'] = df_len[message_col].astype(str).str.len()

    fig = go.Figure()

    spam_data = df_len[df_len[label_col].str.upper() == 'SPAM']['char_length']
    ham_data = df_len[df_len[label_col].str.upper() == 'HAM']['char_length']

    if not ham_data.empty:
        fig.add_trace(go.Box(
            y=ham_data,
            name='Legitimate (Ham)',
            marker_color=THEME_COLORS['ham'],
            boxmean=True
        ))

    if not spam_data.empty:
        fig.add_trace(go.Box(
            y=spam_data,
            name='Spam',
            marker_color=THEME_COLORS['spam'],
            boxmean=True
        ))

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        yaxis_title="Character Length",
        yaxis=dict(gridcolor=THEME_COLORS['grid']),
        margin=dict(l=40, r=20, t=40, b=40),
        height=340,
        paper_bgcolor=THEME_COLORS['paper_bg'],
        plot_bgcolor=THEME_COLORS['paper_bg'],
        font=dict(color=THEME_COLORS['text'])
    )
    return fig


def create_top_keywords_bar(
    keywords_dict: Dict[str, int],
    title: str = "Top Detected Suspicious Terms"
) -> go.Figure:
    """Generate horizontal bar chart of top triggered spam terms."""
    if not keywords_dict:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "No keyword indicators detected yet", 'x': 0.5},
            paper_bgcolor=THEME_COLORS['paper_bg'],
            font=dict(color=THEME_COLORS['text'])
        )
        return fig

    # Sort descending
    sorted_items = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    words = [item[0] for item in sorted_items][::-1]
    counts = [item[1] for item in sorted_items][::-1]

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation='h',
        marker=dict(
            color=counts,
            colorscale=[[0, '#f87171'], [1, '#ef4444']],
            showscale=False
        ),
        text=counts,
        textposition='auto'
    ))

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Frequency",
        yaxis_title="Indicator Keyword",
        xaxis=dict(gridcolor=THEME_COLORS['grid']),
        margin=dict(l=80, r=30, t=40, b=40),
        height=340,
        paper_bgcolor=THEME_COLORS['paper_bg'],
        plot_bgcolor=THEME_COLORS['paper_bg'],
        font=dict(color=THEME_COLORS['text'])
    )
    return fig

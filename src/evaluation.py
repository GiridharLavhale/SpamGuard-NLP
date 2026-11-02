"""
Model evaluation module for SpamGuard NLP.
Calculates performance metrics, confusion matrices, and interactive Plotly figures.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import plotly.graph_objects as go
import plotly.figure_factory as ff


def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for Spam (class 1) vs Ham (class 0).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # Binary metrics for the positive class (SPAM = 1)
    spam_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    spam_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    spam_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    # Binary metrics for the negative class (HAM = 0)
    ham_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    ham_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    ham_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # Macro & Weighted metrics
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    # Full report dict
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=['Ham (Legit)', 'Spam'],
        output_dict=True,
        zero_division=0
    )

    return {
        'accuracy': round(acc, 4),
        'spam_precision': round(spam_prec, 4),
        'spam_recall': round(spam_rec, 4),
        'spam_f1': round(spam_f1, 4),
        'ham_precision': round(ham_prec, 4),
        'ham_recall': round(ham_rec, 4),
        'ham_f1': round(ham_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(weighted_f1, 4),
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'classification_report': report
    }


def create_confusion_matrix_figure(cm: List[List[int]], title: str = "Confusion Matrix") -> go.Figure:
    """Generate an annotated Plotly heatmap for a 2x2 confusion matrix."""
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Ham', 'Predicted Spam']
    y = ['Actual Ham', 'Actual Spam']

    # Custom text annotations with descriptive labels
    annotations = [
        [f"True Ham (TN)<br><b>{tn}</b>", f"False Spam (FP)<br><b>{fp}</b>"],
        [f"False Ham (FN)<br><b>{fn}</b>", f"True Spam (TP)<br><b>{tp}</b>"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[[0, "#1e293b"], [0.5, "#3b82f6"], [1, "#10b981"]],
        showscale=True,
        reversescale=False
    ))

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis={'autorange': 'reversed'},
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc")
    )
    return fig


def create_model_comparison_figure(models_metrics: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Generate a grouped bar chart comparing multiple algorithms across key metrics."""
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('Spam Precision', 'spam_precision'),
        ('Spam Recall', 'spam_recall'),
        ('Spam F1 Score', 'spam_f1')
    ]

    metric_names = [m[0] for m in metrics_to_compare]

    fig = go.Figure()

    palette = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899']

    for i, (model_name, metrics) in enumerate(models_metrics.items()):
        values = [metrics.get(m[1], 0) * 100 for m in metrics_to_compare]
        color = palette[i % len(palette)]
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_names,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
            marker_color=color
        ))

    fig.update_layout(
        title={'text': "Model Performance Comparison (Test Split)", 'x': 0.5, 'xanchor': 'center'},
        barmode='group',
        yaxis=dict(title="Score (%)", range=[80, 102]),
        xaxis_title="Evaluation Metric",
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_precision_recall_figure(models_metrics: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Scatter comparison of Precision vs Recall for the spam class."""
    fig = go.Figure()

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

    for i, (name, metrics) in enumerate(models_metrics.items()):
        prec = metrics.get('spam_precision', 0) * 100
        rec = metrics.get('spam_recall', 0) * 100
        fig.add_trace(go.Scatter(
            x=[rec],
            y=[prec],
            mode='markers+text',
            name=name,
            text=[f"  {name}<br>  P: {prec:.1f}%, R: {rec:.1f}%"],
            textposition="top right",
            marker=dict(size=16, color=colors[i % len(colors)], line=dict(width=2, color='white'))
        ))

    fig.update_layout(
        title={'text': "Spam Class Precision vs. Recall Tradeoff", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Recall (%) — Proportion of Spam Caught",
        yaxis_title="Precision (%) — Proportion of Flagged Messages that are Spam",
        xaxis=dict(range=[80, 100]),
        yaxis=dict(range=[85, 102]),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc")
    )
    return fig

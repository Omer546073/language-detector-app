import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
from utils import load_metrics, inject_custom_css

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
inject_custom_css()

metrics = load_metrics()

st.markdown('<div class="eyebrow">Evaluation · Held-Out Test Set</div>', unsafe_allow_html=True)
st.title("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
col2.metric("Test Loss", f"{metrics['test_loss']:.4f}")
col3.metric("Classes", len(metrics['class_names']))

st.divider()
st.subheader("Per-Language Precision & Recall")

report = metrics['classification_report']
rows = []
for lang in metrics['class_names']:
    r = report[lang]
    rows.append({
        'Language': lang,
        'Precision': round(r['precision'], 3),
        'Recall': round(r['recall'], 3),
        'F1-Score': round(r['f1-score'], 3),
        'Support': int(r['support'])
    })
df = pd.DataFrame(rows).sort_values('F1-Score', ascending=False).reset_index(drop=True)
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "F1-Score": st.column_config.ProgressColumn(
            "F1-Score", min_value=0, max_value=1, format="%.3f"
        ),
    }
)

st.divider()
st.subheader("Confusion Matrix")
st.caption("Rows = actual language · Columns = predicted language")

cm = metrics['confusion_matrix']
classes = metrics['class_names']

fig = ff.create_annotated_heatmap(
    z=cm, x=classes, y=classes,
    colorscale=[[0, '#121B2E'], [0.15, '#1A2540'], [0.5, '#3A5A6E'], [1, '#9C8FE0']],
    showscale=True
)
fig.update_layout(
    height=800,
    xaxis_title="Predicted",
    yaxis_title="Actual",
    plot_bgcolor='#0B1220',
    paper_bgcolor='#0B1220',
    font=dict(family="Inter", color="#F4F6FB", size=10)
)
fig.update_xaxes(side="bottom", tickangle=-45, color="#93A0BC")
fig.update_yaxes(color="#93A0BC")
for annotation in fig.layout.annotations:
    annotation.font.size = 9
st.plotly_chart(fig, use_container_width=True)

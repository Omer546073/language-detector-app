import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import load_artifacts, predict_language, inject_custom_css

st.set_page_config(page_title="Detector | Language Detector", page_icon="🔍", layout="centered")
inject_custom_css()

model, char_to_idx, label_encoder, config = load_artifacts()
MAX_LEN = config['MAX_LEN']

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown('<div class="eyebrow">Live Inference</div>', unsafe_allow_html=True)
st.title("Detector")
st.caption("For the most reliable read, use text longer than ~50 characters.")

user_text = st.text_area(
    "Text to analyze",
    height=150,
    placeholder="Type or paste a sentence in any supported language…",
    label_visibility="collapsed"
)

detect = st.button("Detect Language", type="primary")

if detect:
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        probs = predict_language(user_text, model, char_to_idx, label_encoder, MAX_LEN)
        pred_idx = np.argmax(probs)
        pred_lang = label_encoder.classes_[pred_idx]
        confidence = probs[pred_idx] * 100

        st.session_state.history.append({
            'text': user_text[:80] + ('...' if len(user_text) > 80 else ''),
            'prediction': pred_lang,
            'confidence': f"{confidence:.1f}%"
        })

        st.markdown(
            f'<div class="result-banner">'
            f'<span class="lang">{pred_lang}</span><br>'
            f'<span class="conf">{confidence:.1f}% confidence</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_langs = [label_encoder.classes_[i] for i in top5_idx]
        top5_probs = [probs[i] * 100 for i in top5_idx]

        colors = ['#9C8FE0' if i == 0 else '#2C3A56' for i in range(5)]

        fig = go.Figure(go.Bar(
            x=top5_probs, y=top5_langs, orientation='h',
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{p:.1f}%" for p in top5_probs],
            textposition='outside',
            textfont=dict(family="JetBrains Mono", color="#F4F6FB")
        ))
        fig.update_layout(
            title=dict(text="Top 5 Candidates", font=dict(family="Fraunces", size=18, color="#F4F6FB")),
            xaxis=dict(title="Confidence (%)", range=[0, 105], gridcolor="#26324A", color="#93A0BC"),
            yaxis=dict(autorange="reversed", color="#F4F6FB"),
            height=340,
            plot_bgcolor='#0B1220',
            paper_bgcolor='#0B1220',
            font=dict(family="Inter", color="#F4F6FB"),
            margin=dict(l=10, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(user_text.strip()) < 50:
            st.info("Short inputs carry less signal — try a longer sentence for a more confident read.")

with st.expander("Supported languages"):
    st.markdown(
        f'<span style="font-family: \'JetBrains Mono\'; font-size: 0.85rem; color: #93A0BC;">'
        f'{", ".join(sorted(label_encoder.classes_))}</span>',
        unsafe_allow_html=True
    )

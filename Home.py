import streamlit as st
from utils import inject_custom_css, render_script_strip

st.set_page_config(page_title="Language Detector | Omer Farooq", page_icon="🌐", layout="wide")
inject_custom_css()

st.markdown('<div class="eyebrow">Character-level NLP · TensorFlow / Keras</div>', unsafe_allow_html=True)
st.title("The Multilingual Language Detector")
st.markdown(
    "<p style='font-size:1.05rem; color:#93A0BC; max-width:640px;'>"
    "A recurrent neural network that reads text one character at a time and identifies "
    "which of 22 languages it's written in — no dictionaries, no pretrained embeddings, "
    "just the raw shape of the alphabet."
    "</p>",
    unsafe_allow_html=True
)

render_script_strip()

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", "96.6%")
col2.metric("Languages", "22")
col3.metric("Model Size", "800 KB")

st.markdown("<br>", unsafe_allow_html=True)
st.page_link("pages/1_Detector.py", label="Open the Detector →", icon="🔍")

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    st.markdown("**Detector**")
    st.caption("Paste text, get a live prediction with a full confidence breakdown across all 22 classes.")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    st.markdown("**Performance**")
    st.caption("Precision, recall, and the full confusion matrix from evaluation on 4,400 held-out samples.")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    st.markdown("**How It Works**")
    st.caption("The dataset, the character-vocabulary tradeoffs, and the architecture, explained end to end.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Built by Omer Farooq · Character-level LSTM · TensorFlow/Keras · Streamlit")

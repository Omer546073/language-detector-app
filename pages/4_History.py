import streamlit as st
import pandas as pd
from utils import inject_custom_css

st.set_page_config(page_title="History", page_icon="🕒", layout="centered")
inject_custom_css()

st.markdown('<div class="eyebrow">This Session</div>', unsafe_allow_html=True)
st.title("Prediction History")
st.caption("A log of everything you've tested — resets when you close the tab.")

if 'history' not in st.session_state or len(st.session_state.history) == 0:
    st.markdown(
        '<div class="card">No predictions yet — head to the Detector page to try it out.</div>',
        unsafe_allow_html=True
    )
else:
    df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

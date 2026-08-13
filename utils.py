import streamlit as st
import numpy as np
import pickle
import json
import tensorflow as tf


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('language_detector_model.keras')
    with open('char_to_idx.pkl', 'rb') as f:
        char_to_idx = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('config.pkl', 'rb') as f:
        config = pickle.load(f)
    return model, char_to_idx, label_encoder, config


@st.cache_data
def load_metrics():
    with open('metrics.json', 'r') as f:
        return json.load(f)


def encode_text(text, char_to_idx, max_len):
    text = text.lower()  # must match training preprocessing (lowercase)
    encoded = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text[:max_len]]
    if len(encoded) < max_len:
        encoded += [char_to_idx['<PAD>']] * (max_len - len(encoded))
    return encoded


def predict_language(text, model, char_to_idx, label_encoder, max_len):
    encoded = encode_text(text, char_to_idx, max_len)
    input_array = np.array([encoded])
    probs = model.predict(input_array, verbose=0)[0]
    return probs


# The greeting strip: "hello" rendered in scripts the model actually trains on.
# This is the page's signature element — it IS the subject matter, not decoration.
GREETINGS = [
    ("Hello", "English · Latin"),
    ("Привет", "Russian · Cyrillic"),
    ("مرحبا", "Arabic"),
    ("नमस्ते", "Hindi · Devanagari"),
    ("你好", "Chinese · Han"),
    ("안녕하세요", "Korean · Hangul"),
    ("こんにちは", "Japanese · Kana"),
    ("สวัสดี", "Thai"),
    ("வணக்கம்", "Tamil"),
]


def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: #0B1220;
            --surface: #121B2E;
            --surface-alt: #1A2540;
            --border: #26324A;
            --text-primary: #F4F6FB;
            --text-muted: #93A0BC;
            --accent-violet: #9C8FE0;
            --accent-violet-soft: rgba(156, 143, 224, 0.12);
            --accent-teal: #45C4B0;
            --font-display: 'Fraunces', serif;
            --font-body: 'Inter', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        html, body, [class*="css"] { font-family: var(--font-body); }

        .stApp { background: var(--bg); }
        .main .block-container { padding-top: 2.5rem; max-width: 1100px; }

        /* Headings use the display serif, tightened tracking, gold-tinted h1 */
        h1 {
            font-family: var(--font-display) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
            color: var(--text-primary) !important;
        }
        h2, h3 {
            font-family: var(--font-display) !important;
            font-weight: 500 !important;
            color: var(--text-primary) !important;
        }
        p, li, label, .stMarkdown { color: var(--text-primary); }
        .stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { font-family: var(--font-body); }

        /* Buttons — gold, no default rounded-corporate look, small caps energy */
        .stButton > button, .stLinkButton > a {
            background: var(--accent-violet) !important;
            color: #0B1220 !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            font-family: var(--font-body) !important;
            letter-spacing: 0.01em;
            padding: 0.65rem 1.6rem !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover, .stLinkButton > a:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(156, 143, 224, 0.25);
        }

        /* Text areas / inputs */
        .stTextArea textarea, .stTextInput input {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            color: var(--text-primary) !important;
            font-family: var(--font-body) !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--accent-violet) !important;
            box-shadow: 0 0 0 1px var(--accent-violet) !important;
        }

        /* Metrics — mono numerals, gold value, muted label */
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.1rem 1.3rem;
        }
        [data-testid="stMetricValue"] {
            font-family: var(--font-mono) !important;
            font-size: 1.9rem !important;
            color: var(--accent-violet) !important;
            font-weight: 600;
        }
        [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

        /* Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
        }
        .card-accent {
            border-left: 3px solid var(--accent-violet);
        }

        /* Result banner (replaces default green st.success box) */
        .result-banner {
            background: var(--accent-violet-soft);
            border: 1px solid var(--accent-violet);
            border-radius: 8px;
            padding: 1.1rem 1.4rem;
            margin: 1rem 0;
        }
        .result-banner .lang {
            font-family: var(--font-display);
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent-violet);
        }
        .result-banner .conf {
            font-family: var(--font-mono);
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Dividers */
        hr { border-color: var(--border) !important; }

        /* Dataframes */
        [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

        /* Expander */
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        /* Script strip — the signature hero element */
        .script-strip {
            display: flex;
            overflow-x: auto;
            gap: 0;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            margin: 1.6rem 0 2.2rem 0;
        }
        .script-cell {
            flex: 1 0 auto;
            min-width: 108px;
            text-align: center;
            padding: 1.3rem 0.6rem 1rem 0.6rem;
            border-right: 1px solid var(--border);
        }
        .script-cell:last-child { border-right: none; }
        .script-cell .glyph {
            font-family: var(--font-display);
            font-size: 1.5rem;
            color: var(--text-primary);
            display: block;
            margin-bottom: 0.4rem;
        }
        .script-cell .label {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            letter-spacing: 0.03em;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        /* Eyebrow label used above headlines */
        .eyebrow {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent-teal);
            margin-bottom: 0.4rem;
        }
        </style>
    """, unsafe_allow_html=True)


def render_script_strip():
    """Render the 'Hello' greeting strip across scripts — the app's signature visual."""
    cells = "".join(
        f'<div class="script-cell"><span class="glyph">{glyph}</span>'
        f'<span class="label">{label}</span></div>'
        for glyph, label in GREETINGS
    )
    st.markdown(f'<div class="script-strip">{cells}</div>', unsafe_allow_html=True)

import streamlit as st
from utils import inject_custom_css

st.set_page_config(page_title="About", page_icon="📖", layout="centered")
inject_custom_css()

st.markdown('<div class="eyebrow">Methodology</div>', unsafe_allow_html=True)
st.title("How It Works")

st.markdown("""
### Overview
This project detects the language of a piece of text using a **character-level LSTM
neural network**, trained entirely from scratch — no pretrained embeddings, no external
language-detection library.

### Why character-level, not word-level?
Language identity lives in *character patterns* — letter frequencies, scripts, diacritics —
not vocabulary. A word-level model needs a huge dictionary across 22 languages and breaks on
unseen words. Character-level modeling sidesteps this: the model only needs to recognize
which characters and character sequences are typical of each language.

### Dataset
- **22,000 samples** across **22 languages**, perfectly balanced — 1,000 samples each
- Text sourced from Wikipedia excerpts
- Scripts include Latin, Cyrillic, Arabic, Devanagari, Han, Hangul, Kana, Thai, and Tamil

### Preprocessing
1. Built a character vocabulary from all training text — **7,186** unique characters found
2. Capped to the **top 1,500 most frequent characters** (98.9% coverage) — rare CJK
   characters beyond this were mapped to an `<UNK>` token to keep the model efficient
3. Encoded each text as a sequence of character indices, padded or truncated to **300 characters**

### Architecture
""")

st.code("""Input (300,)
  → Embedding (1502 vocab → 64 dims, mask_zero=True)
  → LSTM (128 units)
  → Dropout (0.3)
  → Dense (64, ReLU)
  → Dropout (0.3)
  → Dense (22, Softmax)""", language="text")

st.markdown("""
~205K parameters total (~800 KB) — light enough to run instantly, even on CPU.

### Training
- Optimizer: Adam, with `ReduceLROnPlateau` shrinking the learning rate as validation loss plateaued
- `EarlyStopping` to halt training and restore the best-performing weights automatically
- Final result: **96.6% accuracy** on a held-out test set

### Known limitation
Short inputs (under ~50 characters) carry less character-level signal, so confidence can drop —
especially among Latin-script languages with overlapping letter patterns (Spanish, Portuguese,
Latin, English). Longer, more natural text gives the most reliable predictions.
""")

st.divider()
st.caption("Built by Omer Farooq · TensorFlow/Keras · Streamlit")

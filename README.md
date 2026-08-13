# Multilingual Language Detector

Character-level LSTM that identifies text across 22 languages. Built with TensorFlow/Keras,
deployed with Streamlit.

## Setup

1. Add these 5 files to the **root** of this folder (same level as `Home.py`) —
   generated from your Colab training notebook:
   - `language_detector_model.keras`
   - `char_to_idx.pkl`
   - `label_encoder.pkl`
   - `config.pkl`
   - `metrics.json` ← generate this with the snippet below if you haven't already

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run Home.py
   ```
   (Note: entry point is `Home.py`, not `app.py`)

## Generating metrics.json (run in Colab, after evaluation)

```python
import json

metrics = {
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'classification_report': classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    ),
    'confusion_matrix': cm.tolist(),
    'class_names': label_encoder.classes_.tolist()
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

from google.colab import files
files.download('metrics.json')
```

## Folder structure

```
language-detector-app/
├── Home.py                          # entry point — landing page
├── pages/
│   ├── 1_🔍_Detector.py             # main detection tool
│   ├── 2_📊_Model_Performance.py    # accuracy, per-language scores, confusion matrix
│   ├── 3_📖_About.py                # architecture + methodology writeup
│   └── 4_🕒_History.py              # session-based prediction log
├── utils.py                         # shared model-loading + prediction logic
├── .streamlit/
│   └── config.toml                  # theme (clean corporate style)
├── requirements.txt
├── language_detector_model.keras    ← you add this
├── char_to_idx.pkl                  ← you add this
├── label_encoder.pkl                ← you add this
├── config.pkl                       ← you add this
└── metrics.json                     ← you add this
```

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (model files included — check they're under
   GitHub's 100MB file limit; at ~800KB this model is fine)
2. Go to share.streamlit.io, connect your repo
3. Set the main file path to `Home.py`
4. Deploy

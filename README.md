# Twitter Sentiment Analyzer

A lightweight pipeline to classify tweet sentiment (Positive / Neutral / Negative) using Python.
Includes data cleaning, tokenization, vectorization (TF-IDF), baseline models (Logistic Regression / Linear SVM),
and simple evaluation.

## ✨ Features
- Clean & preprocess text (lowercasing, URL/mention removal, basic emoji/punctuation handling)
- Train baseline classifiers (LR / Linear SVM) on TF-IDF features
- Evaluate with accuracy/F1 and optional confusion matrix
- Clear, modular layout for quick experimentation

## 🗂️ Project Structure
```
twitter-sentiment-analyzer/
├─ data/
│  ├─ raw/               # raw CSV (not committed)
│  └─ processed/         # cleaned/split data (ignored by git by default)
├─ notebooks/            # exploratory notebooks
├─ src/
│  ├─ preprocess.py      # cleaning & train/valid/test split
│  ├─ features.py        # vectorization (e.g., TF-IDF)
│  ├─ train.py           # train & save model
│  ├─ evaluate.py        # evaluate metrics
│  └─ infer.py           # run sentiment on new text
├─ models/               # saved model/vectorizer (ignored by default)
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

> If your current code differs, keep your structure and adjust paths in the scripts.

## 🚀 Quickstart

### 1) Environment (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data
Put your CSV in `data/raw/` with columns:
- `text`: tweet text
- `label`: sentiment (e.g., `positive`, `neutral`, `negative`)

### 3) Run the pipeline
```bash
# 1) Clean & split (edit args if needed)
python src\preprocess.py --in data\raw\tweets.csv --out data\processed

# 2) Build features
python src\features.py --in data\processed --out data\processed

# 3) Train a model (LR / SVM)
python src\train.py --data data\processed --model_dir models
```

### 4) Evaluate
```bash
python src\evaluate.py --data data\processed --model_dir models
```

### 5) Inference (single text)
```bash
python src\infer.py --text "I love this product!"
```

## 📦 Requirements
See `requirements.txt`. Add/remove packages based on your code.

## 🧪 Notes
- To fetch from API, add a small ingestion script that writes CSV into `data/raw/`.
- To commit models or processed data, remove them from `.gitignore`.

## 🛡️ License
MIT — see `LICENSE`.

## 🙌 Credits
Built by **Lohith B N**. Contributions and issues welcome!

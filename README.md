# SpamGuard NLP — Intelligent SMS Spam Detection & Analysis System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Pytest-23%20Passed-10B981?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

An intelligent, lightweight, and explainable Machine Learning & Natural Language Processing web application that classifies incoming SMS text messages as **SPAM** (unwanted, fraudulent, promotional, phishing) or **HAM** (legitimate, personal, transactional).

Built **100% offline** with traditional NLP techniques and Scikit-learn pipelines — **zero dependence on paid APIs or external LLMs (OpenAI, Gemini, Groq)**.

---

## 📑 Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. System Architecture](#3-system-architecture)
- [4. Technology Stack](#4-technology-stack)
- [5. NLP Preprocessing Pipeline](#5-nlp-preprocessing-pipeline)
- [6. Machine Learning Implementation](#6-machine-learning-implementation)
- [7. Dataset Information](#7-dataset-information)
- [8. Project Directory Structure](#8-project-directory-structure)
- [9. Installation & Quick Start](#9-installation--quick-start)
- [10. Model Training & Benchmarking](#10-model-training--benchmarking)
- [11. Running Automated Tests](#11-running-automated-tests)
- [12. Application Navigation & Pages](#12-application-navigation--pages)
- [13. Streamlit Cloud Deployment](#13-streamlit-cloud-deployment)
- [14. Limitations & Responsible AI](#14-limitations--responsible-ai)
- [15. Future Enhancements](#15-future-enhancements)
- [16. License & Attribution](#16-license--attribution)

---

## 1. Project Overview

Mobile SMS spam causes billions of dollars in phishing, scam, and smishing losses globally every year. **SpamGuard NLP** provides an educational and production-grade demonstration of how classical Natural Language Processing coupled with supervised probabilistic machine learning can reliably filter malicious communications in real-time.

### Why Classical NLP instead of LLMs?
- **Zero API Costs:** Runs completely locally without recurring token billing.
- **Ultra-Low Latency:** Inferences occur in sub-millisecond speeds (< 5ms per message).
- **Absolute Privacy:** User text never leaves your device or server.
- **Explainability:** Transparent heuristic risk indicators and token feature attribution.

---

## 2. Key Features

- **Real-Time Single Message Detection:**
  - Live character and word counters with SMS segment estimation.
  - One-click testing presets for both spam schemes and genuine conversational ham.
  - Class probabilities, model confidence scores, and heuristic risk scoring (0–100).
  - Downloadable audit inspection reports in `.TXT` and `.CSV`.
- **High-Throughput Bulk Batch Processing:**
  - Ingests CSV or TSV datasets with automatic column detection or manual override.
  - High-speed vectorized batch inference.
  - Interactive Plotly charts: Spam/Ham distribution donut, confidence histogram, length box plots.
  - Searchable and filterable data tables with bulk export.
- **Interactive Analytics Dashboard:**
  - Real-time KPI summary cards (Total analyzed, spam count, ham count, spam incidence rate).
  - Time-series timeline activity chart.
  - Top detected suspicious keywords horizontal bar chart.
  - SQLite telemetry with parameterized queries and fallback to in-memory caching.
- **Comparative Model Performance Benchmark:**
  - Rigorous offline holdout test split (20% stratified).
  - Compares **Multinomial Naive Bayes** and **Logistic Regression**.
  - Interactive Confusion Matrix heatmaps (TN, FP, FN, TP).
  - Precision vs. Recall trade-off scatter plot.
  - One-click interactive pipeline retraining controls.
- **Enterprise-Ready Cybersecurity UX:**
  - Dark navy modern SaaS aesthetic (`#0f172a`, `#1e293b`).
  - Mobile, tablet, and desktop responsive layouts.
  - Fully accessible tooltips, status badges, and feedback toasts.

---

## 3. System Architecture

```
                            +---------------------------------+
                            |       USER INTERFACE LAYER      |
                            |   (Streamlit Multi-Page App)    |
                            +----------------+----------------+
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
                   v                                                   v
      [Single Message Inspection]                            [Bulk Dataset Upload]
                   |                                                   |
                   +-------------------------+-------------------------+
                                             |
                                             v
                            +---------------------------------+
                            |      INPUT VALIDATION ENGINE    |
                            | (Empty check, size constraints) |
                            +----------------+----------------+
                                             |
                                             v
                            +---------------------------------+
                            |     NLP PREPROCESSING ENGINE    |
                            | • Lowercase & Whitespace Clean  |
                            | • Regex Entity Normalization    |
                            | • Stopwords + Negation Preserv. |
                            | • Porter Stemming               |
                            +----------------+----------------+
                                             |
                                             v
                            +---------------------------------+
                            |     TF-IDF N-GRAM VECTORIZER    |
                            | (1-2 N-Grams, 5000 max features)|
                            +----------------+----------------+
                                             |
                                             v
                            +---------------------------------+
                            |    MACHINE LEARNING INFERENCE   |
                            |   Multinomial Naive Bayes / LR  |
                            +----------------+----------------+
                                             |
                   +-------------------------+-------------------------+
                   |                                                   |
                   v                                                   v
           [Prediction: SPAM]                                  [Prediction: HAM]
                   |                                                   |
                   +-------------------------+-------------------------+
                                             |
                                             v
                            +---------------------------------+
                            |   EXPLAINABILITY & AUDIT ENGINE |
                            | • Class Probability & Confidence|
                            | • Token Contribution Weights    |
                            | • Heuristic Risk Score (0-100)  |
                            +----------------+----------------+
                                             |
                                             v
                            +---------------------------------+
                            |   SQLITE AUDIT PERSISTENCE DB   |
                            |  (Parameterized local storage)  |
                            +---------------------------------+
```

---

## 4. Technology Stack

| Layer | Component | Version | Role |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.11+ | Core runtime |
| **Frontend** | Streamlit | 1.40+ | Interactive multi-page web framework |
| **Styling** | Custom CSS | CSS3 | Dark navy modern SaaS interface |
| **Data Viz** | Plotly | 5.24+ | Interactive responsive charts |
| **NLP Engine** | NLTK | 3.9+ | Tokenization, stopwords, Porter stemmer |
| **Machine Learning**| Scikit-learn | 1.5+ | TF-IDF, MultinomialNB, LogisticRegression |
| **Data Processing** | Pandas & NumPy | 2.2+ / 1.26+ | Dataframe manipulation & vectorized math |
| **Model Storage** | Joblib | 1.4+ | Serialization of complete fitted pipeline |
| **Database** | SQLite3 | Built-in | Local prediction audit log persistence |
| **Testing** | Pytest | 8.0+ | Automated unit and integration test suite |

---

## 5. NLP Preprocessing Pipeline

The preprocessing engine (`src/preprocessing.py`) handles text cleaning before TF-IDF vectorization:

1. **Normalization:** Converts characters to lowercase and collapses redundant whitespace.
2. **Entity Tokenization:**
   - URLs $\rightarrow$ `url_token`
   - Email Addresses $\rightarrow$ `email_token`
   - Phone Numbers $\rightarrow$ `phone_token`
   - Currency Symbols (£, $, €, ¥) $\rightarrow$ `currency_token`
3. **Character Compression:** Normalizes elongated expressive spellings (e.g., `coooool` $\rightarrow$ `cool`).
4. **Negation Preservation:** NLTK English stopwords are filtered out, **except** critical negation words (*not*, *no*, *never*, *neither*, *cannot*) to prevent semantic flipping.
5. **Stemming:** Applies Porter Stemmer to reduce words to their base roots (e.g., `claim`, `claiming`, `claimed` $\rightarrow$ `claim`).
6. **Feature Extraction:** Sublinear TF-IDF scaling over unigrams and bigrams (`ngram_range=(1, 2)`).

---

## 6. Machine Learning Implementation

Two distinct algorithms were implemented, trained, and benchmarked on identical 80/20 stratified splits:

### Model 1: Multinomial Naive Bayes (MNB) — Selected Champion
- **Configuration:** `alpha=0.15` (Laplace Additive Smoothing)
- **Test Accuracy:** **98.54%**
- **Spam Precision:** **98.29%** (Only 2 False Positives out of 902 legitimate messages)
- **Spam Recall:** **89.84%**
- **Spam F1-Score:** **93.88%**
- **Selection Rationale:** In email and SMS spam detection, **Precision is paramount**. Flagging a legitimate SMS (e.g., doctor appointment, bank OTP, friend message) as Spam (False Positive) is significantly more damaging than letting a rare spam slip through (False Negative). MNB achieved superior precision with minimal false alarms.

### Model 2: Logistic Regression (LR)
- **Configuration:** `C=4.0`, `max_iter=1000`, `class_weight='balanced'`
- **Test Accuracy:** **98.74%**
- **Spam Precision:** **96.75%** (4 False Positives)
- **Spam Recall:** **92.97%**
- **Spam F1-Score:** **94.82%**

---

## 7. Dataset Information

The system utilizes the authoritative **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

- **Total SMS Messages:** 5,574 (4,827 ham, 747 spam)
- **Data Leakage Prevention:** 414 duplicate messages are dropped prior to the train/test split.
- **Final Clean Corpus:** 5,160 unique messages (4,518 ham, 642 spam).
- **Training Subset (80%):** 4,118 messages.
- **Holdout Test Subset (20%):** 1,030 messages.

---

## 8. Project Directory Structure

```
SpamGuard-NLP/
│
├── app.py                         # Main Streamlit entry point and routing coordinator
│
├── pages/
│   ├── 1_Home.py                  # Landing dashboard with KPIs and architecture cards
│   ├── 2_Single_Detection.py      # Real-time single message inspection & explainability
│   ├── 3_Bulk_Detection.py        # Batch CSV/TSV processing & visualization
│   ├── 4_Analytics.py             # Telemetry, Plotly analytics & SQLite history
│   ├── 5_Model_Performance.py     # Benchmark, confusion matrices, and retraining
│   └── 6_About.py                 # System documentation, math foundations & disclaimer
│
├── src/
│   ├── __init__.py                # Package declaration
│   ├── preprocessing.py           # NLP text cleaning, entity masking & indicators
│   ├── data_loader.py             # CSV/TSV ingestion, column detection & deduplication
│   ├── train_model.py             # Pipeline training, evaluation & artifact export
│   ├── prediction.py              # Single and bulk inference service
│   ├── evaluation.py              # Metric calculation and Plotly chart generators
│   ├── analytics.py               # Visual analytics and distribution chart helpers
│   ├── database.py                # Parameterized SQLite persistence & memory fallback
│   └── utils.py                   # UI styling injection, presets & report generators
│
├── models/
│   ├── spam_classifier.joblib     # Persisted champion Scikit-learn Pipeline
│   └── model_metadata.json        # Comprehensive training and benchmark metadata
│
├── data/
│   ├── spam.csv                   # Cleaned SMS Spam Collection dataset
│   ├── spam.tsv                   # Tab-separated dataset
│   └── README.md                  # Dataset attribution and licensing
│
├── assets/
│   ├── styles.css                 # Custom CSS stylesheet
│   └── logo.png                   # SpamGuard brand identity logo
│
├── tests/
│   ├── test_preprocessing.py      # NLP normalization & negation unit tests
│   ├── test_prediction.py         # Model inference & confidence tests
│   ├── test_training.py           # Dataset loading & metric computation tests
│   └── test_database.py           # SQLite operations & fallback tests
│
├── scripts/
│   └── download_dataset.py        # Automated dataset and NLTK downloader
│
├── .streamlit/
│   └── config.toml                # Server and dark theme configuration
│
├── .gitignore                     # Git exclusion rules
├── requirements.txt               # Pinned Python package dependencies
├── runtime.txt                    # Target Python runtime version
├── LICENSE                        # MIT License
├── DEPLOYMENT.md                  # Complete cloud deployment guide
└── README.md                      # Primary project documentation
```

---

## 9. Installation & Quick Start

Follow these exact commands in Windows PowerShell:

### 1. Clone or Open Project
```powershell
cd "c:\Users\ASUS\Desktop\SpamGuard NLP"
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Dataset & NLP Resources
The dataset and pre-trained model artifacts are included. To re-verify or download fresh:
```powershell
python scripts/download_dataset.py
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Your default web browser will automatically open: `http://localhost:8501`.

---

## 10. Model Training & Benchmarking

To execute the offline model training pipeline and generate fresh artifacts:

```powershell
python -m src.train_model
```

**Output:**
```
[*] Loading dataset from: data/spam.csv
    - Cleaned records: 5160 (414 duplicates removed)
    - Class distribution: {'ham': 4518, 'spam': 642}
[*] Preprocessing text corpus with NLP engine...
[*] Performing Stratified Train/Test Split (ratio: 0.8/0.2)...
    - Train samples: 4118 (Spam: 513, Ham: 3605)
    - Test samples:  1030 (Spam: 128, Ham: 902)
[*] Training Multinomial Naive Bayes...
      - Accuracy:       98.54%
      - Spam Precision: 98.29%
      - Spam Recall:    89.84%
      - Spam F1-Score:  93.88%
[*] Training Logistic Regression...
      - Accuracy:       98.74%
      - Spam Precision: 96.75%
      - Spam Recall:    92.97%
      - Spam F1-Score:  94.82%
[OK] Selected Best Model: Multinomial Naive Bayes
[OK] Model artifacts saved successfully.
```

---

## 11. Running Automated Tests

Run the full automated Pytest test suite:

```powershell
python -m pytest -v
```

**Result:**
```
tests/test_database.py::test_init_db PASSED
tests/test_database.py::test_insert_and_retrieve_single_prediction PASSED
tests/test_database.py::test_insert_bulk_predictions PASSED
tests/test_database.py::test_filtering_and_search PASSED
tests/test_database.py::test_clear_history PASSED
tests/test_prediction.py::test_model_artifact_exists PASSED
tests/test_prediction.py::test_predict_single_spam_detection PASSED
tests/test_prediction.py::test_predict_single_ham_detection PASSED
tests/test_prediction.py::test_predict_single_empty_input PASSED
tests/test_prediction.py::test_predict_bulk_dataframe PASSED
tests/test_prediction.py::test_predict_bulk_invalid_column PASSED
tests/test_preprocessing.py::test_lowercase_conversion PASSED
tests/test_preprocessing.py::test_empty_and_whitespace_input PASSED
tests/test_preprocessing.py::test_url_normalization PASSED
tests/test_preprocessing.py::test_email_normalization PASSED
tests/test_preprocessing.py::test_currency_and_phone_normalization PASSED
tests/test_preprocessing.py::test_negation_preservation PASSED
tests/test_preprocessing.py::test_repeated_characters_compression PASSED
tests/test_preprocessing.py::test_heuristic_indicators_extraction PASSED
tests/test_preprocessing.py::test_heuristic_empty_input PASSED
tests/test_training.py::test_detect_columns PASSED
tests/test_training.py::test_dataset_loading_and_deduplication PASSED
tests/test_training.py::test_compute_metrics PASSED

============================= 23 passed in 7.67s =============================
```

---

## 12. Application Navigation & Pages

The application features a modern sidebar navigation with persistent active indicators and model status:

1. **Home Dashboard (`pages/1_Home.py`):**
   - Hero banner with quick action CTAs.
   - Aggregate statistics cards (total analyzed, spam count, ham count, spam percentage).
   - Visual cards explaining How SpamGuard Works and NLP technology overview.
   - Active model specification and latest prediction previews.
2. **Single Message Detection (`pages/2_Single_Detection.py`):**
   - Interactive text area with live character/word counter.
   - 6 one-click test message presets.
   - Prominent Spam (Red) and Ham (Green) result cards with model confidence and class probabilities.
   - Heuristic risk breakdown, token weight explainability, preprocessed text view.
   - Downloadable inspection reports in `.TXT` and `.CSV`.
3. **Bulk Message Detection (`pages/3_Bulk_Detection.py`):**
   - Drag-and-drop CSV/TSV upload with automatic text column detection.
   - Vectorized batch inference with KPI metric cards.
   - Interactive Plotly visualizations (distribution donut, confidence histogram, length box plots).
   - Filter tabs (All, Spam Only, Ham Only) and keyword search.
   - Download complete results as CSV and audit summary as TXT.
4. **Analytics Dashboard (`pages/4_Analytics.py`):**
   - All-time audited telemetry and spam incidence tracking.
   - Plotly timeline trends and top detected suspicious keywords bar chart.
   - Searchable, filterable prediction history table.
   - SQLite persistence management with confirmable history clear controls.
5. **Model Performance (`pages/5_Model_Performance.py`):**
   - Comprehensive comparative evaluation between Multinomial Naive Bayes and Logistic Regression.
   - Interactive Confusion Matrix heatmaps with TN, FP, FN, TP breakdowns.
   - Precision vs. Recall trade-off scatter plot.
   - Detailed per-class classification reports.
   - TF-IDF vocabulary metadata and top predictive n-gram tags.
   - Live interactive pipeline retraining button.
6. **About Project (`pages/6_About.py`):**
   - Executive mission statement and privacy-first architecture.
   - 6-step NLP pipeline deep-dive.
   - Mathematical foundations of Multinomial Naive Bayes and Logistic Regression.
   - Full technology stack inventory and dataset citations.
   - Mandatory responsible AI disclaimer.

---

## 13. Streamlit Cloud Deployment

SpamGuard NLP is configured for **Streamlit Community Cloud**:

1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **"Deploy!"**.
6. The app launches immediately with pre-trained artifacts without requiring any external API keys or secrets!

For complete cloud and Docker deployment steps, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 14. Limitations & Responsible AI

- **Probabilistic Classification:** Statistical machine learning models identify mathematical similarities to historical training data; they do not possess true semantic understanding or legal judgment.
- **Language Scope:** The current model is trained primarily on English SMS messages. Non-English messages may produce lower confidence predictions.
- **Evolving Adversarial Spam:** Sophisticated spammers deliberately obfuscate text (leetspeak, zero-width spaces). While SpamGuard employs regex entity normalization, zero-day evasion tactics require ongoing model updates.
- **Mandatory Disclaimer:** *This application provides automated message classification for educational and informational purposes. Predictions are not guaranteed to identify all spam or fraudulent messages.*

---

## 15. Future Enhancements

- [ ] Support for Email format parsing (`.eml` and MIME headers).
- [ ] Multilingual spam detection (Spanish, French, Hindi).
- [ ] Integrated phishing URL reputation verification via local blocklists.
- [ ] Lightweight deep learning comparison model (e.g., 1D-CNN or BiLSTM via ONNX runtime).
- [ ] Real-time SMS webhook integration for automated messaging gateways.

---

## 16. License & Attribution

- **License:** Released under the [MIT License](LICENSE).
- **Dataset Citation:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *SMS Spam Collection*. UCI Machine Learning Repository.

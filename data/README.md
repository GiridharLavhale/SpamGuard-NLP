# SMS Spam Collection Dataset

This directory contains the dataset used for training, evaluating, and benchmarking the **SpamGuard NLP** intelligent SMS spam classification system.

---

## 1. Dataset Overview

The **SMS Spam Collection** is a public set of SMS tagged messages that have been collected for SMS Spam research.

- **Source:** [UCI Machine Learning Repository: SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Creators:** Tiago A. Almeida and José María Gómez Hidalgo
- **Total Records:** 5,574 English text messages
- **Class Labels:**
  - `ham`: Legitimate, non-spam SMS messages (4,827 raw / 4,518 unique)
  - `spam`: Unwanted, promotional, or malicious SMS messages (747 raw / 642 unique)

---

## 2. Preprocessing & Deduplication

In accordance with strict machine learning best practices:
1. **Duplicate Elimination:** 414 duplicate SMS messages are filtered out prior to train/test splitting to prevent data leakage and inflated metric scores.
2. **Stratification:** A stratified 80/20 train/test split preserves natural class balance across subsets.
3. **Artifact Independence:** The trained model pipeline is persisted in `models/spam_classifier.joblib` and `models/model_metadata.json`, allowing the application to infer immediately without dataset dependencies.

---

## 3. Automatic Download Script

To re-download or refresh the dataset automatically:

```powershell
python scripts/download_dataset.py
```

This script retrieves the official SMS Spam Collection corpus and formats it into both `data/spam.csv` and `data/spam.tsv`.

---

## 4. Licensing and Attribution

The SMS Spam Collection is freely available for scientific, research, and educational purposes. If you use this dataset, cite:

> Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results. *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng'11)*, Mountain View, CA, USA.

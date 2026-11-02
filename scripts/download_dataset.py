"""
Dataset and NLP Resource Downloader for SpamGuard NLP.
Downloads the official SMS Spam Collection dataset and necessary NLTK data.
"""

import os
import sys
import urllib.request
import zipfile
import io
import pandas as pd
import nltk

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")
TSV_PATH = os.path.join(DATA_DIR, "spam.tsv")


def download_nltk_data():
    """Download required NLTK corpora and tokenizers."""
    resources = ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    print("[*] Checking / Downloading NLTK resources...")
    for res in resources:
        try:
            nltk.download(res, quiet=True)
            print(f"  + {res} downloaded/verified.")
        except Exception as e:
            print(f"  - Warning downloading {res}: {e}")


def download_sms_dataset():
    """Download the official SMS Spam Collection dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 1000:
        print(f"[*] SMS dataset already exists at: {CSV_PATH}")
        return True

    print("[*] Downloading SMS Spam Collection dataset...")

    # Method 1: Direct TSV from authoritative github mirror of UCI SMS Spam Collection
    primary_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    # Fallback: UCI official archive zip
    uci_zip_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

    df = None

    try:
        print(f"  -> Attempting download from: {primary_url}")
        req = urllib.request.Request(
            primary_url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(content), sep='\t', names=['label', 'message'], quoting=3)
            print(f"  -> Successfully retrieved {len(df)} records from primary source.")
    except Exception as e:
        print(f"  -> Primary source download failed: {e}")
        try:
            print(f"  -> Attempting UCI Zip source: {uci_zip_url}")
            req = urllib.request.Request(
                uci_zip_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                zip_bytes = resp.read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    for filename in z.namelist():
                        if "SMSSpamCollection" in filename:
                            with z.open(filename) as f:
                                df = pd.read_csv(f, sep='\t', names=['label', 'message'], quoting=3, encoding='utf-8')
                                print(f"  -> Retrieved {len(df)} records from UCI zip.")
                                break
        except Exception as e2:
            print(f"  -> UCI zip download failed: {e2}")

    if df is not None and not df.empty:
        # Standardize columns and labels
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['message'] = df['message'].astype(str)

        # Save as CSV and TSV
        df.to_csv(CSV_PATH, index=False, encoding='utf-8')
        df.to_csv(TSV_PATH, sep='\t', index=False, encoding='utf-8')
        print(f"[OK] Saved dataset successfully to:\n    - {CSV_PATH}\n    - {TSV_PATH}")
        print(f"    Total records: {len(df)}")
        print(f"    Distribution:\n{df['label'].value_counts().to_string()}")
        return True
    else:
        print("[!] ERROR: Unable to automatically download the dataset.")
        print("    Please place the SMS Spam Collection dataset as data/spam.csv")
        return False


if __name__ == "__main__":
    download_nltk_data()
    success = download_sms_dataset()
    sys.exit(0 if success else 1)

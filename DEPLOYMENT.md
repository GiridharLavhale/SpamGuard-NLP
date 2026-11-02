# Deployment Guide: SpamGuard NLP

This guide provides complete, production-tested instructions for deploying **SpamGuard NLP** locally and to cloud platforms, specifically **Streamlit Community Cloud** and containerized environments.

---

## 1. Streamlit Community Cloud (Recommended Free Hosting)

Streamlit Community Cloud provides free, instant cloud deployment for Streamlit repositories directly connected to GitHub.

### Prerequisites

- A [GitHub](https://github.com/) account.
- A [Streamlit Community Cloud](https://share.streamlit.io/) account.
- The `SpamGuard-NLP` repository pushed to GitHub.

### Step-by-Step Deployment Instructions

1. **Commit & Push Repository to GitHub:**
   Ensure all essential files and trained model artifacts are staged and pushed:
   ```powershell
   git init
   git add .
   git commit -m "Initial release: SpamGuard NLP v1.0.0"
   git branch -M main
   git remote add origin https://github.com/<your-username>/SpamGuard-NLP.git
   git push -u origin main
   ```

2. **Verify Critical Files in Repository:**
   Ensure your remote repository contains:
   - `app.py` (Main entry point)
   - `pages/` (All 6 multipage views)
   - `src/` (Core preprocessing, inference, and database modules)
   - `models/spam_classifier.joblib` (Pre-trained champion Scikit-learn pipeline)
   - `models/model_metadata.json` (Model benchmark metadata)
   - `requirements.txt` (Pinned production dependencies)
   - `.streamlit/config.toml` (Theme & server configurations)

3. **Deploy via Streamlit Community Cloud:**
   - Log into [share.streamlit.io](https://share.streamlit.io/).
   - Click **"Create app"** (or **"New app"**).
   - In the **Repository** field, select or paste: `<your-username>/SpamGuard-NLP`.
   - In the **Branch** field, select: `main`.
   - In the **Main file path** field, enter: `app.py`.
   - Click **"Deploy!"**.

4. **Zero-Configuration Verification:**
   - **No Secrets Required:** SpamGuard NLP operates entirely on traditional NLP and Machine Learning. No OpenAI, Gemini, Groq, or third-party API keys are required.
   - **Automatic Artifact Loading:** The application loads `models/spam_classifier.joblib` upon startup. No dataset downloading or retraining is executed during boot.
   - **In-Memory Fallback:** In ephemeral cloud environments where persistent SQLite disk writes may be sandboxed, SpamGuard NLP automatically falls back to in-memory session history without crashing.

---

## 2. Local Production Execution

Run SpamGuard NLP locally on Windows, macOS, or Linux.

### Windows (PowerShell)

```powershell
# 1. Create Python virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify / Run tests
python -m pytest -v

# 5. Launch web application
streamlit run app.py
```

### Linux / macOS (Bash)

```bash
# 1. Create Python virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify / Run tests
pytest -v

# 5. Launch web application
streamlit run app.py
```

---

## 3. Docker Containerization (Optional)

Deploy SpamGuard NLP as an isolated Docker container on AWS, GCP, Azure, or DigitalOcean.

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK resources
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"

# Copy application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Build & Run Commands

```bash
# Build Docker image
docker build -t spamguard-nlp:latest .

# Run container
docker run -d -p 8501:8501 --name spamguard spamguard-nlp:latest

# Open browser at:
# http://localhost:8501
```

---

## 4. Verification Checklist

- [x] Application launches without runtime warnings.
- [x] Model pipeline deserializes from `models/spam_classifier.joblib`.
- [x] Single-message detection responds with confidence and explainability.
- [x] Bulk CSV/TSV upload processes and renders Plotly charts.
- [x] SQLite database creates tables automatically or uses graceful memory fallback.
- [x] Export buttons generate `.TXT` audit reports and `.CSV` prediction dumps.
- [x] Zero external API key dependencies.

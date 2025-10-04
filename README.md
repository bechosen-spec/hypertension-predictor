# 🩺 Hypertension Risk Predictor

**Live demo:** [https://hypertension-predictor.streamlit.app](https://hypertension-predictor.streamlit.app)

A Streamlit app that predicts a patient’s hypertension risk from simple inputs, highlights the key factors influencing the prediction, and generates **personalized health guidance** using Google **Gemini** (Vertex AI). It also includes **Sign Up / Sign In** and a **Dashboard** for viewing and exporting past predictions and LLM feedback.

---

## ✨ Features

* **Prediction UI**
  Collects Age, BMI, Salt Intake, Stress, Sleep, BP history, Medication, Family history, Exercise, and Smoking status → predicts risk.
* **Explainability**
  Shows **global feature importances** (from your model) and a **local snapshot** of top factors for the current case.
* **Personalized Guidance (LLM)**
  Uses Gemini (Vertex AI) to produce clear, non-judgmental, actionable tips tailored to the patient’s inputs and model outcome.
* **Authentication & Dashboard**
  Users can **Sign Up / Sign In**. After sign-in, a **Dashboard** lists past predictions, lets you inspect details, and **export CSV/JSON**.
* **Local logging (SQLite)**
  Stores user, timestamp, patient input, model output, key factors, and LLM guidance.

---

## 🧱 Architecture

* **Frontend:** Streamlit (`app.py`)
* **Inference & Preprocessing:** `inference.py`
  Loads `best_rf_model.joblib`, `scaler.joblib`, and `feature_names.joblib`; aligns/encodes inputs exactly like training.
* **LLM Client:** `vertex_config.py` using `google-genai` (Vertex AI)
* **Database:** SQLite via SQLAlchemy (`db.py`)

  * `users` (simple auth)
  * `prediction_logs` (audit trail)
* **Dashboard:** Built into the main app (gated behind auth)
* **Scripts / Tests:**

  * `scripts/verify_artifacts.py` quick artifact sanity check
  * `scripts/export_feature_names.py` generate `feature_names.joblib`
  * `tests/` basic unit tests & CI (GitHub Actions)

---

## 📁 Repository structure

```
.
├── app.py
├── inference.py
├── vertex_config.py
├── db.py
├── models/
│   ├── best_rf_model.joblib
│   ├── scaler.joblib
│   └── feature_names.joblib
├── scripts/
│   ├── verify_artifacts.py
│   └── export_feature_names.py
├── tests/
│   ├── conftest.py
│   └── test_inference.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml            # DO NOT COMMIT (ignored)
├── .github/workflows/
│   ├── ci.yml
│   └── codeql.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting started (local)

### 1) Clone & create a virtual environment

```bash
git clone <your-repo-url> hypertension-predictor
cd hypertension-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Place model artifacts

Put these **three files** in `models/`:

* `best_rf_model.joblib`
* `scaler.joblib`
* `feature_names.joblib`  ← must match the **exact** columns of your training design matrix.

> Don’t have `feature_names.joblib` yet?
>
> * If you have your **final X_train CSV** (already one-hot encoded):
>
>   ```bash
>   python scripts/export_feature_names.py --design-csv path/to/X_train_final.csv
>   ```
> * If you only have a **raw training CSV**:
>
>   ```bash
>   python scripts/export_feature_names.py --raw-csv path/to/train.csv --target-column TARGET_NAME
>   ```

### 3) Configure Vertex AI (Gemini)

**Option A (recommended for local):** point to a real key file

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/keys/vertex-sa.json"
export PROJECT_ID="your-gcp-project-id"
export LOCATION="us-central1"
```

**Option B (Streamlit secrets):** paste into `.streamlit/secrets.toml` (not committed)

```toml
[gcp]
project_id = "your-gcp-project-id"
location   = "us-central1"
service_account_json = """
{ ...full service account JSON with \\n in the private_key... }
"""
```

> ⚠️ Ensure the `private_key` in the JSON contains **escaped** newlines (`\\n`) — not literal line breaks.

### 4) Initialize DB & run a quick artifact check (optional)

```bash
python scripts/verify_artifacts.py
```

### 5) Run the app

```bash
streamlit run app.py
```

Open the local URL shown, **Sign Up** to create a user, **Sign In**, then try a prediction and visit the **Dashboard** tab.

---

## 🔐 Authentication notes

* The app includes a minimal user system in `db.py` using **bcrypt**.
* First-time: click **Sign Up** in the app to create an account.
* Data is stored in `app.db` (SQLite) by default. Set `DB_URL` to use Postgres if desired, e.g.:

  ```
  export DB_URL="postgresql+psycopg2://user:password@host:5432/dbname"
  ```

---

## ☁️ Deploying to Streamlit Community Cloud

1. **Push** your repo to GitHub (make sure `.streamlit/secrets.toml`, any `*.json` keys, `app.db`, and `models/*.joblib` are **ignored**).
2. On Streamlit Cloud, **Create app** from your repo.
3. In **App → Settings → Secrets**, paste:

   ```toml
   [gcp]
   project_id = "your-gcp-project-id"
   location   = "us-central1"
   service_account_json = """
   { ...full service account JSON with \\n in the private_key... }
   """
   ```
4. (Optional) Upload your model artifacts somewhere accessible (you can commit **non-sensitive** artifacts if licensing allows; otherwise host them privately and load at startup).
5. Deploy. Sign up a user and test.

---

## 🧪 Tests & CI

* Run tests:

  ```bash
  pytest -q
  ```
* CI: GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on pushes/PRs.
* Security scanning: CodeQL workflow (`codeql.yml`) runs weekly.

---

## 🔧 Configuration

* **LLM model & settings:** fixed in `app.py`

  ```python
  LLM_MODEL_NAME = "gemini-2.5-pro"
  LLM_TEMPERATURE = 0.3
  LLM_TOP_P = 0.95
  LLM_MAX_TOKENS = 2048
  ```
* **Theme & server:** `.streamlit/config.toml`
  (Keep `enableXsrfProtection = true`; remove `enableCORS=false` to avoid warnings.)
* **DB URL:** `DB_URL` env var (defaults to `sqlite:///app.db`).

---

## 🧰 Troubleshooting

* **“LLM feedback unavailable: … not a valid json file … Invalid control character …”**
  Your service account JSON in secrets has raw newlines. Use `\\n` escapes or set `GOOGLE_APPLICATION_CREDENTIALS` to point to the key file.

* **Partial/short LLM output**
  We stitch all text parts and increased `max_output_tokens` to 2048. If you still see truncation, raise it further.

* **“feature_names mismatch / wrong column count”**
  `feature_names.joblib` **must** match the exact post-dummies order used in training. Re-export with `scripts/export_feature_names.py`.

* **macOS LibreSSL warning**
  Harmless for local dev. Consider using Python 3.10+ to link against a newer OpenSSL.

* **Nothing appears on Dashboard**
  Make a prediction first; only successful runs are logged. Also ensure you’re **signed in** (Dashboard is gated).

* **GitHub rejects push for secrets**
  Rotate/revoke the leaked key in GCP, purge history with `git filter-repo`, and force-push. Don’t commit `.streamlit/secrets.toml`.

---

## 📄 Data & Privacy

* This tool is for **educational purposes** and does **not** replace professional medical advice.
* Avoid storing personally identifiable information (PII) in free-text fields.
* LLM calls send the minimal necessary context to Vertex AI.

---

## 📦 Requirements (key packages)

See `requirements.txt`, including:

* `streamlit`
* `pandas`
* `scikit-learn` (for scaler compatibility if needed)
* `joblib`
* `sqlalchemy`
* `bcrypt`
* `google-genai` (Vertex AI client)

---

## 📝 License

Add your preferred license (MIT/Apache-2.0/etc.) to a `LICENSE` file.

---

## 🙌 Acknowledgements

* Streamlit for the rapid UI
* Google Vertex AI Gemini for LLM guidance
* scikit-learn for classical ML utilities
* Boniface Emmanuel, My Research Assistant


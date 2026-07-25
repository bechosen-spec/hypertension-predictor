# PulseWise Risk

PulseWise Risk is a polished Streamlit demonstration that estimates hypertension-related risk from ten health and lifestyle inputs. A saved Random Forest classifier produces the classification and probability. Google Gemini only explains that model output and generates cautious wellness guidance—it does **not** make the prediction.

> This project is educational. It is not a medical device, diagnosis, emergency service, or replacement for a qualified healthcare professional.

## User experience

1. Landing page with product explanation, privacy messaging, sign-in, and account creation
2. Private authenticated overview and recent activity
3. Four-stage assessment: basic information, lifestyle, medical background, and review
4. Calm result summary with estimated probability and cautious interpretation
5. Relevant submitted factors and global model-importance visualization
6. Exactly five Gemini recommendations, with a generic fallback when Gemini is unavailable
7. User-owned history, filtering, trend visualization, CSV/JSON exports, and an HTML assessment report
8. General health education and account/privacy information

## Project layout

```text
app.py                         Main application and page flow
inference.py                   Artifact loading, preprocessing, prediction
db.py                          SQLAlchemy users and prediction records
vertex_config.py               Vertex AI / Gemini client configuration
styles.py                      Central visual theme and responsive CSS
ui_components.py               Reusable cards, notices, progress, images
utils/auth.py                  Shared Streamlit authentication state
assets/heart_health_hero.png   Locally generated, optimized hero artwork
models/                        Existing model, scaler, and feature schema
tests/                         Inference, privacy, auth, and UI helper tests
scripts/                       Artifact export and verification helpers
```

## Prediction and explanation

The inference pipeline remains compatible with the saved training workflow:

- Numerical inputs: `Age`, `Salt_Intake`, `Stress_Score`, `Sleep_Duration`, and `BMI`
- Categorical inputs are one-hot encoded using the existing values
- The saved scaler transforms numerical columns
- Every row is aligned to the exact saved feature-name order
- The Random Forest returns the class and class-1 probability

“Global model importance” describes patterns used by the model across its training data. It is not an explanation of one person’s result. “Relevant submitted factors” shows active inputs ranked using that global importance. It does not claim SHAP values or causal patient-level contributions.

## Screenshots

Add deployment screenshots here after a manual browser review:

- `docs/screenshots/landing.png`
- `docs/screenshots/assessment.png`
- `docs/screenshots/results.png`
- `docs/screenshots/history.png`

The hero artwork is stored locally so the landing page does not depend on a remote image URL. Missing artwork falls back to an accessible in-app placeholder.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
streamlit run app.py
```

The existing artifacts must remain in `models/`:

- `best_rf_model.joblib`
- `scaler.joblib`
- `feature_names.joblib`

## Configuration

### Database

SQLite is used by default:

```bash
export DB_URL="sqlite:///app.db"
```

For a supported managed SQLAlchemy database, provide its URL and install the appropriate driver:

```bash
export DB_URL="postgresql+psycopg2://user:password@host:5432/database"
```

### Vertex AI

Environment configuration:

```bash
export PROJECT_ID="your-gcp-project"
export LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

Alternatively, configure Streamlit secrets:

```toml
[gcp]
project_id = "your-gcp-project"
location = "us-central1"
service_account_json = """{ ... }"""
```

Never commit `.streamlit/secrets.toml`, database files, or credentials. If Gemini cannot be reached, the model result is still saved and five clearly labeled generic wellness suggestions are displayed.

## Testing

```bash
python -m pytest -q
python scripts/verify_artifacts.py
python -m compileall -q app.py db.py inference.py vertex_config.py styles.py ui_components.py utils pages tests
```

Tests cover artifact loading, feature alignment, output/probability behavior, importance ordering, shared authentication state, database-level record ownership, missing-image fallback detection, and risk presentation helpers.

## Streamlit Community Cloud

1. Push the repository without secrets or `app.db`.
2. Create a Streamlit app using `app.py`.
3. Add the Vertex configuration in the app’s Secrets settings.
4. Use a persistent managed database through `DB_URL` if prediction history must survive restarts.
5. Confirm that model artifacts are distributed legally and are available at startup.
6. Run the manual privacy, authentication, prediction, fallback, and mobile-layout checks.

## Security and privacy limitations

- User-facing history is restricted by a required username predicate in the database query.
- Passwords are hashed using bcrypt.
- SQLite is intended for a local or portfolio demonstration, not a multi-instance production health system.
- Production deployment needs formal authorization design, encryption strategy, secret management, backups, retention controls, monitoring, incident response, and a legal/privacy review.
- Gemini receives assessment context when guidance generation is enabled.
- The project does not claim HIPAA, GDPR, NDPR, or other regulatory compliance.
- Do not enter unnecessary personally identifying information.

## Medical safety

Model classifications are stored internally as `Has Hypertension` or `No Hypertension` for backward compatibility. The interface consistently presents them as model estimates rather than facts.

Seek urgent medical care for severe chest pain, difficulty breathing, fainting, sudden weakness, confusion, or other serious symptoms. Do not use this application to assess an emergency or to start, stop, or change medication.

## License

No license has been selected yet. Add a `LICENSE` file before redistributing the project or its generated artwork.

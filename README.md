# Cosmic Seekers — Exoplanet Explorer & AI Training

An interactive full‑stack project built for the 2025 NASA Space Apps Challenge. The backend (Flask) serves exoplanet data and training APIs; the frontend (HTML/CSS/JS) renders a multi‑page experience for exploring star systems, inspecting transit light curves, and training temporary machine‑learning models online (Random Forest, Gradient Boosting, LightGBM, Logistic Regression).

## Features
- Interactive star system and planet exploration with custom UI and audio.
- Kepler‑style “Transit View” to visualize brightness dips (light curves).
- Analysis Dashboard to adjust parameters and observe real‑time AI confidence.
- Model Evaluation page for four pre‑trained models (accuracy and confusion matrix).
- Upload CSV for preprocessing; receive a cleaned, standardized output.
- Online temporary training API: choose model and hyperparameters; get logs and confusion matrix.

## Project Structure
```
├── app.py                  # Flask app entry and API routes
├── data_preprocessing.py   # Standardization, cleaning, merging, and analysis pipeline
├── Train.py                # Online training pipeline (called by app.py)
├── Train_paper_2024/       # 2024 paper-style training and interpretation tooling
├── Train_paper_2022/       # 2022 docs and data downloading utilities
├── templates/              # Flask templates (index and partial pages)
├── static/                 # Frontend assets (js/css/images/music/textures)
├── models/                 # Pretrained models and tools (scaler, imputer, feature names)
├── Data/                   # Source CSVs (KOI / TESS)
├── exoplanet_data_processed.csv  # Main dataset loaded by backend
├── Dockerfile, Procfile, requirements.txt, apt.txt
```

## Requirements
- Python 3.11 (Docker image uses `python:3.11-slim`)
- OS: Windows / macOS / Linux
- Packages: `flask`, `flask-cors`, `gunicorn`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `werkzeug`

## Quick Start (Local)
1. Create and activate a virtual environment.
   - Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the development server (default `http://localhost:5000`):
   ```bash
   python app.py
   ```
4. Open the homepage: `http://localhost:5000`

Note: The backend loads `exoplanet_data_processed.csv` as its primary source. If you want to regenerate it, see “Data Preprocessing”.

## Run with Docker
```bash
# Build image
docker build -t cosmic-seekers .

# Run container (Gunicorn binds 8080)
docker run -p 8080:8080 cosmic-seekers
# Visit: http://localhost:8080
```

## Deployment
- Works on Heroku / Render / Cloud Run, etc.
- Gunicorn command in `Procfile`: `web: gunicorn -b :$PORT app:app`

## Frontend Pages
- `Tutorial`: quick tour and concepts.
- `Dashboard`: detailed parameters and real‑time AI confidence changes.
- `Kepler View`: transit perspective and light curve visualization.
- `Training System`: upload CSV, select model, set hyperparameters, and trigger online training.

## API Reference (main endpoints)
- `GET /` — Homepage.
- `GET /partials/<page_name>` — Load a partial page (`tutorial`, `dashboard`, `kepler`, `training_system`).
- `GET /get_all_systems` — List all systems as `{ system_id, name }`.
- `GET /get_triple_system_data` — Retrieve data for three random systems.
- `GET /get_specific_triple_system_data/<system_id>` — Retrieve one specific system plus two random others.
- `GET /get_model_stats` — Pretrained model performance summaries (if generated).
- `POST /upload_and_process` — Upload CSV, run standardization/cleaning/analysis, return output path and logs.
- `POST /run_training` — Online training; returns training logs and confusion matrix.

### Example: Upload and preprocess a CSV
```bash
curl -F "file=@path/to/your.csv" http://localhost:5000/upload_and_process
```
Successful response:
```json
{
  "success": true,
  "message": "The uploaded ... dataset has been successfully processed.",
  "file_url": "/static/processed_data/cleaned_..._data.csv",
  "log_output": "...pipeline logs..."
}
```

### Example: Online temporary training
Accepted `model_choice` values: `LogisticRegression`, `RandomForest`, `GradientBoosting`, `LightGBM`.
```bash
curl -X POST http://localhost:5000/run_training \
  -H "Content-Type: application/json" \
  -d '{
    "model_choice": "RandomForest",
    "hyperparameters": {"n_estimators": 200, "max_depth": 8},
    "input_csv_path": "exoplanet_data_processed.csv"
  }'
```
Successful response:
```json
{
  "success": true,
  "message": "模型 RandomForest 訓練體驗完成！",
  "log_output": "...training logs...",
  "confusion_matrix": {
    "labels": ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"],
    "matrix": [[TN, FP, X],[FN, TP, Y],[Z, W, U]]
  }
}
```

## Data Preprocessing
- Sources: KOI/TESS CSVs under `Data/`; combined outputs can be saved to `static/processed_data/combined_exoplanet_data_cleaned.csv`.
- Pipeline summary (`data_preprocessing.py`):
  - Column standardization across sources.
  - Outlier handling (e.g., `stellar_radius`, `stellar_logg`).
  - Outlier flag and final column order alignment.
  - Common analyses (extremes, threshold counts).
- For single uploads, the server runs the appropriate one‑file pipeline and returns a cleaned CSV path.

Tip: See `run_full_preprocessing_pipeline()` for a complete merging flow (example implementation); or use the upload endpoint for single files.

## Development Guide
- Frontend logic: `static/js/main.js` (page state & nav), `static/js/pages/*` (per‑page logic), `static/js/threeManager.js` (scene management).
- UI and styles: `templates/` (Jinja templates), `static/css/style.css`.
- Models and tools: `models/` (pretrained models, scaler, imputer, feature names).

## FAQ
- LightGBM dependency: Docker installs `libgomp1`. On Linux, ensure it’s present.
- Data files: defaults to `exoplanet_data_processed.csv`. If you change paths, provide a relative path to `input_csv_path` in the training API.
- Windows tips: if PowerShell blocks venv activation, adjust execution policy or run as Administrator.

## Credits
Built for the NASA Space Apps Challenge. Thanks to open‑source communities and academic datasets that made this project possible.
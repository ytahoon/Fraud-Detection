# Fraud Detection and Risk Intelligence System

This project builds a fraud detection workflow using two datasets only:

- IEEE-CIS Fraud Detection
- Synthetic Financial Dataset

No other dataset is part of the cleaned project scope.

## Project Flow

1. Explore and model the two fraud datasets.
2. Evaluate models using fraud-focused metrics such as precision, recall, F1, ROC-AUC, and PR-AUC.
3. Use the best model outputs for scoring and analyst review.
4. Connect scoring results to TiDB through the database scoring script.
5. Expose a chatbot API for analyst support and transaction lookup.
6. Use the Streamlit dashboard to inspect fraud patterns, alerts, and model behavior.

## Clean Folder Structure

```text
data/
  raw/
    ieee_cis/
    synthetic_financial/

notebooks/
  01_ieee_cis_xgboost_modeling.ipynb
  02_ieee_cis_model_comparison_thresholds.ipynb
  03_synthetic_financial_eda_cleaning.ipynb
  04_synthetic_financial_xgboost_modeling.ipynb

src/
  database/tidb_scoring.py
  dashboard/dashboard_app.py
  chatbot/chatbot_api.py

artifacts/
  dashboard/
  models/

reports/
  presentation/
```

## Data

IEEE-CIS raw files are stored in:

```text
data/raw/ieee_cis/
```

Expected files:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Synthetic Financial data should be placed at:

```text
data/raw/synthetic_financial/synthetic_financial.csv
```

The Synthetic Financial notebooks also include a fallback for the older `cleaned_data.csv` name, but `synthetic_financial.csv` is the project-standard filename.

## Notebooks

Run notebooks in order:

1. `notebooks/01_ieee_cis_xgboost_modeling.ipynb`
2. `notebooks/02_ieee_cis_model_comparison_thresholds.ipynb`
3. `notebooks/03_synthetic_financial_eda_cleaning.ipynb`
4. `notebooks/04_synthetic_financial_xgboost_modeling.ipynb`

The IEEE model comparison notebook expects a prepared feature artifact at:

```text
artifacts/dashboard/cleaned_ieee_features.csv
```

If that file is missing, export the cleaned IEEE feature matrix before running the comparison notebook.

## Dashboard

Run the dashboard from the project root:

```bash
streamlit run src/dashboard/dashboard_app.py
```

The dashboard reads:

- IEEE-CIS data from `data/raw/ieee_cis/`
- Synthetic Financial data from `data/raw/synthetic_financial/synthetic_financial.csv`
- validation predictions from `artifacts/dashboard/ieee_cis_val_predictions.csv`
- Synthetic Financial model metadata from `artifacts/models/synthetic_financial_xgboost_features.json`

Use the sidebar dataset selector to switch between IEEE-CIS views and Synthetic Financial views.

## TiDB Scoring

The TiDB scorer is:

```text
src/database/tidb_scoring.py
```

It loads a trained model, streams new transactions from TiDB, predicts fraud probability, and upserts results into `fraud_alerts`.

Run example:

```bash
python src/database/tidb_scoring.py --model-path artifacts/models/ieee_cis_xgboost.json
```

Synthetic Financial scoring example:

```bash
python src/database/tidb_scoring.py \
  --dataset synthetic_financial \
  --model-path artifacts/models/synthetic_financial_xgboost.pkl \
  --feature-list-path artifacts/models/synthetic_financial_xgboost_features.json \
  --threshold 0.971 \
  --model-version Synthetic_XGBoost_v1.0
```

Required database environment variables:

- `DB_USER`
- `DB_PASS`
- `DB_HOST`
- `DB_NAME`
- `DB_PORT` default: `4000`

## Chatbot API

The chatbot API is:

```text
src/chatbot/chatbot_api.py
```

Run it from the project root:

```bash
python src/chatbot/chatbot_api.py
```

Useful environment variables:

- `MODEL_PATH`
- `DATA_SOURCE`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `PORT`

The API works without a Gemini key for transaction lookup and fallback responses. Add `GEMINI_API_KEY` to enable general LLM answers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

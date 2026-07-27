#!/usr/bin/env python3
"""Score new transactions from TiDB and write fraud alerts back (with upsert).

Features:
- Load model and optional preprocessor
- Stream transactions in chunks
- Safe column handling and feature alignment
- Batch upsert using ON DUPLICATE KEY UPDATE
- Configurable via environment variables or CLI
"""
import os
import sys
import logging
from datetime import datetime
import argparse
import json
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_FEATURE_PATH = PROJECT_ROOT / "artifacts/models/synthetic_financial_xgboost_features.json"

# --- Import-check block: fail fast with clear message if deps are missing ---
try:
    import pymysql  # ensure pymysql available for SQLAlchemy MySQL driver
except Exception:
    logger.warning("pymysql not available — ensure 'pymysql' is installed (pip install pymysql)")


def load_model(model_path):
    if model_path is None:
        raise ValueError("model_path must be provided")
    # Try XGBoost JSON model first
    try:
        if model_path.endswith('.json') or model_path.endswith('.model'):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            logger.info(f"Loaded XGBoost model from {model_path}")
            return model
    except Exception:
        logger.exception("Failed to load XGBoost model; falling back to joblib")

    # Try joblib (sklearn pipeline or model)
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model (joblib) from {model_path}")
        return model
    except Exception:
        logger.exception("Failed to load model with joblib")
        raise


def load_preprocessor(preproc_path):
    if not preproc_path or not os.path.exists(preproc_path):
        return None
    try:
        pre = joblib.load(preproc_path)
        logger.info(f"Loaded preprocessor from {preproc_path}")
        return pre
    except Exception:
        logger.exception("Failed to load preprocessor via joblib")
        raise


def infer_feature_list_path(model_path, dataset="auto"):
    model_path = Path(model_path)
    candidates = [model_path.with_name(f"{model_path.stem}_features.json")]
    model_hint = model_path.stem.lower()
    if dataset == "synthetic_financial" or "synthetic" in model_hint or "financial" in model_hint:
        candidates.append(DEFAULT_SYNTHETIC_FEATURE_PATH)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_feature_names(feature_list_path=None, model_path=None, dataset="auto"):
    if feature_list_path:
        path = Path(feature_list_path)
    elif model_path:
        path = infer_feature_list_path(model_path, dataset=dataset)
    else:
        path = None

    if not path or not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        features = json.load(f)
    if not isinstance(features, list) or not all(isinstance(col, str) for col in features):
        raise ValueError(f"Feature list must be a JSON list of strings: {path}")

    logger.info("Loaded %s model features from %s", len(features), path)
    return features


def get_feature_names_from_model(model):
    # XGBoost scikit-learn wrapper may have feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    # Try booster feature names
    try:
        booster = model.get_booster()
        return booster.feature_names
    except Exception:
        return None


def _normalize_transaction_type(values):
    return (
        values.astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def prepare_feature_matrix(raw_df, feature_names=None, preprocessor=None, dataset="auto"):
    """Prepare database rows for model scoring and align columns with the trained model."""
    df = raw_df.copy()

    # Common TiDB schema aliases. The Synthetic model expects "type"; the DB schema may use
    # "transaction_type". Amount already matches the Synthetic model.
    if "type" not in df.columns and "transaction_type" in df.columns:
        df["type"] = df["transaction_type"]
    if "amount" not in df.columns and "TransactionAmt" in df.columns:
        df["amount"] = df["TransactionAmt"]

    drop_cols = [
        "transaction_id",
        "TransactionID",
        "sender_account_id",
        "receiver_account_id",
        "customer_id",
        "account_id",
        "timestamp",
        "detected_at",
        "true_is_fraud",
        "isFraud",
        "prediction",
        "model_version",
        "action_taken",
        "transaction_type",
        "currency",
        "location_city",
        "ip_address",
        "device_id",
        "nameOrig",
        "nameDest",
    ]
    X_predict = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if preprocessor is not None:
        return preprocessor.transform(X_predict)

    expected = feature_names or []
    needs_synthetic_type = any(col.startswith("type_") for col in expected)

    if needs_synthetic_type:
        if "type" in X_predict.columns:
            X_predict["type"] = _normalize_transaction_type(X_predict["type"])
        X_trans = pd.get_dummies(X_predict, columns=["type"] if "type" in X_predict.columns else [], dtype=np.int8)
    else:
        X_trans = pd.get_dummies(X_predict, dtype=np.int8)

    X_trans = X_trans.apply(pd.to_numeric, errors="coerce").fillna(0)

    if expected:
        missing_cols = [col for col in expected if col not in X_trans.columns]
        if missing_cols:
            logger.info("Filling %s missing model features with 0: %s", len(missing_cols), missing_cols)
        X_trans = X_trans.reindex(columns=expected, fill_value=0)

    return X_trans


def connect_engine(user, password, host, port, dbname, ssl_ca=None):
    conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
    connect_args = {}
    if ssl_ca:
        connect_args['ssl'] = {'ca': ssl_ca}
    engine = create_engine(conn_str, connect_args=connect_args, pool_pre_ping=True, pool_recycle=3600)
    return engine


def chunked_score(
    engine,
    model,
    preprocessor=None,
    feature_names=None,
    threshold=0.8,
    chunksize=5000,
    batch_upsert=1000,
    model_version='XGBoost_v1.0',
    dataset="auto",
):
    query = """
    SELECT * FROM transactions
    WHERE transaction_id NOT IN (SELECT transaction_id FROM fraud_alerts)
    """

    # Use a single connection context for reading in chunks so it is closed properly
    conn_for_read = engine.connect()
    try:
        read_iter = pd.read_sql_query(sql=text(query), con=conn_for_read, chunksize=chunksize)
    finally:
        # Note: the iterator may lazily fetch; we'll close the connection after iteration below
        pass
    total_processed = 0

    insert_sql = (
        "INSERT INTO fraud_alerts (transaction_id, fraud_score, prediction, model_version, action_taken, detected_at)"
        " VALUES (%s, %s, %s, %s, %s, %s)"
        " ON DUPLICATE KEY UPDATE fraud_score=VALUES(fraud_score), prediction=VALUES(prediction), model_version=VALUES(model_version), action_taken=VALUES(action_taken), detected_at=VALUES(detected_at)"
    )

    # Use a raw connection/cursor for batched upsert
    conn = engine.raw_connection()
    try:
        cursor = conn.cursor()
        for chunk in read_iter:
            if chunk.empty:
                continue
            logger.info(f"Scoring chunk with {len(chunk)} rows")
            # Keep ids
            if 'transaction_id' not in chunk.columns:
                logger.warning("transaction_id column not present; skipping chunk")
                continue
            transaction_ids = chunk['transaction_id'].astype(str).tolist()

            try:
                X_trans = prepare_feature_matrix(
                    chunk,
                    feature_names=feature_names,
                    preprocessor=preprocessor,
                    dataset=dataset,
                )
            except Exception:
                logger.exception("Feature preparation failed for chunk; skipping")
                continue

            # Ensure shapes align for prediction
            try:
                probs = None
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_trans)[:, 1]
                else:
                    preds = model.predict(X_trans)
                    probs = preds
                probs = probs.ravel()
            except Exception:
                logger.exception("Model prediction failed for chunk; skipping")
                continue

            if len(probs) != len(transaction_ids):
                logger.error("Mismatch between predictions and ids; skipping chunk")
                continue

            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            rows = []
            for tx_id, p in zip(transaction_ids, probs):
                pred_label = 'Fraud' if float(p) > float(threshold) else 'Legitimate'
                action = 'Blocked' if float(p) > float(threshold) else 'Approved'
                rows.append((tx_id, float(p), pred_label, model_version, action, now))

            # Batch upsert
            for i in range(0, len(rows), batch_upsert):
                batch = rows[i:i+batch_upsert]
                try:
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    logger.exception("Failed to upsert batch; rolling back")
            total_processed += len(rows)

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        conn.close()
        try:
            conn_for_read.close()
        except Exception:
            pass

    logger.info(f"Finished scoring: total processed {total_processed} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=os.getenv('MODEL_PATH', 'my_xgboost_fraud_model.json'))
    parser.add_argument('--preprocessor-path', default=os.getenv('PREPROCESSOR_PATH', 'preprocessor.joblib'))
    parser.add_argument('--threshold', type=float, default=float(os.getenv('FRAUD_THRESHOLD', '0.80')))
    parser.add_argument('--chunksize', type=int, default=int(os.getenv('CHUNKSIZE', '5000')))
    parser.add_argument('--batch-upsert', type=int, default=int(os.getenv('BATCH_UPSERT', '1000')))
    parser.add_argument('--model-version', default=os.getenv('MODEL_VERSION', 'XGBoost_v1.0'))
    parser.add_argument('--feature-list-path', default=os.getenv('FEATURE_LIST_PATH'))
    parser.add_argument('--dataset', default=os.getenv('SCORING_DATASET', 'auto'), choices=['auto', 'ieee_cis', 'synthetic_financial'])
    args = parser.parse_args()

    # DB config from env
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASS')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT', '4000')
    DB_NAME = os.getenv('DB_NAME')
    SSL_CA = os.getenv('SSL_CA', certifi.where())

    for var in ['DB_USER', 'DB_PASS', 'DB_HOST', 'DB_NAME']:
        if not os.getenv(var):
            logger.error(f"Environment variable {var} is required")
            sys.exit(2)

    engine = connect_engine(DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME, ssl_ca=SSL_CA)

    model = load_model(args.model_path)
    preprocessor = None
    if args.preprocessor_path and os.path.exists(args.preprocessor_path):
        preprocessor = load_preprocessor(args.preprocessor_path)

    feature_names = load_feature_names(args.feature_list_path, args.model_path, dataset=args.dataset)
    if not feature_names:
        feature_names = get_feature_names_from_model(model)

    if args.dataset == "synthetic_financial" and not feature_names:
        logger.warning(
            "Synthetic Financial scoring works best with a feature list. Expected path: %s",
            DEFAULT_SYNTHETIC_FEATURE_PATH,
        )

    chunked_score(
        engine,
        model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        threshold=args.threshold,
        chunksize=args.chunksize,
        batch_upsert=args.batch_upsert,
        model_version=args.model_version,
        dataset=args.dataset,
    )


if __name__ == '__main__':
    main()

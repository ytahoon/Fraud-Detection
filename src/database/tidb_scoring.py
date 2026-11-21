"""Score transactions in the TiDB demo with the verified release for their dataset."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from lightgbm.basic import LightGBMError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from xgboost.core import XGBoostError

from src import ieee_release, synthetic_release
from src.model_release import DEMO_CURRENCY, predict_risk
from .connection import db_error_code, load_demo_settings, open_engine
from .demo_queries import DEMO_INPUT_SHA256_SQL


RETRYABLE_ERRORS = {1205, 1213, 8002, 9007}
MAX_ATTEMPTS = 3
RELEASES = {"synthetic_financial": synthetic_release, "ieee_cis": ieee_release}

LOCK_RELEASE = text("""
    SELECT manifest_sha256, model_sha256, threshold
    FROM model_releases WHERE dataset = :dataset AND model_version = :version
    FOR UPDATE
""")

PENDING = text(f"""
    SELECT t.transaction_id, t.amount, t.currency, t.transaction_type, {DEMO_INPUT_SHA256_SQL} AS input_sha256
    FROM transactions t
    LEFT JOIN score_rejections r ON r.dataset = t.dataset AND r.transaction_id = t.transaction_id
      AND r.model_version = :version
    WHERE t.dataset = :dataset AND t.source = 'demo'
      AND (:after_id IS NULL OR t.transaction_id > :after_id)
      AND (r.input_sha256 IS NULL OR r.input_sha256 <> {DEMO_INPUT_SHA256_SQL})
      AND NOT EXISTS (
          SELECT 1 FROM transaction_scores s WHERE s.dataset = t.dataset
            AND s.transaction_id = t.transaction_id AND s.model_version = :version
      )
    ORDER BY t.transaction_id LIMIT :batch_size
""")


@dataclass(frozen=True)
class ScoringResult:
    scored: int = 0
    quarantined: int = 0


def build_scores(rows, release):
    if not rows:
        return [], []
    frame = pd.DataFrame(rows).rename(columns={"transaction_type": "type"})
    features, issues = release.prepare_batch(frame)
    correct_currency = frame["currency"].eq(DEMO_CURRENCY).fillna(False).to_numpy(dtype=bool)
    issues = np.where(correct_currency, issues, "invalid_currency")
    features = features.loc[correct_currency[features.index.to_numpy()]]

    scores = []
    if not features.empty:
        predictions = predict_risk(release.model, features)
        scores = [
            {
                "dataset": release.dataset, "transaction_id": rows[i]["transaction_id"],
                "version": release.version, "risk_score": float(score), "threshold": release.threshold,
                "decision": "review" if float(score) >= release.threshold else "no_alert",
            }
            for i, score in zip(features.index, predictions)
        ]
    rejected = [
        {"dataset": release.dataset, "transaction_id": row["transaction_id"], "version": release.version,
         "input_sha256": row["input_sha256"], "reason": str(reason)}
        for row, reason in zip(rows, issues) if reason
    ]
    return scores, rejected


def require_locked_release(conn, release, params):
    stored = conn.execute(LOCK_RELEASE, params).one_or_none()
    if stored is None or tuple(stored) != (release.manifest_sha256, release.model_sha256, release.threshold):
        raise ValueError("Model version is already registered to a different release, or is missing")


def with_retry(operation):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation()
        except SQLAlchemyError as exc:
            if db_error_code(exc) not in RETRYABLE_ERRORS or attempt + 1 == MAX_ATTEMPTS:
                raise
            time.sleep(0.05 * 2**attempt)


def register_release(engine, release, params):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO model_releases (dataset, model_version, manifest_sha256, model_sha256, threshold)
            VALUES (:dataset, :version, :manifest_sha256, :model_sha256, :threshold)
            ON DUPLICATE KEY UPDATE model_version = VALUES(model_version)
        """), {**params, "manifest_sha256": release.manifest_sha256, "model_sha256": release.model_sha256,
               "threshold": release.threshold})
        require_locked_release(conn, release, params)


def commit_batch(engine, release, params, after_id, batch_size):
    with engine.begin() as conn:
        # Serialize writers for this release. With READ COMMITTED, selection after the lock
        # sees scores another writer committed while this batch was waiting.
        require_locked_release(conn, release, params)
        rows = conn.execute(PENDING, {**params, "after_id": after_id, "batch_size": batch_size}).mappings().all()
        if not rows:
            return None, ScoringResult()
        scores, rejected = build_scores(rows, release)
        if scores:
            conn.execute(text("""
                INSERT INTO transaction_scores (dataset, transaction_id, model_version, risk_score, threshold, decision)
                VALUES (:dataset, :transaction_id, :version, :risk_score, :threshold, :decision)
            """), scores)
            conn.execute(text("""
                DELETE FROM score_rejections WHERE dataset = :dataset
                  AND transaction_id = :transaction_id AND model_version = :version
            """), [{"dataset": r["dataset"], "transaction_id": r["transaction_id"], "version": r["version"]}
                   for r in scores])
        if rejected:
            conn.execute(text("""
                INSERT INTO score_rejections (dataset, transaction_id, model_version, input_sha256, reason)
                VALUES (:dataset, :transaction_id, :version, :input_sha256, :reason)
                ON DUPLICATE KEY UPDATE input_sha256 = VALUES(input_sha256), reason = VALUES(reason),
                  rejected_at = CURRENT_TIMESTAMP(6)
            """), rejected)
    return rows[-1]["transaction_id"], ScoringResult(len(scores), len(rejected))


def score_demo(engine, release, *, batch_size=500):
    if not 1 <= batch_size <= 5_000:
        raise ValueError("Batch size must be between 1 and 5000")
    engine = engine.execution_options(isolation_level="READ COMMITTED")
    params = {"dataset": release.dataset, "version": release.version}
    with_retry(lambda: register_release(engine, release, params))
    scored = quarantined = 0
    after_id = None
    while True:
        after_id, result = with_retry(lambda: commit_batch(engine, release, params, after_id, batch_size))
        if after_id is None:
            return ScoringResult(scored, quarantined)
        scored += result.scored
        quarantined += result.quarantined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--dataset", choices=tuple(RELEASES), default="synthetic_financial")
    parser.add_argument("--release", type=Path, help="Override the validated release manifest for this dataset")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    engine = None
    try:
        module = RELEASES[args.dataset]
        release = module.load_release(args.release) if args.release else module.load_release()
        settings = load_demo_settings(env_file=args.env_file)
        engine = open_engine(settings)
        result = score_demo(engine, release, batch_size=args.batch_size)
        print(f"{release.version} in {settings.database}: {result.scored} scores committed, "
              f"{result.quarantined} rows quarantined")
    except (OSError, ValueError, XGBoostError, LightGBMError) as exc:
        print(f"TiDB demo scoring: {exc}", file=sys.stderr)
        return 1
    except SQLAlchemyError as exc:
        print(f"TiDB demo scoring stopped (driver code {db_error_code(exc)}). "
              "Check migrations and grants; rerun to resume after committed batches.", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

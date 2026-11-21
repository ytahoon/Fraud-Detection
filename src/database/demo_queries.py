"""Read the current TiDB model results for both demo interfaces."""

from sqlalchemy import text


DATASETS = {"synthetic_financial": "Synthetic Financial", "ieee_cis": "IEEE-CIS"}
SCORING_STATUSES = {
    "review": "Review",
    "no_alert": "No alert",
    "not_scored": "Not scored",
    "quarantined": "Quarantined",
}
REJECTION_REASONS = {
    "invalid_amount": "Invalid amount",
    "invalid_type": "Missing or unknown transaction type",
    "invalid_currency": "Unexpected demo currency",
}
DEMO_INPUT_SHA256_SQL = "SHA2(JSON_ARRAY(CAST(t.amount AS CHAR), t.currency, t.transaction_type), 256)"

QUERY = text(f"""
    WITH current_release AS (
        SELECT dataset, model_version
        FROM model_releases
        WHERE dataset = :dataset
        ORDER BY registered_at DESC, model_version DESC
        LIMIT 1
    ),
    demo_results AS (
        SELECT t.dataset, t.transaction_id, t.amount, t.currency, t.transaction_type, t.occurred_at,
               labels.fraud_label, s.risk_score, s.threshold, mr.model_version, s.decision, s.scored_at,
               CASE WHEN s.decision IS NOT NULL THEN s.decision
                    WHEN rejected.reason IS NOT NULL THEN 'quarantined'
                    ELSE 'not_scored' END AS scoring_status,
               CASE WHEN s.decision IS NULL THEN rejected.reason END AS rejection_reason
        FROM transactions t
        LEFT JOIN current_release mr ON mr.dataset = t.dataset
        LEFT JOIN transaction_scores s
          ON s.dataset = t.dataset AND s.transaction_id = t.transaction_id
         AND s.model_version = mr.model_version
        LEFT JOIN score_rejections rejected
          ON rejected.dataset = t.dataset AND rejected.transaction_id = t.transaction_id
         AND rejected.model_version = mr.model_version AND rejected.input_sha256 = {DEMO_INPUT_SHA256_SQL}
        LEFT JOIN transaction_labels labels
          ON labels.dataset = t.dataset AND labels.transaction_id = t.transaction_id
        WHERE t.dataset = :dataset AND t.source = 'demo'
          AND (:transaction_id IS NULL OR t.transaction_id = :transaction_id)
    )
    SELECT * FROM demo_results
    WHERE :status IS NULL OR scoring_status = :status
    ORDER BY CASE scoring_status WHEN 'review' THEN 0 WHEN 'quarantined' THEN 1
                                 WHEN 'no_alert' THEN 2 ELSE 3 END,
             risk_score DESC, occurred_at DESC, transaction_id
    LIMIT :limit
""")


def fetch_demo_transactions(engine, dataset, *, transaction_id=None, status=None, limit=100):
    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError("Dataset must be synthetic_financial or ieee_cis")
    if transaction_id is not None and (not isinstance(transaction_id, str) or not 1 <= len(transaction_id) <= 64):
        raise ValueError("Transaction ID must be a string of 1 to 64 characters")
    if status is not None and (not isinstance(status, str) or status not in SCORING_STATUSES):
        raise ValueError("Unknown scoring status")
    if not 1 <= limit <= 500:
        raise ValueError("Limit must be between 1 and 500")
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(
            QUERY, {"dataset": dataset, "transaction_id": transaction_id, "status": status, "limit": limit}
        ).mappings()]

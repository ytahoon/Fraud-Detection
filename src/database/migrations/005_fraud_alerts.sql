CREATE OR REPLACE SQL SECURITY INVOKER VIEW fraud_alerts AS
SELECT dataset, transaction_id, model_version, risk_score, threshold, scored_at
FROM transaction_scores
WHERE decision = 'review'

CREATE TABLE IF NOT EXISTS transaction_scores (
    dataset VARCHAR(32) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    risk_score DOUBLE NOT NULL,
    threshold DOUBLE NOT NULL,
    decision ENUM('review', 'no_alert') NOT NULL,
    scored_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (dataset, transaction_id, model_version),
    KEY idx_scores_queue (dataset, decision, scored_at, transaction_id),
    CONSTRAINT fk_score_transaction FOREIGN KEY (dataset, transaction_id)
        REFERENCES transactions(dataset, transaction_id)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin

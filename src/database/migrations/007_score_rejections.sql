CREATE TABLE IF NOT EXISTS score_rejections (
    dataset VARCHAR(32) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    input_sha256 CHAR(64) NOT NULL,
    reason ENUM('invalid_amount', 'invalid_type', 'invalid_currency') NOT NULL,
    rejected_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (dataset, transaction_id, model_version),
    KEY idx_rejections_release (dataset, model_version),
    CONSTRAINT fk_rejection_transaction FOREIGN KEY (dataset, transaction_id)
        REFERENCES transactions(dataset, transaction_id),
    CONSTRAINT fk_rejection_release FOREIGN KEY (dataset, model_version)
        REFERENCES model_releases(dataset, model_version)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin

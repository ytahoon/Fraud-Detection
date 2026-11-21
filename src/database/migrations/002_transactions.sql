CREATE TABLE IF NOT EXISTS transactions (
    dataset VARCHAR(32) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    occurred_at DATETIME(6) NOT NULL,
    amount DECIMAL(20, 4) UNSIGNED NOT NULL,
    currency CHAR(3) NOT NULL,
    transaction_type VARCHAR(32) NULL,
    source ENUM('demo', 'benchmark', 'live') NOT NULL,
    PRIMARY KEY (dataset, transaction_id),
    KEY idx_transactions_source (dataset, source, transaction_id),
    CONSTRAINT fk_transaction_dataset FOREIGN KEY (dataset) REFERENCES datasets(dataset)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin

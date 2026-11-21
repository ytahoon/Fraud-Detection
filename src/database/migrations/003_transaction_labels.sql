CREATE TABLE IF NOT EXISTS transaction_labels (
    dataset VARCHAR(32) NOT NULL,
    transaction_id VARCHAR(64) NOT NULL,
    fraud_label ENUM('fraud', 'not_fraud') NOT NULL,
    label_source VARCHAR(32) NOT NULL,
    PRIMARY KEY (dataset, transaction_id),
    CONSTRAINT fk_label_transaction FOREIGN KEY (dataset, transaction_id)
        REFERENCES transactions(dataset, transaction_id)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin

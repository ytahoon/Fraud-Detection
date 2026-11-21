CREATE TABLE IF NOT EXISTS model_releases (
    dataset VARCHAR(32) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    manifest_sha256 CHAR(64) NOT NULL,
    model_sha256 CHAR(64) NOT NULL,
    threshold DOUBLE NOT NULL,
    registered_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (dataset, model_version),
    CONSTRAINT fk_release_dataset FOREIGN KEY (dataset) REFERENCES datasets(dataset)
) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin

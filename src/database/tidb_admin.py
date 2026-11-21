"""Create, inspect, or seed the isolated TiDB demo schema."""

import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from .connection import db_error_code, load_demo_settings, open_engine
from .demo_queries import DATASETS


MIGRATIONS = Path(__file__).with_name("migrations")
FIXTURE = Path(__file__).with_name("demo_transactions.json")
TABLES = {
    "schema_migrations", "datasets", "transactions", "transaction_labels", "transaction_scores",
    "model_releases", "score_rejections",
}


def create_demo_database(settings):
    engine = open_engine(settings, without_database=True)
    try:
        with engine.begin() as conn:
            version = conn.exec_driver_sql("SELECT VERSION()").scalar_one()
            if "TiDB" not in version:
                raise ValueError("The server did not identify itself as TiDB")
            name = conn.dialect.identifier_preparer.quote_identifier(settings.database)
            conn.exec_driver_sql(f"CREATE DATABASE IF NOT EXISTS {name} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin")
    finally:
        engine.dispose()


def migrate(engine):
    files = sorted(MIGRATIONS.glob("*.sql"))
    with engine.connect() as conn:
        inspector = inspect(conn)
        tables, views = set(inspector.get_table_names()), set(inspector.get_view_names())
        if ((tables or views) and "schema_migrations" not in tables
                or tables - TABLES or views - {"fraud_alerts"}):
            raise ValueError("Refusing to modify a non-empty or unmanaged schema; use a fresh demo database")
        if "schema_migrations" not in tables:
            conn.exec_driver_sql(files[0].read_text(encoding="utf-8"))
            conn.commit()
        applied = dict(conn.execute(text("SELECT migration, sha256 FROM schema_migrations")).all())
        conn.commit()
        contents = {path.name: path.read_bytes() for path in files}
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
        for name, checksum in applied.items():
            if hashes.get(name) != checksum:
                raise ValueError(f"Applied migration is missing or changed: {name}")
        pending = []
        for name, data in contents.items():
            if name in applied:
                continue
            with conn.begin():
                conn.exec_driver_sql(data.decode("utf-8"))
                conn.execute(
                    text("INSERT INTO schema_migrations (migration, sha256) VALUES (:name, :sha256)"),
                    {"name": name, "sha256": hashes[name]},
                )
            pending.append(name)
        if set(inspect(conn).get_table_names()) != TABLES or "fraud_alerts" not in inspect(conn).get_view_names():
            raise ValueError("The migration registry exists but a required table or view is missing")
    return pending


def load_demo_rows():
    records = json.loads(FIXTURE.read_text(encoding="utf-8"))
    transactions, labels = [], []
    seen = set()
    for record in records:
        row = {**record}
        label = row.pop("fraud_label")
        key = (row["dataset"], row["transaction_id"])
        if key in seen or row["dataset"] not in DATASETS or row["currency"] != "XTS":
            raise ValueError("Invalid demo transaction identity or currency")
        seen.add(key)
        row["amount"] = Decimal(row["amount"])
        row["occurred_at"] = datetime.fromisoformat(row["occurred_at"])
        transactions.append(row)
        if label is not None:
            if label not in {"fraud", "not_fraud"}:
                raise ValueError("Invalid demo fraud label")
            labels.append({"dataset": key[0], "transaction_id": key[1], "fraud_label": label})
    return transactions, labels


def seed_demo(engine):
    transactions, labels = load_demo_rows()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO datasets (dataset, display_name) VALUES (:dataset, :name)
            ON DUPLICATE KEY UPDATE dataset = VALUES(dataset)
        """), [{"dataset": dataset, "name": name} for dataset, name in DATASETS.items()])
        conn.execute(text("""
            INSERT INTO transactions (dataset, transaction_id, occurred_at, amount, currency, transaction_type, source)
            VALUES (:dataset, :transaction_id, :occurred_at, :amount, :currency, :transaction_type, 'demo')
            ON DUPLICATE KEY UPDATE transaction_type = CASE
                WHEN transactions.dataset = 'ieee_cis' AND transactions.source = 'demo'
                 AND transactions.transaction_type IS NULL AND transactions.amount = VALUES(amount)
                 AND transactions.occurred_at = VALUES(occurred_at) AND transactions.currency = VALUES(currency)
                THEN VALUES(transaction_type) ELSE transactions.transaction_type END
        """), transactions)
        conn.execute(text("""
            INSERT INTO transaction_labels (dataset, transaction_id, fraud_label, label_source)
            SELECT :dataset, :transaction_id, :fraud_label, 'demo_fixture'
            FROM transactions WHERE dataset = :dataset AND transaction_id = :transaction_id AND source = 'demo'
            ON DUPLICATE KEY UPDATE transaction_id = VALUES(transaction_id)
        """), labels)
    return len(transactions)


def check(engine):
    with engine.connect() as conn:
        version, database = conn.exec_driver_sql("SELECT VERSION(), DATABASE()").one()
        if "TiDB" not in version:
            raise ValueError("The server did not identify itself as TiDB")
        print(f"Connected to {version}; database: {database}")
        inspector = inspect(conn)
        if not TABLES.issubset(inspector.get_table_names()) or "fraud_alerts" not in inspector.get_view_names():
            raise ValueError("Required demo schema objects are missing; run migrate")
        for dataset, transactions, scores, alerts, rejections in conn.execute(text("""
            SELECT d.dataset,
                (SELECT COUNT(*) FROM transactions t
                 WHERE t.dataset = d.dataset AND t.source = 'demo') AS transactions,
                (SELECT COUNT(*) FROM transaction_scores s JOIN transactions t USING (dataset, transaction_id)
                 WHERE s.dataset = d.dataset AND t.source = 'demo') AS scores,
                (SELECT COUNT(*) FROM fraud_alerts a JOIN transactions t USING (dataset, transaction_id)
                 WHERE a.dataset = d.dataset AND t.source = 'demo') AS alerts,
                (SELECT COUNT(*) FROM score_rejections r JOIN transactions t USING (dataset, transaction_id)
                 WHERE r.dataset = d.dataset AND t.source = 'demo') AS rejections
            FROM datasets d ORDER BY d.dataset
        """)):
            print(f"{dataset}: {transactions} demo transactions, {scores} versioned scores, "
                  f"{alerts} alerts, {rejections} quarantined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("command", choices=("check", "migrate", "seed"))
    parser.add_argument("--create-database", action="store_true")
    args = parser.parse_args()
    if args.create_database and args.command != "migrate":
        parser.error("--create-database is only available with migrate")
    engine = None
    try:
        settings = load_demo_settings(env_file=args.env_file)
        if args.create_database:
            create_demo_database(settings)
        engine = open_engine(settings)
        if args.command == "check":
            check(engine)
        elif args.command == "migrate":
            applied = migrate(engine)
            print(f"Applied {len(applied)} migrations to {settings.database}")
        else:
            print(f"Ensured {seed_demo(engine)} demo transactions in {settings.database}")
    except ValueError as exc:
        print(f"TiDB demo: {exc}", file=sys.stderr)
        return 1
    except SQLAlchemyError as exc:
        print(f"TiDB demo operation failed (driver code {db_error_code(exc)}). Check connection and grants.",
              file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

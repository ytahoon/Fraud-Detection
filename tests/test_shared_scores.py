import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool
from streamlit.testing.v1 import AppTest

from src.database.demo_queries import DEMO_INPUT_SHA256_SQL, fetch_demo_transactions


class SharedScoreReadTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        self.addCleanup(self.engine.dispose)

        @event.listens_for(self.engine, "connect")
        def install_hash(db, _record):
            db.create_function("SHA2", 2, lambda value, _bits: hashlib.sha256(value.encode()).hexdigest())

        with self.engine.begin() as conn:
            for schema in (
                "model_releases (dataset TEXT, model_version TEXT, registered_at TEXT)",
                "transactions (dataset TEXT, transaction_id TEXT, amount TEXT, currency TEXT, "
                "transaction_type TEXT, occurred_at TEXT, source TEXT)",
                "transaction_scores (dataset TEXT, transaction_id TEXT, model_version TEXT, "
                "risk_score REAL, threshold REAL, decision TEXT, scored_at TEXT)",
                "score_rejections (dataset TEXT, transaction_id TEXT, model_version TEXT, input_sha256 TEXT, reason TEXT)",
                "transaction_labels (dataset TEXT, transaction_id TEXT, fraud_label TEXT)",
            ):
                conn.exec_driver_sql(f"CREATE TABLE {schema}")
            conn.exec_driver_sql("""
                INSERT INTO model_releases VALUES ('synthetic_financial', 'v1', '2026-01-01'),
                    ('synthetic_financial', 'v2', '2026-02-01')
            """)
            conn.exec_driver_sql("""
                INSERT INTO transactions VALUES
                    ('synthetic_financial', 'DEMO-001', '2500.0000', 'XTS', 'TRANSFER', '2026-02-01', 'demo'),
                    ('synthetic_financial', 'DEMO-002', '100.0000', 'XTS', NULL, '2026-02-02', 'demo'),
                    ('synthetic_financial', 'BENCH-001', '200.0000', 'XTS', 'TRANSFER', '2026-02-03', 'benchmark'),
                    ('ieee_cis', 'DEMO-001', '80.0000', 'XTS', NULL, '2026-02-04', 'demo')
            """)
            conn.exec_driver_sql("""
                INSERT INTO transaction_scores VALUES
                    ('synthetic_financial', 'DEMO-001', 'v2', 0.6, 0.9, 'no_alert', '2026-02-01'),
                    ('synthetic_financial', 'DEMO-001', 'v1', 0.95, 0.9, 'review', '2026-03-01'),
                    ('synthetic_financial', 'DEMO-002', 'v1', 0.8, 0.9, 'no_alert', '2026-02-01'),
                    ('synthetic_financial', 'BENCH-001', 'v2', 1.0, 0.9, 'review', '2026-02-01')
            """)
            conn.exec_driver_sql(f"""
                INSERT INTO score_rejections
                SELECT t.dataset, t.transaction_id, 'v2', {DEMO_INPUT_SHA256_SQL}, 'invalid_type'
                FROM transactions t WHERE t.transaction_id = 'DEMO-002'
            """)
            conn.exec_driver_sql("INSERT INTO transaction_labels VALUES ('synthetic_financial', 'DEMO-001', 'fraud')")

    def test_same_transaction_id_is_scoped_to_dataset_and_latest_registered_model(self):
        rows = fetch_demo_transactions(self.engine, "synthetic_financial", transaction_id="DEMO-001")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["model_version"], row["risk_score"], row["scoring_status"]), ("v2", 0.6, "no_alert"))
        self.assertEqual(row["fraud_label"], "fraud")
        ieee = fetch_demo_transactions(self.engine, "ieee_cis", transaction_id="DEMO-001")[0]
        self.assertEqual(ieee["scoring_status"], "not_scored")
        self.assertIsNone(ieee["risk_score"])
        self.assertEqual(fetch_demo_transactions(self.engine, "synthetic_financial", status="review"), [])

    def test_current_quarantine_is_visible_until_inputs_change_and_is_filtered_before_limit(self):
        row, = fetch_demo_transactions(self.engine, "synthetic_financial", status="quarantined", limit=1)
        self.assertEqual(row["rejection_reason"], "invalid_type")
        self.assertIsNone(row["risk_score"])
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE transactions SET transaction_type = 'TRANSFER' WHERE transaction_id = 'DEMO-002'"))
        self.assertEqual(fetch_demo_transactions(self.engine, "synthetic_financial", status="quarantined"), [])
        corrected, = fetch_demo_transactions(self.engine, "synthetic_financial", status="not_scored", limit=1)
        self.assertEqual(corrected["transaction_id"], "DEMO-002")
        self.assertIsNone(corrected["risk_score"])

    def test_dashboard_opens_on_tidb_and_keeps_label_separate_from_model_status(self):
        dashboard = Path(__file__).resolve().parents[1] / "src/dashboard/dashboard_app.py"
        with patch("src.database.connection.load_demo_settings", return_value="sqlite-test"), patch(
            "src.database.connection.open_engine", return_value=self.engine
        ):
            app = AppTest.from_file(str(dashboard)).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.sidebar.radio[0].value, "TiDB Demo")
            self.assertEqual(set(app.dataframe[0].value["scoring_status"]), {"No alert", "Quarantined"})
            app.text_input[0].set_value("DEMO-001").run()
            row, = app.dataframe[0].value.to_dict("records")
            self.assertEqual((row["risk_score"], row["scoring_status"], row["fraud_label"]), (0.6, "No alert", "Fraud"))
            app.sidebar.selectbox[0].set_value("ieee_cis").run()
            self.assertEqual(app.dataframe[0].value.iloc[0]["scoring_status"], "Not scored")

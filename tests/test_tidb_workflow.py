"""Opt-in end-to-end tests using a disposable TiDB demo database and no raw CSVs."""

import os
from pathlib import Path
import unittest
from uuid import uuid4

import pandas as pd
from sqlalchemy import text
from streamlit.testing.v1 import AppTest

from src import ieee_release, synthetic_release
from src.chatbot import chatbot_api
from src.database.connection import DEMO_DATABASE, load_demo_settings, open_engine
from src.database.demo_queries import fetch_demo_transactions
from src.database.tidb_admin import create_demo_database, migrate, seed_demo
from src.database.tidb_scoring import ScoringResult, score_demo


@unittest.skipUnless(os.getenv("RUN_TIDB_INTEGRATION") == "1", "requires a disposable TiDB demo database")
class TiDBWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = load_demo_settings()
        if settings.database == DEMO_DATABASE:
            raise ValueError("Set TIDB_DEMO_DATABASE to a disposable _demo database for integration tests")
        create_demo_database(settings)
        cls.engine = open_engine(settings)
        cls.addClassCleanup(cls.engine.dispose)
        chatbot_api.tidb_demo_engine.cache_clear()
        cls.addClassCleanup(cls.cleanup_chatbot_engine)
        migrate(cls.engine)
        cls.initial = seed_demo(cls.engine)
        cls.releases = (synthetic_release.load_release(), ieee_release.load_release(ieee_release.FIXTURE_RELEASE))
        for release in cls.releases:
            score_demo(cls.engine, release, batch_size=2)

    @staticmethod
    def cleanup_chatbot_engine():
        if chatbot_api.tidb_demo_engine.cache_info().currsize:
            chatbot_api.tidb_demo_engine().dispose()
            chatbot_api.tidb_demo_engine.cache_clear()

    def test_migrations_seed_scoring_and_both_apps_agree(self):
        self.assertEqual(migrate(self.engine), [])
        self.assertEqual(seed_demo(self.engine), self.initial)
        self.assertEqual(self.initial, 12)
        expected = {"synthetic_financial": 8, "ieee_cis": 4}
        ui = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "src/dashboard/dashboard_app.py"), default_timeout=25,
        ).run()
        api = chatbot_api.app.test_client()
        returned = {}
        for release in self.releases:
            self.assertEqual(score_demo(self.engine, release, batch_size=2), ScoringResult())
            rows = fetch_demo_transactions(self.engine, release.dataset)
            self.assertEqual(len(rows), expected[release.dataset])
            self.assertTrue(all(row["risk_score"] is not None for row in rows))
            stored = next(row for row in rows if row["transaction_id"] == "DEMO-001")
            self.assertEqual(stored["model_version"], release.version)
            self.assertEqual(stored["threshold"], release.threshold)
            result = api.get(f"/transactions/{release.dataset}/DEMO-001")
            self.assertEqual(result.status_code, 200)
            body = result.get_json()
            self.assertEqual(body["status"], "scored")
            payload = {"dataset": release.dataset, "transaction_id": "DEMO-001"}
            self.assertEqual(api.post("/chat", json=payload).get_json(), body)

            ui.sidebar.selectbox[0].set_value(release.dataset).run()
            ui.text_input[0].set_value("DEMO-001").run()
            self.assertFalse(ui.exception, ui.exception)
            self.assertFalse(ui.error, ui.error)
            displayed, = ui.dataframe[0].value.to_dict("records")
            for key in ("risk_score", "threshold", "model_version"):
                self.assertEqual(displayed[key], stored[key])
                self.assertEqual(body["transaction"][key], stored[key])
            returned[release.dataset] = stored["model_version"]
        self.assertNotEqual(returned["ieee_cis"], returned["synthetic_financial"])
        self.assertEqual(api.get("/transactions/ieee_cis/DOES-NOT-EXIST").status_code, 404)
        with self.engine.connect() as conn:
            self.assertEqual(conn.execute(text("SELECT COUNT(*) FROM transactions WHERE source = 'demo'")).scalar_one(), 12)
            self.assertEqual(conn.execute(text("SELECT COUNT(*) FROM transaction_scores")).scalar_one(), 12)

    def test_unknown_ieee_product_is_quarantined_then_scored_after_correction(self):
        release = self.releases[1]
        params = {"dataset": release.dataset, "id": f"QA-{uuid4().hex}"}
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO transactions (dataset, transaction_id, occurred_at, amount, currency, transaction_type, source)
                VALUES (:dataset, :id, UTC_TIMESTAMP(6), 149.50, 'XTS', 'UNKNOWN', 'demo')
            """), params)
        try:
            self.assertEqual(score_demo(self.engine, release), ScoringResult(quarantined=1))
            self.assertEqual(score_demo(self.engine, release), ScoringResult())
            row, = fetch_demo_transactions(self.engine, release.dataset, transaction_id=params["id"])
            self.assertEqual(row["scoring_status"], "quarantined")
            self.assertEqual(row["rejection_reason"], "invalid_type")
            self.assertIsNone(row["risk_score"])
            with self.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE transactions SET transaction_type = 'C' WHERE dataset = :dataset AND transaction_id = :id
                """), params)
            self.assertEqual(score_demo(self.engine, release), ScoringResult(scored=1))
            row, = fetch_demo_transactions(self.engine, release.dataset, transaction_id=params["id"])
            self.assertTrue(pd.notna(row["risk_score"]))
            self.assertIsNone(row["rejection_reason"])
            self.assertEqual(score_demo(self.engine, release), ScoringResult())
        finally:
            with self.engine.begin() as conn:
                for table in ("score_rejections", "transaction_scores", "transactions"):
                    conn.execute(text(f"DELETE FROM {table} WHERE dataset = :dataset AND transaction_id = :id"), params)

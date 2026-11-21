from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from pymysql.err import OperationalError as DriverError
from sqlalchemy.exc import OperationalError

from src.database.connection import TiDBSettings, load_demo_settings, open_engine
from src.database.tidb_admin import load_demo_rows, migrate
from src.database.tidb_scoring import build_scores, require_locked_release, with_retry
from src.model_evaluation import threshold_options
from src.model_release import DemoRelease
from src.synthetic_financial import prepare_pre_auth_batch
from src.synthetic_release import DEFAULT_RELEASE, load_release


class TiDBSettingsTest(unittest.TestCase):
    def test_cloud_connection_verifies_tls_and_keeps_password_out_of_repr(self):
        settings = TiDBSettings.from_mapping({
            "DB_HOST": "example.test", "DB_USER": "test", "DB_PASS": "example@/secret",
            "DB_NAME": "fraud_detection_demo",
        })
        self.assertNotIn(settings.password, repr(settings))
        with patch("src.database.connection.create_engine") as create:
            open_engine(settings)
        url, = create.call_args.args
        self.assertEqual(url.password, "example@/secret")
        self.assertNotIn(settings.password, str(url))
        self.assertTrue(create.call_args.kwargs["connect_args"]["ssl_verify_cert"])
        self.assertTrue(create.call_args.kwargs["connect_args"]["ssl_verify_identity"])

    def test_insecure_tls_is_loopback_only(self):
        env = {"DB_HOST": "example.test", "DB_USER": "test", "DB_TLS": "disable", "DB_NAME": "project_demo"}
        with self.assertRaisesRegex(ValueError, "loopback"):
            TiDBSettings.from_mapping(env)
        self.assertEqual(TiDBSettings.from_mapping({**env, "DB_HOST": "127.0.0.1"}).tls, "disable")

    @patch.dict("os.environ", {"DB_HOST": "environment.test", "TIDB_DEMO_DATABASE": ""}, clear=True)
    @patch("src.database.connection.dotenv_values", return_value={
        "DB_HOST": "file.test", "DB_PASS": "password", "DB_USER": "name", "DB_NAME": "legacy",
    })
    def test_settings_reuse_credentials_without_targeting_the_legacy_database(self, _dotenv):
        settings = load_demo_settings()
        self.assertEqual(settings.host, "environment.test")
        self.assertEqual(settings.database, "fraud_detection_demo")
        with patch.dict("os.environ", {"TIDB_DEMO_DATABASE": "legacy"}):
            with self.assertRaisesRegex(ValueError, "end in _demo"):
                load_demo_settings()


class TiDBSchemaTest(unittest.TestCase):
    def test_migration_will_not_write_into_an_unmanaged_database(self):
        engine = MagicMock()
        conn = engine.connect.return_value.__enter__.return_value
        with patch("src.database.tidb_admin.inspect") as inspect:
            inspect.return_value.get_table_names.return_value = ["customers", "transactions", "fraud_alerts"]
            inspect.return_value.get_view_names.return_value = []
            with self.assertRaisesRegex(ValueError, "non-empty or unmanaged"):
                migrate(engine)
        conn.exec_driver_sql.assert_not_called()

    def test_fixture_uses_dataset_scoped_identifiers_and_separate_labels(self):
        rows, labels = load_demo_rows()
        identities = {(t["dataset"], t["transaction_id"]) for t in rows}
        self.assertEqual(len(identities), len(rows))
        self.assertIn(("ieee_cis", "DEMO-001"), identities)
        self.assertIn(("synthetic_financial", "DEMO-001"), identities)
        self.assertFalse(any("fraud_label" in row for row in rows))
        self.assertEqual(len(labels), 10)


class TiDBScoringTest(unittest.TestCase):
    def test_scores_use_contract_threshold_inclusively_and_never_block(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.25, 0.75], [0.30, 0.70]])
        rows = [
            {"transaction_id": "DEMO-001", "input_sha256": "1" * 64, "amount": Decimal("2500.00"), "currency": "XTS",
             "transaction_type": "TRANSFER"},
            {"transaction_id": "DEMO-002", "input_sha256": "2" * 64, "amount": Decimal("125.00"), "currency": "XTS",
             "transaction_type": "PAYMENT"},
        ]
        release = DemoRelease(model, "test", 0.75, "3" * 64, "4" * 64, "synthetic_financial", prepare_pre_auth_batch)
        results, rejected = build_scores(rows, release)
        self.assertFalse(rejected)
        self.assertEqual([r["decision"] for r in results], ["review", "no_alert"])
        self.assertEqual(results[0]["dataset"], "synthetic_financial")

    def test_invalid_rows_are_quarantined_without_stopping_valid_rows(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.10, 0.90]])
        release = DemoRelease(model, "test", 0.75, "a" * 64, "b" * 64, "synthetic_financial", prepare_pre_auth_batch)
        row = {"transaction_id": "good", "amount": "0", "transaction_type": "cash-out",
               "currency": "XTS", "input_sha256": "5" * 64}
        results, rejected = build_scores([
            {**row, "transaction_id": "wrong_currency", "currency": "USD"},
            {**row, "transaction_id": "wrong_type", "transaction_type": None},
            row,
            {**row, "transaction_id": "wrong_amount", "amount": "NaN"},
        ], release)
        self.assertEqual([r["transaction_id"] for r in results], ["good"])
        self.assertEqual([r["reason"] for r in rejected], ["invalid_currency", "invalid_type", "invalid_amount"])
        self.assertEqual(model.predict_proba.call_args.args[0]["amount"].tolist(), [0])
        self.assertTrue(all("amount" not in r and "transaction_type" not in r for r in rejected))

    def test_release_registry_rejects_a_changed_manifest_under_the_same_version(self):
        conn = MagicMock()
        release = DemoRelease(MagicMock(), "test", 0.75, "a" * 64, "b" * 64, "synthetic_financial", prepare_pre_auth_batch)
        conn.execute.return_value.one_or_none.return_value = ("c" * 64, release.model_sha256, 0.75)
        with self.assertRaisesRegex(ValueError, "different release"):
            require_locked_release(conn, release, {})

    @patch("src.database.tidb_scoring.time.sleep")
    def test_retries_conflict_but_not_ambiguous_disconnect(self, sleep):
        conflict = OperationalError(None, None, DriverError(1213, "deadlock"))
        denied = OperationalError(None, None, DriverError(2013, "connection gone"))
        succeeds = MagicMock(side_effect=[conflict, "committed"])
        self.assertEqual(with_retry(succeeds), "committed")
        sleep.assert_called_once()
        fails = MagicMock(side_effect=denied)
        with self.assertRaises(OperationalError):
            with_retry(fails)
        fails.assert_called_once()


class SyntheticReleaseTest(unittest.TestCase):
    def test_model_and_validation_hashes_must_match_the_manifest(self):
        self.assertGreater(load_release().threshold, 0)
        with TemporaryDirectory() as folder:
            root = Path(folder) / "release"
            shutil.copytree(DEFAULT_RELEASE.parent, root)
            manifest = root / "manifest.json"
            (root / "model.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Model checksum"):
                load_release(manifest)
            (root / "validation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Validation report checksum"):
                load_release(manifest)

    def test_release_loader_rejects_production_scope(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            manifest = json.loads(DEFAULT_RELEASE.read_bytes())
            manifest["production_eligible"] = True
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "not a supported"):
                load_release(path)

    def test_threshold_selection_includes_all_tied_scores(self):
        y = np.array([1, 0, 1, 0])
        scores = np.array([.9, .9, .4, .1])
        self.assertEqual(threshold_options(y, scores, [500]), {500: .9})
        with self.assertRaisesRegex(ValueError, "No non-empty"):
            threshold_options(y, scores, [250])


class TiDBChatbotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.chatbot import chatbot_api
        cls.api = chatbot_api
        cls.client = chatbot_api.app.test_client()

    def test_browser_interface_serves_without_loading_tidb(self):
        with patch.object(self.api, "tidb_demo_engine") as engine:
            response = self.client.get("/chat")
        engine.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content_type.startswith("text/html"))
        self.assertIn(b"Transaction assistant", response.data)
        self.assertIn(b'value="ieee_cis"', response.data)
        self.assertIn(b'value="synthetic_financial"', response.data)
        self.assertEqual(self.client.get("/").get_json()["interface"], "/chat")
        for name in ("chat.css", "chat.js", "chat-icons.svg"):
            with self.subTest(asset=name), self.client.get(f"/static/{name}") as asset:
                self.assertEqual(asset.status_code, 200)

    def test_selected_chat_dataset_is_a_default_not_an_override(self):
        with patch.object(self.api, "transaction_lookup", return_value={"status": "checked"}) as lookup:
            payload = {"message": "demo-001", "default_dataset": "synthetic_financial"}
            self.assertEqual(self.client.post("/chat", json=payload).status_code, 200)
            lookup.assert_called_once_with("synthetic_financial", "DEMO-001")
            lookup.reset_mock()
            payload["message"] = "Check DEMO-001 in IEEE-CIS"
            self.assertEqual(self.client.post("/chat", json=payload).status_code, 200)
            lookup.assert_called_once_with("ieee_cis", "DEMO-001")
            lookup.reset_mock()
            payload["message"] = "Check DEMO-001 in IEEE-CIS and PaySim"
            self.assertEqual(self.client.post("/chat", json=payload).status_code, 400)
            payload.update(message="DEMO-001", default_dataset="unknown")
            self.assertEqual(self.client.post("/chat", json=payload).status_code, 400)
            lookup.assert_not_called()

    def test_demo_lookup_returns_unscored_without_a_fallback(self):
        row = {
            "dataset": "ieee_cis", "transaction_id": "DEMO-001", "amount": Decimal("85.99"),
            "risk_score": None, "threshold": None, "occurred_at": datetime(2026, 1, 1),
            "scored_at": None, "decision": None, "model_version": None, "scoring_status": "not_scored",
        }
        with patch.object(self.api, "tidb_demo_engine"), patch.object(
            self.api, "fetch_demo_transactions", return_value=[row]
        ) as lookup:
            response = self.client.post("/chat", json={"dataset": "ieee_cis", "transaction_id": "DEMO-001"})
        self.assertEqual(response.get_json()["status"], "not_scored")
        self.assertIsNone(response.get_json()["transaction"]["risk_score"])
        self.assertEqual(lookup.call_args.args[1], "ieee_cis")

    def test_invalid_chat_input_does_not_trigger_a_lookup(self):
        self.assertEqual(self.client.post("/chat", json={"message": 123}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"dataset": [], "transaction_id": "DEMO-001"}).status_code, 400)

    def test_text_lookup_returns_exact_stored_score_without_calling_gemini(self):
        row = {
            "dataset": "synthetic_financial", "transaction_id": "DEMO-001", "amount": Decimal("2500"),
            "risk_score": .9411700367927551, "threshold": .919187605381012, "occurred_at": datetime(2026, 1, 1),
            "scored_at": datetime(2026, 1, 2), "decision": "review", "scoring_status": "review",
            "model_version": "Synthetic_PreAuth_XGBoost_v1.1", "fraud_label": "fraud",
        }
        with patch.object(self.api, "tidb_demo_engine"), patch.object(
            self.api, "fetch_demo_transactions", return_value=[row]
        ) as lookup, patch.object(self.api, "general_reply") as gemini:
            response = self.client.post("/chat", json={"message": "Check demo-001 in PaySim", "mode": "general"})
            direct = self.client.get("/transactions/synthetic_financial/DEMO-001")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), direct.get_json())
        self.assertEqual(response.get_json()["transaction"]["risk_score"], row["risk_score"])
        self.assertEqual(response.get_json()["transaction"]["threshold"], row["threshold"])
        self.assertEqual(response.get_json()["transaction"]["scored_at"], "2026-01-02T00:00:00Z")
        self.assertEqual(lookup.call_args.kwargs["transaction_id"], "DEMO-001")
        gemini.assert_not_called()

    def test_missing_or_ambiguous_lookup_does_not_guess(self):
        with patch.object(self.api, "tidb_demo_engine"), patch.object(
            self.api, "fetch_demo_transactions", return_value=[]
        ) as lookup:
            not_found = self.client.post("/chat", json={"message": "Check DEMO-989 in IEEE-CIS"})
            ambiguous = self.client.post("/chat", json={"message": "Check DEMO-001 in IEEE-CIS and PaySim"})
        self.assertEqual(not_found.status_code, 404)
        self.assertEqual(not_found.get_json()["status"], "not_found")
        self.assertEqual(ambiguous.status_code, 400)
        lookup.assert_called_once()

    def test_quarantined_and_unavailable_are_not_scores(self):
        row = {
            "dataset": "synthetic_financial", "transaction_id": "DEMO-001", "amount": Decimal("2500"),
            "risk_score": None, "threshold": None, "occurred_at": datetime(2026, 1, 1),
            "scored_at": None, "decision": None, "scoring_status": "quarantined", "rejection_reason": "invalid_type",
        }
        with patch.object(self.api, "tidb_demo_engine"), patch.object(
            self.api, "fetch_demo_transactions", side_effect=[[row], OperationalError(None, None, "private error")]
        ):
            quarantined = self.client.get("/transactions/synthetic_financial/DEMO-001").get_json()
            unavailable = self.client.get("/transactions/synthetic_financial/DEMO-001")
        self.assertEqual(quarantined["status"], "quarantined")
        self.assertIsNone(quarantined["transaction"]["risk_score"])
        self.assertIn("Missing or unknown transaction type", quarantined["reply"])
        self.assertEqual(unavailable.status_code, 503)
        self.assertNotIn("private error", unavailable.get_json()["reply"])

from decimal import Decimal
from pathlib import Path
import unittest

import pandas as pd

from src.ieee_cis import FEATURES, prepare_ieee_batch, prepare_ieee_features
from src.ieee_release import DEFAULT_RELEASE, FIXTURE_RELEASE, load_release
from src.model_release import predict_risk, read_json_object
from src.synthetic_release import load_release as load_synthetic


class IEEEReleaseTest(unittest.TestCase):
    def test_db_and_csv_rows_use_the_same_order_and_four_decimal_amount_encoding(self):
        csv_row = pd.DataFrame({"amount": [85.991], "type": ["W"]})
        db_row = pd.DataFrame({"amount": [Decimal("85.9910")], "type": [" w "]})
        features = prepare_ieee_features(csv_row)
        self.assertEqual(tuple(features.columns), FEATURES)
        self.assertEqual(features["amount_fraction_1e4"].tolist(), [9910])
        pd.testing.assert_frame_equal(features, prepare_ieee_features(db_row))
        enriched = csv_row.assign(TransactionID=1, TransactionDT=8, isFraud=1, card1=78, device="TEST")
        pd.testing.assert_frame_equal(features, prepare_ieee_features(enriched))

    def test_invalid_ieee_codes_and_amounts_do_not_get_scores(self):
        batch = pd.DataFrame({"amount": [15, 0, 25], "type": ["TRANSFER", "W", "C"]})
        features, reasons = prepare_ieee_batch(batch)
        self.assertEqual(reasons.tolist(), ["invalid_type", "invalid_amount", ""])
        self.assertEqual(features.index.tolist(), [2])
        with self.assertRaisesRegex(ValueError, "invalid amounts or product codes"):
            prepare_ieee_features(batch)

    def test_packaged_ieee_model_matches_validation_selection_and_rejects_other_dataset(self):
        report = read_json_object(Path(DEFAULT_RELEASE).with_name("validation.json").read_bytes(), "report")
        candidate_scores = report["validation_model_comparison"]
        self.assertEqual(set(candidate_scores), {"xgboost", "lightgbm"})
        self.assertEqual(report["algorithm"], max(candidate_scores, key=lambda k: candidate_scores[k]["average_precision"]))
        self.assertEqual(report["test"]["caught_fraud"] + report["test"]["missed_fraud"], report["test"]["fraud_rows"])
        self.assertTrue(0 <= report["threshold"] <= 1)
        with self.assertRaisesRegex(ValueError, "dataset-specific"):
            load_synthetic(DEFAULT_RELEASE)

    def test_public_fixture_model_has_no_research_data_dependency(self):
        release = load_release(FIXTURE_RELEASE)
        self.assertEqual(release.version, "IEEE_CIS_Fixture_LightGBM_v1.0")
        self.assertTrue(0 <= predict_risk(release.model, prepare_ieee_features(
            pd.DataFrame({"amount": [85.99], "type": ["W"]})
        ))[0] <= 1)

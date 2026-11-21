from io import StringIO
import unittest

import pandas as pd

from src.synthetic_financial import PRE_AUTH_FEATURES, load_paysim, prepare_pre_auth_features


class PreAuthorizationFeaturesTest(unittest.TestCase):
    def test_fixed_order_with_normalized_type_and_zero_amount(self):
        rows = pd.DataFrame({"amount": ["0", "120.5"], "type": [" cash-out ", "transfer"]}, index=[4, 8])
        features = prepare_pre_auth_features(rows)

        self.assertEqual(tuple(features.columns), PRE_AUTH_FEATURES)
        self.assertEqual(features.index.tolist(), [4, 8])
        self.assertEqual(features.iloc[0].tolist(), [0, 0, 1, 0, 0, 0])
        self.assertEqual(features.iloc[1].tolist(), [120.5, 0, 0, 0, 0, 1])
        self.assertEqual(str(features["amount"].dtype), "float32")

    def test_model_inputs_do_not_depend_on_outcome_or_balance_fields(self):
        request = pd.DataFrame({"amount": [10, 20], "type": ["DEBIT", "PAYMENT"]})
        enriched = request.assign(step=736, oldbalanceOrg=900, newbalanceOrig=0,
                                  oldbalanceDest=0, newbalanceDest=900,
                                  isFlaggedFraud=1, isFraud=1, nameOrig="C1", nameDest="C2")
        pd.testing.assert_frame_equal(prepare_pre_auth_features(request), prepare_pre_auth_features(enriched))

    def test_invalid_amount_or_type_is_not_imputed(self):
        for amount in (None, -1, "wrong", float("inf"), 1e50):
            with self.subTest(amount=amount), self.assertRaisesRegex(ValueError, "Invalid amount"):
                prepare_pre_auth_features(pd.DataFrame({"amount": [amount], "type": ["PAYMENT"]}))
        for tx_type in (None, "", "WIRE"):
            with self.subTest(tx_type=tx_type), self.assertRaisesRegex(ValueError, "unknown type"):
                prepare_pre_auth_features(pd.DataFrame({"amount": [20], "type": [tx_type]}))
        with self.assertRaisesRegex(ValueError, "Missing scoring inputs"):
            prepare_pre_auth_features(pd.DataFrame({"amount": [20]}))

    def test_raw_loader_validates_schema_and_binary_labels(self):
        valid = ("step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,"
                 "oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
                 "1,PAYMENT,0,C1,0,0,M2,0,0,0,0\n")
        self.assertEqual(len(load_paysim(StringIO(valid))), 1)
        with self.assertRaisesRegex(ValueError, "isFraud must contain binary values"):
            load_paysim(StringIO(valid.replace(",M2,0,0,0,0", ",M2,0,0,2,0")))
        with self.assertRaisesRegex(ValueError, "Missing PaySim columns"):
            load_paysim(StringIO("step,type\n1,PAYMENT\n"))

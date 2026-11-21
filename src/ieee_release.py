"""Load the verified IEEE-CIS demo package with native XGBoost or LightGBM."""

from dataclasses import dataclass
import inspect

import numpy as np

from .ieee_cis import FEATURES, PRODUCT_CODES, prepare_ieee_batch, prepare_ieee_features
from .model_release import ROOT, load_verified_package, load_xgboost, sha256


DATASET = "ieee_cis"
DEFAULT_RELEASE = ROOT / "artifacts/models/ieee_cis_pre_auth_v1_0/manifest.json"
FIXTURE_RELEASE = ROOT / "artifacts/models/ieee_cis_fixture_v1_0/manifest.json"


@dataclass(frozen=True)
class LightGBMModel:
    booster: object

    def predict_proba(self, features):
        positive = self.booster.predict(features, validate_features=True, num_threads=1)
        return np.column_stack((1 - positive, positive))


def load_lightgbm(data, features):
    from lightgbm import Booster

    model = Booster(model_str=data.decode("utf-8"))
    if model.feature_name() != list(features):
        raise ValueError("The loaded model's feature order does not match the contract")
    return LightGBMModel(model)


LOADERS = {"xgboost": ("model.json", load_xgboost), "lightgbm": ("model.txt", load_lightgbm)}


def preprocessor_contract():
    source = "".join(inspect.getsource(function) for function in (prepare_ieee_batch, prepare_ieee_features))
    return {
        "version": "ieee_cis_pre_auth_v1", "entrypoint": "src.ieee_cis.prepare_ieee_features",
        "source_sha256": sha256(source.encode("utf-8")),
        "raw_inputs": ["amount", "type"], "csv_mapping": {"amount": "TransactionAmt", "type": "ProductCD"},
        "product_codes": list(PRODUCT_CODES), "feature_order": list(FEATURES),
    }


def load_release(manifest_path=None):
    if manifest_path is None:
        weights = DEFAULT_RELEASE.with_name("model.txt")
        manifest_path = DEFAULT_RELEASE if weights.is_file() else FIXTURE_RELEASE
    return load_verified_package(
        manifest_path, dataset=DATASET, preprocessing=preprocessor_contract(),
        prepare_batch=prepare_ieee_batch, prepare_features=prepare_ieee_features, loaders=LOADERS,
    )

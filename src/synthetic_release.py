"""Load the packaged Synthetic Financial model and check its feature contract."""

import inspect

from .model_release import ROOT, load_verified_package, load_xgboost, sha256
from .synthetic_financial import (
    PRE_AUTH_FEATURES, TRANSACTION_TYPES, prepare_pre_auth_batch, prepare_pre_auth_features,
)


DATASET = "synthetic_financial"
PREPROCESSOR_VERSION = "paysim_pre_auth_v1"
DEFAULT_RELEASE = ROOT / "artifacts/models/synthetic_financial_pre_auth_v1_1/manifest.json"
MODEL_FILE = "model.json"


def preprocessor_contract():
    source = "".join(inspect.getsource(function) for function in (prepare_pre_auth_batch, prepare_pre_auth_features))
    return {
        "version": PREPROCESSOR_VERSION, "entrypoint": "src.synthetic_financial.prepare_pre_auth_features",
        "source_sha256": sha256(source.encode("utf-8")),
        "raw_inputs": ["amount", "type"], "transaction_types": list(TRANSACTION_TYPES),
        "feature_order": list(PRE_AUTH_FEATURES),
    }


def load_verified_model(path, expected_hash):
    data = path.read_bytes()
    if sha256(data) != expected_hash:
        raise ValueError("Model checksum does not match the manifest")
    return load_xgboost(data, PRE_AUTH_FEATURES)


def load_release(manifest_path=DEFAULT_RELEASE):
    return load_verified_package(
        manifest_path, dataset=DATASET, preprocessing=preprocessor_contract(),
        prepare_batch=prepare_pre_auth_batch, prepare_features=prepare_pre_auth_features,
        loaders={"xgboost": (MODEL_FILE, load_xgboost)},
    )

"""Load a verified, demo-only fraud model with its runtime feature contract."""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DEMO_CURRENCY = "XTS"
VALIDATION_FILE = "validation.json"


@dataclass(frozen=True)
class DemoRelease:
    model: object
    version: str
    threshold: float
    model_sha256: str
    manifest_sha256: str
    dataset: str
    prepare_batch: Callable


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json_object(data, label):
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def json_bytes(data):
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")


def predict_risk(model, features):
    result = np.asarray(model.predict_proba(features))
    if result.shape != (len(features), 2):
        raise ValueError("Model returned an invalid prediction shape")
    scores = result[:, 1]
    if not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all():
        raise ValueError("Model returned an invalid risk score")
    return scores


def load_xgboost(data, features):
    model = XGBClassifier()
    model.load_model(bytearray(data))
    if model.get_booster().feature_names != list(features):
        raise ValueError("The loaded model's feature order does not match the contract")
    return model


def load_verified_package(path, *, dataset, preprocessing, prepare_batch, prepare_features, loaders):
    path = Path(path)
    content = path.read_bytes()
    manifest = read_json_object(content, "Release manifest")
    version = manifest.get("model_version")
    policy = manifest.get("decision")
    if (
        manifest.get("schema_version") != 1 or manifest.get("dataset") != dataset
        or manifest.get("status") != "portfolio_demo" or manifest.get("production_eligible") is not False
        or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", version)
        or not isinstance(policy, dict) or policy.get("operator") != ">="
        or policy.get("positive_action") != "review" or policy.get("negative_action") != "no_alert"
        or policy.get("source") != "demo" or policy.get("currency") != DEMO_CURRENCY
    ):
        raise ValueError("Release is not a supported dataset-specific demo package")
    threshold = policy.get("threshold")
    if type(threshold) not in (float, int) or not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be a finite number between 0 and 1")
    if manifest.get("preprocessing") != preprocessing:
        raise ValueError("Runtime preprocessing does not match the packaged feature contract")

    report_data = path.with_name(VALIDATION_FILE).read_bytes()
    if sha256(report_data) != manifest.get("validation_sha256"):
        raise ValueError("Validation report checksum does not match the manifest")
    validation = read_json_object(report_data, "Validation report")
    model_hash = manifest.get("model_sha256")
    algorithm = manifest.get("algorithm", "xgboost")
    if (
        validation.get("dataset") != dataset or validation.get("model_sha256") != model_hash
        or validation.get("production_eligible") is not False or validation.get("threshold") != threshold
        or validation.get("algorithm", "xgboost") != algorithm
    ):
        raise ValueError("Validation report does not describe this model and threshold")
    if not isinstance(algorithm, str) or algorithm not in loaders:
        raise ValueError("Release model algorithm is not supported")
    filename, load_model = loaders[algorithm]
    model_data = path.with_name(filename).read_bytes()
    if sha256(model_data) != model_hash:
        raise ValueError("Model checksum does not match the manifest")
    model = load_model(model_data, preprocessing["feature_order"])

    canaries = manifest.get("prediction_checks")
    if not isinstance(canaries, list) or not canaries or not all(
        isinstance(case, dict) and {"amount", "type", "risk_score", "decision"} <= case.keys() for case in canaries
    ):
        raise ValueError("Release must have prediction parity checks")
    observed = predict_risk(model, prepare_features(pd.DataFrame(canaries)))
    for case, score in zip(canaries, observed):
        actual = "review" if float(score) >= threshold else "no_alert"
        if not math.isclose(float(score), case["risk_score"], abs_tol=1e-7, rel_tol=0) or actual != case["decision"]:
            raise ValueError("Saved model failed a release prediction parity check")

    return DemoRelease(model, version, float(threshold), model_hash, sha256(content), dataset, prepare_batch)

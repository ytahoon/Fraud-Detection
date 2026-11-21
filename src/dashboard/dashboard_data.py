"""Read offline dashboard summaries with their matching research release."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASES = {
    "synthetic_financial": "synthetic_financial_pre_auth_v1_1",
    "ieee_cis": "ieee_cis_pre_auth_v1_0",
}
MODEL_NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}
SNAPSHOT_VERSION = 1


def read_release(dataset):
    root = Path(os.getenv("MODEL_ARTIFACT_DIR", ROOT / "artifacts/models")) / RELEASES[dataset]
    manifest = json.loads((root / "manifest.json").read_bytes())
    payload = (root / "validation.json").read_bytes()
    report = json.loads(payload)
    if (
        manifest["dataset"] != dataset or report["dataset"] != dataset
        or hashlib.sha256(payload).hexdigest() != manifest["validation_sha256"]
        or report["model_sha256"] != manifest["model_sha256"]
        or report["threshold"] != manifest["decision"]["threshold"]
    ):
        raise ValueError("Evaluation report does not match the release manifest")
    return manifest, report


def read_snapshot(dataset, manifest, report):
    root = Path(os.getenv("DASHBOARD_ARTIFACT_DIR", ROOT / "artifacts/dashboard"))
    snapshot = json.loads((root / f"{dataset}.json").read_bytes())
    if (
        snapshot["schema_version"] != SNAPSHOT_VERSION or snapshot["dataset"] != dataset
        or snapshot["data_sha256"] != report["data_sha256"]
        or snapshot["validation_sha256"] != manifest["validation_sha256"]
        or snapshot["rows"] != sum(part["rows"] for part in report["windows"].values())
        or snapshot["fraud"] != sum(part["fraud_rows"] for part in report["windows"].values())
    ):
        raise ValueError("Dashboard snapshot does not match the research release")
    return snapshot

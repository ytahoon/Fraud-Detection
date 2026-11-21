"""Build the public demo's IEEE-shaped model from fabricated data only."""

from importlib.metadata import version

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

from .ieee_cis import PRODUCT_CODES, prepare_ieee_features
from .ieee_release import DATASET, FIXTURE_RELEASE, load_release, preprocessor_contract
from .model_release import DEMO_CURRENCY, VALIDATION_FILE, json_bytes, predict_risk, sha256


def build_fixture():
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        "amount": np.round(rng.uniform(1, 600, 1_200), 3),
        "type": rng.choice(PRODUCT_CODES, 1_200),
    })
    rules = {"C": 120, "H": 220, "R": 320, "S": 420, "W": 520}
    labels = (data["amount"] >= data["type"].map(rules)).astype(int)
    model = LGBMClassifier(
        n_estimators=30, max_depth=3, num_leaves=6, min_child_samples=12,
        learning_rate=.2, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
    ).fit(prepare_ieee_features(data), labels)
    model_data = model.booster_.model_to_string().encode("utf-8")
    threshold = 0.5
    samples = [{"amount": 85.99, "type": "w"}, {"amount": 149.50, "type": "C"},
               {"amount": 45.00, "type": "R"}, {"amount": 520.00, "type": "S"}]
    for sample, score in zip(samples, predict_risk(model, prepare_ieee_features(pd.DataFrame(samples)))):
        sample["risk_score"] = float(score)
        sample["decision"] = "review" if score >= threshold else "no_alert"
    model_hash = sha256(model_data)
    validation = json_bytes({
        "dataset": DATASET, "algorithm": "lightgbm", "model_sha256": model_hash, "threshold": threshold,
        "production_eligible": False, "source": "project_generated_demo_rows",
        "checks": "Native-model reload and fabricated input prediction parity only. No research metrics.",
    })
    manifest = json_bytes({
        "schema_version": 1, "dataset": DATASET, "algorithm": "lightgbm",
        "model_version": "IEEE_CIS_Fixture_LightGBM_v1.0", "status": "portfolio_demo", "production_eligible": False,
        "training_source": "fabricated amounts, product codes and labels; no competition data",
        "training_seed": 42, "training_environment": {"lightgbm": version("lightgbm"), "numpy": version("numpy")},
        "model_sha256": model_hash, "validation_sha256": sha256(validation),
        "preprocessing": preprocessor_contract(),
        "decision": {"threshold": threshold, "operator": ">=", "positive_action": "review",
                     "negative_action": "no_alert", "source": "demo", "currency": DEMO_CURRENCY},
        "prediction_checks": samples,
    })
    files = {"model.txt": model_data, VALIDATION_FILE: validation, FIXTURE_RELEASE.name: manifest}
    for filename, content in files.items():
        target = FIXTURE_RELEASE.with_name(filename)
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different fixture package: {filename}")
    FIXTURE_RELEASE.parent.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        FIXTURE_RELEASE.with_name(filename).write_bytes(content)
    print(f"Verified {load_release(FIXTURE_RELEASE).version}")


if __name__ == "__main__":
    build_fixture()

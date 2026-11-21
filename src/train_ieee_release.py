"""Train, compare, and verify the chronological IEEE-CIS demo release."""

import argparse
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import average_precision_score
import xgboost
from xgboost import XGBClassifier

from .ieee_cis import prepare_ieee_features
from .ieee_release import DATASET, DEFAULT_RELEASE, LOADERS, load_release, preprocessor_contract
from .model_evaluation import operating_metrics, ranking_metrics, threshold_options
from .model_release import DEMO_CURRENCY, ROOT, VALIDATION_FILE, json_bytes, predict_risk, read_json_object, sha256


DEFAULT_DATA = ROOT / "data/raw/ieee_cis/train_transaction.csv"
SECONDS_PER_DAY = 86_400
EDGES = (120 * SECONDS_PER_DAY, 150 * SECONDS_PER_DAY)
ALERTS_PER_1000 = 50
SEED = 42
N_JOBS = 4
DISPLAY_NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}


def load_windows(path):
    fields = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"]
    data = pd.read_csv(path, usecols=fields).rename(columns={"TransactionAmt": "amount", "ProductCD": "type"})
    if (data.empty or data.isna().any().any() or data["TransactionID"].duplicated().any()
            or not data["isFraud"].isin([0, 1]).all() or (data["TransactionDT"] < 0).any()):
        raise ValueError("IEEE-CIS file has invalid IDs, timestamps, required fields, or fraud labels")
    first, second = EDGES
    timestamps = data["TransactionDT"]
    parts = {
        "train": data.loc[timestamps < first],
        "validation": data.loc[(timestamps >= first) & (timestamps < second)],
        "test": data.loc[timestamps >= second],
    }
    summary = {}
    for name, rows in parts.items():
        if rows["isFraud"].nunique() != 2:
            raise ValueError(f"{name} must contain both classes")
        summary[name] = {
            "first_relative_second": int(rows["TransactionDT"].min()),
            "last_relative_second": int(rows["TransactionDT"].max()),
            "rows": len(rows), "fraud_rows": int(rows["isFraud"].sum()),
        }
    if not summary["train"]["last_relative_second"] < summary["validation"]["first_relative_second"]:
        raise ValueError("Training and validation windows overlap")
    if not summary["validation"]["last_relative_second"] < summary["test"]["first_relative_second"]:
        raise ValueError("Validation and test windows overlap")
    return parts, summary


def score_rows(model, rows):
    return predict_risk(model, prepare_ieee_features(rows))


def compare_models(train, validation):
    y_train = train["isFraud"]
    features = prepare_ieee_features(train)
    weight = float(y_train.eq(0).sum() / y_train.eq(1).sum())
    common = dict(n_estimators=400, learning_rate=0.05, max_depth=6, scale_pos_weight=weight,
                  random_state=SEED, n_jobs=N_JOBS)
    candidates = {
        "xgboost": XGBClassifier(**common, tree_method="hist", eval_metric="aucpr"),
        "lightgbm": LGBMClassifier(**common, num_leaves=31, deterministic=True, force_col_wise=True, verbosity=-1),
    }
    comparison = {}
    for name, model in candidates.items():
        model.fit(features, y_train)
        comparison[name] = ranking_metrics(validation["isFraud"], score_rows(model, validation))
        print(f"{DISPLAY_NAMES[name]} validation AP: {comparison[name]['average_precision']:.6f}", flush=True)
    winner = max(candidates, key=lambda name: comparison[name]["average_precision"])
    return winner, candidates[winner], comparison


def native_model(algorithm, model):
    if algorithm == "lightgbm":
        return model.booster_.model_to_string().encode("utf-8")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        model.save_model(path)
        return path.read_bytes()


def evaluate(model, parts, threshold):
    validation, test = parts["validation"], parts["test"]
    validation_scores = score_rows(model, validation)
    test_scores = score_rows(model, test)
    test_metrics = operating_metrics(test["isFraud"], test_scores, threshold)
    test_metrics.update(ranking_metrics(test["isFraud"], test_scores))
    type_rates = parts["train"].groupby("type")["isFraud"].mean()
    per_product = {
        name: operating_metrics(rows["isFraud"], rows["risk_score"], threshold)
        for name, rows in test.assign(risk_score=test_scores).groupby("type")
    }
    return {
        "validation": {**operating_metrics(validation["isFraud"], validation_scores, threshold),
                       **ranking_metrics(validation["isFraud"], validation_scores)},
        "test": test_metrics,
        "test_by_product": per_product,
        "baselines": {
            "test_no_skill_average_precision": float(test["isFraud"].mean()),
            "test_product_only_average_precision": float(average_precision_score(
                test["isFraud"], test["type"].map(type_rates)
            )),
        },
    }


def write_package(path, algorithm, data_hash, parts, windows, original, comparison):
    filename, load_native = LOADERS[algorithm]
    payload = native_model(algorithm, original)
    preprocessing = preprocessor_contract()
    model = load_native(payload, preprocessing["feature_order"])
    validation = parts["validation"]
    scores = score_rows(model, validation)
    threshold = threshold_options(validation["isFraud"], scores, [ALERTS_PER_1000])[ALERTS_PER_1000]
    original_threshold = threshold_options(
        validation["isFraud"], score_rows(original, validation), [ALERTS_PER_1000]
    )[ALERTS_PER_1000]
    if threshold != original_threshold:
        raise ValueError("Native model reload changed the selected threshold")
    model_hash = sha256(payload)
    report = {
        "dataset": DATASET, "algorithm": algorithm, "model_sha256": model_hash, "data_sha256": data_hash,
        "windows": windows, "split_edges_relative_seconds": list(EDGES),
        "selection_metric": "validation average precision", "validation_model_comparison": comparison,
        "threshold_selection": {"split": "validation", "max_alerts_per_1000": ALERTS_PER_1000,
                                "ordering": ["highest recall", "highest precision", "highest threshold"]},
        "threshold": threshold, **evaluate(model, parts, threshold), "production_eligible": False,
        "limitations": [
            "Internal chronological holdout from the labeled competition training file, not a Kaggle leaderboard result.",
            "Product codes require a documented source mapping. Test currency and fixtures are fabricated.",
            "Scores are not calibrated probabilities. No real payment decisions are supported.",
        ],
    }
    samples = [{"amount": 85.99, "type": "w"}, {"amount": 149.50, "type": "C"},
               {"amount": 45.001, "type": "R"}, {"amount": 520.00, "type": "S"}]
    for sample, score in zip(samples, score_rows(model, pd.DataFrame(samples))):
        sample["risk_score"] = float(score)
        sample["decision"] = "review" if float(score) >= threshold else "no_alert"
    report_bytes = json_bytes(report)
    manifest = {
        "schema_version": 1, "dataset": DATASET, "algorithm": algorithm,
        "model_version": f"IEEE_CIS_PreAuth_{DISPLAY_NAMES[algorithm]}_v1.0",
        "status": "portfolio_demo", "production_eligible": False, "model_sha256": model_hash,
        "validation_sha256": sha256(report_bytes), "preprocessing": preprocessing,
        "training_environment": {name: xgboost.__version__ if name == "xgboost" else version(name)
                                 for name in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm")},
        "training_parameters": original.get_params(),
        "decision": {"threshold": threshold, "operator": ">=", "positive_action": "review",
                     "negative_action": "no_alert", "source": "demo", "currency": DEMO_CURRENCY},
        "prediction_checks": samples,
    }
    # XGBoost exposes NaN as its default missing-value sentinel; the pipeline rejects missing inputs.
    manifest["training_parameters"] = {
        key: value for key, value in manifest["training_parameters"].items()
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    }
    files = {filename: payload, VALIDATION_FILE: report_bytes, path.name: json_bytes(manifest)}
    for name, content in files.items():
        existing = path.with_name(name)
        if existing.exists() and existing.read_bytes() != content:
            raise ValueError(f"Will not overwrite {name} in a different release; use a new version")
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path.with_name(name).write_bytes(content)


def verify_package(path, data_hash, parts, windows):
    release = load_release(path)
    report = read_json_object(path.with_name(VALIDATION_FILE).read_bytes(), "Validation report")
    if report["data_sha256"] != data_hash or report["windows"] != windows:
        raise ValueError("IEEE-CIS source file or split does not match the model release")
    validation = parts["validation"]
    actual_threshold = threshold_options(
        validation["isFraud"], score_rows(release.model, validation), [ALERTS_PER_1000]
    )[ALERTS_PER_1000]
    if actual_threshold != release.threshold:
        raise ValueError("Validation replay did not reproduce the release threshold")
    for section, actual in evaluate(release.model, parts, release.threshold).items():
        if actual != report[section]:
            raise ValueError(f"IEEE-CIS release replay mismatch in {section}; use the recorded environment")
    test = report["test"]
    print(f"{release.version}: threshold {release.threshold:.8f}")
    print(f"Holdout AP {test['average_precision']:.6f}; precision {test['precision']:.2%}; "
          f"recall {test['recall']:.2%}; {test['alerts_per_1000']:.2f} alerts per 1,000")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--build", action="store_true", help="Train candidates and write the release if absent")
    args = parser.parse_args()
    parts, windows = load_windows(args.data)
    with args.data.open("rb") as data:
        data_hash = hashlib.file_digest(data, "sha256").hexdigest()
    if args.build:
        algorithm, model, comparison = compare_models(parts["train"], parts["validation"])
        write_package(args.manifest, algorithm, data_hash, parts, windows, model, comparison)
    verify_package(args.manifest, data_hash, parts, windows)


if __name__ == "__main__":
    main()

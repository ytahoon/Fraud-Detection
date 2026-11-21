"""Replay chronological evaluation and verify or build the versioned demo model package."""

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from .model_evaluation import operating_metrics, ranking_metrics, threshold_options
from .synthetic_financial import TRANSACTION_TYPES, paysim_path, prepare_pre_auth_features
from .model_release import DEMO_CURRENCY, ROOT, VALIDATION_FILE, json_bytes, predict_risk, read_json_object, sha256
from .synthetic_release import DATASET, DEFAULT_RELEASE, MODEL_FILE, load_release, load_verified_model, preprocessor_contract


BENCHMARK = ROOT / "artifacts/models/synthetic_financial_pre_auth_contract.json"
RELEASE_VERSION = "Synthetic_PreAuth_XGBoost_v1.1"
WINDOWS = {"train": (1, 336), "validation": (337, 504), "test": (505, 743)}
REVIEW_BUDGET = 10


def evaluate_candidate(path, model):
    data_counts = Counter()
    fraud_counts = Counter()
    training_by_type = Counter()
    training_fraud_by_type = Counter()
    predictions = {"validation": [], "test": []}

    for chunk in pd.read_csv(path, usecols=["step", "amount", "type", "isFraud"], chunksize=250_000):
        if not chunk["step"].between(1, 743).all() or not chunk["isFraud"].isin([0, 1]).all():
            raise ValueError("Invalid simulation hour or binary label in the audited dataset")
        for name, (first, last) in WINDOWS.items():
            rows = chunk.loc[chunk["step"].between(first, last)]
            if rows.empty:
                continue
            data_counts[name] += len(rows)
            fraud_counts[name] += int(rows["isFraud"].sum())
            if name == "train":
                training_by_type.update(rows["type"].value_counts().to_dict())
                training_fraud_by_type.update(rows.groupby("type")["isFraud"].sum().to_dict())
            else:
                scores = predict_risk(model, prepare_pre_auth_features(rows))
                predictions[name].append(rows[["type", "isFraud"]].assign(score=scores))

    windows = {
        name: {"first_hour": first, "last_hour": last, "rows": data_counts[name], "fraud_rows": fraud_counts[name]}
        for name, (first, last) in WINDOWS.items()
    }
    type_rates = {t: training_fraud_by_type[t] / training_by_type[t] for t in TRANSACTION_TYPES}
    return windows, type_rates, {name: pd.concat(frames, ignore_index=True) for name, frames in predictions.items()}


def require_benchmark_match(report, candidate):
    for window, actual in report["windows"].items():
        recorded = candidate["split_by_simulated_hour"][window]
        if any(value != recorded[key] for key, value in actual.items()):
            raise ValueError(f"The {window} split does not match the audited benchmark")
    for window in ("validation", "test"):
        for key, value in candidate[window].items():
            if not math.isclose(value, report[window][key], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Benchmark replay mismatch: {window}.{key}")


def make_report(data_path, candidate, model):
    with data_path.open("rb") as source:
        data_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if data_hash != candidate["data_sha256"]:
        raise ValueError("Raw data checksum does not match the audited PaySim release")
    windows, rates, predictions = evaluate_candidate(data_path, model)
    validation, test = predictions["validation"], predictions["test"]
    val_y, val_scores = validation["isFraud"].to_numpy(), validation["score"].to_numpy()
    test_y, test_scores = test["isFraud"].to_numpy(), test["score"].to_numpy()
    options = threshold_options(val_y, val_scores, (5, REVIEW_BUDGET, 20))
    threshold = options[REVIEW_BUDGET]
    if threshold != candidate["decision"]["threshold"]:
        raise ValueError("Validation replay did not reproduce the frozen threshold")

    test_results = operating_metrics(test_y, test_scores, threshold)
    test_results.update(ranking_metrics(test_y, test_scores))
    report = {
        "dataset": DATASET, "model_sha256": candidate["model_sha256"], "data_sha256": data_hash,
        "threshold": threshold, "windows": windows,
        "threshold_selection": {
            "split": "validation", "alerts_per_1000_target": REVIEW_BUDGET,
            "ordering": ["highest recall", "highest precision", "highest threshold"],
            "tie_policy": "equal scores receive the same decision; score >= threshold",
        },
        "validation": operating_metrics(val_y, val_scores, threshold),
        "validation_budget_sensitivity": {
            str(budget): {"threshold": limit, **operating_metrics(val_y, val_scores, limit)}
            for budget, limit in options.items()
        },
        "test": test_results,
        "test_type_baseline_average_precision": float(average_precision_score(test_y, test["type"].map(rates))),
        "test_no_skill_average_precision": float(test_y.mean()),
        "test_by_transaction_type": {
            name: operating_metrics(rows["isFraud"].to_numpy(), rows["score"].to_numpy(), threshold)
            for name, rows in test.groupby("type")
        },
        "test_meets_validation_review_budget": test_results["alerts_per_1000"] <= REVIEW_BUDGET,
        "production_eligible": False,
        "production_limitations": [
            "Later window is held out in the same simulation; no external or live validation.",
            "At the frozen threshold most test fraud was not detected.",
            "Scores are rankings, not calibrated probabilities or payment decisions.",
        ],
    }
    require_benchmark_match(report, candidate)
    return report


def build_package(manifest_path, candidate, model, report):
    samples = [{"amount": 0, "type": "cash-out"}, {"amount": 125, "type": "PAYMENT"},
               {"amount": 2500, "type": "TRANSFER"}, {"amount": 1000000, "type": "CASH_OUT"}]
    scores = predict_risk(model, prepare_pre_auth_features(pd.DataFrame(samples)))
    threshold = report["threshold"]
    for sample, score in zip(samples, scores):
        sample["risk_score"] = float(score)
        sample["decision"] = "review" if float(score) >= threshold else "no_alert"
    report_data = json_bytes(report)
    manifest = {
        "schema_version": 1, "dataset": DATASET, "model_version": RELEASE_VERSION,
        "source_model_version": candidate["model_version"],
        "status": "portfolio_demo", "production_eligible": False,
        "model_sha256": candidate["model_sha256"], "validation_sha256": sha256(report_data),
        "preprocessing": preprocessor_contract(), "training_parameters": candidate["training_parameters"],
        "training_environment": candidate["versions"],
        "decision": {"threshold": threshold, "operator": ">=", "positive_action": "review",
                     "negative_action": "no_alert", "source": "demo", "currency": DEMO_CURRENCY},
        "prediction_checks": samples,
    }
    model_data = BENCHMARK.with_name(candidate["model_file"]).read_bytes()
    if sha256(model_data) != manifest["model_sha256"]:
        raise ValueError("Source model changed between evaluation and packaging")
    files = {MODEL_FILE: model_data, VALIDATION_FILE: report_data, manifest_path.name: json_bytes(manifest)}
    for name, content in files.items():
        path = manifest_path.with_name(name)
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Will not overwrite a different package file ({name}); bump release version")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        manifest_path.with_name(name).write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to the original, checksum-matched PaySim CSV")
    parser.add_argument("--build", action="store_true", help="Materialize the versioned package if absent")
    args = parser.parse_args()
    candidate = read_json_object(BENCHMARK.read_bytes(), "Audited candidate")
    legacy_model = load_verified_model(BENCHMARK.with_name(candidate["model_file"]), candidate["model_sha256"])
    if args.build:
        report = make_report(args.data or paysim_path(ROOT), candidate, legacy_model)
        build_package(DEFAULT_RELEASE, candidate, legacy_model, report)
        release = load_release()
    else:
        release = load_release()
        report = make_report(args.data or paysim_path(ROOT), candidate, release.model)
    if DEFAULT_RELEASE.with_name(VALIDATION_FILE).read_bytes() != json_bytes(report):
        raise ValueError("Packaged evaluation differs from replay; use the recorded training environment")
    test = report["test"]
    print(f"{release.version}: verified at threshold {release.threshold:.8f}")
    print(f"Later test: AP {test['average_precision']:.4f}, precision {test['precision']:.2%}, "
          f"recall {test['recall']:.2%}, {test['alerts_per_1000']:.2f} alerts per 1,000")
    print("Portfolio demo only; production release criteria are not met.")


if __name__ == "__main__":
    main()

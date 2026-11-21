"""Export non-identifying dataset aggregates and verified holdout curve points."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src import ieee_release, synthetic_release
from src.dashboard.dashboard_data import RELEASES, ROOT, SNAPSHOT_VERSION, read_release
from src.ieee_cis import prepare_ieee_features
from src.model_evaluation import operating_metrics, ranking_metrics
from src.model_release import predict_risk
from src.synthetic_financial import paysim_path, prepare_pre_auth_features


AMOUNTS = [1, 10, 50, 100, 500, 1_000, 10_000, 100_000, 1_000_000]
AMOUNT_LABELS = ["0 to <1", "1 to <10", "10 to <50", "50 to <100", "100 to <500",
                 "500 to <1k", "1k to <10k", "10k to <100k", "100k to <1M", "1M+"]
BALANCES = [-1_000_000, -100_000, -10_000, -1_000, 0, 1, 1_000, 10_000, 100_000, 1_000_000]
BALANCE_LABELS = ["<-1M", "-1M to <-100k", "-100k to <-10k", "-10k to <-1k", "-1k to <0",
                  "0 to <1", "1 to <1k", "1k to <10k", "10k to <100k", "100k to <1M", "1M+"]


def sha_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def grouped_counts(data, column):
    items = data.groupby(column, observed=True)["isFraud"].agg(rows="size", fraud="sum")
    return [{"group": group, "rows": int(row.rows), "fraud": int(row.fraud)}
            for group, row in items.iterrows()]


def distribution(data, column, edges, labels):
    bins = pd.cut(data[column], [-np.inf, *edges, np.inf], labels=labels, right=False)
    counts = pd.crosstab(bins, data["isFraud"]).reindex(index=labels, columns=[0, 1], fill_value=0)
    summary = data.groupby("isFraud")[column].agg(["min", "median", "mean", "max"])
    summary["p95"] = data.groupby("isFraud")[column].quantile(0.95)
    return {
        "bins": [{"range": label, "not_fraud": int(row[0]), "fraud": int(row[1])}
                 for label, row in counts.iterrows()],
        "summary": [{"class": "Fraud" if group else "Not fraud", **row.to_dict()}
                    for group, row in summary.iterrows()],
    }


def read_data(dataset, source, identity):
    if dataset == "synthetic_financial":
        data = pd.read_csv(source, usecols=["step", "type", "amount", "isFraud", "isFlaggedFraud",
                                            "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
                           dtype={"type": "category", "isFraud": "int8"})
        data["day"] = (data["step"] - 1) // 24 + 1
        data["sender_change"] = data.pop("newbalanceOrig") - data.pop("oldbalanceOrg")
        data["receiver_change"] = data.pop("newbalanceDest") - data.pop("oldbalanceDest")
        return data, {"Transaction type": "type"}

    data = pd.read_csv(source, usecols=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
                                       "ProductCD", "card4", "card6"], dtype={"isFraud": "int8"})
    devices = pd.read_csv(identity, usecols=["TransactionID", "DeviceType"])
    data = data.merge(devices, on="TransactionID", how="left", validate="one_to_one")
    data = data.rename(columns={"TransactionDT": "step", "TransactionAmt": "amount", "ProductCD": "type"})
    data["day"] = data["step"] // 86_400
    for name in ("DeviceType", "card4", "card6"):
        data[name] = data[name].fillna("Unknown")
    return data, {"Product code": "type", "Device type": "DeviceType", "Card network": "card4", "Card type": "card6"}


def points(x, y):
    indices = np.linspace(0, len(x) - 1, min(len(x), 450), dtype=int)
    return np.column_stack([x[indices], y[indices]]).tolist()


def evaluate_curves(data, report, release, prepare):
    result = {}
    bounds = ("first_hour", "last_hour") if release.dataset == "synthetic_financial" else (
        "first_relative_second", "last_relative_second")
    for split in ("validation", "test"):
        window = report["windows"][split]
        rows = data.loc[data["step"].between(*(window[edge] for edge in bounds)), ["amount", "type", "isFraud"]]
        actual = rows["isFraud"]
        scores = predict_risk(release.model, prepare(rows))
        ranking = ranking_metrics(actual, scores)
        metrics = {**operating_metrics(actual, scores, release.threshold), **ranking}
        for key, expected in report[split].items():
            matched = metrics[key] == expected if isinstance(expected, int) else np.isclose(
                metrics[key], expected, atol=1e-10, rtol=0)
            if not matched:
                raise ValueError(f"Replay mismatch: {release.dataset}.{split}.{key}")
        precision, recall, _ = precision_recall_curve(actual, scores)
        fpr, tpr, _ = roc_curve(actual, scores)
        result[split] = {"pr": points(recall[::-1], precision[::-1]), "roc": points(fpr, tpr), "ranking": ranking}
    return result


def export(dataset, source, identity, output):
    manifest, report = read_release(dataset)
    if sha_file(source) != report["data_sha256"]:
        raise ValueError("Transaction CSV does not match the audited research source")
    data, segments = read_data(dataset, source, identity)
    if not data["isFraud"].isin([0, 1]).all() or not np.isfinite(data["amount"]).all() or data["amount"].lt(0).any():
        raise ValueError("Unexpected invalid label or amount in research source")
    synthetic = dataset == "synthetic_financial"
    release = synthetic_release.load_release() if synthetic else ieee_release.load_release()
    prepare = prepare_pre_auth_features if synthetic else prepare_ieee_features
    snapshot = {
        "schema_version": SNAPSHOT_VERSION, "dataset": dataset, "data_sha256": report["data_sha256"],
        "validation_sha256": manifest["validation_sha256"], "rows": len(data), "fraud": int(data["isFraud"].sum()),
        "total_amount": float(data["amount"].sum()), "median_amount": float(data["amount"].median()),
        "by_day": grouped_counts(data, "day"),
        "segments": {title: grouped_counts(data, column) for title, column in segments.items()},
        "distributions": {"Transaction amount": distribution(data, "amount", AMOUNTS, AMOUNT_LABELS)},
        "curves": evaluate_curves(data, report, release, prepare),
    }
    if synthetic:
        snapshot["flagged_by_rule"] = int(data["isFlaggedFraud"].sum())
        for title, column in (("Sender balance change", "sender_change"), ("Receiver balance change", "receiver_change")):
            snapshot["distributions"][title] = distribution(data, column, BALANCES, BALANCE_LABELS)
    else:
        snapshot["identity_sha256"] = sha_file(identity)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{dataset}.json"
    path.write_text(json.dumps(snapshot, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{dataset}: {len(data):,} transactions, {snapshot['fraud']:,} fraud labels -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*RELEASES, "all"], default="all")
    parser.add_argument("--paysim", type=Path)
    parser.add_argument("--ieee-transactions", type=Path, default=ROOT / "data/raw/ieee_cis/train_transaction.csv")
    parser.add_argument("--ieee-identity", type=Path, default=ROOT / "data/raw/ieee_cis/train_identity.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dashboard")
    args = parser.parse_args()
    for dataset in RELEASES if args.dataset == "all" else [args.dataset]:
        source = (args.paysim or paysim_path(ROOT)) if dataset == "synthetic_financial" else args.ieee_transactions
        export(dataset, source, args.ieee_identity, args.output)


if __name__ == "__main__":
    main()

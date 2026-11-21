"""PaySim data validation and the pre-authorization feature contract."""

from pathlib import Path

import numpy as np
import pandas as pd


TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
BALANCE_COLUMNS = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
ACCOUNT_COLUMNS = ("nameOrig", "nameDest")
RAW_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
)
TYPE_FEATURES = tuple(f"type_{value}" for value in TRANSACTION_TYPES)
PRE_AUTH_FEATURES = ("amount", *TYPE_FEATURES)
HISTORICAL_FEATURES = ("step", "amount", *BALANCE_COLUMNS, "isFlaggedFraud", *TYPE_FEATURES)


def paysim_path(project_root):
    folder = Path(project_root) / "data" / "raw" / "synthetic_financial"
    for filename in ("synthetic_financial.csv", "cleaned_data.csv"):
        path = folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"Place the PaySim CSV at {folder / 'synthetic_financial.csv'}")


def load_paysim(path, *, include_accounts=True):
    """Load typed fields, rejecting schema, domain, and completeness violations."""
    missing = set(RAW_COLUMNS) - set(pd.read_csv(path, nrows=0).columns)
    if missing:
        raise ValueError(f"Missing PaySim columns: {sorted(missing)}")
    if hasattr(path, "seek"):
        path.seek(0)
    columns = [col for col in RAW_COLUMNS if include_accounts or col not in ACCOUNT_COLUMNS]
    dtypes = {
        "step": "Int32",
        "type": pd.CategoricalDtype(TRANSACTION_TYPES),
        "amount": "float64",
        **{col: "float64" for col in BALANCE_COLUMNS},
        **{col: "string" for col in ACCOUNT_COLUMNS if include_accounts},
        "isFraud": "Int32",
        "isFlaggedFraud": "Int32",
    }
    data = pd.read_csv(path, usecols=columns, dtype=dtypes)
    if data.empty:
        raise ValueError("PaySim file has no transactions")
    nulls = data.isna().sum()
    if nulls.any():
        raise ValueError(f"Missing PaySim values or unknown transaction types: {nulls[nulls > 0].to_dict()}")
    if data["step"].lt(1).any():
        raise ValueError("PaySim step must be a positive hours-since-start index")
    for column in ("isFraud", "isFlaggedFraud"):
        if not data[column].isin((0, 1)).all():
            raise ValueError(f"{column} must contain binary values")
        data[column] = data[column].astype("int8")
    for column in ("amount", *BALANCE_COLUMNS):
        values = data[column].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} has non-finite or negative values")
    data["step"] = data["step"].astype("int32")
    return data


def prepare_pre_auth_batch(rows):
    """Return features for valid rows and one issue code for each invalid row."""
    missing = {"amount", "type"} - set(rows.columns)
    if missing:
        raise ValueError(f"Missing scoring inputs: {sorted(missing)}")

    amount = pd.to_numeric(rows["amount"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    invalid_amount = ~np.isfinite(amount) | (amount < 0) | (amount > np.finfo(np.float32).max)

    types = rows["type"]
    if not isinstance(types.dtype, pd.CategoricalDtype) or tuple(types.cat.categories) != TRANSACTION_TYPES:
        types = types.astype("string").str.strip().str.upper().str.replace(r"[ -]+", "_", regex=True)
    invalid_type = ~types.isin(TRANSACTION_TYPES).to_numpy(dtype=bool)
    issues = np.select([invalid_amount, invalid_type], ["invalid_amount", "invalid_type"], default="")
    valid = ~(invalid_amount | invalid_type)
    types = types.iloc[valid]
    features = pd.DataFrame(
        {
            "amount": amount[valid].astype(np.float32),
            **{f"type_{value}": types.eq(value).to_numpy(dtype=np.int8) for value in TRANSACTION_TYPES},
        },
        index=rows.index[valid],
        columns=PRE_AUTH_FEATURES,
    )
    return features, issues


def prepare_pre_auth_features(rows):
    """Transform only the two raw scoring inputs; never impute missing data."""
    features, issues = prepare_pre_auth_batch(rows)
    if (bad_amount := np.count_nonzero(issues == "invalid_amount")):
        raise ValueError(f"Invalid amount in {bad_amount} row(s)")
    if (bad_type := np.count_nonzero(issues == "invalid_type")):
        raise ValueError(f"Missing or unknown type in {bad_type} row(s)")
    return features

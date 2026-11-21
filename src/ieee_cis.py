"""Point-in-time IEEE-CIS features computed from the amount and ProductCD only."""

import numpy as np
import pandas as pd


PRODUCT_CODES = ("C", "H", "R", "S", "W")
FEATURES = ("amount", "amount_fraction_1e4", *(f"product_{code}" for code in PRODUCT_CODES))


def prepare_ieee_batch(rows):
    """Map transaction_type to ProductCD upstream; reject unknown codes and bad amounts."""
    if not {"amount", "type"}.issubset(rows.columns):
        raise ValueError("IEEE-CIS scoring requires amount and ProductCD")
    amounts = pd.to_numeric(rows["amount"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    amounts = np.round(amounts, 4)
    bad_amount = ~np.isfinite(amounts) | (amounts <= 0) | (amounts >= 10**16)
    products = rows["type"].astype("string").str.strip().str.upper()
    bad_product = ~products.isin(PRODUCT_CODES).to_numpy(dtype=bool)
    issues = np.select([bad_amount, bad_product], ["invalid_amount", "invalid_type"], default="")
    valid = ~(bad_amount | bad_product)
    amounts = amounts[valid]
    products = products.iloc[valid]
    fractions = np.rint((amounts - np.floor(amounts)) * 10_000).astype(np.int16) % 10_000
    features = pd.DataFrame({
        "amount": amounts.astype(np.float32),
        "amount_fraction_1e4": fractions,
        **{f"product_{code}": products.eq(code).to_numpy(dtype=np.int8) for code in PRODUCT_CODES},
    }, index=rows.index[valid], columns=FEATURES)
    return features, issues


def prepare_ieee_features(rows):
    features, issues = prepare_ieee_batch(rows)
    if (issues != "").any():
        raise ValueError("IEEE-CIS input contains invalid amounts or product codes")
    return features

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IEEE_DATA_DIR = Path(os.getenv("IEEE_CIS_DATA_DIR", PROJECT_ROOT / "data/raw/ieee_cis"))
SYNTHETIC_DATA_FILE = Path(
    os.getenv("SYNTHETIC_FINANCIAL_DATA_FILE", PROJECT_ROOT / "data/raw/synthetic_financial/synthetic_financial.csv")
)
SYNTHETIC_FALLBACK_FILE = PROJECT_ROOT / "data/raw/synthetic_financial/cleaned_data.csv"
ARTIFACT_DIR = Path(os.getenv("DASHBOARD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts/dashboard"))
MODEL_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", PROJECT_ROOT / "artifacts/models"))
IEEE_VAL_PRED_FILE = ARTIFACT_DIR / "ieee_cis_val_predictions.csv"
SYNTHETIC_MODEL_FILE = MODEL_DIR / "synthetic_financial_xgboost.pkl"
SYNTHETIC_FEATURE_FILE = MODEL_DIR / "synthetic_financial_xgboost_features.json"

PRIMARY_COLOR = "#2563EB"
SUCCESS_COLOR = "#10B981"
DANGER_COLOR = "#EF4444"
WARNING_COLOR = "#F59E0B"
NEUTRAL_COLOR = "#64748B"


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.4rem; padding-bottom: 1.2rem;}
            .stSidebar {border-right: 1px solid #E2E8F0;}
            .kpi-card {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
            }
            .kpi-label {font-size: 0.86rem; color: #475569; margin-bottom: 0.25rem;}
            .kpi-value {font-size: 1.55rem; font-weight: 700; color: #0F172A; line-height: 1.2;}
            .section-note {color: #64748B; font-size: 0.92rem; margin-top: -0.25rem; margin-bottom: 0.65rem;}
            .page-subtitle {color: #475569; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_note(text: str) -> None:
    st.markdown(f'<div class="section-note"><strong>How to read:</strong> {text}</div>', unsafe_allow_html=True)


def apply_plot_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    return fig


def sample_for_plot(df: pd.DataFrame, size: int = 20_000) -> pd.DataFrame:
    if len(df) > size:
        return df.sample(size, random_state=42)
    return df


@st.cache_data
def load_ieee_data() -> pd.DataFrame:
    train_transaction = pd.read_csv(IEEE_DATA_DIR / "train_transaction.csv")
    train_identity = pd.read_csv(IEEE_DATA_DIR / "train_identity.csv")
    df = train_transaction.merge(train_identity, on="TransactionID", how="left")
    df["day_idx"] = (df["TransactionDT"] // 86400).astype(int)
    df["DeviceInfo"] = df["DeviceInfo"].fillna("Unknown")
    return df


@st.cache_data
def load_ieee_validation_artifacts() -> pd.DataFrame | None:
    if not IEEE_VAL_PRED_FILE.exists():
        return None
    return pd.read_csv(IEEE_VAL_PRED_FILE)


@st.cache_data
def load_synthetic_data() -> pd.DataFrame:
    data_file = SYNTHETIC_DATA_FILE if SYNTHETIC_DATA_FILE.exists() else SYNTHETIC_FALLBACK_FILE
    usecols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]
    dtype = {
        "step": "int16",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(data_file, usecols=usecols, dtype=dtype)
    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))
    df["origin_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["destination_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df


@st.cache_data
def load_synthetic_feature_list() -> list[str]:
    if not SYNTHETIC_FEATURE_FILE.exists():
        return []
    return json.loads(SYNTHETIC_FEATURE_FILE.read_text())


def add_amount_filter(df: pd.DataFrame, amount_col: str, label: str = "Transaction Amount") -> pd.DataFrame:
    st.sidebar.markdown("### Filters")
    st.sidebar.caption("Use these filters to narrow all pages at once.")
    min_amt = float(df[amount_col].min())
    max_amt = float(df[amount_col].max())
    amount_range = st.sidebar.slider(label, min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))
    return df[(df[amount_col] >= amount_range[0]) & (df[amount_col] <= amount_range[1])]


def add_synthetic_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = add_amount_filter(df, "amount")
    tx_types = sorted(filtered["type"].astype(str).unique().tolist())
    selected_types = st.sidebar.multiselect("Transaction Type", tx_types, default=tx_types)
    if selected_types:
        filtered = filtered[filtered["type"].astype(str).isin(selected_types)]
    return filtered


def render_stakeholder_note(page: str) -> None:
    notes = {
        "Overview": "Executive snapshot of volume, fraud count, fraud rate, and transaction value.",
        "Fraud Monitoring": "Track fraud behavior over time and check how amount relates to fraud labels.",
        "Device Risk": "Identify device groups with higher observed fraud rates.",
        "Card / User Risk": "Highlight card groups with concentrated fraud behavior.",
        "Alerts": "Review highest-risk transactions first, then drill down by risk level.",
        "Model Performance": "Compare model quality using validation metrics and threshold behavior.",
        "Transaction Type Risk": "Compare fraud concentration across payment, transfer, and cash movement types.",
        "Amount / Balance Analysis": "Inspect how transaction amounts and balance movements differ by fraud class.",
        "Synthetic XGBoost Results": "Review the final Synthetic Financial XGBoost validation results and model features.",
    }
    st.info(notes.get(page, "This page provides decision support for fraud analysis."))


def page_ieee_overview(df: pd.DataFrame) -> None:
    total_tx = len(df)
    fraud_count = int(df["isFraud"].sum())
    fraud_rate = (df["isFraud"].mean() * 100.0) if total_tx else 0.0
    total_amount = float(df["TransactionAmt"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Total Transactions", f"{total_tx:,}")
    with c2:
        render_kpi("Fraud Count", f"{fraud_count:,}")
    with c3:
        render_kpi("Fraud Rate", f"{fraud_rate:.2f}%")
    with c4:
        render_kpi("Total Amount", f"${total_amount:,.2f}")

    st.subheader("Fraud Share")
    fraud_share = df["isFraud"].map({0: "Not Fraud", 1: "Fraud"}).value_counts().reset_index()
    fraud_share.columns = ["Class", "Count"]
    fig = px.pie(
        fraud_share,
        names="Class",
        values="Count",
        color="Class",
        color_discrete_map={"Fraud": DANGER_COLOR, "Not Fraud": SUCCESS_COLOR},
        hole=0.45,
    )
    render_chart_note("Each slice is the share of transactions in that class.")
    st.plotly_chart(apply_plot_style(fig), use_container_width=True)


def page_ieee_fraud_monitoring(df: pd.DataFrame) -> None:
    st.subheader("Fraud Over Time")
    fraud_time = df.groupby("day_idx", as_index=False)["isFraud"].sum()
    fig_time = px.line(
        fraud_time,
        x="day_idx",
        y="isFraud",
        markers=True,
        color_discrete_sequence=[PRIMARY_COLOR],
        title="Fraud Count by Day Index",
    )
    render_chart_note("X axis is day index, Y axis is number of fraud transactions on that day.")
    st.plotly_chart(apply_plot_style(fig_time), use_container_width=True)

    st.subheader("Fraud vs Amount")
    sample = sample_for_plot(df)
    fig = px.scatter(
        sample,
        x="TransactionAmt",
        y="isFraud",
        opacity=0.5,
        title="Transaction Amount vs Fraud Label",
        labels={"isFraud": "Fraud (0/1)"},
        color="isFraud",
        color_discrete_map={0: SUCCESS_COLOR, 1: DANGER_COLOR},
    )
    render_chart_note("Each dot is one transaction; red points are fraud and green points are not fraud.")
    st.plotly_chart(apply_plot_style(fig), use_container_width=True)


def page_ieee_device_risk(df: pd.DataFrame) -> None:
    min_tx = st.slider("Min transactions per device", min_value=20, max_value=1000, value=100, step=20)
    device_stats = (
        df.groupby("DeviceInfo", as_index=False)
        .agg(tx_count=("isFraud", "size"), fraud_count=("isFraud", "sum"), fraud_rate=("isFraud", "mean"))
    )
    device_stats = device_stats[device_stats["tx_count"] >= min_tx]
    top_risky = device_stats.sort_values("fraud_rate", ascending=False).head(20)

    st.subheader("Top Risky Devices")
    fig_top = px.bar(
        top_risky,
        x="DeviceInfo",
        y="fraud_rate",
        hover_data=["tx_count", "fraud_count"],
        color_discrete_sequence=[DANGER_COLOR],
    )
    render_chart_note("Bars show devices with the highest fraud rate after applying the minimum transaction filter.")
    st.plotly_chart(apply_plot_style(fig_top), use_container_width=True)

    st.subheader("Fraud Rate Per Device")
    fig_dist = px.histogram(device_stats, x="fraud_rate", nbins=40, color_discrete_sequence=[PRIMARY_COLOR])
    render_chart_note("X axis is device fraud rate; Y axis is number of devices in that range.")
    st.plotly_chart(apply_plot_style(fig_dist), use_container_width=True)


def page_ieee_card_user_risk(df: pd.DataFrame) -> None:
    min_tx = st.slider("Min transactions per card key", min_value=20, max_value=1000, value=100, step=20)
    temp = df.copy()
    temp["card_key"] = temp["card1"].astype(str) + "_" + temp["card2"].fillna(-1).astype(int).astype(str)

    card_stats = (
        temp.groupby("card_key", as_index=False)
        .agg(tx_count=("isFraud", "size"), fraud_count=("isFraud", "sum"), fraud_rate=("isFraud", "mean"))
    )
    card_stats = card_stats[card_stats["tx_count"] >= min_tx]
    top_cards = card_stats.sort_values("fraud_rate", ascending=False).head(20)

    st.subheader("High-Risk Cards")
    fig_cards = px.bar(
        top_cards,
        x="card_key",
        y="fraud_rate",
        hover_data=["tx_count", "fraud_count"],
        color_discrete_sequence=[DANGER_COLOR],
    )
    render_chart_note("Bars show card groups with the highest fraud rate.")
    st.plotly_chart(apply_plot_style(fig_cards), use_container_width=True)

    st.subheader("Fraud Distribution")
    dist = df["isFraud"].map({0: "Not Fraud", 1: "Fraud"}).value_counts().reset_index()
    dist.columns = ["Class", "Count"]
    fig_dist = px.pie(
        dist,
        names="Class",
        values="Count",
        color="Class",
        color_discrete_map={"Fraud": DANGER_COLOR, "Not Fraud": SUCCESS_COLOR},
        hole=0.45,
    )
    render_chart_note("Shows class balance after filters are applied.")
    st.plotly_chart(apply_plot_style(fig_dist), use_container_width=True)


def _amount_risk_score(amount: pd.Series) -> pd.Series:
    z = (amount - amount.median()) / (amount.std() + 1e-9)
    z = z.clip(-3, 3)
    return (z - z.min()) / (z.max() - z.min() + 1e-9)


def page_ieee_alerts(df: pd.DataFrame, val_df: pd.DataFrame | None) -> None:
    work = df.copy()

    if val_df is not None and {"TransactionID", "risk_score"}.issubset(val_df.columns):
        vf = val_df[["TransactionID", "risk_score"]].copy()
        work = work.merge(vf, on="TransactionID", how="left")
        if work["risk_score"].notna().sum() == 0 and len(vf) > 0:
            vtid = pd.to_numeric(vf["TransactionID"], errors="coerce")
            if vtid.notna().all() and vtid.max() < work["TransactionID"].min():
                base = load_ieee_data().reset_index(drop=True)
                pos = vtid.astype(int).clip(lower=0, upper=len(base) - 1)
                vf = vf.copy()
                vf["TransactionID"] = base.iloc[pos]["TransactionID"].values
                work = df.copy()
                work = work.merge(vf[["TransactionID", "risk_score"]], on="TransactionID", how="left")
                st.warning("Prediction file used row indices; remapped using training row order.")
        if work["risk_score"].isna().any():
            m = work["risk_score"].isna()
            work.loc[m, "risk_score"] = _amount_risk_score(work.loc[m, "TransactionAmt"])
    else:
        work["risk_score"] = _amount_risk_score(work["TransactionAmt"])

    work["risk_level"] = pd.cut(work["risk_score"], bins=[-1, 0.33, 0.66, 2], labels=["Low", "Medium", "High"])
    level = st.selectbox("Risk level", ["All", "High", "Medium", "Low"])
    if level != "All":
        work = work[work["risk_level"] == level]

    st.subheader("Flagged Transactions")
    cols = ["TransactionID", "TransactionAmt", "isFraud", "DeviceInfo", "card1", "card2", "risk_score", "risk_level"]
    display_df = work[cols].sort_values("risk_score", ascending=False).head(1000)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=460,
        hide_index=True,
        column_config={
            "TransactionAmt": st.column_config.NumberColumn("Transaction Amount", format="$%.2f"),
            "risk_score": st.column_config.NumberColumn("Risk Score", format="%.3f"),
        },
    )


def _ieee_model_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["XGBoost Final", 0.9630, 0.41, 0.83, 0.55],
            ["XGBoost PR Variant", np.nan, 0.10, 0.95, 0.18],
            ["LightGBM", np.nan, 0.23, 0.88, 0.36],
        ],
        columns=["Model", "ROC-AUC", "Precision", "Recall", "F1"],
    )


def page_ieee_model_performance(val_df: pd.DataFrame | None) -> None:
    has_live = val_df is not None and {"y_true", "y_pred", "risk_score"}.issubset(val_df.columns)
    comp = _ieee_model_comparison_df()
    st.subheader("Model Comparison")
    st.dataframe(comp, use_container_width=True, hide_index=True)

    fig0 = px.bar(
        comp,
        x="Model",
        y=["Precision", "Recall", "F1"],
        barmode="group",
        title="IEEE-CIS Model Comparison",
        color_discrete_sequence=[PRIMARY_COLOR, SUCCESS_COLOR, DANGER_COLOR],
    )
    render_chart_note("Compare precision, recall, and F1 to understand the alert trade-off.")
    st.plotly_chart(apply_plot_style(fig0), use_container_width=True)

    if not has_live:
        st.info("Run the IEEE-CIS export cell to create validation predictions for live CM and ROC views.")
        return

    y_true = val_df["y_true"].astype(int)
    y_pred = val_df["y_pred"].astype(int)
    y_prob = val_df["risk_score"].astype(float)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            title="IEEE-CIS Validation Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale="Blues",
        )
        render_chart_note("Cells show correct and wrong predictions on the validation data.")
        st.plotly_chart(apply_plot_style(fig_cm), use_container_width=True)

    with col_b:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.4f})", line=dict(color=PRIMARY_COLOR, width=3))
        )
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(color=NEUTRAL_COLOR, dash="dash")))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", title="IEEE-CIS ROC Curve")
        render_chart_note("Curve closer to the top-left means better ranking quality.")
        st.plotly_chart(apply_plot_style(fig), use_container_width=True)


def page_synthetic_overview(df: pd.DataFrame) -> None:
    total_tx = len(df)
    fraud_count = int(df["isFraud"].sum())
    flagged_count = int(df["isFlaggedFraud"].sum())
    fraud_rate = (df["isFraud"].mean() * 100.0) if total_tx else 0.0
    total_amount = float(df["amount"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi("Total Transactions", f"{total_tx:,}")
    with c2:
        render_kpi("Fraud Count", f"{fraud_count:,}")
    with c3:
        render_kpi("Fraud Rate", f"{fraud_rate:.4f}%")
    with c4:
        render_kpi("Flagged by Rule", f"{flagged_count:,}")
    with c5:
        render_kpi("Total Amount", f"${total_amount:,.0f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fraud Distribution")
        fraud_share = df["isFraud"].map({0: "Not Fraud", 1: "Fraud"}).value_counts().reset_index()
        fraud_share.columns = ["Class", "Count"]
        fig = px.pie(
            fraud_share,
            names="Class",
            values="Count",
            color="Class",
            color_discrete_map={"Fraud": DANGER_COLOR, "Not Fraud": SUCCESS_COLOR},
            hole=0.45,
        )
        render_chart_note("Shows the extreme class imbalance in the Synthetic Financial data.")
        st.plotly_chart(apply_plot_style(fig), use_container_width=True)

    with col_b:
        st.subheader("Transaction Volume by Type")
        type_counts = df["type"].astype(str).value_counts().reset_index()
        type_counts.columns = ["Type", "Transactions"]
        fig = px.bar(type_counts, x="Type", y="Transactions", color_discrete_sequence=[PRIMARY_COLOR])
        render_chart_note("Bars compare the number of transactions in each transaction type.")
        st.plotly_chart(apply_plot_style(fig), use_container_width=True)


def page_synthetic_type_risk(df: pd.DataFrame) -> None:
    min_tx = st.slider("Min transactions per type", min_value=1, max_value=1000, value=1, step=1)
    stats = (
        df.groupby("type", observed=False)
        .agg(
            tx_count=("isFraud", "size"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )
    stats = stats[stats["tx_count"] >= min_tx].sort_values("fraud_rate", ascending=False)

    st.subheader("Transaction-Type Risk Table")
    display = stats.copy()
    display["fraud_rate_%"] = display["fraud_rate"] * 100
    st.dataframe(
        display[["type", "tx_count", "fraud_count", "fraud_rate_%", "avg_amount", "total_amount"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_rate_%": st.column_config.NumberColumn("Fraud Rate", format="%.4f%%"),
            "avg_amount": st.column_config.NumberColumn("Average Amount", format="$%.2f"),
            "total_amount": st.column_config.NumberColumn("Total Amount", format="$%.0f"),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_rate = px.bar(
            stats,
            x="type",
            y="fraud_rate",
            hover_data=["tx_count", "fraud_count"],
            title="Fraud Rate by Transaction Type",
            color_discrete_sequence=[DANGER_COLOR],
        )
        render_chart_note("Taller bars mean a larger share of that type is fraudulent.")
        st.plotly_chart(apply_plot_style(fig_rate), use_container_width=True)

    with col_b:
        type_class = df.groupby(["type", "isFraud"], observed=False).size().reset_index(name="count")
        type_class["Class"] = type_class["isFraud"].map({0: "Not Fraud", 1: "Fraud"})
        fig_count = px.bar(
            type_class,
            x="type",
            y="count",
            color="Class",
            barmode="group",
            title="Fraud vs Not Fraud by Type",
            color_discrete_map={"Fraud": DANGER_COLOR, "Not Fraud": SUCCESS_COLOR},
        )
        render_chart_note("Grouped bars show how rare fraud is inside each transaction type.")
        st.plotly_chart(apply_plot_style(fig_count), use_container_width=True)


def page_synthetic_amount_balance(df: pd.DataFrame) -> None:
    sample = sample_for_plot(df, 30_000)
    summary = (
        df.groupby("isFraud")
        .agg(
            tx_count=("isFraud", "size"),
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            avg_origin_delta=("origin_balance_delta", "mean"),
            avg_destination_delta=("destination_balance_delta", "mean"),
        )
        .reset_index()
    )
    summary["Class"] = summary["isFraud"].map({0: "Not Fraud", 1: "Fraud"})

    st.subheader("Amount and Balance Summary")
    st.dataframe(
        summary[["Class", "tx_count", "avg_amount", "median_amount", "avg_origin_delta", "avg_destination_delta"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "avg_amount": st.column_config.NumberColumn("Average Amount", format="$%.2f"),
            "median_amount": st.column_config.NumberColumn("Median Amount", format="$%.2f"),
            "avg_origin_delta": st.column_config.NumberColumn("Avg Origin Balance Change", format="$%.2f"),
            "avg_destination_delta": st.column_config.NumberColumn("Avg Destination Balance Change", format="$%.2f"),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_amount = px.box(
            sample,
            x="isFraud",
            y="amount_log",
            color="isFraud",
            title="Log Amount by Fraud Class",
            labels={"isFraud": "Fraud (0/1)", "amount_log": "log1p(amount)"},
            color_discrete_map={0: SUCCESS_COLOR, 1: DANGER_COLOR},
        )
        render_chart_note("Fraud often appears in different amount ranges; log scale keeps large values readable.")
        st.plotly_chart(apply_plot_style(fig_amount), use_container_width=True)

    with col_b:
        fig_delta = px.scatter(
            sample,
            x="amount",
            y="origin_balance_delta",
            color="isFraud",
            opacity=0.45,
            title="Amount vs Origin Balance Change",
            labels={"origin_balance_delta": "Old Origin Balance - New Origin Balance"},
            color_discrete_map={0: SUCCESS_COLOR, 1: DANGER_COLOR},
        )
        render_chart_note("Fraudulent transfers often move the full origin balance out of the account.")
        st.plotly_chart(apply_plot_style(fig_delta), use_container_width=True)

    st.subheader("Destination Balance Movement")
    fig_dest = px.scatter(
        sample,
        x="amount",
        y="destination_balance_delta",
        color="type",
        opacity=0.5,
        title="Amount vs Destination Balance Change by Type",
    )
    render_chart_note("Compares the transaction amount with how much the destination balance changed.")
    st.plotly_chart(apply_plot_style(fig_dest), use_container_width=True)


def page_synthetic_model_results() -> None:
    metrics = pd.DataFrame(
        [
            ["PR-AUC", 0.9651, "Full precision-recall ranking quality"],
            ["Fraud Precision", 0.84, "Share of fraud alerts that were truly fraud"],
            ["Fraud Recall", 0.93, "Share of real fraud cases caught"],
            ["Fraud F1-score", 0.88, "Balance between fraud precision and recall"],
        ],
        columns=["Metric", "Value", "Meaning"],
    )
    class_report = pd.DataFrame(
        [
            ["Not Fraud", 1.00, 1.00, 1.00, 1_270_881],
            ["Fraud", 0.84, 0.93, 0.88, 1_643],
            ["Macro Avg", 0.92, 0.97, 0.94, 1_272_524],
            ["Weighted Avg", 1.00, 1.00, 1.00, 1_272_524],
        ],
        columns=["Class", "Precision", "Recall", "F1-score", "Support"],
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi("Model", "XGBoost")
    with c2:
        render_kpi("PR-AUC", "0.9651")
    with c3:
        render_kpi("Fraud Precision", "0.84")
    with c4:
        render_kpi("Fraud Recall", "0.93")
    with c5:
        render_kpi("Threshold", "0.971")

    st.subheader("Validation Metrics")
    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        color="Metric",
        title="Synthetic Financial XGBoost Results",
        color_discrete_sequence=[PRIMARY_COLOR, SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR],
        range_y=[0, 1],
    )
    render_chart_note("Higher values mean better fraud detection performance for the selected metric.")
    st.plotly_chart(apply_plot_style(fig), use_container_width=True)

    st.subheader("Classification Report")
    st.dataframe(class_report, use_container_width=True, hide_index=True)

    st.subheader("Model Artifacts and Features")
    features = load_synthetic_feature_list()
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("Model Artifact", "Available" if SYNTHETIC_MODEL_FILE.exists() else "Missing")
    with c2:
        render_kpi("Feature Count", str(len(features)))
    with c3:
        render_kpi("Validation Support", "1,272,524")

    if features:
        feature_df = pd.DataFrame({"Expected XGBoost Feature": features})
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Feature list not found at {SYNTHETIC_FEATURE_FILE}")


def render_ieee_dashboard(page: str) -> None:
    if not IEEE_DATA_DIR.exists():
        st.error(f"Data folder not found. Expected IEEE-CIS files at: {IEEE_DATA_DIR}")
        return
    df = load_ieee_data()
    val_df = load_ieee_validation_artifacts()
    filtered_df = add_amount_filter(df, "TransactionAmt")

    if page == "Overview":
        page_ieee_overview(filtered_df)
    elif page == "Fraud Monitoring":
        page_ieee_fraud_monitoring(filtered_df)
    elif page == "Device Risk":
        page_ieee_device_risk(filtered_df)
    elif page == "Card / User Risk":
        page_ieee_card_user_risk(filtered_df)
    elif page == "Alerts":
        page_ieee_alerts(filtered_df, val_df)
    elif page == "Model Performance":
        page_ieee_model_performance(val_df)


def render_synthetic_dashboard(page: str) -> None:
    data_file = SYNTHETIC_DATA_FILE if SYNTHETIC_DATA_FILE.exists() else SYNTHETIC_FALLBACK_FILE
    if not data_file.exists():
        st.error(f"Synthetic Financial data not found. Expected: {SYNTHETIC_DATA_FILE}")
        return
    df = load_synthetic_data()
    filtered_df = add_synthetic_filters(df)

    if page == "Overview":
        page_synthetic_overview(filtered_df)
    elif page == "Transaction Type Risk":
        page_synthetic_type_risk(filtered_df)
    elif page == "Amount / Balance Analysis":
        page_synthetic_amount_balance(filtered_df)
    elif page == "Synthetic XGBoost Results":
        page_synthetic_model_results()


def main() -> None:
    apply_custom_style()
    st.title("Fraud Detection Dashboard")
    st.markdown(
        '<div class="page-subtitle">Interactive monitoring for IEEE-CIS and Synthetic Financial fraud analytics.</div>',
        unsafe_allow_html=True,
    )

    dataset = st.sidebar.selectbox("Dataset", ["IEEE-CIS", "Synthetic Financial"])
    if dataset == "IEEE-CIS":
        pages = ["Overview", "Fraud Monitoring", "Device Risk", "Card / User Risk", "Alerts", "Model Performance"]
    else:
        pages = ["Overview", "Transaction Type Risk", "Amount / Balance Analysis", "Synthetic XGBoost Results"]

    page = st.sidebar.radio("Pages", pages)

    with st.sidebar.expander("Dashboard guide", expanded=False):
        st.markdown(
            "- Choose a dataset first.\n"
            "- Apply filters before reading the charts.\n"
            "- Use model pages to compare detection quality and alert thresholds."
        )

    render_stakeholder_note(page)

    if dataset == "IEEE-CIS":
        render_ieee_dashboard(page)
    else:
        render_synthetic_dashboard(page)


if __name__ == "__main__":
    main()

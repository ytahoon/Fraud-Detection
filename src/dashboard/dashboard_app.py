"""Streamlit views for dataset research and stored TiDB demo scores."""

from html import escape
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.dashboard_data import MODEL_NAMES, read_release, read_snapshot
from src.database.connection import load_demo_settings, open_engine
from src.database.demo_queries import DATASETS, REJECTION_REASONS, SCORING_STATUSES, fetch_demo_transactions


st.set_page_config(page_title="Fraud Detection | Risk Analytics", layout="wide")

TEAL = "#17796D"
RED = "#BB4F48"
GOLD = "#AC6D1C"
INK = "#243330"
MUTED = "#596C65"
GRID = "#DDE7E1"
CLASSES = {"Not fraud": TEAL, "Fraud": RED}
SNAPSHOT_ERRORS = (OSError, ValueError, KeyError, TypeError)


def apply_style():
    st.markdown("""
        <style>
          [data-testid="stDecoration"], [data-testid="stAppDeployButton"] { display: none; }
          [data-testid="stMainBlockContainer"] { max-width: 90rem; padding-top: 2.6rem; padding-bottom: 2rem; }
          [data-testid="stSidebar"] { border-right: 1px solid #DDE7E1; }
          [data-testid="stSidebarUserContent"] { padding-top: 2rem; }
          [data-testid="stMetric"] { background: #FFF; border-color: #DDE7E1; border-radius: 8px; }
          [data-testid="stMetricValue"] { font-size: 1.38rem; }
          [data-testid="stMetricValue"] > div { overflow-wrap: anywhere; white-space: normal; }
          .risk-brand { color: #164E44; font-weight: 750; font-size: 1.08rem; padding-bottom: .2rem; }
          .risk-kicker { color: #22695F; font-size: .76rem; font-weight: 700; text-transform: uppercase; }
          .risk-header { border-bottom: 1px solid #DDE7E1; padding-bottom: .85rem; }
          .risk-header h1 { color: #172D27; font-size: 2rem; padding: .18rem 0 .1rem; margin: 0; }
          .risk-header p { color: #526761; font-size: .92rem; margin: 0; }
          @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] { padding-top: 3.1rem; }
            .risk-header h1 { font-size: 1.6rem; }
            .st-key-kpi-strip [data-testid="stHorizontalBlock"] {
              display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem;
            }
            .st-key-kpi-strip [data-testid="stColumn"] { width: auto !important; min-width: 0; }
          }
        </style>
        """, unsafe_allow_html=True)


def header(dataset, scope, subtitle):
    st.markdown(
        f'<div class="risk-header"><div class="risk-kicker">{escape(scope)}</div>'
        f'<h1>{escape(DATASETS[dataset])}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True,
    )


def metric_row(items):
    with st.container(key="kpi-strip"):
        for column, (label, value, help_text) in zip(st.columns(len(items)), items):
            column.metric(label, value, border=True, help=help_text)


def chart(fig, *, height=290, legend=False):
    fig.update_layout(
        height=height, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=INK), margin=dict(l=10, r=14, t=35 if legend else 12, b=12),
        showlegend=legend, legend=dict(orientation="h", x=0, y=1.04, yanchor="bottom", title_text=""),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=INK),
    )
    fig.update_xaxes(automargin=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(automargin=True, gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, theme=None, use_container_width=True, config={"displaylogo": False})


def table(frame, formats=None, height=None):
    config = {name: st.column_config.NumberColumn(format=fmt) for name, fmt in (formats or {}).items()}
    st.dataframe(frame, hide_index=True, use_container_width=True, column_config=config, height=height)


def segment_frame(items):
    frame = pd.DataFrame(items).rename(columns={"group": "Segment", "rows": "Transactions", "fraud": "Fraud"})
    frame["Not fraud"] = frame["Transactions"] - frame["Fraud"]
    frame["Fraud rate"] = frame["Fraud"] / frame["Transactions"]
    return frame


def download_table(frame, name):
    st.download_button("Download summary CSV", frame.to_csv(index=False), f"{name}.csv", "text/csv",
                       icon=":material/download:")


def page_overview(dataset, snapshot):
    rows, fraud = snapshot["rows"], snapshot["fraud"]
    metric_row([
        ("Transactions", f"{rows:,}", "All labeled transactions in the source dataset."),
        ("Labeled fraud", f"{fraud:,}", "Transactions marked as fraud in the source data."),
        ("Fraud rate", f"{fraud / rows:.3%}", "Labeled fraud as a share of all transactions."),
        ("Median amount", f"{snapshot['median_amount']:,.2f}", "In source units."),
    ])
    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("Class distribution")
        class_fig = go.Figure(go.Scatter(
            x=[rows - fraud, fraud], y=list(CLASSES), mode="markers",
            marker=dict(size=17, color=list(CLASSES.values()), line=dict(width=2, color="#FFFFFF")),
            text=[f"{rows - fraud:,}  ({1 - fraud / rows:.3%})", f"{fraud:,}  ({fraud / rows:.3%})"],
            hovertemplate="%{y}<br>%{text}<extra></extra>",
        ))
        class_fig.update_xaxes(type="log", title="Transactions (log scale)", range=[3, 7],
                               tickvals=[1_000, 10_000, 100_000, 1_000_000, 10_000_000],
                               ticktext=["1k", "10k", "100k", "1M", "10M"])
        class_fig.update_yaxes(autorange="reversed")
        chart(class_fig, height=264)
        st.caption(f"{rows - fraud:,} not fraud  |  {fraud:,} fraud")
    with right:
        title = "Transaction type" if dataset == "synthetic_financial" else "Product code"
        st.subheader(f"Volume by {title.lower()}")
        segments = segment_frame(snapshot["segments"][title]).sort_values("Transactions")
        fig = px.bar(segments, x="Transactions", y="Segment", orientation="h", color_discrete_sequence=[TEAL],
                     hover_data={"Fraud": ":,d", "Fraud rate": ":.3%"})
        fig.update_yaxes(title="")
        chart(fig, height=264)
        st.caption(f"Total amount: {snapshot['total_amount']:,.2f} in source units.")

    st.subheader("Daily activity")
    options = {"Volume": ("Transactions", TEAL), "Fraud count": ("Fraud", RED), "Fraud rate": ("Fraud rate", GOLD)}
    selection = st.radio("Daily measure", list(options), index=1, horizontal=True, label_visibility="collapsed")
    measure, color = options[selection]
    daily = segment_frame(snapshot["by_day"]).rename(columns={"Segment": "Relative day"})
    figure = px.line(daily, x="Relative day", y=measure, color_discrete_sequence=[color],
                     hover_data={"Transactions": ":,d", "Fraud": ":,d", "Fraud rate": ":.3%"})
    if measure == "Fraud rate":
        figure.update_yaxes(tickformat=".1%")
    chart(figure)
    if dataset == "synthetic_financial":
        st.caption(f"Simulated days. The dataset's original rule flagged {snapshot['flagged_by_rule']:,} transactions.")
    else:
        st.caption("Days are relative to the competition dataset's time origin, not calendar dates.")
    with st.expander("Daily counts and fraud rates"):
        table(daily, {"Fraud rate": "percent"}, height=265)
        download_table(daily, f"{dataset}-daily-summary")


def page_segments(dataset, snapshot):
    st.subheader("Observed risk by segment")
    groups = snapshot["segments"]
    segment = st.selectbox("Group by", list(groups)) if len(groups) > 1 else next(iter(groups))
    frame = segment_frame(groups[segment]).sort_values("Fraud rate", ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown("**Fraud rate**")
        fig = px.bar(frame[::-1], x="Fraud rate", y="Segment", orientation="h", color_discrete_sequence=[RED],
                     hover_data={"Transactions": ":,d", "Fraud": ":,d", "Fraud rate": ":.3%"})
        fig.update_xaxes(tickformat=".2%")
        fig.update_yaxes(title="")
        chart(fig)
    with right:
        st.markdown("**Fraud count**")
        fig = px.bar(frame[::-1], x="Fraud", y="Segment", orientation="h", color_discrete_sequence=[GOLD],
                     hover_data={"Transactions": ":,d", "Fraud rate": ":.3%"})
        fig.update_yaxes(title="")
        chart(fig)
    table(frame, {"Fraud rate": "percent"})
    if segment in {"Device type", "Card network", "Card type"}:
        st.caption("Unknown includes transactions with no recorded value.")
    download_table(frame, f"{dataset}-segment-summary")


def page_amounts(dataset, snapshot):
    st.subheader("Distributions by fraud class")
    groups = snapshot["distributions"]
    selected = st.selectbox("Measure", list(groups)) if len(groups) > 1 else next(iter(groups))
    mode = st.radio("Distribution scale", ["Within-class %", "Counts"], horizontal=True)
    group = groups[selected]
    counts = pd.DataFrame(group["bins"]).rename(columns={"range": "Range", "not_fraud": "Not fraud", "fraud": "Fraud"})
    shares = counts.copy()
    figure = go.Figure()
    for label, color in CLASSES.items():
        shares[label] = counts[label] / counts[label].sum()
        pct = mode != "Counts"
        figure.add_bar(x=counts["Range"], y=shares[label] if pct else counts[label], name=label, marker_color=color,
                       customdata=np.column_stack([counts[label], shares[label]]),
                       hovertemplate="%{x}<br>%{customdata[0]:,.0f} transactions<br>%{customdata[1]:.3%} of class")
    figure.update_layout(barmode="group", xaxis_title=f"{selected} (source units)",
                         yaxis_title="Share of class" if mode != "Counts" else "Transactions")
    if mode != "Counts":
        figure.update_yaxes(tickformat=".0%")
    chart(figure, height=335, legend=True)
    if "balance" in selected:
        st.caption("Change = balance after minus balance before. These post-transaction values are not scoring inputs.")
    st.subheader("Range and outliers")
    stats = pd.DataFrame(group["summary"]).rename(columns={"class": "Class", "min": "Minimum", "max": "Maximum",
                                                           "mean": "Mean", "median": "Median", "p95": "95th percentile"})
    stats = stats[["Class", "Minimum", "Median", "Mean", "95th percentile", "Maximum"]]
    table(stats, {col: "localized" for col in stats.columns if col != "Class"})
    with st.expander("Exact distribution counts"):
        table(counts)
        download_table(counts, f"{dataset}-distribution")


def confusion_chart(result, *, normalized):
    counts = np.array([[result["true_negatives"], result["false_alerts"]],
                       [result["missed_fraud"], result["caught_fraud"]]])
    shares = counts / counts.sum(axis=1, keepdims=True)
    labels = [[f"{value:.1%}" if normalized else f"{value:,}" for value in row]
              for row in (shares if normalized else counts)]
    figure = go.Figure(go.Heatmap(
        z=shares, x=["No alert", "Review"], y=["Not fraud", "Fraud"], text=labels, texttemplate="%{text}",
        textfont=dict(size=18, color=INK), customdata=counts, xgap=3, ygap=3, showscale=False, zmin=0, zmax=1,
        colorscale=[[0, "#ECF4F0"], [1, "#83B9A8"]],
        hovertemplate="Actual: %{y}<br>Decision: %{x}<br>%{customdata:,} transactions<br>%{z:.2%} of actual class<extra></extra>",
    ))
    figure.update_xaxes(title="Model decision", side="bottom", showgrid=False)
    figure.update_yaxes(title="Actual label", autorange="reversed", showgrid=False)
    return figure


def model_curve(curves, kind, result):
    pr = kind == "Precision-recall"
    x, y = zip(*curves["pr" if pr else "roc"])
    base = result["fraud_rows"] / result["rows"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[base, base] if pr else [0, 1], mode="lines", name="No-skill baseline",
                    line=dict(color=MUTED, dash="dash", width=1.5), hoverinfo="skip")
    fig.add_scatter(x=x, y=y, mode="lines", name=kind, line=dict(color=TEAL, width=3),
                    hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>")
    fig.update_xaxes(range=[0, 1.01], tickformat=".0%", title="Recall" if pr else "False positive rate")
    fig.update_yaxes(range=[0, 1.03], tickformat=".0%", title="Precision" if pr else "True positive rate")
    return fig


def page_model(dataset, snapshot, manifest, report):
    model_name = MODEL_NAMES[manifest.get("algorithm", "xgboost")]
    st.subheader(f"{model_name} evaluation")
    split_name = st.radio("Evaluation split", ["Test holdout", "Validation"], horizontal=True)
    split = "validation" if split_name == "Validation" else "test"
    result = report[split]
    threshold = report["threshold"]
    budget = report["threshold_selection"]
    st.caption(f"{manifest['model_version']}  |  {result['rows']:,} rows, {result['fraud_rows']:,} fraud  |  Review at score \u2265 {threshold:.6f}")
    metric_row([
        ("Accuracy", f"{(result['true_negatives'] + result['caught_fraud']) / result['rows']:.2%}",
         "All correct decisions. This can be high when most transactions are not fraud."),
        ("Fraud precision", f"{result['precision']:.2%}", "Share of review alerts that actually were fraud."),
        ("Fraud recall", f"{result['recall']:.2%}", "Share of known fraud cases sent for review."),
        ("Fraud F1", f"{result['f1']:.2%}", "Harmonic mean of fraud precision and recall."),
    ])
    st.markdown(f"**{result['caught_fraud']:,} fraud cases caught**, {result['missed_fraud']:,} missed, and "
                f"{result['false_alerts']:,} false alerts per {result['rows']:,} evaluated transactions.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        mode = st.radio("Matrix values", ["Count", "Row %"], horizontal=True, label_visibility="collapsed")
        chart(confusion_chart(result, normalized=mode == "Row %"), height=287)
    with right:
        st.markdown("**Ranking curves**")
        if snapshot:
            curve = st.radio("Curve", ["Precision-recall", "ROC"], horizontal=True, label_visibility="collapsed")
            chart(model_curve(snapshot["curves"][split], curve, result), height=287, legend=True)
        else:
            st.info("Curve points are unavailable. The confusion matrix and report metrics are still shown.")

    st.subheader("Scores and review load")
    metrics = [
        ("Average precision", "average_precision", "Precision weighted by changes in recall."),
        ("PR-AUC (trapezoidal)", "pr_auc_trapezoidal", "Area under the precision-recall curve."),
        ("ROC-AUC", "roc_auc", "How well fraud ranks above non-fraud."),
        ("Alerts per 1,000", "alerts_per_1000", "Review workload at the frozen threshold."),
    ]
    ranking = snapshot["curves"][split]["ranking"] if snapshot else {}
    display = {**ranking, **result}
    table(pd.DataFrame([{"Metric": title, "Value": f"{display[key]:.4f}", "Meaning": meaning}
                        for title, key, meaning in metrics if key in display]))
    st.caption(f"Threshold selected on validation with a target of {budget.get('alerts_per_1000_target', budget.get('max_alerts_per_1000'))} alerts per 1,000. "
               "The threshold is unchanged on the later test holdout.")
    if split == "test":
        baselines = report.get("baselines", report)
        group_ap = baselines.get("test_type_baseline_average_precision", baselines.get("test_product_only_average_precision"))
        st.caption(f"Holdout AP baselines: {baselines['test_no_skill_average_precision']:.4f} for random ranking; "
                   f"{group_ap:.4f} for ranking by segment fraud rate alone.")
        with st.expander("Holdout performance by segment"):
            by_segment = report.get("test_by_transaction_type", report.get("test_by_product"))
            table(pd.DataFrame([
                {"Segment": key, "Transactions": row["rows"], "Fraud": row["fraud_rows"], "Alerts": row["alerts"],
                 "Caught": row["caught_fraud"], "Missed": row["missed_fraud"],
                 "Precision": row["precision"] if row["alerts"] else None,
                 "Recall": row["recall"] if row["fraud_rows"] else None}
                for key, row in by_segment.items()
            ]), {"Precision": "percent", "Recall": "percent"})
    elif sensitivity := report.get("validation_budget_sensitivity"):
        with st.expander("Validation review budget sensitivity"):
            table(pd.DataFrame([{"Budget per 1,000": int(key), "Threshold": row["threshold"],
                                 "Actual alerts per 1,000": row["alerts_per_1000"], "Precision": row["precision"],
                                 "Recall": row["recall"]} for key, row in sensitivity.items()]),
                  {"Threshold": "%.6f", "Actual alerts per 1,000": "%.2f", "Precision": "percent", "Recall": "percent"})
    if comparison := report.get("validation_model_comparison"):
        with st.expander("Model comparison on validation"):
            table(pd.DataFrame([{"Model": MODEL_NAMES[key], "Average precision": row["average_precision"],
                                 "PR-AUC": row["pr_auc_trapezoidal"], "ROC-AUC": row["roc_auc"]}
                                for key, row in comparison.items()]),
                  {"Average precision": "%.4f", "PR-AUC": "%.4f", "ROC-AUC": "%.4f"})
    with st.expander("Data splits and evaluation limits"):
        table(pd.DataFrame([{"Split": name.title(), "Transactions": part["rows"], "Fraud": part["fraud_rows"]}
                            for name, part in report["windows"].items()]))
        for limitation in report.get("production_limitations", report.get("limitations", [])):
            st.caption(limitation)
        if dataset == "ieee_cis":
            st.caption("TiDB Demo uses a separate LightGBM fixture trained on fabricated data. These research metrics do not describe that fixture.")


@st.cache_resource
def tidb_demo_engine(settings):
    return open_engine(settings)


def page_tidb_demo(dataset):
    header(dataset, "Fraud detection / TiDB demo", "Stored scores for fabricated demo transactions")
    id_column, status_column, action_column = st.columns([2, 2, 1], vertical_alignment="bottom")
    with id_column:
        transaction_id = st.text_input("Transaction ID", placeholder="DEMO-001", max_chars=64).strip()
    with status_column:
        selected = st.selectbox("Scoring status", [None, *SCORING_STATUSES],
                                format_func=lambda value: SCORING_STATUSES[value] if value else "All statuses")
    with action_column:
        st.button("Refresh", icon=":material/refresh:")

    try:
        rows = fetch_demo_transactions(tidb_demo_engine(load_demo_settings()), dataset,
                                       transaction_id=transaction_id or None, status=selected)
    except (ValueError, SQLAlchemyError):
        st.warning("TiDB is unavailable. Research Analytics is available without the database.")
        return
    if not rows:
        st.info("No transactions match these filters.")
        return

    frame = pd.DataFrame(rows)
    counts = frame["scoring_status"].value_counts()
    metric_row([
        ("Shown", f"{len(frame):,}", "Stored demo transactions matching the current filters, up to 100."),
        ("Review", str(counts.get("review", 0)), "Current score meets the threshold for review."),
        ("No alert", str(counts.get("no_alert", 0)), "Current score falls below the review threshold."),
        ("Unscored", str(counts.get("quarantined", 0) + counts.get("not_scored", 0)), "Includes quarantined inputs."),
    ])
    frame["scoring_status"] = frame["scoring_status"].map(SCORING_STATUSES)
    for column in ("amount", "risk_score", "threshold"):
        frame[column] = pd.to_numeric(frame[column])
    frame["fraud_label"] = frame["fraud_label"].map({"fraud": "Fraud", "not_fraud": "Not fraud"}).fillna("Unlabeled")
    frame["rejection_reason"] = frame["rejection_reason"].map(REJECTION_REASONS)
    columns = ["transaction_id", "scoring_status", "risk_score", "threshold", "model_version", "fraud_label",
               "amount", "currency", "transaction_type", "occurred_at"]
    if frame["rejection_reason"].notna().any():
        columns.insert(2, "rejection_reason")
    st.subheader("Transaction scores")
    st.dataframe(frame[columns], hide_index=True, use_container_width=True, column_config={
        "transaction_id": "Transaction ID", "occurred_at": "Occurred at (UTC)",
        "transaction_type": "Product code" if dataset == "ieee_cis" else "Transaction type",
        "currency": "Currency", "amount": st.column_config.NumberColumn("Amount", format="%.4f"),
        "risk_score": st.column_config.NumberColumn("Stored score", format="%.6f"),
        "threshold": st.column_config.NumberColumn("Review threshold", format="%.6f"),
        "model_version": "Model version", "scoring_status": "Decision", "fraud_label": "Demo label",
        "rejection_reason": "Input issue",
    })
    if len(frame) == 100:
        st.caption("Showing the first 100 matches.")


def main():
    apply_style()
    st.sidebar.markdown('<div class="risk-brand">Fraud Detection</div>', unsafe_allow_html=True)
    st.sidebar.caption("RISK ANALYTICS")
    st.sidebar.divider()
    view = st.sidebar.radio("View", ["TiDB Demo", "Research Analytics"])
    dataset = st.sidebar.selectbox("Dataset", list(DATASETS), format_func=DATASETS.__getitem__)
    if view == "TiDB Demo":
        page_tidb_demo(dataset)
        return

    page = st.sidebar.radio("Analysis", ["Overview", "Segment Risk", "Distributions", "Model Performance"])
    header(dataset, "Fraud detection / Research analytics", "PaySim payment simulation" if dataset == "synthetic_financial"
           else "Labeled competition transactions")
    try:
        manifest, report = read_release(dataset)
    except SNAPSHOT_ERRORS:
        st.warning("The research release report is unavailable or does not match its manifest.")
        return
    try:
        snapshot = read_snapshot(dataset, manifest, report)
    except SNAPSHOT_ERRORS:
        snapshot = None
    if page == "Model Performance":
        page_model(dataset, snapshot, manifest, report)
    elif not snapshot:
        st.warning("Dataset summaries are unavailable for this release. Model Performance still uses the verified report.")
    elif page == "Overview":
        page_overview(dataset, snapshot)
    elif page == "Segment Risk":
        page_segments(dataset, snapshot)
    else:
        page_amounts(dataset, snapshot)


if __name__ == "__main__":
    main()

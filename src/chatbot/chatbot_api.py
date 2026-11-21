"""Reply to transaction questions using stored TiDB demo scores."""

from datetime import timezone
from functools import lru_cache
import os
from pathlib import Path
import re
import sys

from flask import Flask, jsonify, make_response, render_template, request, url_for
from sqlalchemy.exc import SQLAlchemyError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import load_demo_settings, open_engine
from src.database.demo_queries import DATASETS, REJECTION_REASONS, SCORING_STATUSES, fetch_demo_transactions


DATASET_PATTERNS = {
    "synthetic_financial": r"\b(?:synthetic(?:[ _-]+financial)?|paysim)\b",
    "ieee_cis": r"\bieee(?:[ _-]+cis)?\b",
}
ID_PATTERN = re.compile(r"\b(?:DEMO-\d+|\d{7,64})\b", re.IGNORECASE)
EXPLICIT_ID = re.compile(r"\b(?:transaction[ _-]+id|tx[ _-]+id)\s*[:#=]?\s*([A-Za-z0-9][A-Za-z0-9_.-]{0,63})\b",
                         re.IGNORECASE)
LOOKUP_EXAMPLE = "Check DEMO-001 in Synthetic Financial"
app = Flask(__name__)


def matching_datasets(text):
    return [name for name, pattern in DATASET_PATTERNS.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def format_timestamp(value):
    if value is None:
        return None
    utc = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return utc.isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=1)
def tidb_demo_engine():
    return open_engine(load_demo_settings())


@app.get("/")
def index():
    return jsonify({
        "service": "fraud-detection-chatbot-api",
        "transaction_source": "tidb_demo",
        "datasets": DATASETS,
        "interface": url_for("chat_interface"),
        "endpoints": ["/chat", "/transactions/<dataset>/<transaction_id>"],
        "example": {"message": LOOKUP_EXAMPLE},
    })


@app.get("/chat")
def chat_interface():
    response = make_response(render_template("chat.html", datasets=DATASETS))
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return response


@app.get("/get_next_stream")
def retired_stream():
    return jsonify({"status": "retired", "reply": "Use a TiDB transaction lookup instead."}), 410


@app.get("/transactions/<dataset>/<transaction_id>")
def transaction_lookup(dataset, transaction_id):
    if isinstance(dataset, str):
        normalized = dataset.strip().lower().replace("-", "_").replace(" ", "_")
        dataset = {"paysim": "synthetic_financial", "synthetic": "synthetic_financial", "ieee": "ieee_cis"}.get(
            normalized, normalized
        )
    if not isinstance(dataset, str) or dataset not in DATASETS:
        return jsonify({"status": "invalid_request", "reply": "Select Synthetic Financial or IEEE-CIS."}), 400
    transaction_id = transaction_id.strip() if isinstance(transaction_id, str) else ""
    if not 1 <= len(transaction_id) <= 64 or not transaction_id.isprintable():
        return jsonify({"status": "invalid_request", "reply": "Provide a valid transaction ID."}), 400
    try:
        rows = fetch_demo_transactions(tidb_demo_engine(), dataset, transaction_id=transaction_id, limit=1)
    except (ValueError, SQLAlchemyError):
        return jsonify({"status": "unavailable", "reply": "TiDB scores are temporarily unavailable."}), 503
    if not rows:
        return jsonify({"status": "not_found", "reply": f"No transaction {transaction_id} in {DATASETS[dataset]}."}), 404

    row = dict(rows[0])
    row["amount"] = str(row["amount"])
    for name in ("occurred_at", "scored_at"):
        row[name] = format_timestamp(row[name])
    status = row["scoring_status"]
    if status == "quarantined":
        reason = REJECTION_REASONS.get(row["rejection_reason"], "Invalid input")
        details = f"Quarantined: {reason}. No model score is available."
    elif status == "not_scored":
        details = "No score has been stored for the current model."
    else:
        details = (f"Stored score: {row['risk_score']:.6f}. Threshold: {row['threshold']:.6f}. "
                   f"Model: {row['model_version']}. Recommendation: {SCORING_STATUSES[status]}.")
    return jsonify({"status": "scored" if row["decision"] is not None else status, "source": "tidb",
                    "transaction": row, "reply": f"{DATASETS[dataset]} transaction {transaction_id}. {details}"})


def general_reply(message):
    key, model_name = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL")
    if not key or not model_name:
        return jsonify({"status": "unavailable", "reply": "General Gemini answers are not configured."}), 503
    try:
        from google import genai
        from google.genai import types

        with genai.Client(api_key=key) as client:
            response = client.models.generate_content(
                model=model_name,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction="Answer general fraud-analysis questions. You have no transaction data or scores.",
                ),
            )
        return jsonify({"status": "general", "source": "gemini", "reply": response.text})
    except Exception:
        return jsonify({"status": "unavailable", "reply": "General Gemini answers are temporarily unavailable."}), 503


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "invalid_request", "reply": "Send a JSON object."}), 400

    if "dataset" in payload or "transaction_id" in payload:
        return transaction_lookup(payload.get("dataset"), payload.get("transaction_id"))

    message = payload.get("message", "")
    if not isinstance(message, str) or not message.strip() or len(message) > 2_000:
        return jsonify({"status": "invalid_request", "reply": "Enter a text message of 1 to 2,000 characters."}), 400

    datasets = matching_datasets(message)
    default_dataset = payload.get("default_dataset")
    if not datasets and isinstance(default_dataset, str) and default_dataset in DATASETS:
        datasets = [default_dataset]
    ids = {value.upper() if value.upper().startswith("DEMO-") else value
           for value in [*ID_PATTERN.findall(message), *EXPLICIT_ID.findall(message)]}
    if len(datasets) == 1 and len(ids) == 1:
        return transaction_lookup(datasets[0], ids.pop())
    if ids or payload.get("mode") != "general":
        return jsonify({"status": "lookup_details_required",
                        "reply": f"Provide one dataset and one transaction ID. Example: {LOOKUP_EXAMPLE}."}), 400
    return general_reply(message)


if __name__ == "__main__":
    app.run(host=os.getenv("CHATBOT_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))

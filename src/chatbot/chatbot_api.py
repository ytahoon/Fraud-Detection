from pathlib import Path
import os
import random
import re
import time

from flask import Flask, jsonify, request
import joblib
import pandas as pd

try:
    import google.generativeai as genai
except ImportError:
    genai = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts/models/xgb_precision_model.pkl"))
DATA_SOURCE = Path(os.getenv("DATA_SOURCE", PROJECT_ROOT / "data/raw/ieee_cis/train_transaction.csv"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

model = None
expected_features = ["TransactionAmt", "card1", "TransactionDT"]

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "get_booster"):
            expected_features = model.get_booster().feature_names or expected_features
        elif hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_.tolist()
        print(f"Model loaded from {MODEL_PATH}. Tracking {len(expected_features)} features.")
    except Exception as exc:
        print(f"Model loading failed: {exc}")
else:
    print(f"Model artifact not found at {MODEL_PATH}; using fallback scoring.")

df_stream = pd.DataFrame()
if DATA_SOURCE.exists():
    df_stream = pd.read_csv(DATA_SOURCE)
    print(f"Transaction stream loaded from {DATA_SOURCE}. Rows: {len(df_stream)}")
else:
    print(f"Transaction data not found at {DATA_SOURCE}.")

gemini_model = None
if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

current_ptr = 0


def get_score(row_dict):
    """Return model fraud probability, with a simple fallback when no model artifact exists."""
    try:
        aligned_dict = {col: 0 for col in expected_features}
        for col in expected_features:
            if col in row_dict:
                aligned_dict[col] = row_dict[col]

        final_df = pd.DataFrame([aligned_dict])
        final_df = final_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if model is not None:
            try:
                model.get_booster().set_param({"device": "cpu"})
            except Exception:
                pass
            return float(model.predict_proba(final_df)[0][1])
    except Exception:
        pass

    amount = float(row_dict.get("TransactionAmt", 0) or 0)
    return 0.95 if amount > 3000 else random.uniform(0.002, 0.05)


@app.route("/")
def index():
    return jsonify({
        "service": "fraud-detection-chatbot-api",
        "endpoints": ["/get_next_stream", "/chat"],
        "model_loaded": model is not None,
        "stream_rows": len(df_stream),
    })


@app.route("/get_next_stream")
def get_next_stream():
    global current_ptr
    if df_stream.empty:
        return jsonify({"status": "error", "message": "transaction stream is empty"}), 404

    if current_ptr >= len(df_stream):
        current_ptr = 0

    row = df_stream.iloc[current_ptr].to_dict()
    score = get_score(row)
    status = "CRITICAL" if score >= 0.75 else ("SUSPICIOUS" if score >= 0.40 else "APPROVED")

    clean_data = {
        "id": int(row.get("TransactionID", 0)),
        "amount": round(float(row.get("TransactionAmt", 0) or 0), 2),
        "score": round(float(score), 4),
        "status": status,
        "time": time.strftime("%H:%M:%S"),
        "progress": f"{current_ptr + 1}/{len(df_stream)}",
    }

    current_ptr += 1
    return jsonify(clean_data)


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("message", "")

    match = re.search(r"\d{7}", user_input)
    if match and not df_stream.empty:
        tx_id = int(match.group())
        search = df_stream[df_stream["TransactionID"] == tx_id]
        if not search.empty:
            row = search.iloc[0].to_dict()
            score = get_score(row)
            return jsonify({
                "reply": (
                    f"System audit: transaction #{tx_id} was found. "
                    f"Amount: ${row.get('TransactionAmt', 0):.2f}. "
                    f"Fraud score: {round(score, 4)}."
                )
            })

    if gemini_model is None:
        return jsonify({
            "reply": (
                "Falcon Copilot is online. Set GEMINI_API_KEY to enable general "
                "LLM responses, or ask about a 7-digit transaction ID."
            )
        })

    try:
        response = gemini_model.start_chat().send_message(user_input)
        return jsonify({"reply": response.text.replace("\n", "<br>")})
    except Exception:
        return jsonify({"reply": "Falcon Copilot is listening."})


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5000")))

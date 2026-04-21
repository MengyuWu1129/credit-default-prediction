from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Import custom transformer so that joblib can unpickle the pipeline
from custom_transformers import TargetEncoder

MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"
TARGET_COL = "isDefault"

# Load pipeline (preprocessor + model)
model = joblib.load(MODEL_PATH)

# Load raw data only for building the HTML form choices and numeric stats
raw_df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in raw_df.columns if c != TARGET_COL]

numeric_cols = raw_df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

cat_values = {c: sorted(raw_df[c].dropna().unique().tolist()) for c in categorical_cols}
num_stats = {
    c: {
        "mean": float(raw_df[c].mean()),
        "median": float(raw_df[c].median()),
        "min": float(raw_df[c].min()),
        "max": float(raw_df[c].max()),
    } for c in numeric_cols
}

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_values=cat_values,
        num_stats=num_stats
    )

@app.route("/predict", methods=["POST"])
def predict_form():
    # Collect inputs from HTML form
    row = {}
    # Numeric: cast to float, use median as fallback if empty
    for col in numeric_cols:
        v = request.form.get(col, "")
        if v == "":
            v = num_stats[col]["median"]
        row[col] = float(v)
    # Categorical: raw string
    for col in categorical_cols:
        row[col] = request.form.get(col, "")
    df_input = pd.DataFrame([row])

    pred = model.predict(df_input)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df_input)[0][1])

    return render_template("result.html", input_data=row, prediction=int(pred), proba=proba)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "msg": "send JSON body"}), 400

    row = {}
    missing = []
    for col in feature_cols:
        if col not in data:
            missing.append(col)
        else:
            row[col] = data[col]
    if missing:
        return jsonify({"status": "error", "msg": f"missing fields: {missing}"}), 400

    # Cast numeric fields
    for col in numeric_cols:
        try:
            row[col] = float(row[col])
        except Exception:
            return jsonify({"status": "error", "msg": f"{col} cannot be cast to float"}), 400

    df_input = pd.DataFrame([row])

    pred = model.predict(df_input)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df_input)[0][1])

    return jsonify({"status": "ok", "prediction": int(pred), "probability": proba})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

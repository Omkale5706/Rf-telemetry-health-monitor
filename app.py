from pathlib import Path
from datetime import datetime
import json
import uuid

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

from train_model import train_model, load_data, get_feature_columns, MODEL_PATH, DATA_PATH, OUTPUT_DIR

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-local-development-secret"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def get_artifact():
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def read_tabular(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Upload an Excel (.xlsx/.xls) or CSV (.csv) file.")


def normalize_input_columns(df):
    df = df.copy()
    # Keep compatibility with the user's preferred term.
    if "Cable_Loss_dB" in df.columns and "Insertion_Loss_dB" not in df.columns:
        df = df.rename(columns={"Cable_Loss_dB": "Insertion_Loss_dB"})
    return df


def prediction_table(df, artifact):
    df = normalize_input_columns(df)
    pipeline = artifact["pipeline"]
    features = artifact["feature_columns"]
    X = df.drop(columns=artifact["excluded_input_columns"], errors="ignore")
    X = X.reindex(columns=features)
    pred = pipeline.predict(X)
    prob = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)

    out = df.copy()
    out["Predicted_Fault_Type"] = pred
    out["Predicted_Health_Status"] = np.where(pred == "NORMAL", "NORMAL", "FAULT SUSPECTED")
    out["Model_Confidence"] = prob.max(axis=1).round(4)
    for i, label in enumerate(classes):
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in str(label))
        out[f"Probability_{safe_label}"] = prob[:, i].round(4)
    # DTF distance is an instrument measurement, not an ML-estimated location.
    if "DTF_Peak_Distance_m" in out.columns:
        out["DTF_Measured_Peak_Distance_m"] = out["DTF_Peak_Distance_m"]
    return out


def dashboard_metrics(artifact):
    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return artifact.get("metrics", {})


@app.route("/")
def index():
    artifact = get_artifact()
    metrics = dashboard_metrics(artifact)
    features = artifact["feature_columns"]
    defaults = {
        "Frequency_MHz": "150",
        "VSWR": "1.3",
        "Return_Loss_dB": "17",
        "DTF_Peak_Distance_m": "20",
        "DTF_Peak_Magnitude_dB": "-28",
        "Insertion_Loss_dB": "2.5",
        "Signal_Power_dBm": "-35",
        "Telemetry_Value_V": "2.5",
        "Operating_Condition": "Normal",
        "Temperature_C": "25",
        "Communication_Status": "OK",
        "Data_Logger_Status": "OK",
        "Telemetry_Missing_Flag": "0",
    }
    feature_importance = artifact.get("top_features", [])
    classes = artifact.get("classes", [])
    return render_template(
        "index.html",
        metrics=metrics,
        features=features,
        defaults=defaults,
        feature_importance=feature_importance,
        classes=classes,
        model_name="Random Forest",
    )


@app.route("/predict/manual", methods=["POST"])
def predict_manual():
    artifact = get_artifact()
    values = {}
    for feature in artifact["feature_columns"]:
        raw = request.form.get(feature, "").strip()
        if raw == "":
            values[feature] = np.nan
        else:
            # Feature types are stored by the training pipeline.
            kind = artifact["feature_types"].get(feature, "numeric")
            if kind == "numeric":
                try:
                    values[feature] = float(raw)
                except ValueError:
                    flash(f"'{feature}' must be numeric.", "error")
                    return redirect(url_for("index"))
            else:
                values[feature] = raw
    df = pd.DataFrame([values])
    result = prediction_table(df, artifact)
    record = result.iloc[0].to_dict()
    probabilities = []
    for cls in artifact["classes"]:
        col = "Probability_" + "".join(ch if ch.isalnum() else "_" for ch in str(cls))
        probabilities.append({"label": cls, "value": float(record.get(col, 0.0))})
    probabilities.sort(key=lambda x: x["value"], reverse=True)
    safe_record = {}
    for k, v in record.items():
        if pd.isna(v):
            safe_record[k] = None
        elif isinstance(v, (np.integer,)):
            safe_record[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe_record[k] = float(v)
        else:
            safe_record[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
    return render_template(
        "result.html",
        result=safe_record,
        probabilities=probabilities,
        is_normal=record.get("Predicted_Fault_Type") == "NORMAL",
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/predict/upload", methods=["POST"])
def predict_upload():
    if "file" not in request.files:
        flash("Choose a file first.", "error")
        return redirect(url_for("index"))
    file = request.files["file"]
    if not file or not file.filename:
        flash("Choose a file first.", "error")
        return redirect(url_for("index"))
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        flash("Supported file types: .xlsx, .xls, .csv", "error")
        return redirect(url_for("index"))

    artifact = get_artifact()
    filename = secure_filename(file.filename)
    saved = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
    file.save(saved)
    try:
        df = read_tabular(saved)
        if df.empty:
            raise ValueError("The uploaded file has no rows.")
        predictions = prediction_table(df, artifact)
        out_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.csv"
        predictions.to_csv(OUTPUT_DIR / out_name, index=False)
        preview = predictions.head(25).replace({np.nan: None}).to_dict(orient="records")
        columns = list(predictions.columns)
        summary = predictions["Predicted_Fault_Type"].value_counts().to_dict()
        return render_template(
            "batch_result.html",
            rows=preview,
            columns=columns,
            summary=summary,
            output_file=out_name,
            total=len(predictions),
        )
    except Exception as exc:
        flash(f"Could not process file: {exc}", "error")
        return redirect(url_for("index"))
    finally:
        try:
            saved.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/download/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/health")
def health():
    artifact = get_artifact()
    return {
        "status": "ok",
        "model": "Random Forest",
        "classes": artifact["classes"],
        "features": artifact["feature_columns"],
    }


if __name__ == "__main__":
    # Local development only. For network deployment, configure a production WSGI server.
    app.run(host="127.0.0.1", port=5000, debug=False)

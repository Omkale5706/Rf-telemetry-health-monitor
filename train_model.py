from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "RF_Telemetry_10000_Synthetic_ML_Dataset.xlsx"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "rf_telemetry_model.joblib"
TARGET = "Fault_Type"

# Never use labels, sample/session IDs, timestamps, or dataset partition as inputs.
IDENTIFIER_AND_TARGET_COLUMNS = [
    "Sample_ID", "Session_ID", "Timestamp", "Dataset_Split",
    "Fault_Type", "Fault_Location",
]
# The deployed CWPRS R&S ZVH does not have the spectrum-analyzer option.
# Therefore, do not make spectrum-style fields required model inputs.
UNAVAILABLE_SPECTRUM_COLUMNS = ["Dominant_Frequency_MHz", "Bandwidth_MHz"]


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name="Dataset")
    else:
        df = pd.read_csv(path)
    if "Cable_Loss_dB" in df.columns and "Insertion_Loss_dB" not in df.columns:
        df = df.rename(columns={"Cable_Loss_dB": "Insertion_Loss_dB"})
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column {TARGET}.")
    df[TARGET] = df[TARGET].astype("string").str.strip()
    return df.dropna(subset=[TARGET]).copy()


def get_feature_columns(df):
    excluded = set(IDENTIFIER_AND_TARGET_COLUMNS + UNAVAILABLE_SPECTRUM_COLUMNS)
    return [c for c in df.columns if c not in excluded]


def split_data(df):
    # Use the dataset's existing session-based split labels when present.
    if "Dataset_Split" in df.columns:
        split = df["Dataset_Split"].astype("string").str.upper().str.strip()
        train = df.loc[split.eq("TRAIN")].copy()
        val = df.loc[split.eq("VALIDATION")].copy()
        test = df.loc[split.eq("TEST")].copy()
        if len(train) and len(val) and len(test):
            return train, val, test

    # Fallback: keep sessions intact to reduce leakage.
    if "Session_ID" not in df.columns:
        raise ValueError("Dataset_Split is invalid and Session_ID is missing.")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, hold_idx = next(outer.split(df, groups=df["Session_ID"]))
    train = df.iloc[train_idx].copy()
    hold = df.iloc[hold_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
    val_idx, test_idx = next(inner.split(hold, groups=hold["Session_ID"]))
    return train, hold.iloc[val_idx].copy(), hold.iloc[test_idx].copy()


def make_pipeline(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=22,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def train_model(data_path=DATA_PATH):
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data(data_path)
    train, val, test = split_data(df)
    feature_columns = get_feature_columns(df)
    X_train = train.reindex(columns=feature_columns)
    X_val = val.reindex(columns=feature_columns)
    X_test = test.reindex(columns=feature_columns)
    y_train = train[TARGET].astype(str)
    y_val = val[TARGET].astype(str)
    y_test = test[TARGET].astype(str)

    pipeline = make_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    classes = sorted(df[TARGET].astype(str).unique().tolist())

    def evaluate(X, y):
        pred = pipeline.predict(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "report": classification_report(y, pred, labels=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        }

    val_metrics = evaluate(X_val, y_val)
    test_metrics = evaluate(X_test, y_test)

    # Feature type schema supports manual entry in the Flask UI.
    feature_types = {}
    for c in feature_columns:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            feature_types[c] = "numeric"
        else:
            feature_types[c] = "categorical"

    # Importance is for display only; one-hot fields are expanded by the pipeline.
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        imps = pipeline.named_steps["model"].feature_importances_
        top_features = sorted(
            [{"name": str(n).replace("numeric__", "").replace("categorical__", ""),
              "importance": float(v)} for n, v in zip(names, imps)],
            key=lambda x: x["importance"], reverse=True
        )[:12]
    except Exception:
        top_features = []

    # Save evaluation metrics.
    metrics = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "total_rows": int(len(df)),
        "class_count": int(len(classes)),
        "classes": classes,
        "validation": val_metrics,
        "test": test_metrics,
        "note": "Scores are on synthetic data; real-world accuracy is not established.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=False, values_format="d"
    )
    ax.set_title("RF Telemetry Fault Classification — Test Set")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    pd.DataFrame(top_features).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "feature_types": feature_types,
        "excluded_input_columns": IDENTIFIER_AND_TARGET_COLUMNS + UNAVAILABLE_SPECTRUM_COLUMNS,
        "classes": classes,
        "metrics": metrics,
        "top_features": top_features,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    return artifact


if __name__ == "__main__":
    train_model()

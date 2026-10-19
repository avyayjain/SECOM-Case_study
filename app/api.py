
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile,File
from fastapi.responses import Response
import urllib.request
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "results" / "final_model"

DATA_PATH = DATA_DIR / "secom.data"
LABEL_PATH = DATA_DIR / "secom_labels.data"
MODEL_PATH = MODEL_DIR / "secom_random_forest.joblib"
CONFIG_PATH = MODEL_DIR / "decision_config.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"

app = FastAPI(
    title="SECOM Case Study API",
    description="Step-by-step semiconductor fault detection workflow",
    version="2.0.0",
)


# ---------------------------------------------------------
# Shared data/model loading
# ---------------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists() or not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Dataset files not found. Expected:\n{DATA_PATH}\n{LABEL_PATH}"
        )

    X = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
    labels = pd.read_csv(LABEL_PATH, sep=r"\s+", header=None)

    if X.shape[0] != labels.shape[0]:
        raise ValueError("Sensor rows and label rows do not match.")

    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    # Original SECOM labels: -1 = pass, +1 = fault.
    y = labels.iloc[:, 0].map({-1: 0, 1: 1})

    if y.isna().any():
        raise ValueError("Unexpected labels found. Expected -1 and +1.")

    return X, y.astype(int)


def load_model_and_config():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Decision config not found: {CONFIG_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    threshold = float(config.get("threshold", 0.12))
    return model, config, threshold


def load_saved_metrics():
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_json(value):
    """Convert numpy values and NaN/inf into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    return value


# ---------------------------------------------------------
# 0. Service health
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "SECOM Fault Detection Case Study",
        "version": "2.0.0",
        "workflow": [
            "/case-study/overview",
            "/case-study/data-quality",
            "/case-study/preprocessing",
            "/case-study/experiments",
            "/case-study/threshold",
            "/case-study/final-model",
            "/predict",
        ],
    }


@app.get("/health")
def health():
    try:
        model, config, threshold = load_model_and_config()
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "threshold": threshold,
            "dataset_available": DATA_PATH.exists() and LABEL_PATH.exists(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "dataset_available": DATA_PATH.exists() and LABEL_PATH.exists(),
        }


# ---------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------

@app.get("/case-study/overview")
def overview():
    try:
        X, y = load_dataset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    total = len(X)
    faults = int(y.sum())
    passes = int((y == 0).sum())

    return {
        "title": "Understand the SECOM dataset",
        "observations": total,
        "features": X.shape[1],
        "pass_count": passes,
        "fault_count": faults,
        "pass_percent": round(100 * passes / total, 2),
        "fault_percent": round(100 * faults / total, 2),
        "label_mapping": {"-1": "Pass", "+1": "Fault"},
        "lesson": (
            "The dataset is strongly imbalanced. Accuracy alone can be misleading "
            "because a model can predict Pass for every row and still achieve high accuracy."
        ),
    }


# ---------------------------------------------------------
# 2. Data quality and exploratory analysis
# ---------------------------------------------------------

@app.get("/case-study/data-quality")
def data_quality():
    try:
        X, y = load_dataset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    missing_by_feature = X.isna().sum()
    all_missing = int((missing_by_feature == len(X)).sum())
    some_missing = int(
        ((missing_by_feature > 0) & (missing_by_feature < len(X))).sum()
    )
    total_cells = X.shape[0] * X.shape[1]
    missing_cells = int(X.isna().sum().sum())

    # Only numeric sensor columns are present in the source dataset.
    stats = X.describe().T.replace({np.nan: None})
    feature_table = []
    for feature in X.columns:
        feature_table.append({
            "feature": feature,
            "missing_count": int(X[feature].isna().sum()),
            "missing_percent": round(
                100 * X[feature].isna().mean(), 2
            ),
            "unique_non_missing": int(X[feature].nunique(dropna=True)),
            "mean": clean_json(X[feature].mean()),
            "std": clean_json(X[feature].std()),
        })

    feature_table.sort(
        key=lambda row: row["missing_count"], reverse=True
    )

    return {
        "title": "Data quality and EDA",
        "rows": len(X),
        "features": X.shape[1],
        "missing_cells": missing_cells,
        "missing_cell_percent": round(
            100 * missing_cells / total_cells, 2
        ),
        "features_entirely_missing": all_missing,
        "features_partially_missing": some_missing,
        "duplicate_rows": int(X.duplicated().sum()),
        "label_counts": {
            "Pass": int((y == 0).sum()),
            "Fault": int((y == 1).sum()),
        },
        "features_by_missingness": feature_table,
        "lesson": (
            "Missing readings need a consistent treatment. Features that are entirely "
            "missing cannot provide direct measurement information. Any imputation "
            "must be fitted on training data only."
        ),
    }


# ---------------------------------------------------------
# 3. Preprocessing and experimental design
# ---------------------------------------------------------

@app.get("/case-study/preprocessing")
def preprocessing():
    try:
        X, y = load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        train_missing = int(X_train.isna().sum().sum())
        test_missing = int(X_test.isna().sum().sum())

        return {
            "title": "Preprocessing and split",
            "split_method": "Stratified train/test split",
            "random_state": 42,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_faults": int(y_train.sum()),
            "train_passes": int((y_train == 0).sum()),
            "test_faults": int(y_test.sum()),
            "test_passes": int((y_test == 0).sum()),
            "train_missing_cells": train_missing,
            "test_missing_cells": test_missing,
            "imputation": "Median imputation inside the model pipeline",
            "leakage_prevention": (
                "Fit transformations on training data only; do not fit imputers, "
                "scalers, feature selection, or thresholds on the held-out test set."
            ),
            "lesson": (
                "Stratification keeps the fault proportion similar in both splits. "
                "The test split is for final evaluation, not repeated model selection."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# 4. Previously completed model experiments
# ---------------------------------------------------------

@app.get("/case-study/experiments")
def experiments():
    # Historical nested-CV results from the case study lessons.
    # These are reported results, not recalculated on each request.
    rows = [
        {
            "model": "Logistic Regression",
            "average_precision": 0.148,
            "roc_auc": 0.664,
            "balanced_accuracy": 0.565,
            "precision": 0.123,
            "recall": 0.270,
            "f1": 0.166,
            "threshold_mean": 0.435,
        },
        {
            "model": "Random Forest",
            "average_precision": 0.212,
            "roc_auc": 0.730,
            "balanced_accuracy": 0.660,
            "precision": 0.182,
            "recall": 0.481,
            "f1": 0.261,
            "threshold_mean": 0.107,
        },
        {
            "model": "XGBoost",
            "average_precision": 0.198,
            "roc_auc": 0.705,
            "balanced_accuracy": 0.664,
            "precision": 0.146,
            "recall": 0.567,
            "f1": 0.231,
            "threshold_mean": 0.072,
        },
    ]

    return {
        "title": "Model comparison from previous nested cross-validation",
        "results": rows,
        "important_note": (
            "These are historical case-study results. They are not recalculated "
            "by this endpoint. XGBoost results are included only as a prior experiment; "
            "the deployed model is the saved Random Forest."
        ),
        "lesson": (
            "Average precision is useful for imbalanced classification. Recall measures "
            "the proportion of actual faults found, while precision measures how many "
            "flagged items are actually faults."
        ),
    }


# ---------------------------------------------------------
# 5. Threshold and cost trade-off
# ---------------------------------------------------------

@app.get("/case-study/threshold")
def threshold_analysis():
    _, config, threshold = load_model_and_config()

    return {
        "title": "Decision threshold and inspection trade-offs",
        "selected_threshold": threshold,
        "threshold_source": (
            "Loaded from decision_config.json. In this project the threshold was "
            "selected using training out-of-fold predictions."
        ),
        "cost_scenarios": [
            {
                "false_negative_to_false_positive_cost_ratio": 5,
                "interpretation": "Missing a fault costs 5 times a false alarm.",
            },
            {
                "false_negative_to_false_positive_cost_ratio": 10,
                "interpretation": "Missing a fault costs 10 times a false alarm.",
            },
            {
                "false_negative_to_false_positive_cost_ratio": 20,
                "interpretation": "Missing a fault costs 20 times a false alarm.",
            },
        ],
        "config": clean_json(config),
        "lesson": (
            "A lower threshold usually catches more faults but can create more false "
            "alarms. The right threshold depends on real operational costs and must be "
            "validated before use in manufacturing."
        ),
    }


# ---------------------------------------------------------
# 6. Final model and saved holdout evaluation
# ---------------------------------------------------------

@app.get("/case-study/final-model")
def final_model():
    try:
        model, config, threshold = load_model_and_config()
        metrics = load_saved_metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "title": "Final Random Forest model",
        "model_type": type(model).__name__,
        "threshold": threshold,
        "configuration": clean_json(config),
        "saved_test_metrics": clean_json(metrics),
        "known_test_results": {
            "test_rows": 314,
            "test_faults": 21,
            "true_pass": 262,
            "false_alarm": 31,
            "missed_fault": 13,
            "detected_fault": 8,
            "average_precision": 0.2438,
            "balanced_accuracy": 0.6376,
            "precision": 0.2051,
            "recall": 0.3810,
            "f1": 0.2667,
        },
        "lesson": (
            "On the held-out test split, 13 of 21 faults were missed and 31 passing "
            "items were flagged. This is a useful educational result, not sufficient "
            "evidence for production release decisions. The same test split has been "
            "used during previous experiments, so treat this evaluation cautiously."
        ),
    }


# ---------------------------------------------------------
# Download and use the official UCI SECOM dataset
# ---------------------------------------------------------

UCI_FEATURES_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/secom/secom.data"
)
UCI_LABELS_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/secom/secom_labels.data"
)

UCI_CACHE_DIR = DATA_DIR / "uci_cache"
UCI_FEATURES_PATH = UCI_CACHE_DIR / "secom.data"
UCI_LABELS_PATH = UCI_CACHE_DIR / "secom_labels.data"


def download_uci_file(url: str, destination: Path):
    """Download a public UCI file once and cache it locally."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SECOM-Educational-Case-Study/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()

        if not content:
            raise ValueError("The downloaded file was empty.")

        destination.write_bytes(content)

    except Exception as exc:
        raise RuntimeError(
            f"Could not download {url}. Check your internet connection. "
            f"Details: {exc}"
        ) from exc


def load_uci_source_dataset():
    """Load the original sensor file and corresponding labels from UCI."""
    download_uci_file(UCI_FEATURES_URL, UCI_FEATURES_PATH)
    download_uci_file(UCI_LABELS_URL, UCI_LABELS_PATH)

    X = pd.read_csv(
        UCI_FEATURES_PATH,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan"],
    )

    labels = pd.read_csv(
        UCI_LABELS_PATH,
        sep=r"\s+",
        header=None,
    )

    if len(X) != len(labels):
        raise ValueError(
            f"Source data has {len(X)} rows but labels have {len(labels)} rows."
        )

    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    # The first label column is -1 (pass) or +1 (fault).
    y = labels.iloc[:, 0].map({-1: 0, 1: 1})

    if y.isna().any():
        raise ValueError("Unexpected target labels in the UCI source.")

    return X, y.astype(int)


@app.get("/source-data/summary")
def source_data_summary():
    """Download/cache the official data and report its structure."""
    try:
        X, y = load_uci_source_dataset()

        return {
            "source": "UCI Machine Learning Repository — SECOM",
            "source_page": "https://archive.ics.uci.edu/dataset/179/secom",
            "rows": len(X),
            "features": X.shape[1],
            "pass_count": int((y == 0).sum()),
            "fault_count": int((y == 1).sum()),
            "missing_cells": int(X.isna().sum().sum()),
            "downloaded": True,
            "cached_locally": True,
            "message": "Source dataset is ready for row selection.",
        }

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/source-data/download-csv")
def download_source_csv():
    """Provide the complete source dataset as a downloadable CSV."""
    try:
        X, y = load_uci_source_dataset()

        combined = X.copy()
        combined["actual_label"] = y.map({0: "Pass", 1: "Fault"})

        csv_content = combined.to_csv(index=False)

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="secom_uci_dataset.csv"'
            },
        )

    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/source-data/download-row/{row_index}")
def download_source_row(row_index: int):
    """Download one source observation as a model-ready CSV."""
    try:
        X, y = load_uci_source_dataset()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if row_index < 0 or row_index >= len(X):
        raise HTTPException(
            status_code=404,
            detail=f"Row index must be between 0 and {len(X) - 1}.",
        )

    csv_content = X.iloc[[row_index]].to_csv(index=False)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="secom_source_row_{row_index}.csv"'
        },
    )


@app.post("/predict-source-row/{row_index}")
def predict_source_row(row_index: int):
    """Predict a source observation and compare with its known source label."""
    try:
        X, y = load_uci_source_dataset()
        model, config, threshold = load_model_and_config()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if row_index < 0 or row_index >= len(X):
        raise HTTPException(
            status_code=404,
            detail=f"Row index must be between 0 and {len(X) - 1}.",
        )

    row = X.iloc[[row_index]]

    try:
        probabilities = model.predict_proba(row)
        fault_index = list(model.classes_).index(1)
        score = float(probabilities[0][fault_index])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        )

    predicted_fault = score >= threshold
    actual_fault = int(y.iloc[row_index]) == 1

    return {
        "source": "UCI SECOM",
        "row_index": row_index,
        "score": score,
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "prediction": "Fault" if predicted_fault else "Pass",
        "decision": (
            "FLAG FOR INSPECTION" if predicted_fault else "NOT FLAGGED"
        ),
        "actual_label": "Fault" if actual_fault else "Pass",
        "matches_source_label": predicted_fault == actual_fault,
        "missing_readings": int(row.isna().sum().sum()),
        "note": (
            "This is a historical source observation. It may have been included "
            "in model training or previous experimentation. Agreement with its "
            "label is not independent validation."
        ),
    }



# ---------------------------------------------------------
# 7. Schema for incoming sensor data
# ---------------------------------------------------------

@app.get("/schema")
def schema():
    try:
        X, _ = load_dataset()
        return {
            "features": list(X.columns),
            "feature_count": len(X.columns),
            "expected_columns": "sensor_0 through sensor_589",
            "input_format": "CSV with one row and one column per sensor feature",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# 8. Final prediction
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    readings: dict[str, float | None] = Field(
        ...,
        description="Sensor readings keyed by sensor_0 ... sensor_589",
    )


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, config, threshold = load_model_and_config()
        X, _ = load_dataset()
        expected = list(X.columns)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    supplied = request.readings
    unknown = sorted(set(supplied) - set(expected))
    missing = sorted(set(expected) - set(supplied))

    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"message": "Unknown sensor columns", "columns": unknown[:20]},
        )

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing sensor columns. Include every expected column; "
                           "unknown readings may be null.",
                "missing_count": len(missing),
                "sample_missing": missing[:20],
            },
        )

    values = []
    for feature in expected:
        value = supplied[feature]
        if value is None:
            values.append(np.nan)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail=f"{feature} must be numeric or blank.",
                )
            if not math.isfinite(number):
                raise HTTPException(
                    status_code=422,
                    detail=f"{feature} must be finite or blank.",
                )
            values.append(number)

    row = pd.DataFrame([values], columns=expected)

    try:
        scores = model.predict_proba(row)
        fault_index = list(model.classes_).index(1)
        score = float(scores[0][fault_index])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        )

    flagged = score >= threshold

    return {
        "score": score,
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "flagged": flagged,
        "decision": "FLAG FOR INSPECTION" if flagged else "NOT FLAGGED",
        "missing_readings": int(row.isna().sum().sum()),
        "model": "Random Forest",
        "caution": (
            "Model score is not a guaranteed real-world fault probability. "
            "Validate independently before operational use."
        ),
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Predict every row in an uploaded sensor CSV.
    Returns all predictions and the subset flagged for inspection.
    """
    import io

    if not file.filename:
        raise HTTPException(status_code=400, detail="Please select a file.")

    if not file.filename.lower().endswith((".csv", ".data", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV, .data, or .txt sensor file.",
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 100 MB upload limit.",
            )

        # Detect standard CSV versus original whitespace-separated UCI data.
        first_line = content.decode("utf-8-sig", errors="replace").splitlines()[0]

        if "," in first_line:
            uploaded = pd.read_csv(io.BytesIO(content))
            has_headers = True
        else:
            uploaded = pd.read_csv(
                io.BytesIO(content),
                sep=r"\s+",
                header=None,
            )
            has_headers = False

        model, config, threshold = load_model_and_config()
        X_reference, _ = load_dataset()
        expected = list(X_reference.columns)

        if uploaded.empty:
            raise HTTPException(
                status_code=400,
                detail="No sensor observations were found in the file.",
            )

        # The raw UCI data has no column names.
        if not has_headers:
            if uploaded.shape[1] != len(expected):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Expected {len(expected)} sensor columns in raw UCI format, "
                        f"but found {uploaded.shape[1]}."
                    ),
                )
            uploaded.columns = expected

        else:
            # Permit an optional label column, but never use it as a feature.
            label_columns = [
                c for c in uploaded.columns
                if str(c).strip().lower() in {
                    "actual_label", "label", "class", "target"
                }
            ]

            input_labels = (
                uploaded[label_columns[0]].copy()
                if label_columns else None
            )

            if label_columns:
                uploaded = uploaded.drop(columns=label_columns)

            # Do not silently guess the order of arbitrary unnamed CSV columns.
            missing = sorted(set(expected) - set(uploaded.columns))
            unknown = sorted(set(uploaded.columns) - set(expected))

            if missing or unknown:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": (
                            "CSV columns do not match the expected sensor schema. "
                            "Use the downloadable template or upload the original "
                            "headerless UCI sensor file."
                        ),
                        "missing_columns_count": len(missing),
                        "sample_missing_columns": missing[:10],
                        "unknown_columns": unknown[:10],
                    },
                )

            uploaded = uploaded[expected]

        # Convert sensor values to numeric. Blank readings become NaN.
        original = uploaded.copy()
        numeric = uploaded.apply(pd.to_numeric, errors="coerce")

        invalid_text = original.notna() & numeric.isna()
        if invalid_text.any().any():
            bad_columns = invalid_text.any()
            bad_names = bad_columns[bad_columns].index.tolist()
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Non-numeric sensor values were found.",
                    "columns": bad_names[:20],
                },
            )

        # Reject infinities, but allow missing values for pipeline imputation.
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise HTTPException(
                status_code=422,
                detail="Infinite sensor values are not allowed.",
            )

        if numeric.isna().all(axis=1).any():
            bad_rows = (
                np.flatnonzero(numeric.isna().all(axis=1).to_numpy()) + 1
            ).tolist()
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Some rows contain no usable sensor readings.",
                    "csv_row_numbers": bad_rows[:20],
                },
            )

        # Run the saved model on the complete file.
        probabilities = model.predict_proba(numeric)
        fault_index = list(model.classes_).index(1)
        scores = probabilities[:, fault_index]
        flags = scores >= threshold

        result = pd.DataFrame({
            # CSV row number includes the header row when headers exist.
            "row_number": np.arange(1, len(numeric) + 1),
            "fault_score": scores,
            "fault_score_percent": np.round(scores * 100, 2),
            "decision_threshold": threshold,
            "prediction": np.where(flags, "Fault", "Pass"),
            "inspection_decision": np.where(
                flags, "FLAG FOR INSPECTION", "NOT FLAGGED"
            ),
            "missing_readings": numeric.isna().sum(axis=1).to_numpy(),
        })

        # If the user uploaded known labels, compare them separately.
        # Labels never enter the model feature matrix.
        if input_labels is not None:
            actual = (
                input_labels.astype(str).str.strip().str.lower()
                .replace({
                    "-1": "Pass",
                    "1": "Fault",
                    "0": "Pass",
                    "pass": "Pass",
                    "fault": "Fault",
                    "fail": "Fault",
                })
            )
            result["actual_label"] = actual.to_numpy()
            result["matches_label"] = (
                result["prediction"] == result["actual_label"]
            )

        faulty = result[result["prediction"] == "Fault"]

        return {
            "filename": file.filename,
            "rows_processed": len(result),
            "features_used": len(expected),
            "threshold": threshold,
            "fault_count": int(flags.sum()),
            "pass_count": int((~flags).sum()),
            "fault_rate_percent": round(float(flags.mean() * 100), 2),
            "missing_readings_total": int(numeric.isna().sum().sum()),
            "results": result.to_dict(orient="records"),
            "faulty_samples": faulty.to_dict(orient="records"),
            "note": (
                "Predictions use the saved model and configured threshold. "
                "A flagged row is a candidate for inspection, not proof of a defect. "
                "If the file contains historical SECOM observations, some rows may "
                "have been used during model development."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process the uploaded file: {exc}",
        )
    finally:
        await file.close()


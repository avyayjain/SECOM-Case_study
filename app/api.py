from pathlib import Path
import json
import math
import io
import urllib.request

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "final_model"

DATA_PATH = DATA_DIR / "secom.data"
LABEL_PATH = DATA_DIR / "secom_labels.data"
MODEL_PATH = MODEL_DIR / "secom_random_forest.joblib"
CONFIG_PATH = MODEL_DIR / "decision_config.json"
METRICS_PATH = MODEL_DIR / "test_metrics.json"

app = FastAPI(
    title="SECOM Semiconductor Fault Detection API",
    description="Full CRISP-DM workflow from raw sensor readings to cost-sensitive wafer inspection",
    version="2.1.0",
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

    threshold = float(config.get("threshold", 0.495))
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


def load_csv_table(filename: str):
    file_path = TABLES_DIR / filename
    if not file_path.exists():
        return []
    df = pd.read_csv(file_path)
    df = df.replace({np.nan: None})
    return clean_json(df.to_dict(orient="records"))


# ---------------------------------------------------------
# 0. Service health & figures
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "application": "SECOM Fault Detection Case Study",
        "version": "2.1.0",
        "workflow": [
            "/case-study/overview",
            "/case-study/data-quality",
            "/case-study/sip-log",
            "/case-study/reduction-summary",
            "/case-study/selected-features",
            "/case-study/preprocessing",
            "/case-study/experiments",
            "/case-study/grid-search",
            "/case-study/threshold",
            "/case-study/final-model",
            "/case-study/figures/{figure_name}",
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
            "results_available": TABLES_DIR.exists() and FIGURES_DIR.exists(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "dataset_available": DATA_PATH.exists() and LABEL_PATH.exists(),
        }


@app.get("/case-study/figures/{figure_name}")
def get_figure(figure_name: str):
    """Serve generated visual artifact figures."""
    file_path = FIGURES_DIR / figure_name
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Figure '{figure_name}' not found in {FIGURES_DIR}",
        )
    return FileResponse(file_path, media_type="image/png")


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
        "title": "SECOM Dataset Understanding",
        "observations": total,
        "features": X.shape[1],
        "pass_count": passes,
        "fault_count": faults,
        "pass_percent": round(100 * passes / total, 2),
        "fault_percent": round(100 * faults / total, 2),
        "label_mapping": {"-1": "Pass", "+1": "Fault"},
        "imbalance_ratio": f"{round(passes / faults, 1)}:1",
        "lesson": (
            "The dataset is severely imbalanced (6.64% faults, ~14:1 ratio). "
            "A naive model predicting Pass for every wafer scores 93.36% accuracy "
            "while missing 100% of defective chips. Recall and economic cost are the real criteria."
        ),
    }


# ---------------------------------------------------------
# 2. Data quality & EDA
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

    # Features exceeding 60% missingness threshold
    gt_60_missing = int((missing_by_feature / len(X) > 0.60).sum())

    # Top missing features table
    feature_table = []
    for feature in X.columns:
        m_count = int(missing_by_feature[feature])
        if m_count > 0:
            feature_table.append({
                "feature": feature,
                "missing_count": m_count,
                "missing_percent": round(100 * m_count / len(X), 2),
            })

    feature_table.sort(key=lambda row: row["missing_count"], reverse=True)

    return {
        "title": "Data Quality and Exploratory Data Analysis",
        "rows": len(X),
        "features": X.shape[1],
        "missing_cells": missing_cells,
        "missing_cell_percent": round(100 * missing_cells / total_cells, 2),
        "features_entirely_missing": all_missing,
        "features_partially_missing": some_missing,
        "features_gt_60_percent_missing": gt_60_missing,
        "duplicate_rows": int(X.duplicated().sum()),
        "label_counts": {
            "Pass": int((y == 0).sum()),
            "Fault": int((y == 1).sum()),
        },
        "features_by_missingness": feature_table[:30],
        "lesson": (
            "Sensor dropouts account for 4.5% of all values. 24 features have >60% missingness "
            "and are removed. Outlier treatment uses 3-sigma capping (winsorising) rather than IQR, "
            "as extreme skewness in semiconductor readings causes IQR to over-trim genuine process excursions."
        ),
    }


# ---------------------------------------------------------
# 3. Step Impact Protocol (SIP) & Reduction Summary
# ---------------------------------------------------------

@app.get("/case-study/sip-log")
def sip_log():
    """Return the complete 4-column Step Impact Protocol table from the manual."""
    records = load_csv_table("sip_log.csv")
    return {
        "title": "Step Impact Protocol (SIP) Log",
        "description": "Standardized 4-column audit log recording every data engineering and modeling decision.",
        "columns": [
            "Options / Approaches Available",
            "Pro & Con of Each Method",
            "Decision Made",
            "Consequence",
        ],
        "records": records,
    }


@app.get("/case-study/reduction-summary")
def reduction_summary():
    """Return feature reduction progression: 590 -> 566 -> 450 -> 20."""
    records = load_csv_table("reduction_summary.csv")
    return {
        "title": "Feature Reduction Progression",
        "records": records,
        "lesson": (
            "Rough reduction safely eliminates 140 uninformative columns (24 high-missingness, "
            "116 constant zero-variance) on training data alone, followed by Boruta shadow-feature "
            "selection isolating the top 20 truly informative sensors."
        ),
    }


@app.get("/case-study/selected-features")
def selected_features():
    """Return the 20 features selected by Boruta shadow feature comparison."""
    records = load_csv_table("selected_features.csv")
    features = [r["selected_sensor"] for r in records if "selected_sensor" in r]
    return {
        "title": "Boruta Selected Sensor Features",
        "count": len(features),
        "features": features,
        "selection_method": "Boruta random shadow feature comparison (target: 20 sensors)",
        "cv_recall_jump": {
            "all_450_features_baseline_cv_recall": 0.0243,
            "boruta_20_features_baseline_cv_recall": 0.1566,
            "tuned_random_forest_cv_recall": 0.3257,
        },
    }


# ---------------------------------------------------------
# 4. Preprocessing and Experimental Design
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

        return {
            "title": "Data Preparation & Preprocessing Pipeline",
            "split_method": "Stratified 80/20 train/test split (random_state=42)",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_faults": int(y_train.sum()),
            "train_passes": int((y_train == 0).sum()),
            "test_faults": int(y_test.sum()),
            "test_passes": int((y_test == 0).sum()),
            "outlier_strategy": "3-Sigma capping (winsorising) bounded at +/- 3 std deviations",
            "imputation_strategy": "MinMaxScaler followed by KNNImputer(n_neighbors=5, weights='distance')",
            "balancing_strategy": "SMOTE(k_neighbors=3) applied strictly to training folds",
            "leakage_prevention": (
                "Zero test leakage: all thresholds, statistics (mean, std), scalers, "
                "kNN imputation graphs, and SMOTE synthesis are fitted purely on training splits."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# 5. Model Experiments & GridSearch
# ---------------------------------------------------------

@app.get("/case-study/experiments")
def experiments():
    """Return rigorous 5-fold CV candidate algorithm comparisons."""
    records = load_csv_table("rigorous_model_comparison.csv")
    return {
        "title": "Candidate Algorithms Benchmark (Stratified 5-Fold Cross Validation)",
        "metric_evaluated": "Minority Class Recall & F1-Score",
        "results": records,
        "lesson": (
            "Random Forest provides superior generalization on noisy tabular sensor data. "
            "While unregularized linear models or high-bias SVMs can trade precision for recall, "
            "Random Forest maintains high AUC and calibrates cleanly with threshold optimization."
        ),
    }


@app.get("/case-study/grid-search")
def grid_search():
    """Return GridSearchCV tuning progression on Random Forest."""
    records = load_csv_table("grid_search_results.csv")
    return {
        "title": "Hyperparameter Tuning via Stratified 5-Fold GridSearchCV",
        "target_metric": "Recall on minority fault class",
        "best_parameters": {
            "classifier__max_depth": 10,
            "classifier__min_samples_leaf": 2,
            "classifier__n_estimators": 100,
        },
        "best_cv_recall": 0.3257,
        "tuning_summary": (
            "Bounding max_depth to 10 prevents deep-tree memorization of noisy sensor spikes. "
            "Setting min_samples_leaf to 2 smooths leaf probability estimates for robust thresholding."
        ),
        "top_trials": records[:8] if records else [],
    }


# ---------------------------------------------------------
# 6. Threshold & Cost Trade-off Analysis
# ---------------------------------------------------------

@app.get("/case-study/threshold")
def threshold_analysis():
    _, config, threshold = load_model_and_config()
    curve_data = load_csv_table("threshold_selection_training_cv.csv")

    return {
        "title": "Cost-Sensitive Operating Threshold Optimization",
        "selected_threshold": threshold,
        "optimization_criterion": "Minimize Total Cost = 100 * FP + 400 * FN on Out-Of-Fold CV",
        "cost_weights": {
            "cost_fp": 100,
            "cost_fn": 400,
            "cost_ratio": "4:1 penalty for missed faults",
        },
        "manufacturing_impact_weights": {
            "mfg_fp_eur": 200,
            "mfg_fn_eur": 5000,
            "interpretation": "€200 per unnecessary wafer teardown vs €5,000 per escaped defect",
        },
        "threshold_curve": curve_data[:40] if curve_data else [],
        "lesson": (
            "The standard 0.50 cutoff ignores the asymmetric economic cost of defects. "
            "Optimizing threshold over training out-of-fold predictions yields an optimal cutoff "
            f"of {threshold:.3f}, cutting escaped faults while keeping false alarms economically controlled."
        ),
    }


# ---------------------------------------------------------
# 7. Final Model Evaluation (Holdout Test Set)
# ---------------------------------------------------------

@app.get("/case-study/final-model")
def final_model():
    try:
        model, config, threshold = load_model_and_config()
        metrics = load_saved_metrics()
        comparison = load_csv_table("training_test_comparison.csv")
        cm_records = load_csv_table("confusion_matrix.csv")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "title": "Final Random Forest Pipeline — Held-Out Test Evaluation",
        "model_type": type(model).__name__,
        "threshold": threshold,
        "test_sample_count": 314,
        "test_fault_count": 21,
        "metrics": metrics,
        "confusion_matrix": {
            "true_negatives": 272,
            "false_positives": 21,
            "false_negatives": 13,
            "true_positives": 8,
            "table": cm_records,
        },
        "training_vs_test": comparison,
        "economic_performance": {
            "total_inspection_cost": 7300,
            "manufacturing_impact_eur": 69200,
            "baseline_all_pass_cost": 21 * 400,  # 8,400
            "cost_saved": (21 * 400) - 7300,     # $1,100 saved on inspection
        },
        "lesson": (
            "On the held-out test split of 314 wafers (21 actual defects), the tuned pipeline "
            "achieves 38.10% recall (8 faults caught) and 92.83% specificity (272 passes cleared) "
            "at the cost-optimal threshold of 0.495. This generates a total cost of $7,300 "
            "compared to $8,400 under a naive all-pass baseline."
        ),
    }


# ---------------------------------------------------------
# 8. Download & Query Official UCI SECOM Dataset
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
            f"Could not download {url}. Check internet connection. Details: {exc}"
        ) from exc


def load_uci_source_dataset():
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
    y = labels.iloc[:, 0].map({-1: 0, 1: 1})

    return X, y.astype(int)


@app.get("/source-data/summary")
def source_data_summary():
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
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/source-data/download-csv")
def download_source_csv():
    try:
        X, y = load_uci_source_dataset()
        combined = X.copy()
        combined["actual_label"] = y.map({0: "Pass", 1: "Fault"})
        csv_content = combined.to_csv(index=False)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="secom_uci_dataset.csv"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/source-data/download-row/{row_index}")
def download_source_row(row_index: int):
    try:
        X, y = load_uci_source_dataset()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if row_index < 0 or row_index >= len(X):
        raise HTTPException(status_code=404, detail="Invalid row index")

    csv_content = X.iloc[[row_index]].to_csv(index=False)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="secom_source_row_{row_index}.csv"'},
    )


@app.post("/predict-source-row/{row_index}")
def predict_source_row(row_index: int):
    try:
        X, y = load_uci_source_dataset()
        model, config, threshold = load_model_and_config()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if row_index < 0 or row_index >= len(X):
        raise HTTPException(status_code=404, detail="Invalid row index")

    row = X.iloc[[row_index]]
    probabilities = model.predict_proba(row)
    fault_index = list(model.classes_).index(1)
    score = float(probabilities[0][fault_index])

    predicted_fault = score >= threshold
    actual_fault = int(y.iloc[row_index]) == 1

    return {
        "source": "UCI SECOM",
        "row_index": row_index,
        "score": score,
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "prediction": "Fault" if predicted_fault else "Pass",
        "decision": "FLAG FOR INSPECTION" if predicted_fault else "NOT FLAGGED",
        "actual_label": "Fault" if actual_fault else "Pass",
        "matches_source_label": predicted_fault == actual_fault,
        "missing_readings": int(row.isna().sum().sum()),
    }


# ---------------------------------------------------------
# 9. Schema & Interactive Predictions
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
                "message": "Missing sensor columns",
                "missing_count": len(missing),
                "sample_missing": missing[:20],
            },
        )

    values = []
    for feature in expected:
        val = supplied[feature]
        if val is None:
            values.append(np.nan)
        else:
            try:
                num = float(val)
                if not math.isfinite(num):
                    values.append(np.nan)
                else:
                    values.append(num)
            except (ValueError, TypeError):
                raise HTTPException(status_code=422, detail=f"Invalid value for {feature}")

    row = pd.DataFrame([values], columns=expected)
    scores = model.predict_proba(row)
    fault_index = list(model.classes_).index(1)
    score = float(scores[0][fault_index])
    flagged = score >= threshold

    return {
        "score": score,
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "flagged": flagged,
        "decision": "FLAG FOR INSPECTION" if flagged else "NOT FLAGGED",
        "missing_readings": int(row.isna().sum().sum()),
        "model": "Random Forest",
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please select a file.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        first_line = content.decode("utf-8-sig", errors="replace").splitlines()[0]
        if "," in first_line:
            uploaded = pd.read_csv(io.BytesIO(content))
            has_headers = True
        else:
            uploaded = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
            has_headers = False

        model, config, threshold = load_model_and_config()
        X_ref, _ = load_dataset()
        expected = list(X_ref.columns)

        if not has_headers:
            if uploaded.shape[1] != len(expected):
                raise HTTPException(
                    status_code=422,
                    detail=f"Expected {len(expected)} sensor columns, got {uploaded.shape[1]}.",
                )
            uploaded.columns = expected
        else:
            label_cols = [
                c for c in uploaded.columns
                if str(c).strip().lower() in {"actual_label", "label", "class", "target"}
            ]
            if label_cols:
                uploaded = uploaded.drop(columns=label_cols)
            uploaded = uploaded[expected]

        numeric = uploaded.apply(pd.to_numeric, errors="coerce")
        probabilities = model.predict_proba(numeric)
        fault_index = list(model.classes_).index(1)
        scores = probabilities[:, fault_index]
        flags = scores >= threshold

        result = pd.DataFrame({
            "row_number": np.arange(1, len(numeric) + 1),
            "fault_score": scores,
            "fault_score_percent": np.round(scores * 100, 2),
            "decision_threshold": threshold,
            "prediction": np.where(flags, "Fault", "Pass"),
            "inspection_decision": np.where(flags, "FLAG FOR INSPECTION", "NOT FLAGGED"),
            "missing_readings": numeric.isna().sum(axis=1).to_numpy(),
        })

        faulty = result[result["prediction"] == "Fault"]

        return {
            "filename": file.filename,
            "rows_processed": len(result),
            "features_used": len(expected),
            "threshold": threshold,
            "fault_count": int(flags.sum()),
            "pass_count": int((~flags).sum()),
            "fault_rate_percent": round(float(flags.mean() * 100), 2),
            "results": result.to_dict(orient="records"),
            "faulty_samples": faulty.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await file.close()

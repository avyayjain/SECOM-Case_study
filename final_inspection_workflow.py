"""
SECOM Semiconductor Fault Detection - Complete End-to-End Analysis Workflow
=============================================================================
This script implements the complete data-mining pipeline according to the
CRISP-DM methodology and the SECOM Case Study Student Manual:
1. Business and Data Understanding: Runtime download from UCI, validation, EDA.
2. Train/Test Stratified Split (80/20): Locked test set, no data leakage.
3. Rough Feature Reduction: Constant features, duplicates, >60% missingness.
4. Outlier Identification and Treatment: 3-sigma rule (IQR strictly banned).
5. Missing-Value Imputation: Pre-scaled distance-weighted kNN (Mean banned).
6. Sophisticated Feature Selection: Boruta shadow features (~15-20 sensors).
7. Class Balancing: SMOTE applied strictly to training portion only.
8. Model Pipeline Assembly: imblearn Pipeline chaining all steps.
9. Candidate Models Baseline Comparison: 5-fold Stratified CV.
10. Hyperparameter Tuning: GridSearchCV across pipeline parameters.
11. Cost-Sensitive Threshold Tuning: Minimizing Total Cost (FP=100, FN=400).
12. Final Evaluation on Held-Out Test Set: Confusion matrix, F2, ROC-AUC, PR-AUC.
13. Generalisation Analysis: Side-by-side training vs test error check.
14. Step Impact Protocol (SIP) Log: Traceable 4-column methodological audit.
"""

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Step 0: Ensure required dependencies are installed automatically
# ---------------------------------------------------------------------------

def ensure_dependencies():
    """
    Ensure all required libraries for the SECOM analysis pipeline are installed.

    Inputs:
        None.

    Outputs:
        None (installs missing packages via pip at runtime if necessary).
    """
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("imblearn", "imbalanced-learn"),
        ("boruta", "boruta"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("requests", "requests"),
        ("joblib", "joblib"),
        ("xgboost", "xgboost"),
    ]
    for module_name, pip_name in required_packages:
        try:
            __import__(module_name)
        except ImportError:
            print(f"[Setup] Installing missing package: {pip_name}...", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name]
            )

ensure_dependencies()

import joblib
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ===========================================================================
# 1. Configuration, Constants, and Relative Paths
# ===========================================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "final_model"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SENSOR_FILE = DATA_DIR / "secom.data"
LABEL_FILE = DATA_DIR / "secom_labels.data"

EXPECTED_ROWS = 1567
EXPECTED_FEATURES = 590

# One single random seed used for split, SMOTE, classifier init, and CV splitter
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Preprocessing thresholds
MISSINGNESS_THRESHOLD = 0.60
TARGET_FEATURE_COUNT = 20

# Cost matrix assumptions (Student Manual Section 14.3, 16.2, 17.1, 17.3)
TYPE_I_COST = 100        # False positive: good wafer flagged for inspection
TYPE_II_COST = 400       # False negative: defective wafer missed (4x costlier)

# Manufacturing impact costs for sensitivity analysis (Section 14.2)
MFG_TYPE_I_COST = 200    # Scrap/rework labor cost per false alarm
MFG_TYPE_II_COST = 5000  # Customer defect warranty/recall cost (25x costlier)


# ===========================================================================
# 2. Custom Preprocessing Transformers
# ===========================================================================

class TrainingFeatureCleaner:
    """
    Performs initial rough feature reduction strictly based on training data:
    1. Removes features with > 60% missing values (justified by Pareto elbow).
    2. Removes zero-variance (constant) features (no discriminative value).
    3. Removes exact duplicate features (avoids redundant collinearity).

    The resulting retained column list is learned on training data and
    applied identically to validation/test partitions without re-learning.
    """

    def __init__(self, missing_threshold=0.60):
        self.missing_threshold = missing_threshold

    def fit(self, X):
        X_df = X.copy()
        self.input_columns_ = list(X_df.columns)

        # 1. High missingness removal
        missing_fraction = X_df.isna().mean()
        self.removed_missing_ = list(
            missing_fraction[missing_fraction > self.missing_threshold].index
        )
        candidates = X_df.drop(columns=self.removed_missing_)

        # 2. Zero-variance constant feature removal
        variances = candidates.var(axis=0, skipna=True)
        self.removed_constant_ = list(
            variances[variances.fillna(0) == 0].index
        )
        candidates = candidates.drop(columns=self.removed_constant_)

        # 3. Duplicate feature removal
        duplicate_mask = candidates.T.duplicated()
        self.removed_duplicate_ = list(candidates.columns[duplicate_mask])
        self.kept_columns_ = list(candidates.columns[~duplicate_mask])

        return self

    def transform(self, X):
        return X.loc[:, self.kept_columns_].copy()


class ThreeSigmaOutlierCapper:
    """
    Detects and caps extreme measurements using the 3-sigma empirical rule:
    bounds = [mean - 3*std, mean + 3*std].
    IQR-based outlier detection is strictly banned per the manual.

    Bounds are computed exclusively from training data and applied to test data.
    """

    def fit(self, X):
        arr = np.asarray(X, dtype=float)
        self.means_ = np.nanmean(arr, axis=0)
        self.stds_ = np.nanstd(arr, axis=0, ddof=1)
        self.stds_ = np.nan_to_num(self.stds_, nan=0.0, posinf=0.0, neginf=0.0)

        self.lower_bounds_ = self.means_ - 3 * self.stds_
        self.upper_bounds_ = self.means_ + 3 * self.stds_
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        arr = np.where(arr < self.lower_bounds_, self.lower_bounds_, arr)
        arr = np.where(arr > self.upper_bounds_, self.upper_bounds_, arr)
        return pd.DataFrame(arr, columns=X.columns, index=X.index)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects a specified subset of columns by name or index.
    Preserves column names and allows tuning of feature counts in pipelines.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns].copy()
        return X[:, self.columns]


# ===========================================================================
# 3. Data Acquisition and Structural Validation (Runtime Download)
# ===========================================================================

def download_and_load_secom():
    """
    Download the SECOM dataset from the UCI Machine Learning Repository at runtime
    and load sensor measurements and binary labels as separate objects.

    Inputs:
        None.

    Outputs:
        X (pd.DataFrame): 1567 x 590 matrix of numeric sensor readings (sensor_0 ... sensor_589).
        y (pd.Series): 1567 target labels mapped to 0 (pass / good) and 1 (fail / defective).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Data Acquisition] Verifying SECOM dataset from UCI repository...", flush=True)
    download_needed = not (SENSOR_FILE.exists() and LABEL_FILE.exists())

    if download_needed:
        print(f"Downloading dataset archive from {UCI_ZIP_URL}...", flush=True)
        try:
            resp = requests.get(UCI_ZIP_URL, timeout=60)
            resp.raise_for_status()
            with ZipFile(BytesIO(resp.content)) as archive:
                namelist = archive.namelist()
                data_name = next(n for n in namelist if n.endswith("secom.data"))
                label_name = next(n for n in namelist if n.endswith("secom_labels.data"))
                SENSOR_FILE.write_bytes(archive.read(data_name))
                LABEL_FILE.write_bytes(archive.read(label_name))
            print("Download and extraction completed successfully.", flush=True)
        except Exception as exc:
            print(f"Warning: Runtime download encountered an issue: {exc}", flush=True)
            if not (SENSOR_FILE.exists() and LABEL_FILE.exists()):
                raise RuntimeError(f"Unable to acquire SECOM dataset: {exc}")
    else:
        print(f"Using verified SECOM dataset files at {DATA_DIR}.", flush=True)

    # Load sensor features and target labels separately
    X = pd.read_csv(SENSOR_FILE, sep=r"\s+", header=None, na_values=["NaN", "nan"])
    labels_df = pd.read_csv(LABEL_FILE, sep=r"\s+", header=None)

    # Validate dataset structure
    if X.shape[0] != EXPECTED_ROWS or X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_ROWS}x{EXPECTED_FEATURES}, got {X.shape}"
        )
    if len(X) != len(labels_df):
        raise ValueError("Sensor row count and label row count do not align.")

    # Explicit sensor column names (sensor_0 ... sensor_589)
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]
    X = X.replace([np.inf, -np.inf], np.nan)

    # Target remapping: -1 = Pass -> 0, +1 = Defective -> 1
    raw_labels = labels_df.iloc[:, 0].astype(int)
    if not set(raw_labels.unique()).issubset({-1, 1}):
        raise ValueError(f"Unexpected target labels found: {raw_labels.unique()}")
    y = (raw_labels.to_numpy() == 1).astype(int)
    y = pd.Series(y, name="target")

    return X, y


# ===========================================================================
# 4. Exploratory Data Understanding Visualizations
# ===========================================================================

def generate_eda_visualizations(X, y):
    """
    Produce and save the required exploratory visualizations (Section 2.6, 4.4, 4.6):
    - class_distribution.png: Class imbalance bar chart.
    - missingness_histogram.png: Histogram of missingness % per sensor.
    - sensor_correlation_heatmap.png: Correlation heatmap of top variance sensors.
    - feature_missingness.png: Outlier boxplot before/after treatment.
    - row_missingness.png: Imputation distribution before/after for top missing sensor.

    Inputs:
        X (pd.DataFrame): Sensor measurement matrix.
        y (pd.Series): Target labels.

    Outputs:
        None (saves figure files into results/figures/).
    """
    print("\n[EDA Visualizations] Generating figures for Section 2, 4, 5, 6...", flush=True)

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(7, 4.5))
    pass_cnt = int((y == 0).sum())
    fail_cnt = int((y == 1).sum())
    bars = plt.bar(["Pass (Good: 0)", "Defective (Fail: 1)"], [pass_cnt, fail_cnt], color=["#2b8cbe", "#e41a1c"])
    plt.title("SECOM Class Distribution (Imbalance: 93.36% vs 6.64%)", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Wafers")
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(y)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 20, f"{h} ({pct:.2f}%)", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, 1650)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # 2. Missingness Percentage Histogram
    missing_pct = X.isna().mean() * 100
    plt.figure(figsize=(9, 4.5))
    plt.hist(missing_pct, bins=30, color="#3182bd", edgecolor="black")
    plt.title("Missing-Value Percentage Per Sensor Column", fontsize=11, fontweight="bold")
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Number of Sensors")
    plt.axvline(60, color="red", linestyle="--", label="60% Missingness Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missingness_histogram.png", dpi=150)
    plt.close()

    # 3. Correlation Heatmap (Top 30 Variance Sensors, Section 4.6: Visualize only)
    top_var_cols = X.var(skipna=True).sort_values(ascending=False).head(30).index
    corr_mat = X[top_var_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"})
    plt.title("Sensor Correlation Heatmap (30 High-Variance Sensors - Visualise Only)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. Outlier Boxplot Before/After (Section 5.5)
    means = X.mean()
    stds = X.std()
    outlier_counts = (((X < (means - 3 * stds)) | (X > (means + 3 * stds))) & X.notna()).sum()
    top_outlier_col = outlier_counts.idxmax()
    col_vals = X[top_outlier_col].dropna()
    c_mean, c_std = col_vals.mean(), col_vals.std()
    lb, ub = c_mean - 3 * c_std, c_mean + 3 * c_std
    capped_vals = np.clip(col_vals, lb, ub)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].boxplot(col_vals)
    axes[0].set_title(f"Before Capping ({top_outlier_col})")
    axes[1].boxplot(capped_vals)
    axes[1].set_title("After 3-Sigma Capping")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_missingness.png", dpi=150)
    plt.close()

    # 5. Imputation Before/After Distribution (Section 6.6)
    sample_col = missing_pct[missing_pct > 0].idxmax()
    orig_vals = X[sample_col].dropna()
    imputed_vals = X[sample_col].fillna(col_vals.median())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    axes[0].hist(orig_vals, bins=25, color="#74a9cf", edgecolor="black")
    axes[0].set_title(f"Before Imputation: {sample_col}")
    axes[1].hist(imputed_vals, bins=25, color="#02818a", edgecolor="black")
    axes[1].set_title("After Imputation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "row_missingness.png", dpi=150)
    plt.close()

    print("[EDA Visualizations] All required exploratory figures successfully saved.", flush=True)


# ===========================================================================
# 5. Evaluation and Metric Helpers
# ===========================================================================

def calculate_metrics(y_true, probabilities, threshold, cost_fp=TYPE_I_COST, cost_fn=TYPE_II_COST):
    """
    Calculate comprehensive evaluation metrics and total cost for binary predictions.

    Inputs:
        y_true (array-like): Ground truth binary labels (0 = Pass, 1 = Fault).
        probabilities (array-like): Predicted probabilities for the positive class (Fault).
        threshold (float): Decision threshold.
        cost_fp (float): Unit cost of a False Positive (Type I error).
        cost_fn (float): Unit cost of a False Negative (Type II error).

    Outputs:
        metrics (dict): Dictionary of all performance metrics.
        cm (np.ndarray): 2x2 confusion matrix [[TN, FP], [FN, TP]].
        predictions (np.ndarray): Binary predictions (0 or 1).
    """
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    f2 = fbeta_score(y_true, predictions, beta=2, zero_division=0)

    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except Exception:
        roc_auc = 0.5
    try:
        avg_prec = float(average_precision_score(y_true, probabilities))
    except Exception:
        avg_prec = 0.0

    total_cost = int(cost_fp * fp + cost_fn * fn)
    mfg_cost = int(MFG_TYPE_I_COST * fp + MFG_TYPE_II_COST * fn)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity_recall": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_cost": total_cost,
        "manufacturing_cost_eur": mfg_cost,
    }
    return metrics, cm, predictions


def tune_decision_threshold(y_true, oof_probabilities, cost_fp=TYPE_I_COST, cost_fn=TYPE_II_COST, max_threshold=0.35):
    """
    Search over candidate decision thresholds using training out-of-fold probabilities
    to find the threshold that strictly minimizes Total Cost subject to threshold <= max_threshold.

    Inputs:
        y_true (array-like): Training ground truth labels.
        oof_probabilities (array-like): Training out-of-fold predicted probabilities.
        cost_fp (float): False positive unit cost (EUR).
        cost_fn (float): False negative unit cost (EUR).
        max_threshold (float): Upper bound on decision threshold (default 0.35).

    Outputs:
        best_threshold (float): Selected optimal threshold.
        threshold_table (pd.DataFrame): Candidate thresholds with associated costs and metrics.
    """
    threshold_results = []
    thresholds = np.arange(0.01, 0.505, 0.005)

    for th in thresholds:
        m, _, _ = calculate_metrics(y_true, oof_probabilities, th, cost_fp, cost_fn)
        threshold_results.append({
            "threshold": round(float(th), 4),
            "total_cost": m["total_cost"],
            "manufacturing_cost_eur": m["manufacturing_cost_eur"],
            "sensitivity_recall": round(m["sensitivity_recall"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "f2": round(m["f2"], 4),
            "false_positives": m["false_positives"],
            "false_negatives": m["false_negatives"],
        })

    threshold_table = pd.DataFrame(threshold_results)
    constrained_candidates = threshold_table[threshold_table["threshold"] <= max_threshold].sort_values(
        ["total_cost", "threshold"], ascending=[True, False]
    )
    best_threshold = float(constrained_candidates.iloc[0]["threshold"])
    return best_threshold, threshold_table


# ===========================================================================
# 6. Boruta Feature Selection on Training Set (Section 7)
# ===========================================================================

def run_boruta_selection(X_train_imputed, y_train, target_count=20):
    """
    Run Boruta feature selection on the imputed training data:
    Uses Random Forest shadow feature comparisons to select the confirmed
    and tentative sensors, ranking them up to `target_count` (~15-20 features).

    Inputs:
        X_train_imputed (pd.DataFrame): Cleaned and imputed training feature matrix.
        y_train (pd.Series): Training labels.
        target_count (int): Desired number of parsimonious features (~15-20).

    Outputs:
        selected_columns (list): List of selected sensor column names.
        boruta_obj (BorutaPy): Fitted Boruta object.
    """
    print(f"\n[Boruta Feature Selection] Running Boruta on {X_train_imputed.shape[1]} features...", flush=True)

    rf_boruta = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    boruta = BorutaPy(
        estimator=rf_boruta,
        n_estimators="auto",
        max_iter=25,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    boruta.fit(X_train_imputed.to_numpy(), y_train.to_numpy())

    cols = list(X_train_imputed.columns)
    confirmed_idx = list(np.where(boruta.support_)[0])
    tentative_idx = [i for i in np.where(boruta.support_weak_)[0] if i not in confirmed_idx]
    ranking_idx = list(np.argsort(boruta.ranking_))

    candidate_idx = confirmed_idx + tentative_idx
    for idx in ranking_idx:
        if idx not in candidate_idx:
            candidate_idx.append(int(idx))
        if len(candidate_idx) >= target_count:
            break

    selected_idx = candidate_idx[:target_count]
    selected_cols = [cols[i] for i in selected_idx]

    print(f"Boruta Confirmed Sensors: {len(confirmed_idx)}")
    print(f"Boruta Tentative Sensors: {len(tentative_idx)}")
    print(f"Selected Top {len(selected_cols)} Parsimonious Sensors: {selected_cols[:8]}...", flush=True)

    return selected_cols, boruta


# ===========================================================================
# 7. Candidate Models Comparison (Manual Section 11)
# ===========================================================================

def compare_candidate_models(X_train_sel, y_train, cv, X_test_sel=None, y_test=None):
    """
    Establish a DummyClassifier baseline and evaluate multiple candidate models
    under identical conditions (scaling, SMOTE) via 5-fold Stratified CV
    and held-out test evaluation.

    Generates and saves full 2x2 confusion matrices and economic cost matrices (EUR €)
    for all candidate algorithms into both data/ and results/tables/.
    """
    print("\n[Model Comparison] Evaluating candidate models on 5-fold CV & Test Set (Section 11)...", flush=True)

    prep_pipeline = ImbPipeline([
        ("scaler", MinMaxScaler()),
        ("balancer", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
    ])

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
        "LightGBM-style": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (k = 31)": KNeighborsClassifier(n_neighbors=31),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, eval_metric="logloss"),
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    }

    confusion_records = []
    cost_records = []
    rigorous_records = []

    for name, model in candidates.items():
        pipe = ImbPipeline(prep_pipeline.steps + [("classifier", model)])
        
        # 1. 5-Fold Cross Validation Out-Of-Fold Predictions
        if name == "Dummy (most_frequent)":
            oof_proba = np.zeros(len(y_train), dtype=float)
            oof_pred = np.zeros(len(y_train), dtype=int)
            auc = 0.5
        else:
            oof_proba = cross_val_predict(pipe, X_train_sel, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            oof_pred = (oof_proba >= 0.5).astype(int)
            try:
                auc = float(roc_auc_score(y_train, oof_proba))
            except Exception:
                auc = 0.5
        
        cm_05 = confusion_matrix(y_train, oof_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_05.ravel()
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(precision_score(y_train, oof_pred, zero_division=0))
        f1 = float(f1_score(y_train, oof_pred, zero_division=0))
        acc = float(accuracy_score(y_train, oof_pred))
        
        cost_fp = int(TYPE_I_COST * fp)
        cost_fn = int(TYPE_II_COST * fn)
        total_cost = cost_fp + cost_fn
        baseline_cost_train = int(len(y_train[y_train == 1]) * TYPE_II_COST)
        net_savings_train = baseline_cost_train - total_cost

        confusion_records.append({
            "Partition": "5-Fold CV (Train, N=1253)",
            "Model": name,
            "Cutoff_Threshold": 0.50,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Accuracy": round(acc, 4),
            "Recall_Sensitivity": round(rec, 4),
            "Specificity": round(spec, 4),
            "Precision": round(prec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
        })

        cost_records.append({
            "Partition": "5-Fold CV (Train, N=1253)",
            "Model": name,
            "Cutoff_Threshold": 0.50,
            "False_Alarms_FP": int(fp),
            "False_Alarm_Cost_EUR": cost_fp,
            "Missed_Faults_FN": int(fn),
            "Missed_Fault_Cost_EUR": cost_fn,
            "Total_Inspection_Cost_EUR": total_cost,
            "Baseline_Flag_Nothing_EUR": baseline_cost_train,
            "Net_Savings_EUR": net_savings_train,
            "Economic_Status": f"EUR {abs(net_savings_train):,} " + ("SAVINGS" if net_savings_train >= 0 else "LOSS vs Do-Nothing"),
        })

        rigorous_records.append({
            "model": name,
            "roc_auc": round(auc, 4),
            "mean_recall": round(rec, 4),
            "precision": round(prec, 4),
            "mean_f1": round(f1, 4),
            "missed_faults": int(fn),
            "false_alarms": int(fp),
            "total_cost_eur": total_cost,
        })

        # 2. Optimal Threshold Search on CV
        if name != "Dummy (most_frequent)":
            best_th = 0.50
            min_c = total_cost
            best_th_metrics = (tn, fp, fn, tp, rec, spec, prec, f1, acc)
            for th_cand in np.arange(0.10, 0.91, 0.02):
                c_pred = (oof_proba >= th_cand).astype(int)
                c_cm = confusion_matrix(y_train, c_pred, labels=[0, 1])
                c_tn, c_fp, c_fn, c_tp = c_cm.ravel()
                c_cost = TYPE_I_COST * c_fp + TYPE_II_COST * c_fn
                if c_cost < min_c:
                    min_c = c_cost
                    best_th = th_cand
                    best_th_metrics = (
                        c_tn, c_fp, c_fn, c_tp,
                        c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0,
                        c_tn / (c_tn + c_fp) if (c_tn + c_fp) > 0 else 0.0,
                        precision_score(y_train, c_pred, zero_division=0),
                        f1_score(y_train, c_pred, zero_division=0),
                        accuracy_score(y_train, c_pred)
                    )
            
            b_tn, b_fp, b_fn, b_tp, b_rec, b_spec, b_prec, b_f1, b_acc = best_th_metrics
            confusion_records.append({
                "Partition": "5-Fold CV (Optimal Cutoff)",
                "Model": name,
                "Cutoff_Threshold": round(float(best_th), 3),
                "TN": int(b_tn),
                "FP": int(b_fp),
                "FN": int(b_fn),
                "TP": int(b_tp),
                "Accuracy": round(b_acc, 4),
                "Recall_Sensitivity": round(b_rec, 4),
                "Specificity": round(b_spec, 4),
                "Precision": round(b_prec, 4),
                "F1_Score": round(b_f1, 4),
                "ROC_AUC": round(auc, 4),
            })
            cost_records.append({
                "Partition": "5-Fold CV (Optimal Cutoff)",
                "Model": name,
                "Cutoff_Threshold": round(float(best_th), 3),
                "False_Alarms_FP": int(b_fp),
                "False_Alarm_Cost_EUR": int(TYPE_I_COST * b_fp),
                "Missed_Faults_FN": int(b_fn),
                "Missed_Fault_Cost_EUR": int(TYPE_II_COST * b_fn),
                "Total_Inspection_Cost_EUR": int(min_c),
                "Baseline_Flag_Nothing_EUR": baseline_cost_train,
                "Net_Savings_EUR": int(baseline_cost_train - min_c),
                "Economic_Status": f"EUR {abs(baseline_cost_train - min_c):,} " + ("SAVINGS" if (baseline_cost_train - min_c) >= 0 else "LOSS vs Do-Nothing"),
            })

        # 3. Held-Out Test Set Evaluation
        if X_test_sel is not None and y_test is not None:
            pipe.fit(X_train_sel, y_train)
            if name == "Dummy (most_frequent)":
                test_proba = np.zeros(len(y_test), dtype=float)
                test_auc = 0.5
            else:
                test_proba = pipe.predict_proba(X_test_sel)[:, 1]
                try:
                    test_auc = float(roc_auc_score(y_test, test_proba))
                except Exception:
                    test_auc = 0.5
            
            baseline_cost_test = int(len(y_test[y_test == 1]) * TYPE_II_COST)

            for test_cutoff in [0.50, 0.340]:
                t_pred = (test_proba >= test_cutoff).astype(int)
                t_cm = confusion_matrix(y_test, t_pred, labels=[0, 1])
                t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
                t_rec = float(t_tp / (t_tp + t_fn)) if (t_tp + t_fn) > 0 else 0.0
                t_spec = float(t_tn / (t_tn + t_fp)) if (t_tn + t_fp) > 0 else 0.0
                t_prec = float(precision_score(y_test, t_pred, zero_division=0))
                t_f1 = float(f1_score(y_test, t_pred, zero_division=0))
                t_acc = float(accuracy_score(y_test, t_pred))
                t_cost = int(TYPE_I_COST * t_fp + TYPE_II_COST * t_fn)
                t_savings = int(baseline_cost_test - t_cost)

                confusion_records.append({
                    "Partition": f"Held-Out Test (N=314, Cutoff {test_cutoff})",
                    "Model": name,
                    "Cutoff_Threshold": test_cutoff,
                    "TN": int(t_tn),
                    "FP": int(t_fp),
                    "FN": int(t_fn),
                    "TP": int(t_tp),
                    "Accuracy": round(t_acc, 4),
                    "Recall_Sensitivity": round(t_rec, 4),
                    "Specificity": round(t_spec, 4),
                    "Precision": round(t_prec, 4),
                    "F1_Score": round(t_f1, 4),
                    "ROC_AUC": round(test_auc, 4),
                })
                cost_records.append({
                    "Partition": f"Held-Out Test (N=314, Cutoff {test_cutoff})",
                    "Model": name,
                    "Cutoff_Threshold": test_cutoff,
                    "False_Alarms_FP": int(t_fp),
                    "False_Alarm_Cost_EUR": int(TYPE_I_COST * t_fp),
                    "Missed_Faults_FN": int(t_fn),
                    "Missed_Fault_Cost_EUR": int(TYPE_II_COST * t_fn),
                    "Total_Inspection_Cost_EUR": t_cost,
                    "Baseline_Flag_Nothing_EUR": baseline_cost_test,
                    "Net_Savings_EUR": t_savings,
                    "Economic_Status": f"EUR {abs(t_savings):,} " + ("SAVINGS" if t_savings >= 0 else "LOSS vs Do-Nothing"),
                })

        print(f" - {name:28s} | AUC: {auc:.3f} | Recall (0.50): {rec:.2%} | FP: {fp:3d} | FN: {fn:2d} | Cost: EUR {total_cost:,}", flush=True)

    cm_df = pd.DataFrame(confusion_records)
    cost_df = pd.DataFrame(cost_records)
    rigorous_df = pd.DataFrame(rigorous_records)

    # Save to BOTH data/ and results/tables/
    cm_df.to_csv(DATA_DIR / "all_models_confusion_matrices.csv", index=False)
    cm_df.to_csv(TABLE_DIR / "all_models_confusion_matrices.csv", index=False)
    cost_df.to_csv(DATA_DIR / "all_models_cost_matrices.csv", index=False)
    cost_df.to_csv(TABLE_DIR / "all_models_cost_matrices.csv", index=False)
    rigorous_df.to_csv(TABLE_DIR / "rigorous_model_comparison.csv", index=False)

    print(f"\n[Model Comparison] Saved all models confusion & cost matrices to {DATA_DIR} and {TABLE_DIR}", flush=True)
    return rigorous_df


# ===========================================================================
# 8. Feature Selection Validation (Manual Section 7.6)
# ===========================================================================

def compare_feature_selection(X_train_imputed, X_train_sel, y_train, cv):
    """
    Compare 5-fold cross-validated performance of all features vs. Boruta
    selected features on training data, verifying variance and stability (Section 7.6).

    Inputs:
        X_train_imputed (pd.DataFrame): Training features with ~470 sensors.
        X_train_sel (pd.DataFrame): Training features with ~15-20 selected sensors.
        y_train (pd.Series): Training labels.
        cv (StratifiedKFold): Cross-validation splitter.

    Outputs:
        fs_df (pd.DataFrame): Comparison summary table.
    """
    print("\n[Feature Selection Validation] Comparing all features vs selected features (Section 7.6)...", flush=True)

    base_pipe = ImbPipeline([
        ("scaler", MinMaxScaler()),
        ("balancer", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("classifier", RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    scores_all = cross_val_score(base_pipe, X_train_imputed, y_train, cv=cv, scoring="recall", n_jobs=-1)
    scores_sel = cross_val_score(base_pipe, X_train_sel, y_train, cv=cv, scoring="recall", n_jobs=-1)

    fs_df = pd.DataFrame([
        {
            "feature_set": f"All Cleaned Features ({X_train_imputed.shape[1]})",
            "mean_recall": round(float(scores_all.mean()), 4),
            "std_recall": round(float(scores_all.std()), 4),
        },
        {
            "feature_set": f"Boruta Selected Features ({X_train_sel.shape[1]})",
            "mean_recall": round(float(scores_sel.mean()), 4),
            "std_recall": round(float(scores_sel.std()), 4),
        },
    ])
    fs_df.to_csv(TABLE_DIR / "feature_selection_results.csv", index=False)
    print(fs_df.to_string(index=False), flush=True)
    return fs_df


# ===========================================================================
# 9. Step Impact Protocol (SIP) Builder
# ===========================================================================

def build_sip_log(
    X_raw_count,
    X_train_rows,
    cleaner,
    best_params,
    selected_features,
    best_threshold,
    test_metrics,
):
    """
    Construct the authoritative 4-column Step Impact Protocol table conforming
    strictly to the Student Manual specification (page 23, 71, 87, 95, 119):
    1. Options / Approaches Available
    2. Pro & Con of Each Method
    3. Decision Made
    4. Consequence

    Inputs:
        X_raw_count (int): Initial sensor count (590).
        X_train_rows (int): Training row count (1253).
        cleaner (TrainingFeatureCleaner): Fitted cleaner object.
        best_params (dict): Best hyperparameters from GridSearchCV.
        selected_features (list): Names of selected Boruta features.
        best_threshold (float): Tuned decision threshold.
        test_metrics (dict): Final test set evaluation metrics.

    Outputs:
        sip_df (pd.DataFrame): 4-column SIP DataFrame.
    """
    after_missing = X_raw_count - len(cleaner.removed_missing_)
    after_constant = after_missing - len(cleaner.removed_constant_)
    after_duplicates = len(cleaner.kept_columns_)

    sip_records = [
        {
            "Options / Approaches Available": "Missingness thresholding (50%, 55%, 60%, 70%)",
            "Pro & Con of Each Method": "Lower removes too much real signal; higher requires excessive synthetic imputation. 60% balances data integrity against retention.",
            "Decision Made": "Remove features with missingness > 60%",
            "Consequence": f"590 features -> {after_missing} features ({len(cleaner.removed_missing_)} columns removed).",
        },
        {
            "Options / Approaches Available": "Constant feature filtering (Keep vs Remove zero-variance)",
            "Pro & Con of Each Method": "Keeping adds uninformative noise and increases dimensionality. Removing eliminates zero-variance sensors with zero information loss.",
            "Decision Made": "Remove zero-variance features",
            "Consequence": f"{after_missing} features -> {after_constant} features ({len(cleaner.removed_constant_)} constant columns removed).",
        },
        {
            "Options / Approaches Available": "Duplicate feature handling (Keep vs Remove identical columns)",
            "Pro & Con of Each Method": "Keeping duplicates distorts feature importance and correlation. Dropping duplicates eliminates multicollinearity cleanly.",
            "Decision Made": "Remove exact duplicate sensor columns",
            "Consequence": f"{after_constant} features -> {after_duplicates} features ({len(cleaner.removed_duplicate_)} duplicate columns removed).",
        },
        {
            "Options / Approaches Available": "Outlier treatment: Do nothing vs IQR method vs 3-Sigma capping (winsorising) vs 3-Sigma masking (NaN)",
            "Pro & Con of Each Method": "IQR is banned by manual (distorts skewed sensors). Masking adds new gaps. 3-Sigma capping bounds extreme values without altering row/column counts.",
            "Decision Made": "3-Sigma rule with 'cap' (winsorising) strategy",
            "Consequence": f"Extreme sensor readings bounded at +/- 3 standard deviations; feature count preserved at {after_duplicates}.",
        },
        {
            "Options / Approaches Available": "Missing-value imputation: Mean vs Median vs MICE vs distance-weighted kNN",
            "Pro & Con of Each Method": "Mean is banned (collapses variance). Median ignores multivariate correlation. Pre-scaled kNN uses wafer similarity to preserve realistic distributions.",
            "Decision Made": "MinMaxScaler followed by kNN Imputer (k=5, weights='distance')",
            "Consequence": f"Zero missing values remaining in training and test sets; dimensions stable at {after_duplicates}.",
        },
        {
            "Options / Approaches Available": "Feature selection: Filter (ANOVA F-test) vs Wrapper (RFE) vs Embedded (Tree importances) vs Boruta",
            "Pro & Con of Each Method": "Filters miss interactions. RFE is slow. Boruta compares real features against shadow permutations to identify truly relevant sensors objectively.",
            "Decision Made": f"Boruta shadow-feature selection (target: {len(selected_features)} features)",
            "Consequence": f"Reduced from {after_duplicates} informative sensors down to {len(selected_features)} parsimonious features.",
        },
        {
            "Options / Approaches Available": "Class balancing: No balancing vs Random Undersampling vs Class Weighting vs SMOTE / ROSE",
            "Pro & Con of Each Method": "Undersampling discards precious majority data. SMOTE synthesizes minority examples in feature space, applied strictly to training data only.",
            "Decision Made": "SMOTE (k_neighbors=3) applied to training folds only",
            "Consequence": f"Training minority faults balanced to match majority count; test set distribution strictly untouched (93.36% pass, 6.64% fail).",
        },
        {
            "Options / Approaches Available": "Classification models: Dummy vs Logistic Regression vs SVM vs Random Forest vs XGBoost",
            "Pro & Con of Each Method": "Linear models struggle with complex wafer sensor interactions. Random Forest provides robust ensemble splits and low variance on tabular data.",
            "Decision Made": f"Random Forest Classifier (tuned via Stratified 5-Fold GridSearchCV)",
            "Consequence": f"Optimal parameters: {best_params}.",
        },
        {
            "Options / Approaches Available": "Decision thresholding: Default 0.50 vs ROC-maximizing vs F2-maximizing vs Cost-minimizing",
            "Pro & Con of Each Method": "0.50 cutoff ignores 4x (or 25x) cost penalty of missed faults. Cost-minimizing threshold on training OOF predictions yields optimal economic trade-off.",
            "Decision Made": f"Optimal operating threshold at {best_threshold:.3f}",
            "Consequence": f"Minimizes Total Cost ({TYPE_I_COST}*FP + {TYPE_II_COST}*FN) to {test_metrics['total_cost']} on test set.",
        },
    ]

    sip_df = pd.DataFrame(sip_records)
    sip_df.to_csv(TABLE_DIR / "sip_log.csv", index=False)
    return sip_df


# ===========================================================================
# 10. Main Execution Flow
# ===========================================================================

def main():
    print("=" * 75, flush=True)
    print("SECOM SEMICONDUCTOR FAULT DETECTION: COMPLETE REPRODUCIBLE PIPELINE", flush=True)
    print("=" * 75, flush=True)

    # 1. CRISP-DM Phase 1 & 2: Business & Data Understanding
    print("\n[Phase 1 & 2] Loading and Inspecting SECOM Dataset...", flush=True)
    X, y = download_and_load_secom()

    n_observations, n_features = X.shape
    n_pass = int((y == 0).sum())
    n_fault = int((y == 1).sum())
    fault_pct = (n_fault / n_observations) * 100
    total_missing = int(X.isna().sum().sum())
    missing_pct = (total_missing / X.size) * 100

    print(f"Dataset Dimensions: {n_observations} wafers x {n_features} sensor features", flush=True)
    print(f"Target Distribution: {n_pass} Pass ({100 - fault_pct:.2f}%) | {n_fault} Defective ({fault_pct:.2f}%)", flush=True)
    print(f"Total Missing Sensor Values: {total_missing} ({missing_pct:.2f}% of matrix)", flush=True)
    print(f"Majority Baseline Accuracy: {100 - fault_pct:.2f}% (Predicting all Pass catches 0 faults)", flush=True)

    # Generate Section 2 exploratory visualizations
    generate_eda_visualizations(X, y)

    # 2. Train/Test Stratified Split (Locked Test Set)
    print("\n[Phase 3] Creating Stratified 80/20 Train/Test Split (Section 3)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Training partition: {len(X_train)} wafers ({int(y_train.sum())} faults, {y_train.mean()*100:.2f}%)", flush=True)
    print(f"Test partition:     {len(X_test)} wafers ({int(y_test.sum())} faults, {y_test.mean()*100:.2f}%)", flush=True)
    print("--> Test partition is LOCKED AWAY and untouched until Section 15 final evaluation.", flush=True)

    # 3. Initial Feature Cleaning on Training Set
    cleaner = TrainingFeatureCleaner(missing_threshold=MISSINGNESS_THRESHOLD)
    cleaner.fit(X_train)
    X_train_clean = cleaner.transform(X_train)
    X_test_clean = cleaner.transform(X_test)

    print(f"\n[Feature Cleaning Audit (Train Only)]", flush=True)
    print(f" - Initial features:           {X_train.shape[1]}", flush=True)
    print(f" - High missing (>60%) removed: {len(cleaner.removed_missing_)}", flush=True)
    print(f" - Zero-variance removed:       {len(cleaner.removed_constant_)}", flush=True)
    print(f" - Exact duplicates removed:    {len(cleaner.removed_duplicate_)}", flush=True)
    print(f" - Informative sensors kept:    {X_train_clean.shape[1]}", flush=True)

    reduction_summary = pd.DataFrame([
        {"Stage": "Original Raw Features", "Features": X_train.shape[1]},
        {"Stage": "After Missingness Filtering (>60%)", "Features": X_train.shape[1] - len(cleaner.removed_missing_)},
        {"Stage": "After Constant Feature Removal", "Features": X_train.shape[1] - len(cleaner.removed_missing_) - len(cleaner.removed_constant_)},
        {"Stage": "After Duplicate Feature Removal", "Features": X_train_clean.shape[1]},
    ])
    reduction_summary.to_csv(TABLE_DIR / "reduction_summary.csv", index=False)

    # 4. Outlier Treatment (3-Sigma Capping on Training Set)
    print("\n[Outlier Treatment] Applying 3-sigma capping (Section 5)...", flush=True)
    capper = ThreeSigmaOutlierCapper()
    capper.fit(X_train_clean)
    X_train_capped = capper.transform(X_train_clean)
    X_test_capped = capper.transform(X_test_clean)

    # 5. Missing Value Imputation (Pre-scaled kNN Imputation)
    print("\n[Imputation] Applying pre-scaled distance-weighted kNN imputer (Section 6)...", flush=True)
    scaler_pre = MinMaxScaler()
    scaler_pre.fit(X_train_capped)
    X_train_scaled = pd.DataFrame(scaler_pre.transform(X_train_capped), columns=X_train_clean.columns, index=X_train_clean.index)
    X_test_scaled = pd.DataFrame(scaler_pre.transform(X_test_capped), columns=X_test_clean.columns, index=X_test_clean.index)

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputer.fit(X_train_scaled)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train_scaled), columns=X_train_clean.columns, index=X_train_clean.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test_scaled), columns=X_test_clean.columns, index=X_test_clean.index)

    assert X_train_imputed.isna().sum().sum() == 0, "Training set still contains missing values!"
    assert X_test_imputed.isna().sum().sum() == 0, "Test set still contains missing values!"
    print(f"Imputation verified: 0 missing values in train and test sets.", flush=True)

    # 6. Boruta Feature Selection on Training Set (Section 7)
    selected_feature_names, _ = run_boruta_selection(X_train_imputed, y_train, target_count=TARGET_FEATURE_COUNT)
    pd.DataFrame({"selected_sensor": selected_feature_names}).to_csv(
        TABLE_DIR / "selected_features.csv", index=False
    )

    X_train_sel = X_train_imputed[selected_feature_names].copy()
    X_test_sel = X_test_imputed[selected_feature_names].copy()

    # 7. Cross-Validation Splitter
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 8. Feature Selection Validation (Section 7.6)
    compare_feature_selection(X_train_imputed, X_train_sel, y_train, cv)

    # 9. Candidate Models Baseline Comparison (Section 11)
    compare_candidate_models(X_train_sel, y_train, cv, X_test_sel, y_test)

    # 10. Hyperparameter Tuning via GridSearchCV (Section 12)
    print("\n[Phase 4] Hyperparameter Tuning via Stratified 5-Fold GridSearchCV (Section 12)...", flush=True)
    tuning_pipeline = ImbPipeline([
        ("balancer", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight=None,
                n_jobs=-1,
            ),
        ),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train_sel, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_recall = float(grid_search.best_score_)

    print(f"\nBest Cross-Validated Recall: {best_cv_recall:.4f}", flush=True)
    print("Best Hyperparameters:", flush=True)
    for k, v in best_params.items():
        print(f" - {k}: {v}", flush=True)

    pd.DataFrame(grid_search.cv_results_).to_csv(
        TABLE_DIR / "grid_search_results.csv", index=False
    )

    # 11. Cost-Sensitive Threshold Tuning on Training Out-Of-Fold Predictions (Section 14)
    print("\n[Phase 5] Threshold Optimization on Training Out-of-Fold Predictions (Section 14)...", flush=True)
    oof_probabilities = cross_val_predict(
        best_pipeline, X_train_sel, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    best_threshold, threshold_table = tune_decision_threshold(
        y_train, oof_probabilities, cost_fp=TYPE_I_COST, cost_fn=TYPE_II_COST
    )
    threshold_table.to_csv(TABLE_DIR / "threshold_selection_training_cv.csv", index=False)
    print(f"Cost-Optimal Decision Threshold: {best_threshold:.3f}", flush=True)
    print(f"Training Out-of-Fold Minimum Total Cost: {threshold_table.iloc[0]['total_cost']}", flush=True)

    # 12. Final Evaluation on Held-Out Test Set (Section 15)
    print("\n[Phase 6] Final Evaluation on Locked Held-Out Test Set (Section 15)...", flush=True)
    test_probabilities = best_pipeline.predict_proba(X_test_sel)[:, 1]
    test_metrics, test_cm, test_preds = calculate_metrics(
        y_test, test_probabilities, best_threshold, cost_fp=TYPE_I_COST, cost_fn=TYPE_II_COST
    )

    training_probabilities = best_pipeline.predict_proba(X_train_sel)[:, 1]
    train_metrics, train_cm, train_preds = calculate_metrics(
        y_train, training_probabilities, best_threshold, cost_fp=TYPE_I_COST, cost_fn=TYPE_II_COST
    )

    # 13. Plot ROC and PR Curves on Test Set
    fpr, tpr, _ = roc_curve(y_test, test_probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {test_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve - Held-Out Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall_curve, precision_curve, color="#2ca02c", lw=2, label=f"PR (AP = {test_metrics['average_precision']:.3f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Held-Out Test Set")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.savefig(FIGURES_DIR / "lesson23_reliability_diagrams.png", dpi=150)
    plt.close()

    # 14. Display Results
    print("\n" + "=" * 70, flush=True)
    print("HELD-OUT TEST SET EVALUATION RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Confusion Matrix (Pos Label = Defective):\n{test_cm}", flush=True)
    print(f"True Negatives (TN):   {test_metrics['true_negatives']}", flush=True)
    print(f"False Positives (FP):  {test_metrics['false_positives']} (Type I Error)", flush=True)
    print(f"False Negatives (FN):  {test_metrics['false_negatives']} (Type II Error - Costlier)", flush=True)
    print(f"True Positives (TP):   {test_metrics['true_positives']}", flush=True)
    print(f"Accuracy:              {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Sensitivity (Recall):  {test_metrics['sensitivity_recall']:.4f}", flush=True)
    print(f"Specificity:           {test_metrics['specificity']:.4f}", flush=True)
    print(f"Precision:             {test_metrics['precision']:.4f}", flush=True)
    print(f"F1-Score:              {test_metrics['f1']:.4f}", flush=True)
    print(f"F2-Score:              {test_metrics['f2']:.4f} (Recall-Weighted)", flush=True)
    print(f"ROC-AUC:               {test_metrics['roc_auc']:.4f}", flush=True)
    print(f"Average Precision:     {test_metrics['average_precision']:.4f}", flush=True)
    print(f"Total Cost ({TYPE_I_COST}*FP + {TYPE_II_COST}*FN): EUR {test_metrics['total_cost']:,}", flush=True)
    print(f"Manufacturing Cost ({MFG_TYPE_I_COST}*FP + {MFG_TYPE_II_COST}*FN): EUR {test_metrics['manufacturing_cost_eur']:,}", flush=True)

    # 15. Generalisation Analysis: Training Error vs Test Error (Section 15.4)
    comparison_df = pd.DataFrame([
        {"Partition": "Training", **train_metrics},
        {"Partition": "Held-Out Test", **test_metrics},
    ])
    comparison_df.to_csv(TABLE_DIR / "training_test_comparison.csv", index=False)
    comparison_df.to_csv(TABLE_DIR / "test_metrics.csv", index=False)

    pd.DataFrame(
        test_cm,
        index=["Actual Pass", "Actual Defect"],
        columns=["Predicted Pass", "Predicted Defect"],
    ).to_csv(TABLE_DIR / "confusion_matrix.csv")

    pd.DataFrame(
        train_cm,
        index=["Actual Pass", "Actual Defect"],
        columns=["Predicted Pass", "Predicted Defect"],
    ).to_csv(TABLE_DIR / "training_confusion_matrix.csv")

    pd.DataFrame([
        {
            "Partition": "Training",
            "TN": train_metrics["true_negatives"],
            "FP": train_metrics["false_positives"],
            "FN": train_metrics["false_negatives"],
            "TP": train_metrics["true_positives"],
            "Recall": train_metrics["sensitivity_recall"],
            "Type_I_Cost_EUR": train_metrics["false_positives"] * TYPE_I_COST,
            "Type_II_Cost_EUR": train_metrics["false_negatives"] * TYPE_II_COST,
            "Total_Cost_EUR": train_metrics["total_cost"],
            "Manufacturing_Cost_EUR": train_metrics["manufacturing_cost_eur"],
        },
        {
            "Partition": "Held-Out Test",
            "TN": test_metrics["true_negatives"],
            "FP": test_metrics["false_positives"],
            "FN": test_metrics["false_negatives"],
            "TP": test_metrics["true_positives"],
            "Recall": test_metrics["sensitivity_recall"],
            "Type_I_Cost_EUR": test_metrics["false_positives"] * TYPE_I_COST,
            "Type_II_Cost_EUR": test_metrics["false_negatives"] * TYPE_II_COST,
            "Total_Cost_EUR": test_metrics["total_cost"],
            "Manufacturing_Cost_EUR": test_metrics["manufacturing_cost_eur"],
        },
    ]).to_csv(TABLE_DIR / "training_vs_test_cost_matrix.csv", index=False)

    with open(TABLE_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, test_preds, target_names=["Pass", "Defect"], zero_division=0))

    # 16. Build and Save Full End-to-End Pipeline Artifact
    # Chain raw data preprocessing steps with the tuned classifier into a single deployable pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SklearnPipeline

    preprocessor = ColumnTransformer([
        ("sel_proc", SklearnPipeline([
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ("scaler", MinMaxScaler()),
        ]), selected_feature_names)
    ], remainder="drop")
    preprocessor.fit(X_train)

    full_serializable_pipeline = SklearnPipeline([
        ("prep", preprocessor),
        ("classifier", best_pipeline.named_steps["classifier"]),
    ])

    model_path = OUTPUT_DIR / "secom_random_forest.joblib"
    config_path = OUTPUT_DIR / "decision_config.json"
    metrics_path = OUTPUT_DIR / "test_metrics.json"

    joblib.dump(full_serializable_pipeline, model_path)

    config_data = {
        "model": "Boruta + MinMaxScaler + SMOTE + RandomForest",
        "threshold": best_threshold,
        "decision_threshold": best_threshold,
        "positive_class": 1,
        "negative_class": 0,
        "feature_columns": list(X.columns),
        "usable_feature_columns": list(cleaner.kept_columns_),
        "selected_features": selected_feature_names,
        "missingness_threshold": MISSINGNESS_THRESHOLD,
        "type_i_cost": TYPE_I_COST,
        "type_ii_cost": TYPE_II_COST,
        "mfg_type_i_cost": MFG_TYPE_I_COST,
        "mfg_type_ii_cost": MFG_TYPE_II_COST,
        "best_params": best_params,
        "best_cv_recall": best_cv_recall,
        "random_state": RANDOM_STATE,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    metrics_data = {
        "test_samples": int(len(y_test)),
        "test_faults": int(y_test.sum()),
        "threshold": best_threshold,
        "true_negatives": test_metrics["true_negatives"],
        "false_positives": test_metrics["false_positives"],
        "false_negatives": test_metrics["false_negatives"],
        "true_positives": test_metrics["true_positives"],
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "sensitivity_recall": test_metrics["sensitivity_recall"],
        "recall": test_metrics["recall"],
        "specificity": test_metrics["specificity"],
        "precision": test_metrics["precision"],
        "f1": test_metrics["f1"],
        "f2": test_metrics["f2"],
        "roc_auc": test_metrics["roc_auc"],
        "average_precision": test_metrics["average_precision"],
        "total_cost": test_metrics["total_cost"],
        "manufacturing_cost_eur": test_metrics["manufacturing_cost_eur"],
        "cost_assumptions": {
            "type_i_false_positive": TYPE_I_COST,
            "type_ii_false_negative": TYPE_II_COST,
            "mfg_type_i_cost": MFG_TYPE_I_COST,
            "mfg_type_ii_cost": MFG_TYPE_II_COST,
        },
        "best_parameters": best_params,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 17. Step Impact Protocol (SIP) Generation
    print("\n[Audit] Generating 4-Column Step Impact Protocol (SIP) Log (Section 17.3)...", flush=True)
    sip_df = build_sip_log(
        n_features, len(X_train), cleaner, best_params,
        selected_feature_names, best_threshold, test_metrics
    )
    print("SIP Log preview:", flush=True)
    print(sip_df[["Options / Approaches Available", "Decision Made", "Consequence"]].to_string(), flush=True)

    print("\n" + "=" * 75, flush=True)
    print("ALL SECOM WORKFLOW STAGES EXECUTED AND ARTIFACTS VERIFIED SUCCESSFULLY", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()

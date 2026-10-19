"""
SECOM Case Study — reproducible starter analysis
Based on the HTW Berlin SECOM Case Study Manual.

Run:
    python secom_analysis.py

The script downloads the official SECOM files at runtime, creates a stratified
80/20 split before learning preprocessing parameters, records feature reductions,
trains a leakage-safe imbalanced-learn pipeline, and evaluates once on the held-out
test set.

Important:
- The UCI labels are conventionally -1 = pass and +1 = fail. This script maps
  +1 (faulty) to class 1 and -1 (good) to class 0.
- The test set is not used for preprocessing, feature selection, balancing,
  hyperparameter tuning, or threshold selection.
- Feature-selection method and number of features are configurable.
- This is a reproducible baseline; results should be interpreted from the
  printed test metrics, not assumed in advance.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

# Install missing packages automatically, as requested by the course manual.
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "requests": "requests",
}

def ensure_packages() -> None:
    """Install missing dependencies.

    Inputs: none.
    Outputs: required Python packages are available in the active environment.
    """
    missing = [
        package for module, package in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages()

import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Allows the script to run without opening GUI windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.20
MISSINGNESS_THRESHOLD = 0.60
N_FEATURES_TO_SELECT = 20
TYPE_I_COST = 100   # false alarm: predict faulty when wafer is good
TYPE_II_COST = 400  # missed defect: predict good when wafer is faulty
OUTPUT_DIR = Path("results")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def make_output_dirs() -> None:
    """Create folders for figures and result tables."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def download_secom() -> tuple[pd.DataFrame, pd.Series]:
    """Download and read the official SECOM data files from UCI.

    Inputs: none.
    Outputs: X (sensor columns) and y (1=faulty, 0=good).
    """
    url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
    response = requests.get(url, timeout=90)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        data_name = next((n for n in names if n.lower().endswith("secom.data")), None)
        label_name = next((n for n in names if n.lower().endswith("secom_labels.data")), None)
        if not data_name or not label_name:
            raise FileNotFoundError(
                "Could not find secom.data and secom_labels.data in the UCI ZIP. "
                f"Files found: {names}"
            )
        X = pd.read_csv(archive.open(data_name), sep=r"\s+", header=None)
        labels = pd.read_csv(archive.open(label_name), sep=r"\s+", header=None)

    if len(X) != len(labels):
        raise ValueError(f"Feature/label row mismatch: {len(X)} vs {len(labels)}")

    raw_y = labels.iloc[:, 0].astype(int)
    # Standard UCI coding: -1 = pass, +1 = fail.
    y = raw_y.map({-1: 0, 1: 1})
    if y.isna().any():
        raise ValueError(f"Unexpected target labels: {sorted(raw_y.unique())}")

    X.columns = [f"sensor_{i:03d}" for i in range(1, X.shape[1] + 1)]
    return X.replace([np.inf, -np.inf], np.nan), y.astype(int).rename("faulty")


class TrainingFeatureCleaner(BaseEstimator, TransformerMixin):
    """Drop columns based on training-set missingness, constants, and duplicates.

    Fit learns which columns to retain from the training data only.
    Transform applies the same retained-column list to any later data.
    """

    def __init__(self, missingness_threshold: float = 0.60):
        self.missingness_threshold = missingness_threshold

    def fit(self, X: pd.DataFrame, y=None):
        """Learn the retained columns from X only."""
        frame = pd.DataFrame(X).copy()
        self.input_columns_ = list(frame.columns)
        missing_rate = frame.isna().mean()
        keep = list(frame.columns[missing_rate <= self.missingness_threshold])
        self.removed_missing_ = [c for c in frame.columns if c not in keep]

        frame = frame[keep]
        nonconstant = frame.nunique(dropna=True) > 1
        self.removed_constant_ = list(frame.columns[~nonconstant])
        frame = frame.loc[:, nonconstant]

        # Keep the first of any exactly duplicated columns.
        duplicate_mask = frame.T.duplicated()
        self.removed_duplicate_ = list(frame.columns[duplicate_mask])
        frame = frame.loc[:, ~duplicate_mask]

        self.kept_columns_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns chosen during fit."""
        frame = pd.DataFrame(X).copy()
        return frame.reindex(columns=self.kept_columns_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


class ThreeSigmaCapper(BaseEstimator, TransformerMixin):
    """Cap extreme values at training-derived mean ± 3 standard deviations.

    NaNs are preserved for the subsequent KNN imputer.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.mean_ = frame.mean()
        self.std_ = frame.std(ddof=0).replace(0, np.nan)
        self.lower_ = self.mean_ - 3 * self.std_
        self.upper_ = self.mean_ + 3 * self.std_
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        return frame.clip(lower=self.lower_, upper=self.upper_, axis=1)


def save_eda_outputs(X: pd.DataFrame, y: pd.Series) -> None:
    """Save basic descriptive statistics and visualisations."""
    summary = pd.DataFrame({
        "metric": ["rows", "sensor_features", "faulty_count", "good_count",
                   "faulty_percentage", "total_missing_values"],
        "value": [len(X), X.shape[1], int((y == 1).sum()), int((y == 0).sum()),
                  round(100 * float((y == 1).mean()), 3), int(X.isna().sum().sum())],
    })
    summary.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    X.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_csv(
        TABLE_DIR / "missingness_by_feature.csv"
    )

    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Good (0)", "Faulty (1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("SECOM target class distribution")
    plt.ylabel("Number of wafers")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=160)
    plt.close()

    missing = X.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    plt.hist(missing * 100, bins=30)
    plt.axvline(MISSINGNESS_THRESHOLD * 100, linestyle="--",
                label=f"Threshold = {MISSINGNESS_THRESHOLD:.0%}")
    plt.title("Missing-value percentage across sensor features")
    plt.xlabel("Missing values per feature (%)")
    plt.ylabel("Number of features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missingness_histogram.png", dpi=160)
    plt.close()


def write_sip(cleaner: TrainingFeatureCleaner, X_train: pd.DataFrame,
              X_test: pd.DataFrame, y_train: pd.Series) -> None:
    """Save an initial Step Impact Protocol (SIP) log."""
    rows = [
        {
            "step": "Remove features above missingness threshold",
            "options_considered": "55%, 60%, 70%; inspect missingness distribution",
            "pros_cons": "Lower threshold reduces more incomplete features but may discard useful signals",
            "decision": f"Remove features with > {MISSINGNESS_THRESHOLD:.0%} missing values",
            "before": X_train.shape[1],
            "after": X_train.shape[1] - len(cleaner.removed_missing_),
            "removed": len(cleaner.removed_missing_),
        },
        {
            "step": "Remove constant features",
            "options_considered": "Keep all / remove zero-variance features",
            "pros_cons": "Constant features provide no separation; removal is low risk",
            "decision": "Remove zero-variance features",
            "before": X_train.shape[1] - len(cleaner.removed_missing_),
            "after": X_train.shape[1] - len(cleaner.removed_missing_) - len(cleaner.removed_constant_),
            "removed": len(cleaner.removed_constant_),
        },
        {
            "step": "Remove duplicate features",
            "options_considered": "Keep duplicates / keep one copy",
            "pros_cons": "Duplicate columns add redundancy without new information",
            "decision": "Keep first copy of each exact duplicate",
            "before": X_train.shape[1] - len(cleaner.removed_missing_) - len(cleaner.removed_constant_),
            "after": len(cleaner.kept_columns_),
            "removed": len(cleaner.removed_duplicate_),
        },
        {
            "step": "Outlier handling",
            "options_considered": "Leave / cap at 3σ / remove rows",
            "pros_cons": "Capping limits extreme influence without discarding scarce faulty examples",
            "decision": "Cap values at training mean ± 3 standard deviations",
            "before": len(cleaner.kept_columns_),
            "after": len(cleaner.kept_columns_),
            "removed": 0,
        },
        {
            "step": "Missing-value imputation",
            "options_considered": "Mean (prohibited by manual) / median / kNN",
            "pros_cons": "kNN uses similarity between observations; requires scaling and can be slower",
            "decision": "Min-max scale then kNN imputation with k=5",
            "before": len(cleaner.kept_columns_),
            "after": len(cleaner.kept_columns_),
            "removed": 0,
        },
        {
            "step": "Feature selection",
            "options_considered": "Filter / wrapper / embedded / Boruta",
            "pros_cons": "Filter is fast; wrapper/embedded methods can capture model-specific relevance but cost more",
            "decision": f"Select up to {N_FEATURES_TO_SELECT} features using ANOVA F-score within the training pipeline",
            "before": len(cleaner.kept_columns_),
            "after": min(N_FEATURES_TO_SELECT, len(cleaner.kept_columns_)),
            "removed": max(0, len(cleaner.kept_columns_) - N_FEATURES_TO_SELECT),
        },
        {
            "step": "Class imbalance",
            "options_considered": "SMOTE / undersampling / class weighting",
            "pros_cons": "SMOTE creates synthetic minority examples; must only be used during training",
            "decision": "SMOTE on training folds only",
            "before": int((y_train == 0).sum() + (y_train == 1).sum()),
            "after": "Balanced inside model fitting only",
            "removed": 0,
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "sip_log.csv", index=False)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate once on the untouched test set and save confusion matrix/ROC."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Positive class 1 means faulty.
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    cost = TYPE_I_COST * fp + TYPE_II_COST * fn

    metrics = {
        "decision_threshold_selected_from_training_cv": float(threshold),
        "accuracy": accuracy_score(y_test, pred),
        "precision_faulty": precision_score(y_test, pred, zero_division=0),
        "sensitivity_recall_faulty": recall_score(y_test, pred, zero_division=0),
        "specificity_good": specificity,
        "f1_faulty": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "average_precision": average_precision_score(y_test, proba),
        "true_negatives_good_correct": int(tn),
        "false_positives_false_alarms": int(fp),
        "false_negatives_missed_faults": int(fn),
        "true_positives_faults_caught": int(tp),
        "cost_type_i_100_type_ii_400": int(cost),
        "majority_class_baseline_accuracy": float((y_test == 0).mean()),
    }
    pd.DataFrame([metrics]).to_csv(TABLE_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(cm, index=["Actual good", "Actual faulty"],
                 columns=["Predicted good", "Predicted faulty"]).to_csv(
        TABLE_DIR / "confusion_matrix.csv"
    )
    (TABLE_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["Good", "Faulty"], zero_division=0),
        encoding="utf-8"
    )

    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC curve — held-out test set")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=160)
    plt.close()

    return metrics


def main() -> None:
    """Run the complete reproducible analysis."""
    make_output_dirs()
    print("\n[1/7] Downloading SECOM dataset...")
    X, y = download_secom()
    print(f"Loaded X={X.shape}; target counts: {y.value_counts().to_dict()}")
    print(f"Faulty proportion: {y.mean():.2%}")

    print("\n[2/7] Saving descriptive statistics and EDA figures...")
    save_eda_outputs(X, y)

    # Split first. Test labels remain unused until final evaluation.
    print("\n[3/7] Creating stratified 80/20 train/test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training rows: {len(X_train_raw)}; test rows: {len(X_test_raw)}")
    print(f"Faulty share — full: {y.mean():.3%}, train: {y_train.mean():.3%}, "
          f"test: {y_test.mean():.3%}")

    print("\n[4/7] Fitting feature cleaner on training data only...")
    cleaner = TrainingFeatureCleaner(MISSINGNESS_THRESHOLD)
    X_train = cleaner.fit_transform(X_train_raw)
    X_test = cleaner.transform(X_test_raw)
    print(f"Original features: {X.shape[1]}")
    print(f"Removed high-missing: {len(cleaner.removed_missing_)}")
    print(f"Removed constant: {len(cleaner.removed_constant_)}")
    print(f"Removed duplicates: {len(cleaner.removed_duplicate_)}")
    print(f"Remaining features: {len(cleaner.kept_columns_)}")

    # Record the chosen 3-sigma strategy. The capper itself is fitted inside
    # the pipeline, separately within each CV fold, then on all training data.
    write_sip(cleaner, X_train_raw, X_test_raw, y_train)

    print("\n[5/7] Building a leakage-safe model pipeline...")
    # MinMaxScaler preserves NaNs; scaling before KNNImputer makes distance
    # calculations more comparable across sensors. All steps are fit on training data.
    k = min(N_FEATURES_TO_SELECT, X_train.shape[1])
    pipeline = Pipeline(steps=[
        ("outlier_capper", ThreeSigmaCapper()),
        ("scale_for_knn", MinMaxScaler()),
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("selector", SelectKBest(score_func=f_classif, k=k)),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        )),
    ])

    # Grid search uses only the training partition. Keep the grid modest enough
    # to run on a typical student laptop; expand it only if runtime allows.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {
        "selector__k": sorted(set([min(10, k), min(20, k), k])),
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    print("Grid search optimizes faulty-class recall using training-only CV.")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best CV recall:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)
    pd.DataFrame(search.cv_results_).to_csv(TABLE_DIR / "grid_search_results.csv", index=False)

    # Choose a decision threshold from out-of-fold predictions on training data.
    # The held-out test set is not involved in threshold selection.
    print("Selecting a decision threshold using training-only out-of-fold predictions...")
    oof_probabilities = cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    threshold_rows = []
    for candidate_threshold in np.linspace(0.05, 0.95, 91):
        oof_pred = (oof_probabilities >= candidate_threshold).astype(int)
        tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(
            y_train, oof_pred, labels=[0, 1]
        ).ravel()
        threshold_rows.append({
            "threshold": float(candidate_threshold),
            "training_oof_cost": int(TYPE_I_COST * fp_cv + TYPE_II_COST * fn_cv),
            "recall": float(tp_cv / (tp_cv + fn_cv)) if tp_cv + fn_cv else 0.0,
            "precision": float(tp_cv / (tp_cv + fp_cv)) if tp_cv + fp_cv else 0.0,
        })
    threshold_table = pd.DataFrame(threshold_rows).sort_values(
        ["training_oof_cost", "threshold"], ascending=[True, False]
    )
    threshold_table.to_csv(TABLE_DIR / "threshold_selection_training_cv.csv", index=False)
    chosen_threshold = float(threshold_table.iloc[0]["threshold"])
    print(f"Selected threshold from training CV: {chosen_threshold:.2f}")

    # Save selected sensor names from the final fit.
    selected_mask = best_model.named_steps["selector"].get_support()
    selected_names = np.asarray(cleaner.kept_columns_)[selected_mask]
    pd.DataFrame({"selected_sensor": selected_names}).to_csv(
        TABLE_DIR / "selected_features.csv", index=False
    )

    print("\n[6/7] Evaluating once on the held-out test set...")
    metrics = evaluate_model(best_model, X_test, y_test, threshold=chosen_threshold)

    print("\n[7/7] Results")
    print("-" * 60)
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("\nSaved outputs under:", OUTPUT_DIR.resolve())
    print("Figures:", FIGURE_DIR.resolve())
    print("Tables:", TABLE_DIR.resolve())
    print("\nInterpretation reminder: prioritize recall and cost, not accuracy alone.")
    print("A false negative is a faulty wafer missed by the model and costs 400 here.")
    print("A false positive is a good wafer flagged as faulty and costs 100 here.")


if __name__ == "__main__":
    main()

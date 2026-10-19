
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
COST_RATIOS = [5, 10, 20]

# Explicitly use the dedicated NVIDIA GPU.
XGB_DEVICE = "cuda"

print("scikit-learn:", sklearn.__version__)
print("XGBoost:", xgb.__version__)
print("XGBoost device:", XGB_DEVICE)

build_info = xgb.build_info()
print("XGBoost CUDA build support:", build_info.get("USE_CUDA"))

if not build_info.get("USE_CUDA", False):
    raise RuntimeError(
        "This XGBoost installation does not report CUDA support. "
        "Fix the installation before running the GPU experiment."
    )

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="?"
)

labels = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None
)

y = (labels.iloc[:, 0] == 1).astype(int)

print("\nDataset:", X.shape)
print("Faults:", int(y.sum()))
print("Passing:", int((y == 0).sum()))


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


def mutual_information_scores(X, y):
    return mutual_info_classif(X, y, random_state=SEED)


# --------------------------------------------------
# 4. Models
# --------------------------------------------------

def make_logistic_100():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mutual_information_scores, k=100)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED,
        )),
    ])


def make_random_forest():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        )),
    ])


def make_xgboost():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=XGB_DEVICE,
            n_jobs=1,
            random_state=SEED,
        )),
    ])


models = {
    "Logistic - 100 features": make_logistic_100,
    "Random Forest - all": make_random_forest,
    "XGBoost - GPU": make_xgboost,
}


# --------------------------------------------------
# 5. Cost and threshold functions
# --------------------------------------------------

def calculate_cost(y_true, y_pred, fn_cost, fp_cost=1):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    total_cost = fn * fn_cost + fp * fp_cost

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "total_cost": int(total_cost),
        "cost_per_observation": total_cost / len(y_true),
    }


def choose_threshold_by_cost(y_true, probabilities, fn_cost):
    thresholds = np.arange(0.01, 0.501, 0.01)

    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        cost = calculate_cost(
            y_true, predictions, fn_cost
        )["cost_per_observation"]

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold


# --------------------------------------------------
# 6. Nested cross-validation
# --------------------------------------------------

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=SEED
)

fold_results = []

for model_name, model_factory in models.items():
    print(f"\nEvaluating {model_name}")
    model_start = perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=100 + fold,
        )

        # Avoid concurrent GPU jobs. CPU models can use all cores
        # for their inner cross-validation.
        cv_jobs = 1 if model_name == "XGBoost - GPU" else -1

        inner_probabilities = cross_val_predict(
            model_factory(),
            X_train,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=cv_jobs,
        )[:, 1]

        fitted_model = model_factory()
        fitted_model.fit(X_train, y_train)

        valid_probabilities = fitted_model.predict_proba(X_valid)[:, 1]

        for ratio in COST_RATIOS:
            threshold = choose_threshold_by_cost(
                y_train,
                inner_probabilities,
                fn_cost=ratio,
            )

            predictions = (
                valid_probabilities >= threshold
            ).astype(int)

            costs = calculate_cost(
                y_valid, predictions, fn_cost=ratio
            )

            fold_results.append({
                "model": model_name,
                "fold": fold,
                "FN_cost_ratio": ratio,
                "threshold": threshold,
                **costs,
                "average_precision": average_precision_score(
                    y_valid, valid_probabilities
                ),
                "roc_auc": roc_auc_score(
                    y_valid, valid_probabilities
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_valid, predictions
                ),
                "precision": precision_score(
                    y_valid, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_valid, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_valid, predictions, zero_division=0
                ),
            })

            print(
                f"  Fold {fold} | cost {ratio}:1"
                f" | threshold={threshold:.2f}"
                f" | FN={costs['FN']}"
                f" | FP={costs['FP']}"
                f" | cost={costs['total_cost']}"
            )

    print(f"Elapsed: {perf_counter() - model_start:.1f} seconds")


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

summary = (
    fold_df.groupby(["model", "FN_cost_ratio"])
    .agg(
        mean_cost=("cost_per_observation", "mean"),
        std_cost=("cost_per_observation", "std"),
        mean_FN=("FN", "mean"),
        mean_FP=("FP", "mean"),
        mean_TP=("TP", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_AP=("average_precision", "mean"),
        mean_ROC_AUC=("roc_auc", "mean"),
        mean_threshold=("threshold", "mean"),
    )
    .reset_index()
)

fold_df.to_csv(
    OUTPUT_DIR / "xgboost_comparison_folds.csv", index=False
)
summary.to_csv(
    OUTPUT_DIR / "xgboost_comparison_summary.csv", index=False
)

print("\nFINAL SUMMARY")
print(summary.round(3).to_string(index=False))
print("\nSaved results under:", OUTPUT_DIR)

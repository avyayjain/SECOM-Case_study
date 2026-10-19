
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Load all observations for nested cross-validation.
X = pd.read_csv(
    "data/secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN",
)
labels = pd.read_csv(
    "data/secom_labels.data",
    sep=r"\s+",
    header=None,
)
y = (labels.iloc[:, 0] == 1).astype(int)


# 2. Remove entirely missing features within each fitted fold.
class DropAllMissingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.keep_mask_ = ~X.isna().all(axis=0).to_numpy()
        return self

    def transform(self, X):
        return X.iloc[:, self.keep_mask_]


def mi_score(X, y):
    return mutual_info_classif(X, y, random_state=42)


# 3. Define the model pipelines.
def build_logistic(k=None):
    steps = [
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
    ]

    if k is not None:
        steps.append(
            ("select", SelectKBest(score_func=mi_score, k=k))
        )

    steps.extend([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )),
    ])

    return Pipeline(steps)


def build_forest():
    return Pipeline([
        ("drop_empty", DropAllMissingFeatures()),
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )),
    ])


models = {
    "Logistic - all": lambda: build_logistic(),
    "Logistic - 100 features": lambda: build_logistic(k=100),
    "Random Forest - all": build_forest,
}


# 4. Outer folds estimate performance.
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Thresholds are selected using inner folds only.
inner_cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=123,
)

threshold_grid = np.arange(0.02, 0.501, 0.01)
all_results = []

for model_name, model_factory in models.items():
    print(f"\nEvaluating {model_name}", flush=True)

    for fold, (train_idx, valid_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_outer_train = X.iloc[train_idx]
        y_outer_train = y.iloc[train_idx]
        X_outer_valid = X.iloc[valid_idx]
        y_outer_valid = y.iloc[valid_idx]

        # Inner out-of-fold probabilities: each inner prediction
        # comes from a model that did not train on that observation.
        inner_probabilities = cross_val_predict(
            model_factory(),
            X_outer_train,
            y_outer_train,
            cv=inner_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        # Choose the threshold using inner training predictions.
        threshold_scores = []

        for threshold in threshold_grid:
            inner_predictions = (
                inner_probabilities >= threshold
            ).astype(int)

            threshold_scores.append(
                f1_score(
                    y_outer_train,
                    inner_predictions,
                    zero_division=0,
                )
            )

        best_threshold = float(
            threshold_grid[int(np.argmax(threshold_scores))]
        )

        # Refit on the full outer training fold.
        model = model_factory()
        model.fit(X_outer_train, y_outer_train)

        # Evaluate once on the outer validation fold.
        probabilities = model.predict_proba(X_outer_valid)[:, 1]
        predictions = (probabilities >= best_threshold).astype(int)

        result = {
            "model": model_name,
            "outer_fold": fold,
            "selected_threshold": best_threshold,
            "average_precision": average_precision_score(
                y_outer_valid, probabilities
            ),
            "roc_auc": roc_auc_score(
                y_outer_valid, probabilities
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_outer_valid, predictions
            ),
            "precision": precision_score(
                y_outer_valid, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_outer_valid, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_outer_valid, predictions, zero_division=0
            ),
        }

        all_results.append(result)

        print(
            f"  Fold {fold}: threshold={best_threshold:.2f}, "
            f"recall={result['recall']:.3f}, "
            f"F1={result['f1']:.3f}, "
            f"AP={result['average_precision']:.3f}",
            flush=True,
        )


# 5. Summarize outer-fold results.
fold_results = pd.DataFrame(all_results)

metrics = [
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
]

summary_rows = []

for model_name, group in fold_results.groupby("model", sort=False):
    row = {"model": model_name}

    for metric in metrics:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)

    row["threshold_mean"] = group["selected_threshold"].mean()
    row["threshold_std"] = group["selected_threshold"].std(ddof=1)
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

print("\nNESTED CROSS-VALIDATION SUMMARY")
print(summary.round(3).to_string(index=False))

from pathlib import Path

Path("results/tables").mkdir(parents=True, exist_ok=True)
fold_results.to_csv(
    "results/tables/nested_cv_fold_results.csv",
    index=False,
)
summary.to_csv(
    "results/tables/nested_cv_summary.csv",
    index=False,
)

print("\nSaved fold-level and summary results to results/tables/")


from pathlib import Path
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
)
labels = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None,
)

y = labels.iloc[:, 0].map({-1: 0, 1: 1}).to_numpy()
X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

# Remove completely missing columns.
X = X.dropna(axis=1, how="all")

print("XGBoost:", xgb.__version__)
print("Dataset:", X.shape)
print("Faults:", int(y.sum()))
print("Fault rate:", round(y.mean() * 100, 2), "%")


# --------------------------------------------------
# 2. Cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []
start_all = time.perf_counter()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):
    fold_start = time.perf_counter()

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit imputation on training fold only.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # float32 reduces memory use.
    X_train_imp = np.asarray(
        X_train_imp, dtype=np.float32
    )
    X_test_imp = np.asarray(
        X_test_imp, dtype=np.float32
    )

    # Balance classes using training-fold labels only.
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    scale_pos_weight = negatives / max(positives, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    model.fit(X_train_imp, y_train, verbose=False)

    # Convert evaluation data to the GPU too, avoiding
    # CPU-input prediction warnings.
    X_test_gpu = xgb.DeviceQuantileDMatrix if False else None

    # predict_proba accepts NumPy input; XGBoost may issue
    # a device mismatch warning on some versions. It does
    # not invalidate predictions, but can add overhead.
    probabilities = model.predict_proba(X_test_imp)[:, 1]

    # Default threshold metrics are reported only as a
    # reference; AP and ROC-AUC are threshold-independent.
    predictions = (probabilities >= 0.5).astype(int)

    scores = {
        "fold": fold,
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision_at_0.5": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_at_0.5": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_at_0.5": f1_score(
            y_test, predictions, zero_division=0
        ),
        "seconds": round(
            time.perf_counter() - fold_start, 2
        ),
    }

    fold_results.append(scores)
    print(
        f"Fold {fold}: "
        f"AP={scores['average_precision']:.3f}, "
        f"ROC-AUC={scores['roc_auc']:.3f}, "
        f"Recall@0.5={scores['recall_at_0.5']:.3f}, "
        f"time={scores['seconds']}s"
    )

results = pd.DataFrame(fold_results)
results.to_csv(
    RESULTS_DIR / "gpu_xgboost_cv.csv",
    index=False,
)

print("\nMEAN CROSS-VALIDATION RESULTS")
for column in [
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "precision_at_0.5",
    "recall_at_0.5",
    "f1_at_0.5",
]:
    print(
        f"{column}: "
        f"{results[column].mean():.3f} "
        f"+/- {results[column].std():.3f}"
    )

print(
    "\nTotal elapsed:",
    round(time.perf_counter() - start_all, 1),
    "seconds",
)
print("Saved:", RESULTS_DIR / "gpu_xgboost_cv.csv")

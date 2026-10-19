
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Paths and configuration
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results") / "final_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


# ============================================================
# 2. Load SECOM
# ============================================================

X = pd.read_csv(
    DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN",
)

labels = pd.read_csv(
    DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None,
)

# 0 = pass, 1 = fault
y = (labels.iloc[:, 0].to_numpy() == 1).astype(int)

X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

print("Dataset:", X.shape)
print("Faults:", int(y.sum()))
print("Fault rate:", round(y.mean() * 100, 2), "%")


# ============================================================
# 3. Split before learning preprocessing rules
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Remove features that are entirely missing in TRAINING.
# The test set is then restricted to those same columns.
usable_features = X_train.columns[
    X_train.notna().any(axis=0)
]

X_train = X_train.loc[:, usable_features].copy()
X_test = X_test.loc[:, usable_features].copy()

print("\nTraining shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Training faults:", int(y_train.sum()))
print("Test faults:", int(y_test.sum()))
print("Usable features:", len(usable_features))


# ============================================================
# 4. Define the model
# ============================================================

def make_model():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# ============================================================
# 5. Choose a threshold using training data only
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Every training sample receives a probability from a model
# that did not train on that sample.
oof_probabilities = cross_val_predict(
    make_model(),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

# Search thresholds for maximum F1.
# This is a demonstration objective, not a known factory cost.
thresholds = np.arange(0.01, 0.501, 0.005)

threshold_scores = [
    f1_score(
        y_train,
        (oof_probabilities >= threshold).astype(int),
        zero_division=0,
    )
    for threshold in thresholds
]

best_index = int(np.argmax(threshold_scores))
best_threshold = float(thresholds[best_index])

print("\nThreshold selected from training OOF predictions")
print("Selected threshold:", round(best_threshold, 3))
print("OOF F1:", round(threshold_scores[best_index], 3))


# ============================================================
# 6. Fit the final model on all training data
# ============================================================

final_model = make_model()
final_model.fit(X_train, y_train)

test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# 7. Evaluate once on the held-out test set
# ============================================================

cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 55)
print("HELD-OUT TEST RESULTS")
print("=" * 55)

print("\nConfusion matrix:")
print(cm)
print("\nRows = actual [pass, fault]")
print("Columns = predicted [pass, fault]")

print("\nTrue negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)

print("\nAccuracy-independent metrics:")
print("Average Precision:", round(
    average_precision_score(y_test, test_probabilities), 4
))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, test_predictions), 4
))
print("Precision:", round(
    precision_score(y_test, test_predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_predictions, zero_division=0), 4
))
print("F1:", round(
    f1_score(y_test, test_predictions, zero_division=0), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Pass", "Fault"],
    zero_division=0,
))


# ============================================================
# 8. Save the model and its decision configuration
# ============================================================

model_path = OUTPUT_DIR / "secom_random_forest.joblib"
config_path = OUTPUT_DIR / "decision_config.json"
metrics_path = OUTPUT_DIR / "test_metrics.json"

joblib.dump(final_model, model_path)

config = {
    "model": "RandomForestClassifier",
    "decision_threshold": best_threshold,
    "positive_class": 1,
    "negative_class": 0,
    "feature_columns": list(usable_features),
    "threshold_objective": "maximum training OOF F1",
    "random_state": RANDOM_STATE,
}

config_path.write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

metrics = {
    "test_samples": int(len(y_test)),
    "test_faults": int(y_test.sum()),
    "threshold": best_threshold,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "average_precision": float(
        average_precision_score(y_test, test_probabilities)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
}

metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\nSaved model:", model_path)
print("Saved configuration:", config_path)
print("Saved test metrics:", metrics_path)
print("\nWorkflow complete.")

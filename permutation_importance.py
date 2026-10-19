
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load the data
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

y = labels.iloc[:, 0].map({-1: 0, 1: 1})

X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

# Exclude features that are completely missing in the
# full dataset. The test set is not used to fit imputation.
X = X.dropna(axis=1, how="all")

print("Dataset shape:", X.shape)
print("Faulty observations:", int(y.sum()))
print("Fault rate:", round(y.mean() * 100, 2), "%")


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\nTraining observations:", len(X_train))
print("Test observations:", len(X_test))
print("Faults in test set:", int(y_test.sum()))


# --------------------------------------------------
# 4. Build and fit the model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]
baseline_ap = average_precision_score(y_test, test_probabilities)

print("\nHeld-out average precision:", round(baseline_ap, 4))


# --------------------------------------------------
# 5. Calculate permutation importance
# --------------------------------------------------

# Shuffle one feature at a time and measure the reduction
# in average precision on the held-out test set.
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

importance = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
).sort_values("importance_mean", ascending=False)

importance["rank"] = np.arange(1, len(importance) + 1)

importance.to_csv(
    RESULTS_DIR / "permutation_importance.csv",
    index=False,
)

print("\nTOP 20 PERMUTATION IMPORTANCE")
print(
    importance[
        ["feature", "importance_mean", "importance_std", "rank"]
    ].head(20).to_string(index=False)
)


# --------------------------------------------------
# 6. Compare with Lesson 17 feature stability
# --------------------------------------------------

stability_path = RESULTS_DIR / "feature_stability_summary.csv"

if stability_path.exists():
    stability = pd.read_csv(stability_path)

    # Use the features that reached the top 20 in every fold.
    stable_features = set(
        stability.loc[
            stability["top20_frequency"] >= 1.0,
            "feature",
        ]
    )

    importance["stable_in_all_folds"] = (
        importance["feature"].isin(stable_features)
    )

    print("\nSTABLE FEATURES AND THEIR PERMUTATION IMPORTANCE")
    print(
        importance[
            importance["stable_in_all_folds"]
        ][
            [
                "feature",
                "importance_mean",
                "importance_std",
                "rank",
            ]
        ].to_string(index=False)
    )

    importance.to_csv(
        RESULTS_DIR / "permutation_importance.csv",
        index=False,
    )
else:
    print(
        "\nLesson 17 stability file not found. "
        "Run 17_feature_stability.py first to enable comparison."
    )


print("\nSaved results to:", RESULTS_DIR / "permutation_importance.csv")

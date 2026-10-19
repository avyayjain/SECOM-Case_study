
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K = 20

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
X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

print("Dataset:", X.shape)
print("Faulty observations:", int(y.sum()))

# --------------------------------------------------
# 3. Cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

importance_rows = []

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y), start=1
):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    # Fit missingness filtering using training data only.
    keep_columns = X_train.columns[
        ~X_train.isna().all(axis=0)
    ].tolist()

    X_train_kept = X_train[keep_columns]

    # Fit imputation using training data only.
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_kept)

    # Train the model.
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=SEED + fold,
        n_jobs=-1,
    )

    model.fit(X_train_imputed, y_train)

    # Rank importance within this fold.
    fold_importance = pd.DataFrame({
        "feature": keep_columns,
        "importance": model.feature_importances_,
    })

    fold_importance = fold_importance.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    fold_importance["rank"] = np.arange(
        1, len(fold_importance) + 1
    )
    fold_importance["fold"] = fold

    importance_rows.append(fold_importance)

    print(f"\nFold {fold}: top 10 features")
    print(
        fold_importance.head(10)[
            ["feature", "importance", "rank"]
        ].to_string(index=False)
    )

# --------------------------------------------------
# 4. Combine fold-level results
# --------------------------------------------------

all_importance = pd.concat(
    importance_rows, ignore_index=True
)

# A feature is in the top 20 if its rank is <= TOP_K.
all_importance["in_top_k"] = (
    all_importance["rank"] <= TOP_K
)

summary = (
    all_importance.groupby("feature")
    .agg(
        folds_present=("fold", "nunique"),
        top20_count=("in_top_k", "sum"),
        mean_rank=("rank", "mean"),
        rank_std=("rank", "std"),
        mean_importance=("importance", "mean"),
        importance_std=("importance", "std"),
    )
    .reset_index()
)

summary["top20_frequency"] = (
    summary["top20_count"] / 5
)

summary = summary.sort_values(
    ["top20_count", "mean_rank", "mean_importance"],
    ascending=[False, True, False]
)

# Save complete fold rankings and summary.
all_importance.to_csv(
    OUTPUT_DIR / "feature_stability_by_fold.csv",
    index=False
)
summary.to_csv(
    OUTPUT_DIR / "feature_stability_summary.csv",
    index=False
)

print("\nFEATURE STABILITY SUMMARY")
print(summary.head(30).round(4).to_string(index=False))

# --------------------------------------------------
# 5. Measure top-20 overlap between folds
# --------------------------------------------------

top_sets = {}

for fold in range(1, 6):
    fold_df = all_importance[
        all_importance["fold"] == fold
    ]

    top_sets[fold] = set(
        fold_df.nsmallest(TOP_K, "rank")["feature"]
    )

print("\nTOP-20 FEATURE OVERLAP")
overlap_rows = []

for fold_a in range(1, 6):
    for fold_b in range(fold_a + 1, 6):
        intersection = top_sets[fold_a] & top_sets[fold_b]
        union = top_sets[fold_a] | top_sets[fold_b]

        jaccard = (
            len(intersection) / len(union)
            if union else 0.0
        )

        overlap_rows.append({
            "fold_a": fold_a,
            "fold_b": fold_b,
            "shared_features": len(intersection),
            "jaccard_similarity": jaccard,
        })

overlap_df = pd.DataFrame(overlap_rows)

print(overlap_df.round(3).to_string(index=False))

print(
    "\nMean top-20 Jaccard similarity:",
    round(overlap_df["jaccard_similarity"].mean(), 3)
)

overlap_df.to_csv(
    OUTPUT_DIR / "feature_top20_overlap.csv",
    index=False
)

print("\nSaved results to:", OUTPUT_DIR)

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
X_FILE = BASE_DIR / "expression_matrix_supplementary.csv"
Y_FILE = BASE_DIR / "ml_sample_labels.csv"

OUT_RESULTS = BASE_DIR / "cv_results_fast.csv"
OUT_PRED = BASE_DIR / "predictions_cv_fast.csv"

# ----------------------------
# Settings
# ----------------------------
TARGET_COL = "tissue"
MIN_SAMPLES_PER_CLASS = 8
TOPK_GENES = 5000          # try 3000 if still slow
N_SPLITS = 5
RANDOM_STATE = 42

# ----------------------------
# Load X (genes x samples) and y
# ----------------------------
print("[INFO] Loading X:", X_FILE)
X = pd.read_csv(X_FILE, index_col=0)
print("[INFO] X shape (genes, samples):", X.shape)

print("[INFO] Loading y:", Y_FILE)
y_df = pd.read_csv(Y_FILE)
y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
y_df[TARGET_COL] = y_df[TARGET_COL].astype(str).str.strip()

# Remove duplicate sample_id (important)
y_df = y_df.drop_duplicates(subset=["sample_id"], keep="first")

# Align samples
common = sorted(set(X.columns).intersection(set(y_df["sample_id"])))
print("[INFO] Common samples:", len(common))
if len(common) == 0:
    raise RuntimeError("No overlapping samples between X and labels.")

X_samp = X[common].T  # samples x genes
y = y_df.set_index("sample_id").loc[common, TARGET_COL]

# ----------------------------
# Filter rare classes
# ----------------------------
counts = y.value_counts()
keep_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
keep_samples = y[y.isin(keep_classes)].index

X_samp = X_samp.loc[keep_samples]
y = y.loc[keep_samples]

print("[INFO] After filtering:")
print("  Samples:", len(y))
print("  Classes:", y.nunique())
print("  Class counts:\n", y.value_counts())

if y.nunique() < 2:
    raise RuntimeError("Not enough classes after filtering.")

# ----------------------------
# Clean X for ML speed
# ----------------------------
# 1) Make gene names strings (avoid sklearn warnings)
X_samp.columns = X_samp.columns.astype(str)

# 2) Fill NaNs (because merged studies have missing genes)
#    For a baseline, fill with 0 is acceptable and very common.
X_samp = X_samp.fillna(0.0)

# 3) Convert to float32 to reduce memory
X_samp = X_samp.astype(np.float32)

# ----------------------------
# Feature selection: Top-K genes by variance (global)
# ----------------------------
print("[INFO] Selecting Top-K genes by variance:", TOPK_GENES)
variances = X_samp.var(axis=0)  # variance per gene
top_genes = variances.sort_values(ascending=False).head(TOPK_GENES).index

X_red = X_samp.loc[:, top_genes]
print("[INFO] Reduced X shape (samples, genes):", X_red.shape)


# CV

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

models = {
    "sgd_logreg": Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE
        ))
    ]),
    "linear_svm": Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True)),
    ("clf", LinearSVC(max_iter=20000))
])
}

rows = []
pred_rows = []

for name, model in models.items():
    print(f"\n[MODEL] {name}")
    y_pred = cross_val_predict(model, X_red, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    print("  Accuracy:", acc)
    print("  F1-macro:", f1_macro)
    print("  F1-weighted:", f1_weighted)

    rows.append({
        "model": name,
        "n_samples": len(y),
        "n_classes": y.nunique(),
        "topk_genes": TOPK_GENES,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    })

    pred_rows.append(pd.DataFrame({
        "sample_id": X_red.index,
        "true_label": y.values,
        "pred_label": y_pred,
        "model": name
    }))

    # save confusion matrix per model
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true::{c}" for c in labels], columns=[f"pred::{c}" for c in labels])
    cm_df.to_csv(BASE_DIR / f"confusion_matrix_{name}_fast.csv")

results = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False)
preds = pd.concat(pred_rows, ignore_index=True)

results.to_csv(OUT_RESULTS, index=False)
preds.to_csv(OUT_PRED, index=False)

print("\n[SAVED] Results:", OUT_RESULTS)
print("[SAVED] Predictions:", OUT_PRED)
print("\nTop results:")
print(results)

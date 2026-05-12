from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# Paths

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

X_FILE = BASE_DIR / "expression_matrix_supplementary.csv"   # genes x samples
Y_FILE = BASE_DIR / "ml_sample_labels.csv"                 # sample_id, type, tissue

OUT_RESULTS = BASE_DIR / "cross_domain_results.csv"
OUT_PRED = BASE_DIR / "cross_domain_predictions.csv"


# Settings

TARGET_COL = "tissue"     # prediction target
DOMAIN_COL = "type"       # Organoid vs Tissue
MIN_TRAIN_PER_CLASS = 5   # minimum samples in TRAIN domain for a tissue class
TOPK_GENES = 5000         # reduce features for speed and robustness
RANDOM_STATE = 42


# Helpers

def load_expression_matrix(path: Path) -> pd.DataFrame:
    """
    Load expression matrix saved as CSV:
    rows = genes
    cols = samples
    """
    X = pd.read_csv(path, index_col=0)
    # Make gene names strings (avoid sklearn warnings later)
    X.index = X.index.astype(str)
    return X

def align_X_y(X_genes_samples: pd.DataFrame, y_df: pd.DataFrame):
    """
    Align sample IDs between expression matrix and labels.
    Returns:
      X_samp_genes: (samples x genes) float32
      y_aligned: labels DataFrame indexed by sample_id
    """
    # Clean sample IDs
    y_df = y_df.copy()
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()

    # Drop duplicates in labels (very important)
    y_df = y_df.drop_duplicates(subset=["sample_id"], keep="first")

    common = sorted(set(X_genes_samples.columns).intersection(set(y_df["sample_id"])))
    if len(common) == 0:
        raise RuntimeError("No overlapping sample IDs between X and y.")

    # Convert to samples x genes
    X_samp = X_genes_samples[common].T
    X_samp.columns = X_samp.columns.astype(str)

    # Align y rows in same order as X_samp
    y_aligned = y_df.set_index("sample_id").loc[common]

    # Fill NaNs (multi-study union genes -> missing values are normal)
    X_samp = X_samp.fillna(0.0).astype(np.float32)

    return X_samp, y_aligned

def topk_variance_genes(X_train: pd.DataFrame, k: int) -> list[str]:
    """
    Select top-k genes by variance computed ONLY on training data (no leakage).
    """
    v = X_train.var(axis=0)
    return v.sort_values(ascending=False).head(k).index.tolist()

def evaluate_train_test(X_train, y_train, X_test, y_test, model, model_name, exp_name):
    """
    Fit model on train and evaluate on test.
    Returns metrics dict and prediction dataframe.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "experiment": exp_name,
        "model": model_name,
        "train_n": len(y_train),
        "test_n": len(y_test),
        "n_classes_train": y_train.nunique(),
        "n_classes_test": y_test.nunique(),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    pred_df = pd.DataFrame({
        "sample_id": X_test.index.astype(str),
        "true_label": y_test.values,
        "pred_label": y_pred,
        "experiment": exp_name,
        "model": model_name
    })

    # Save confusion matrix for this experiment/model
    labels = sorted(pd.unique(y_train))  # train classes (after filtering)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true::{c}" for c in labels], columns=[f"pred::{c}" for c in labels])
    cm_path = BASE_DIR / f"confusion_matrix_{exp_name}_{model_name}.csv"
    cm_df.to_csv(cm_path)

    # Print short report
    print(f"\n[{exp_name} | {model_name}]")
    print("Accuracy:", acc)
    print("F1-macro:", f1_macro)
    print("F1-weighted:", f1_weighted)
    print("Confusion matrix saved:", cm_path)

    # Optional detailed report (printed)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics, pred_df


# Main

def main():
    print("[INFO] Loading X:", X_FILE)
    X = load_expression_matrix(X_FILE)
    print("[INFO] X shape (genes, samples):", X.shape)

    print("[INFO] Loading labels:", Y_FILE)
    y_df = pd.read_csv(Y_FILE)

    # Basic cleaning for label columns
    for col in ["type", "tissue"]:
        if col in y_df.columns:
            y_df[col] = y_df[col].astype(str).str.strip()

    # Align
    X_samp, y_aligned = align_X_y(X, y_df)
    print("[INFO] Aligned samples:", X_samp.shape[0], "Aligned genes:", X_samp.shape[1])

    # Check domains present
    if DOMAIN_COL not in y_aligned.columns:
        raise RuntimeError(f"'{DOMAIN_COL}' column not found in labels file: {Y_FILE}")

    domain_counts = y_aligned[DOMAIN_COL].value_counts(dropna=False)
    print("\n[INFO] Domain counts:")
    print(domain_counts)

    have_organoid = (y_aligned[DOMAIN_COL] == "Organoid").sum()
    have_tissue = (y_aligned[DOMAIN_COL] == "Tissue").sum()

    if have_organoid == 0 or have_tissue == 0:
        print("\n[STOP] Cross-domain (3b) needs BOTH Tissue and Organoid expression samples.")
        print("Right now you have:")
        print("  Organoid samples:", have_organoid)
        print("  Tissue samples  :", have_tissue)
        print("\nSo 3(b) cannot be executed with this expression matrix.")
        print("Next step to enable 3(b): build/obtain Tissue expression matrix too (bulk tissue processed files) and merge.")
        return

    # Split into domains
    organoid_samples = y_aligned.index[y_aligned[DOMAIN_COL] == "Organoid"]
    tissue_samples = y_aligned.index[y_aligned[DOMAIN_COL] == "Tissue"]

    # Two experiments
    experiments = [
        ("train_tissue_test_organoid", tissue_samples, organoid_samples),
        ("train_organoid_test_tissue", organoid_samples, tissue_samples),
    ]

    # Models (fast & robust for p>>n)
    models = {
        "linear_svm": Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LinearSVC(max_iter=20000))
        ]),
        "sgd_logreg": Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=5000,
                tol=1e-4,
                random_state=RANDOM_STATE
            ))
        ])
    }

    all_results = []
    all_preds = []

    for exp_name, train_ids, test_ids in experiments:
        # Prepare train/test
        X_train = X_samp.loc[train_ids]
        y_train = y_aligned.loc[train_ids, TARGET_COL]

        X_test = X_samp.loc[test_ids]
        y_test = y_aligned.loc[test_ids, TARGET_COL]

        # Keep only tissue classes that exist in BOTH train and test
        common_classes = sorted(set(y_train.unique()).intersection(set(y_test.unique())))
        y_train = y_train[y_train.isin(common_classes)]
        X_train = X_train.loc[y_train.index]

        y_test = y_test[y_test.isin(common_classes)]
        X_test = X_test.loc[y_test.index]

        # Enforce minimum train samples per class (stability)
        train_counts = y_train.value_counts()
        keep_classes = train_counts[train_counts >= MIN_TRAIN_PER_CLASS].index.tolist()

        y_train = y_train[y_train.isin(keep_classes)]
        X_train = X_train.loc[y_train.index]

        y_test = y_test[y_test.isin(keep_classes)]
        X_test = X_test.loc[y_test.index]

        print(f"\n[INFO] {exp_name}")
        print("  Train samples:", len(y_train), "Test samples:", len(y_test))
        print("  Classes used:", len(keep_classes))
        print("  Train class counts:\n", y_train.value_counts())

        if len(keep_classes) < 2 or len(y_test) == 0:
            print("[SKIP] Not enough overlapping classes/samples for this experiment.")
            continue

        # Feature selection based ONLY on training set (no leakage)
        top_genes = topk_variance_genes(X_train, TOPK_GENES)
        X_train_red = X_train.loc[:, top_genes]
        X_test_red = X_test.loc[:, top_genes]

        for model_name, model in models.items():
            metrics, pred_df = evaluate_train_test(
                X_train_red, y_train,
                X_test_red, y_test,
                model, model_name, exp_name
            )
            all_results.append(metrics)
            all_preds.append(pred_df)

    # Save outputs
    if all_results:
        res_df = pd.DataFrame(all_results).sort_values(["experiment", "f1_macro"], ascending=[True, False])
        res_df.to_csv(OUT_RESULTS, index=False)
        print("\n[SAVED] Cross-domain results:", OUT_RESULTS)
        print(res_df)
    else:
        print("\n[WARN] No experiments produced results. Likely no Tissue samples or no overlapping tissue classes.")

    if all_preds:
        pred_all = pd.concat(all_preds, ignore_index=True)
        pred_all.to_csv(OUT_PRED, index=False)
        print("[SAVED] Cross-domain predictions:", OUT_PRED)

if __name__ == "__main__":
    main()

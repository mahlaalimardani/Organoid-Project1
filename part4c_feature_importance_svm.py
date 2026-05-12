from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Paths

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
X_FILE = BASE_DIR / "expression_matrix_supplementary.csv"
Y_FILE = BASE_DIR / "ml_sample_labels.csv"

OUT_OVERALL = BASE_DIR / "svm_top_genes_overall.csv"
OUT_PERCLASS = BASE_DIR / "svm_top_genes_per_class.csv"


# Settings

TARGET_COL = "tissue"
MIN_SAMPLES_PER_CLASS = 8
TOPK_GENES = 5000
RANDOM_STATE = 42

def main():
    # Load
    X = pd.read_csv(X_FILE, index_col=0)      # genes x samples
    y_df = pd.read_csv(Y_FILE)

    # Clean labels
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
    y_df[TARGET_COL] = y_df[TARGET_COL].astype(str).str.strip()
    y_df = y_df.drop_duplicates(subset=["sample_id"], keep="first")

    # Align
    common = sorted(set(X.columns).intersection(set(y_df["sample_id"])))
    X_samp = X[common].T                      # samples x genes
    y = y_df.set_index("sample_id").loc[common, TARGET_COL]

    # Filter rare classes
    counts = y.value_counts()
    keep_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    keep_samples = y[y.isin(keep_classes)].index

    X_samp = X_samp.loc[keep_samples]
    y = y.loc[keep_samples]

    # Clean matrix
    X_samp.columns = X_samp.columns.astype(str)
    X_samp = X_samp.fillna(0.0).astype(np.float32)

    print("[INFO] Samples:", len(y), "Classes:", y.nunique())
    print("[INFO] Class counts:\n", y.value_counts())

    # Top-K variance genes (computed globally for interpretation)
    variances = X_samp.var(axis=0)
    top_genes = variances.sort_values(ascending=False).head(TOPK_GENES).index
    X_red = X_samp.loc[:, top_genes]

    # Scale
    scaler = StandardScaler(with_mean=True)
    X_scaled = scaler.fit_transform(X_red)

    # Fit Linear SVM on full data
    clf = LinearSVC(max_iter=20000)
    clf.fit(X_scaled, y)

    classes = clf.classes_
    coefs = clf.coef_  # shape: (n_classes, n_features)
    genes = np.array(X_red.columns)

    # Overall importance
    overall = np.mean(np.abs(coefs), axis=0)
    overall_df = pd.DataFrame({"gene": genes, "importance": overall})
    overall_df = overall_df.sort_values("importance", ascending=False)
    overall_df.to_csv(OUT_OVERALL, index=False)
    print("Saved:", OUT_OVERALL)

    # Per-class top genes (one-vs-rest weight vector)
    rows = []
    for i, cls in enumerate(classes):
        imp = np.abs(coefs[i])
        top_idx = np.argsort(-imp)[:100]  # top 100 per class
        for rank, j in enumerate(top_idx, start=1):
            rows.append({
                "class": cls,
                "rank": rank,
                "gene": genes[j],
                "importance": float(imp[j])
            })

    perclass_df = pd.DataFrame(rows)
    perclass_df.to_csv(OUT_PERCLASS, index=False)
    print("Saved:", OUT_PERCLASS)

    print("\nTop 15 overall genes:")
    print(overall_df.head(15))

if __name__ == "__main__":
    main()

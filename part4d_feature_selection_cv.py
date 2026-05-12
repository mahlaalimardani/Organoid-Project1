from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
X_FILE = BASE_DIR / "expression_matrix_supplementary.csv"
Y_FILE = BASE_DIR / "ml_sample_labels.csv"

OUT = BASE_DIR / "feature_selection_results.csv"

TARGET_COL = "tissue"
MIN_SAMPLES_PER_CLASS = 8
TOPN_LIST = [200, 500, 1000, 2000, 5000]
N_SPLITS = 5
RANDOM_STATE = 42

def main():
    X = pd.read_csv(X_FILE, index_col=0)   # genes x samples
    y_df = pd.read_csv(Y_FILE)

    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
    y_df[TARGET_COL] = y_df[TARGET_COL].astype(str).str.strip()
    y_df = y_df.drop_duplicates(subset=["sample_id"], keep="first")

    common = sorted(set(X.columns).intersection(set(y_df["sample_id"])))
    X_samp = X[common].T
    y = y_df.set_index("sample_id").loc[common, TARGET_COL]

    # Filter rare
    counts = y.value_counts()
    keep_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    keep_samples = y[y.isin(keep_classes)].index
    X_samp = X_samp.loc[keep_samples]
    y = y.loc[keep_samples]

    # Clean
    X_samp.columns = X_samp.columns.astype(str)
    X_samp = X_samp.fillna(0.0).astype(np.float32)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    for topN in TOPN_LIST:
        print("\n[TEST] TopN genes:", topN)

        # Select TopN by variance (global baseline selection)
        variances = X_samp.var(axis=0)
        genes = variances.sort_values(ascending=False).head(topN).index
        X_red = X_samp.loc[:, genes]

        model = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LinearSVC(max_iter=20000))
        ])

        y_pred = cross_val_predict(model, X_red, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="weighted")

        print("  Accuracy:", acc)
        print("  F1-macro:", f1_macro)
        print("  F1-weighted:", f1_weighted)

        rows.append({
            "topN_genes": topN,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "n_samples": len(y),
            "n_classes": y.nunique()
        })

    res = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False)
    res.to_csv(OUT, index=False)
    print("\nSaved:", OUT)
    print(res)

if __name__ == "__main__":
    main()

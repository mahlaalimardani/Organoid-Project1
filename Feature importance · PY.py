#!/usr/bin/env python3
"""
Feature importance (workplan 3b.iii): which genes drive the tissue assignment?

Run after Train_Classifier.py, from the same folder:
    python Feature_Importance.py

Uses the linear SVM (the model that transfers best). A linear SVM has one
weight per gene and class; because genes are standardised first, the weights
can be compared between genes. A large positive weight = high expression of
that gene pushes a sample towards that class.

Two models are trained (no cross-validation, so this is fast):
  1. on organoids          -> genes that define each class in organoids
  2. on epithelium-rich GTEx tissue -> genes that define each class in tissue
Genes in the top list of BOTH models are tissue-identity genes that organoids
and real tissue share. Genes only in the organoid list may be culture-specific.

Needs Train_Classifier.py in the same folder.
Optional: Human.GRCh38.p13.annot.tsv.gz in this folder, for gene symbols.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import Train_Classifier as tc

# ---- settings you may want to change -------------------------------------
N_TOP = 50          # top genes per class that are compared between the two models
N_SHOW = 10         # genes per class printed on screen
EPI_MAIN = 5.0      # epithelial threshold for the tissue model
MIN_TISSUE_TRAIN = 20
ANNOT = tc.Path("Human.GRCh38.p13.annot.tsv.gz")
# Epithelial / muscle / stroma marker genes (Entrez) - flagged in the output
COMPOSITION = {4072: "epi", 3856: "epi", 3875: "epi", 999: "epi",
               1674: "muscle", 59: "muscle", 4629: "muscle", 1264: "muscle",
               1277: "stroma", 1278: "stroma", 1634: "stroma", 7431: "stroma"}
# ---------------------------------------------------------------------------


def gene_symbols() -> dict:
    if not ANNOT.exists():
        print(f"({ANNOT} not found - gene symbols left empty)")
        return {}
    a = pd.read_csv(ANNOT, sep="\t", usecols=["GeneID", "Symbol"])
    return dict(zip(a["GeneID"].astype(int), a["Symbol"]))


def fit_weights(X, y, gene_ids) -> pd.DataFrame:
    """Train SVM; return genes x classes weight table (only the selected genes)."""
    pipe = make_pipeline(tc.TopVariance(tc.N_GENES), StandardScaler(),
                         LinearSVC(C=0.01, class_weight="balanced", max_iter=5000))
    pipe.fit(X, y)
    svm = pipe[-1]
    return pd.DataFrame(svm.coef_.T, index=gene_ids[pipe[0].idx_], columns=svm.classes_)


def top_genes(w: pd.DataFrame, n: int) -> dict:
    return {c: list(w[c].sort_values(ascending=False).index[:n]) for c in w.columns}


def main() -> None:
    tc.OUT.mkdir(exist_ok=True)
    out = tc.OUT
    sym = gene_symbols()

    tc.log("Loading matrices ...")
    tis = tc.load_counts(tc.DATA / "tissue_counts.tsv.gz")
    org = tc.load_counts(tc.DATA / "organoid_counts.tsv.gz")
    genes = org.index.intersection(tis.index)
    org, tis = org.loc[genes], tis.loc[genes]
    gene_ids = np.asarray(genes)

    org_meta = pd.read_csv(tc.DATA / "metadata" / "samples.csv").set_index("gsm").loc[org.columns]
    tis_meta = pd.read_csv(tc.DATA / "metadata" / "tissue_samples.csv").drop_duplicates("sample_id")
    tis_meta = tis_meta.set_index("sample_id").loc[tis.columns]

    sizes = org_meta["tissue_class"].value_counts()
    classes = sorted(c for c, n in sizes.items() if n >= tc.MIN_ORGANOID_SAMPLES and c not in tc.EXCLUDE_CLASSES)
    keep_o = org_meta["tissue_class"].isin(classes).to_numpy()
    keep_t = (tis_meta["tissue_class"].isin(classes) & ~tis_meta["gtex_tissue"].isin(tc.EXCLUDE_GTEX)).to_numpy()
    X_org, X_tis = tc.normalise(org.loc[:, keep_o]), tc.normalise(tis.loc[:, keep_t])
    y_org = org_meta.loc[keep_o, "tissue_class"].to_numpy()
    y_tis = tis_meta.loc[keep_t, "tissue_class"].to_numpy()
    epi = tc.epithelial_score(tis.loc[:, keep_t])
    ok = (epi >= EPI_MAIN) | np.isin(y_tis, list(tc.EPI_EXEMPT))
    counts = pd.Series(y_tis[ok]).value_counts()
    tis_classes = sorted(counts[counts >= MIN_TISSUE_TRAIN].index)
    ok &= np.isin(y_tis, tis_classes)
    del org, tis

    tc.log("Training SVM on organoids ...")
    w_org = fit_weights(X_org, y_org, gene_ids)
    tc.log(f"Training SVM on epithelium-rich tissue (classes: {tis_classes}) ...")
    w_tis = fit_weights(X_tis[ok], y_tis[ok], gene_ids)

    top_o, top_t = top_genes(w_org, N_TOP), top_genes(w_tis, N_TOP)
    name = lambda g: sym.get(int(g), str(g))

    # long table: one row per class and top organoid gene
    rows = []
    for c in w_org.columns:
        shared = set(top_o[c]) & set(top_t.get(c, []))
        for rank, g in enumerate(top_o[c], 1):
            rows.append({"class": c, "rank_organoid": rank, "GeneID": g, "symbol": name(g),
                         "weight_organoid": round(w_org.at[g, c], 4),
                         "weight_tissue": round(w_tis.at[g, c], 4) if c in w_tis.columns and g in w_tis.index else np.nan,
                         "in_tissue_top": g in shared,
                         "composition_marker": COMPOSITION.get(int(g), "")})
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "feature_importance_svm_organoid_top.csv", index=False)

    # tissue model top genes as well
    rows_t = []
    for c in w_tis.columns:
        for rank, g in enumerate(top_t[c], 1):
            rows_t.append({"class": c, "rank_tissue": rank, "GeneID": g, "symbol": name(g),
                           "weight_tissue": round(w_tis.at[g, c], 4),
                           "in_organoid_top": g in set(top_o.get(c, []))})
    pd.DataFrame(rows_t).to_csv(out / "feature_importance_svm_tissue_top.csv", index=False)

    # full weight tables for later analysis
    w_org.rename(index=name).to_csv(out / "svm_weights_organoid.csv")
    w_tis.rename(index=name).to_csv(out / "svm_weights_tissue.csv")

    # ---- screen output -----------------------------------------------------
    print(f"\n=== Top {N_SHOW} genes per class (SVM trained on organoids); * = also top {N_TOP} in tissue model ===")
    for c in w_org.columns:
        d = detail[detail["class"] == c].head(N_SHOW)
        print(f"{c:16s} " + ", ".join(f"{s}{'*' if t else ''}" for s, t in zip(d["symbol"], d["in_tissue_top"])))

    print(f"\n=== Overlap of top {N_TOP} genes: organoid model vs tissue model ===")
    ov = []
    for c in w_org.columns:
        if c not in w_tis.columns:
            ov.append({"class": c, "shared": "-", "of": N_TOP, "shared_genes": "(no tissue class)"})
            continue
        shared = [g for g in top_o[c] if g in set(top_t[c])]
        ov.append({"class": c, "shared": len(shared), "of": N_TOP,
                   "shared_genes": ", ".join(name(g) for g in shared[:12])})
    ov = pd.DataFrame(ov)
    ov.to_csv(out / "feature_importance_overlap.csv", index=False)
    print(ov.to_string(index=False))

    print("\n=== Weight correlation per class (organoid vs tissue model, all selected genes in both) ===")
    common = w_org.index.intersection(w_tis.index)
    corr = {c: round(np.corrcoef(w_org.loc[common, c], w_tis.loc[common, c])[0, 1], 3)
            for c in w_org.columns if c in w_tis.columns}
    print(pd.Series(corr, name="pearson_r").to_string())
    print(f"(genes in both models' feature sets: {len(common)})")

    print(f"\nSaved in {out}/: feature_importance_svm_*_top.csv, feature_importance_overlap.csv, svm_weights_*.csv")


if __name__ == "__main__":
    main()
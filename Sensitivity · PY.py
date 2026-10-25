#!/usr/bin/env python3
"""
Sensitivity analysis: do the main results hold without known confounders?

Run after Train_Classifier.py, from the same folder:
    python Sensitivity.py

Four variants, each with the linear SVM (the main model):
  baseline     - same data as the main analysis
  no_sex_genes - Y-chromosome genes and XIST/TSIX removed (they encode donor
                 sex, e.g. PCDH11Y was a top Brain gene)
  no_flagged   - organoids flagged as cancer-derived or genetically modified
                 removed (flag_cancer / flag_modified from get_organoid_data.py)
  both         - both removals

For each variant: organoid -> tissue (all GTEx and epithelium-rich), and the
reverse direction (epithelium-rich tissue -> organoid). The class list is kept
the same as in the main analysis so the variants are comparable.

Needs Train_Classifier.py in the same folder.
Uses Human.GRCh38.p13.annot.tsv.gz (if present) to find Y-chromosome genes.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import Train_Classifier as tc

# ---- settings you may want to change -------------------------------------
RUN_CV = False          # organoid cross-validation per variant (+12 min each)
EPI_MAIN = 5.0
MIN_TISSUE_TRAIN = 20   # reverse direction: tissue classes with fewer samples dropped
ANNOT = tc.Path("Human.GRCh38.p13.annot.tsv.gz")
# Used in addition to the annotation file (Entrez ids): XIST, TSIX and common
# Y-chromosome genes, so the filter works even without the annotation file.
SEX_GENES_FALLBACK = {7503, 9383, 6192, 8653, 8284, 7404, 9086, 7544, 8287, 22829, 83259, 246126, 6736}
# ---------------------------------------------------------------------------


def sex_genes() -> set:
    genes = set(SEX_GENES_FALLBACK)
    if not ANNOT.exists():
        print(f"({ANNOT} not found - using the built-in list of {len(genes)} sex genes only)")
        return genes
    a = pd.read_csv(ANNOT, sep="\t", dtype=str)
    if "ChrAcc" in a.columns:          # NCBI annotation: NC_000024.x = chromosome Y
        y = a["ChrAcc"].fillna("").str.startswith("NC_000024")
    elif "Chromosome" in a.columns:
        y = a["Chromosome"].fillna("").str.upper().isin(["Y", "CHRY"])
    else:
        print(f"(no chromosome column in {ANNOT}; columns: {list(a.columns)[:10]} ...)")
        return genes
    genes |= set(a.loc[y, "GeneID"].dropna().astype(int))
    return genes


def flag(series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().eq("true").to_numpy()


def svm_pipe():
    return make_pipeline(tc.TopVariance(tc.N_GENES), StandardScaler(),
                         LinearSVC(C=0.01, class_weight="balanced", max_iter=5000))


def recall_table(y_true, y_pred) -> pd.Series:
    d = pd.DataFrame({"t": y_true, "ok": np.asarray(y_true) == np.asarray(y_pred)})
    return d.groupby("t")["ok"].mean().round(2)


def main() -> None:
    tc.OUT.mkdir(exist_ok=True)
    out = tc.OUT

    tc.log("Loading matrices ...")
    tis = tc.load_counts(tc.DATA / "tissue_counts.tsv.gz")
    org = tc.load_counts(tc.DATA / "organoid_counts.tsv.gz")
    genes = org.index.intersection(tis.index)
    org, tis = org.loc[genes], tis.loc[genes]

    org_meta = pd.read_csv(tc.DATA / "metadata" / "samples.csv").set_index("gsm").loc[org.columns]
    tis_meta = pd.read_csv(tc.DATA / "metadata" / "tissue_samples.csv").drop_duplicates("sample_id")
    tis_meta = tis_meta.set_index("sample_id").loc[tis.columns]

    # same classes as the main analysis (fixed for all variants)
    sizes = org_meta["tissue_class"].value_counts()
    classes = sorted(c for c, n in sizes.items() if n >= tc.MIN_ORGANOID_SAMPLES and c not in tc.EXCLUDE_CLASSES)
    keep_o = org_meta["tissue_class"].isin(classes).to_numpy()
    keep_t = (tis_meta["tissue_class"].isin(classes) & ~tis_meta["gtex_tissue"].isin(tc.EXCLUDE_GTEX)).to_numpy()

    meta_o = org_meta.loc[keep_o]
    y_org_all = meta_o["tissue_class"].to_numpy()
    groups_all = meta_o["gse"].to_numpy()
    y_tis = tis_meta.loc[keep_t, "tissue_class"].to_numpy()
    gtex = tis_meta.loc[keep_t, "gtex_tissue"].to_numpy()

    epi = tc.epithelial_score(tis.loc[:, keep_t])
    epi_ok = (epi >= EPI_MAIN) | np.isin(y_tis, list(tc.EPI_EXEMPT))
    counts = pd.Series(y_tis[epi_ok]).value_counts()
    rev_classes = sorted(counts[counts >= MIN_TISSUE_TRAIN].index)
    rev_tr = epi_ok & np.isin(y_tis, rev_classes)

    # confounder masks
    sg = sex_genes()
    gene_sex = np.isin(np.asarray(genes), list(sg))
    for col in ("flag_cancer", "flag_modified"):
        if col not in meta_o.columns:
            raise SystemExit(f"Column {col} missing in samples.csv - rerun get_organoid_data.py list/download.")
    flagged = flag(meta_o["flag_cancer"]) | flag(meta_o["flag_modified"])

    print(f"\nSex-chromosome genes found in the data: {gene_sex.sum()}")
    fl = pd.DataFrame({"class": y_org_all, "flagged": flagged}).groupby("class")["flagged"].agg(["sum", "size"])
    fl.columns = ["flagged", "total"]
    fl["kept"] = fl["total"] - fl["flagged"]
    print("\nOrganoids flagged as cancer-derived or modified:")
    print(fl.to_string())
    fl.to_csv(out / "sensitivity_flagged_organoids.csv")
    if (fl["kept"] < 20).any():
        print("WARNING: some classes keep fewer than 20 organoids in the 'no_flagged' variants.")

    X_org_full = tc.normalise(org.loc[:, keep_o])
    X_tis_full = tc.normalise(tis.loc[:, keep_t])
    del org, tis

    variants = {
        "baseline": (np.ones(len(genes), bool), np.ones(len(y_org_all), bool)),
        "no_sex_genes": (~gene_sex, np.ones(len(y_org_all), bool)),
        "no_flagged": (np.ones(len(genes), bool), ~flagged),
        "both": (~gene_sex, ~flagged),
    }
    cv = StratifiedGroupKFold(n_splits=tc.N_SPLITS, shuffle=True, random_state=0)
    rows, rec_fwd, rec_rev = [], {}, {}

    for vname, (gmask, smask) in variants.items():
        # Normalisation is per sample over all genes; dropping a few sex genes
        # afterwards changes the z-scores negligibly, so columns are just removed.
        X_o, X_t = X_org_full[smask][:, gmask], X_tis_full[:, gmask]
        y_o, g_o = y_org_all[smask], groups_all[smask]
        row = {"variant": vname, "organoids": int(smask.sum()), "genes": int(gmask.sum())}

        if RUN_CV:
            tc.log(f"{vname}: cross-validation ...")
            pred = cross_val_predict(svm_pipe(), X_o, y_o, groups=g_o, cv=cv)
            row["cv_organoid"] = round(float(np.mean(pred == y_o)), 3)

        tc.log(f"{vname}: organoid -> tissue ...")
        pipe = svm_pipe().fit(X_o, y_o)
        z = pipe[0].transform(X_t)
        pred_all = pipe[-1].predict(tc.domain_scale(z))
        pred_epi = pipe[-1].predict(tc.domain_scale(z[epi_ok]))
        row["org->tissue"] = round(float(np.mean(pred_all == y_tis)), 3)
        row[f"org->tissue epi{EPI_MAIN:g}"] = round(float(np.mean(pred_epi == y_tis[epi_ok])), 3)
        rec_fwd[vname] = recall_table(gtex[epi_ok], np.where(pred_epi == y_tis[epi_ok], gtex[epi_ok], "x"))

        tc.log(f"{vname}: tissue -> organoid ...")
        te = np.isin(y_o, rev_classes)
        pipe_r = svm_pipe().fit(X_t[rev_tr], y_tis[rev_tr])
        zr = pipe_r[0].transform(X_o[te])
        pred_r = pipe_r[-1].predict(tc.domain_scale(zr))
        row[f"tissue epi{EPI_MAIN:g}->org"] = round(float(np.mean(pred_r == y_o[te])), 3)
        rec = recall_table(y_o[te], pred_r)
        row[f"tissue epi{EPI_MAIN:g}->org balanced"] = round(float(rec.mean()), 3)
        rec_rev[vname] = rec
        rows.append(row)

    res = pd.DataFrame(rows)
    fwd = pd.DataFrame(rec_fwd)
    rev = pd.DataFrame(rec_rev)
    res.to_csv(out / "sensitivity_summary.csv", index=False)
    fwd.to_csv(out / "sensitivity_recall_org_to_tissue.csv")
    rev.to_csv(out / "sensitivity_recall_tissue_to_org.csv")

    print(f"\n=== Recall per GTEx tissue, organoid -> epithelium-rich tissue (epi >= {EPI_MAIN:g}) ===")
    print(fwd.to_string())
    print("\n=== Recall per organoid class, tissue -> organoid ===")
    print(rev.to_string())
    print("\n=== Sensitivity summary (linear SVM, domain-scaled accuracy) ===")
    print(res.to_string(index=False))
    print(f"\nSaved in {out}/sensitivity_*.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
More models and the reverse direction (workplan 1c and 3b.ii).

Run after Train_Classifier.py, from the same folder:
    python Train_Models.py

Part A: Random Forest and XGBoost, organoid -> tissue
        (same setup as Train_Classifier.py, which already has the SVM).
Part B: the reverse direction: train on GTEx tissue, test on organoids
        (SVM, Random Forest, XGBoost).
Also saves the genes the Random Forest relies on most (start of 3b.iii).

Needs Train_Classifier.py in the same folder (uses its settings and functions).
Requires: pip install xgboost      (if missing, XGBoost is skipped)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

import Train_Classifier as tc

# ---- settings you may want to change -------------------------------------
RUN_TREE_CV = True     # organoid cross-validation for RF/XGB (the slow part)
EPI_MAIN = 5.0         # epithelial threshold used for the filtered runs
MIN_REVERSE_TRAIN = 20  # reverse direction: tissue classes with fewer train samples are dropped
N_TOP_GENES = 50       # genes saved in the feature-importance file
ANNOT = tc.Path("Human.GRCh38.p13.annot.tsv.gz")  # optional, for gene symbols
# ---------------------------------------------------------------------------

try:
    import xgboost  # noqa: F401
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False


class BalancedXGB(ClassifierMixin, BaseEstimator):
    """XGBoost that takes text labels and weights classes like class_weight='balanced'."""

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.le_ = LabelEncoder().fit(y)
        yi = self.le_.transform(y)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=0.8, colsample_bytree=0.3,
            tree_method="hist", n_jobs=-1, random_state=0, eval_metric="mlogloss")
        self.model_.fit(X, yi, sample_weight=compute_sample_weight("balanced", yi))
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.model_.predict(X)).astype(int).ravel())


def tree_models() -> dict:
    m = {"random_forest": RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=0)}
    if HAVE_XGB:
        m["xgboost"] = BalancedXGB()
    return m


def svm():
    return LinearSVC(C=0.01, class_weight="balanced", max_iter=5000)


def gene_symbols() -> dict:
    if not ANNOT.exists():
        return {}
    a = pd.read_csv(ANNOT, sep="\t", usecols=["GeneID", "Symbol"])
    return dict(zip(a["GeneID"], a["Symbol"]))


def main() -> None:
    tc.OUT.mkdir(exist_ok=True)
    out = tc.OUT
    if not HAVE_XGB:
        print("xgboost not installed -> XGBoost skipped (pip install xgboost)")

    # ---- same data preparation as Train_Classifier.py ----------------------
    tc.log("Loading matrices ...")
    tis = tc.load_counts(tc.DATA / "tissue_counts.tsv.gz")
    org = tc.load_counts(tc.DATA / "organoid_counts.tsv.gz")
    genes = org.index.intersection(tis.index)
    org, tis = org.loc[genes], tis.loc[genes]

    org_meta = pd.read_csv(tc.DATA / "metadata" / "samples.csv").set_index("gsm").loc[org.columns]
    tis_meta = pd.read_csv(tc.DATA / "metadata" / "tissue_samples.csv").drop_duplicates("sample_id")
    tis_meta = tis_meta.set_index("sample_id").loc[tis.columns]

    sizes = org_meta["tissue_class"].value_counts()
    classes = sorted(c for c, n in sizes.items() if n >= tc.MIN_ORGANOID_SAMPLES and c not in tc.EXCLUDE_CLASSES)
    keep_o = org_meta["tissue_class"].isin(classes).to_numpy()
    keep_t = (tis_meta["tissue_class"].isin(classes) & ~tis_meta["gtex_tissue"].isin(tc.EXCLUDE_GTEX)).to_numpy()
    X_org, X_tis = tc.normalise(org.loc[:, keep_o]), tc.normalise(tis.loc[:, keep_t])
    y_org = org_meta.loc[keep_o, "tissue_class"].to_numpy()
    groups = org_meta.loc[keep_o, "gse"].to_numpy()
    y_tis = tis_meta.loc[keep_t, "tissue_class"].to_numpy()
    gtex = tis_meta.loc[keep_t, "gtex_tissue"].to_numpy()
    epi = tc.epithelial_score(tis.loc[:, keep_t])
    epi_ok = (epi >= EPI_MAIN) | np.isin(y_tis, list(tc.EPI_EXEMPT))
    gene_ids = np.asarray(genes)
    del org, tis
    tc.log(f"Organoid {X_org.shape[0]} | tissue {X_tis.shape[0]} | epithelium-rich tissue {epi_ok.sum()}")

    summary, tables = [], {}
    cv = StratifiedGroupKFold(n_splits=tc.N_SPLITS, shuffle=True, random_state=0)
    tissue_classes = sorted(set(y_tis))
    epi_classes = sorted(set(y_tis[epi_ok]))

    # ---- Part A: RF / XGB, organoid -> tissue ------------------------------
    for name, clf in tree_models().items():
        pipe = make_pipeline(tc.TopVariance(tc.N_GENES), StandardScaler(), clf)
        if RUN_TREE_CV:
            tc.log(f"{name}: organoid cross-validation ({tc.N_SPLITS} folds) ...")
            pred = cross_val_predict(pipe, X_org, y_org, groups=groups, cv=cv, verbose=2)
            summary.append(tc.report(f"cv_organoid_{name}", y_org, pred, classes))

        tc.log(f"{name}: train on organoids, test on tissue ...")
        pipe.fit(X_org, y_org)
        z = pipe[0].transform(X_tis)
        pred_t = pipe.predict(X_tis)
        pred_d = pipe[-1].predict(tc.domain_scale(z))
        pred_ed = pipe[-1].predict(tc.domain_scale(z[epi_ok]))
        summary.append(tc.report(f"organoid_to_tissue_{name}", y_tis, pred_t, tissue_classes))
        summary.append(tc.report(f"organoid_to_tissue_domainscaled_{name}", y_tis, pred_d, tissue_classes))
        summary.append(tc.report(f"epi{EPI_MAIN:g}_organoid_to_tissue_domainscaled_{name}",
                                 y_tis[epi_ok], pred_ed, epi_classes))
        tables[f"epi{EPI_MAIN:g}_organoid_to_tissue_domainscaled_{name}"] = pd.crosstab(
            pd.Series(gtex[epi_ok], name="gtex_tissue"), pd.Series(pred_ed, name="pred"))

        if name == "random_forest":
            imp = pipe[-1].feature_importances_
            ids = gene_ids[pipe[0].idx_]
            top = np.argsort(imp)[::-1][:N_TOP_GENES]
            sym = gene_symbols()
            fi = pd.DataFrame({"GeneID": ids[top], "symbol": [sym.get(int(g), "") for g in ids[top]],
                               "importance": imp[top].round(5)})
            fi.to_csv(out / "feature_importance_random_forest_organoid.csv", index=False)
            print(f"\nTop 15 genes (random forest, trained on organoids):\n{fi.head(15).to_string(index=False)}")

    # ---- Part B: reverse, tissue -> organoid -------------------------------
    for set_name, mask in [("all", np.ones(len(y_tis), bool)), (f"epi{EPI_MAIN:g}", epi_ok)]:
        y_train = y_tis[mask]
        counts = pd.Series(y_train).value_counts()
        rev_classes = sorted(counts[counts >= MIN_REVERSE_TRAIN].index)
        tr = mask & np.isin(y_tis, rev_classes)
        te = np.isin(y_org, rev_classes)
        print(f"\nReverse ({set_name}): train on {tr.sum()} tissue samples, classes {rev_classes}")
        print(f"   test on {te.sum()} organoids (organoid classes without a tissue class, e.g. Retina, left out)")

        models = {"linear_svm": svm(), **tree_models()}
        for name, clf in models.items():
            tc.log(f"reverse {set_name}: {name} ...")
            pipe = make_pipeline(tc.TopVariance(tc.N_GENES), StandardScaler(), clf)
            pipe.fit(X_tis[tr], y_tis[tr])
            z = pipe[0].transform(X_org[te])
            pred_r = pipe.predict(X_org[te])
            pred_rd = pipe[-1].predict(tc.domain_scale(z))
            summary.append(tc.report(f"rev_{set_name}_tissue_to_organoid_{name}", y_org[te], pred_r, rev_classes))
            key = f"rev_{set_name}_tissue_to_organoid_domainscaled_{name}"
            summary.append(tc.report(key, y_org[te], pred_rd, rev_classes))
            tables[key] = pd.crosstab(pd.Series(y_org[te], name="organoid_class"),
                                      pd.Series(pred_rd, name="pred"))

    # ---- output ------------------------------------------------------------
    summary = pd.DataFrame(summary).round(3)
    summary.to_csv(out / "summary_models.csv", index=False)

    for key, t in tables.items():
        if key.startswith("rev_") and not key.endswith("linear_svm"):
            continue  # keep the screen short: tree tables are in the csv files
        print(f"\n=== {key} ===")
        print(t.to_string())
    for key, t in tables.items():
        t.to_csv(out / f"crosstab_{key}.csv")

    print("\n=== Summary (all models) ===")
    print(summary.to_string(index=False))

    # compact view: one row per model and direction (domain-scaled)
    s = summary.set_index("run")["accuracy"]
    rows = []
    names = ["linear_svm", *tree_models().keys()]
    svm_prev = pd.read_csv(out / "summary.csv").set_index("run")["accuracy"] if (out / "summary.csv").exists() else pd.Series(dtype=float)
    for n in names:
        src = svm_prev if n == "linear_svm" else s
        rows.append({
            "model": n,
            "cv_organoid": src.get(f"cv_organoid_{n}", np.nan),
            "org->tissue": src.get(f"organoid_to_tissue_domainscaled_{n}", np.nan),
            f"org->tissue epi{EPI_MAIN:g}": src.get(f"epi{EPI_MAIN:g}_organoid_to_tissue_domainscaled_{n}", np.nan),
            "tissue->org": s.get(f"rev_all_tissue_to_organoid_domainscaled_{n}", np.nan),
            f"tissue epi{EPI_MAIN:g}->org": s.get(f"rev_epi{EPI_MAIN:g}_tissue_to_organoid_domainscaled_{n}", np.nan),
        })
    print("\n=== Overview: accuracy, domain-scaled (SVM organoid runs from summary.csv) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nAll results saved in {out}/ (summary_models.csv, crosstab_*.csv)")


if __name__ == "__main__":
    main()
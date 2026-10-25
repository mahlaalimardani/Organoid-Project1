#!/usr/bin/env python3

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA = Path("data")
OUT = Path("results")

# ---- settings
MIN_ORGANOID_SAMPLES = 50      # classes with fewer organoid samples are dropped
EXCLUDE_CLASSES = {"Intestine"}  # label does not say small or large intestine
N_GENES = 5000                 # most variable genes kept (chosen on train only)
N_SPLITS = 5
# GTEx sub-tissues left out of the test: sigmoid colon samples are the muscle
# layer of the gut wall (no mucosa), so they are not comparable to organoids.
EXCLUDE_GTEX = {"colon_sigmoid"}

# Part 3: epithelial filter for GTEx samples.
# Score = mean log2-CPM of EPCAM, KRT8, KRT18, CDH1 (Entrez ids below).
# From diagnose.py: correctly predicted gut/prostate samples score ~7-9,
# wrong ones ~3.5-4.5, so 5.0 separates them. All thresholds listed here are
# tested in ONE run (the slow training does not depend on the threshold).
EPI_THRESHOLDS = [4.0, 5.0, 6.0]
# Logistic regression was always worse than the SVM. Set to False to halve
# the run time.
RUN_LOGREG = True
EPI_GENES = [4072, 3856, 3875, 999]   # EPCAM, KRT8, KRT18, CDH1
# Brain is not an epithelial tissue, so the filter is not applied to it.
EPI_EXEMPT = {"Brain"}
# ---------------------------------------------------------------------------


class TopVariance(BaseEstimator, TransformerMixin):
    """Keep the k most variable genes, chosen on the training data only."""

    def __init__(self, k=5000):
        self.k = k

    def fit(self, X, y=None):
        self.idx_ = np.argsort(X.var(axis=0))[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.idx_]


def normalise(counts: pd.DataFrame) -> np.ndarray:
    """genes x samples raw counts -> samples x genes, log-CPM, z-scored per sample.

    Everything here is computed inside one sample, so it removes differences
    in sequencing depth and pipeline scale without looking at other samples.
    """
    x = counts.to_numpy(dtype=np.float32).T
    x = np.log2(x / x.sum(axis=1, keepdims=True) * 1e6 + 1)
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x


def epithelial_score(counts: pd.DataFrame) -> np.ndarray:
    """Mean log2-CPM of the epithelial marker genes, one value per sample."""
    genes = [g for g in EPI_GENES if g in counts.index]
    if not genes:
        raise SystemExit("None of the epithelial marker genes are in the tissue matrix.")
    cpm = counts.loc[genes] / counts.sum(axis=0) * 1e6
    return np.log2(cpm + 1).mean(axis=0).to_numpy()


def domain_scale(z: np.ndarray) -> np.ndarray:
    return (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)


def report(name: str, y_true, y_pred, labels) -> dict:
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    cm.to_csv(OUT / f"confusion_{name}.csv")
    return {
        "run": name,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


_T0 = time.time()


def log(msg: str) -> None:
    """Print with elapsed time, flushed at once so the IDE console shows it."""
    m, s = divmod(int(time.time() - _T0), 60)
    print(f"[{m:3d}m{s:02d}s] {msg}", flush=True)


def load_counts(path: Path) -> pd.DataFrame:
    """Read a .tsv.gz count matrix; keep a fast pickle copy next to it.

    Reading the big gzip text file takes many minutes. The first run saves a
    .pkl copy; later runs load that in seconds. If the .tsv.gz is newer
    (e.g. after a new download), the copy is rebuilt.
    """
    cache = path.with_suffix("").with_suffix(".v2.pkl")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_pickle(cache)
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Gene ids must stay integers: as float32, large Entrez ids (> 16.7 million)
    # get rounded and different genes end up with the same id.
    df.index = df.index.astype(np.int64)
    df = df.astype(np.float32)
    if not df.index.is_unique:
        raise SystemExit(f"{path}: gene ids are not unique.")
    df.to_pickle(cache)
    return df


def main() -> None:
    OUT.mkdir(exist_ok=True)

    log("Loading tissue matrix ...")
    tis = load_counts(DATA / "tissue_counts.tsv.gz")
    log("Loading organoid matrix (slow the first time, fast after that) ...")
    org = load_counts(DATA / "organoid_counts.tsv.gz")
    log(f"Loaded: organoid {org.shape}, tissue {tis.shape}")

    genes = org.index.intersection(tis.index)
    org, tis = org.loc[genes], tis.loc[genes]
    print(f"Shared genes: {len(genes)}")

    org_meta = pd.read_csv(DATA / "metadata" / "samples.csv").set_index("gsm").loc[org.columns]
    tis_meta = pd.read_csv(DATA / "metadata" / "tissue_samples.csv").drop_duplicates("sample_id")
    tis_meta = tis_meta.set_index("sample_id").loc[tis.columns]

    # classes to train on
    sizes = org_meta["tissue_class"].value_counts()
    classes = sorted(c for c, n in sizes.items() if n >= MIN_ORGANOID_SAMPLES and c not in EXCLUDE_CLASSES)
    print(f"Training classes ({len(classes)}): {classes}")
    print(f"Dropped: {sorted(set(sizes.index) - set(classes))}")

    keep_o = org_meta["tissue_class"].isin(classes).to_numpy()
    keep_t = (tis_meta["tissue_class"].isin(classes) & ~tis_meta["gtex_tissue"].isin(EXCLUDE_GTEX)).to_numpy()
    X_org, X_tis = normalise(org.loc[:, keep_o]), normalise(tis.loc[:, keep_t])
    y_org = org_meta.loc[keep_o, "tissue_class"].to_numpy()
    groups = org_meta.loc[keep_o, "gse"].to_numpy()
    y_tis = tis_meta.loc[keep_t, "tissue_class"].to_numpy()
    gtex = tis_meta.loc[keep_t, "gtex_tissue"].to_numpy()
    tissue_classes = sorted(set(y_tis))
    print(f"Organoid: {X_org.shape[0]} samples, {len(set(groups))} studies | Tissue: {X_tis.shape[0]} samples")

    # Part 3 filters: epithelium-rich tissue samples (Brain always kept)
    epi = epithelial_score(tis.loc[:, keep_t])
    exempt = np.isin(y_tis, list(EPI_EXEMPT))
    masks = {t: (epi >= t) | exempt for t in EPI_THRESHOLDS}
    kept = pd.DataFrame({f"kept_{t:g}": m for t, m in masks.items()}, index=gtex).groupby(level=0).sum()
    kept["total"] = pd.Series(gtex).value_counts()
    print(f"\nEpithelial filter (thresholds {EPI_THRESHOLDS}, exempt: {sorted(EPI_EXEMPT)}):")
    print(kept.to_string())
    kept.to_csv(OUT / "epithelial_filter_kept.csv")
    del org, tis

    models = {"linear_svm": LinearSVC(C=0.01, class_weight="balanced", max_iter=5000)}
    if RUN_LOGREG:
        models["logreg_l2"] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    summary, tables = [], {}

    for name, clf in models.items():
        pipe = make_pipeline(TopVariance(N_GENES), StandardScaler(), clf)

        # Part 1: organoid cross-validation, split by study
        log(f"{name}: cross-validation ({N_SPLITS} folds) ...")
        pred = cross_val_predict(pipe, X_org, y_org, groups=groups, cv=cv, verbose=2)
        log(f"{name}: cross-validation done")
        summary.append(report(f"cv_organoid_{name}", y_org, pred, classes))
        pd.DataFrame({"gsm": org_meta.index[keep_o], "gse": groups, "true": y_org, "pred": pred}).to_csv(
            OUT / f"predictions_cv_organoid_{name}.csv", index=False)

        # Part 2: train on organoid, test on tissue
        log(f"{name}: training on all organoids, testing on tissue ...")
        pipe.fit(X_org, y_org)
        pred_t = pipe.predict(X_tis)
        summary.append(report(f"organoid_to_tissue_{name}", y_tis, pred_t, tissue_classes))

        # Same trained model, but each gene is z-scored inside the tissue set
        # instead of with the organoid mean/sd. This removes the constant
        # organoid-vs-tissue offset of every gene. It uses the tissue samples'
        # expression values (never their labels).
        z = pipe[0].transform(X_tis)
        pred_d = pipe[-1].predict(domain_scale(z))
        summary.append(report(f"organoid_to_tissue_domainscaled_{name}", y_tis, pred_d, tissue_classes))

        preds = pd.DataFrame({"sample_id": tis_meta.index[keep_t], "gtex_tissue": gtex, "true": y_tis,
                              "epithelial_score": epi.round(2),
                              "pred": pred_t, "pred_domainscaled": pred_d})
        tables[f"organoid_to_tissue_{name}"] = pd.crosstab(preds["gtex_tissue"], preds["pred"])
        tables[f"organoid_to_tissue_domainscaled_{name}"] = pd.crosstab(preds["gtex_tissue"], preds["pred_domainscaled"])

        # Part 3: epithelium-rich tissue samples only, once per threshold.
        # Uses the same trained model (no retraining). Domain scaling is
        # recomputed on each subset, since that subset is the test set.
        for t, ok in masks.items():
            y_e, g_e = y_tis[ok], gtex[ok]
            labels_e = sorted(set(y_e))
            pred_e = pred_t[ok]
            summary.append(report(f"epi{t:g}_organoid_to_tissue_{name}", y_e, pred_e, labels_e))
            pred_ed = pipe[-1].predict(domain_scale(z[ok]))
            summary.append(report(f"epi{t:g}_organoid_to_tissue_domainscaled_{name}", y_e, pred_ed, labels_e))
            col = np.full(len(y_tis), "", dtype=object)
            col[ok] = pred_ed
            preds[f"pred_epi{t:g}_domainscaled"] = col
            tables[f"epi{t:g}_organoid_to_tissue_domainscaled_{name}"] = pd.crosstab(
                pd.Series(g_e, name="gtex_tissue"), pd.Series(pred_ed, name="pred_domainscaled"))

        preds.to_csv(OUT / f"predictions_organoid_to_tissue_{name}.csv", index=False)

    summary = pd.DataFrame(summary).round(3)
    summary.to_csv(OUT / "summary.csv", index=False)

    runs = summary[summary["run"].str.startswith("organoid_to_tissue")]
    best = runs.sort_values("accuracy").iloc[-1]["run"]
    print(f"\n=== Predictions per GTEx tissue, all GTEx samples, best run: {best} ===")
    print(tables[best].to_string())
    for t in EPI_THRESHOLDS:
        key = f"epi{t:g}_organoid_to_tissue_domainscaled_linear_svm"
        print(f"\n=== Predictions per GTEx tissue, epithelial score >= {t:g}: {key} ===")
        print(tables[key].to_string())

    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # one line per threshold: does the result depend on the threshold?
    rows = []
    for t in EPI_THRESHOLDS:
        r = summary.set_index("run").loc[f"epi{t:g}_organoid_to_tissue_domainscaled_linear_svm"]
        rows.append({"threshold": t, "n": int(r["n"]), "accuracy": r["accuracy"],
                     "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"]})
    print("\n=== Threshold check (linear SVM, domain-scaled) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nChance level: {1 / len(classes):.3f}")
    print(f"All results saved in {OUT}/")


if __name__ == "__main__":
    main()

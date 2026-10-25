#!/usr/bin/env python3
"""
Diagnose where organoid -> tissue errors come from.

Question: are the wrong predictions (e.g. uterus -> Brain/Retina, colon -> Retina)
caused by the TISSUE samples being made of other cell types (muscle, stroma)
than the organoids (epithelium)? If yes, the problem is the test reference,
not a lack of organoid data.

Run from the project folder (the one with data/ and results/), after
Train_Classifier.py:
    python diagnose.py

Requires: pip install pandas numpy
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
RES = Path("results")
PRED = RES / "predictions_organoid_to_tissue_linear_svm.csv"

# Entrez GeneIDs
MARKERS = {
    "epithelial": {4072: "EPCAM", 3856: "KRT8", 3875: "KRT18", 999: "CDH1"},
    "smooth_muscle": {1674: "DES", 59: "ACTA2", 4629: "MYH11", 1264: "CNN1"},
    "stroma": {1277: "COL1A1", 1278: "COL1A2", 1634: "DCN", 7431: "VIM"},
}
ALL_IDS = {g for d in MARKERS.values() for g in d}


def marker_scores(path: Path) -> pd.DataFrame:
    """Read a genes x samples count matrix in chunks; return log-CPM marker scores per sample."""
    totals, keep = None, []
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=5000):
        s = chunk.sum(axis=0)
        totals = s if totals is None else totals + s
        hit = chunk[chunk.index.isin(ALL_IDS)]
        if len(hit):
            keep.append(hit)
    m = pd.concat(keep)
    logcpm = np.log2(m / totals * 1e6 + 1)
    out = {}
    for name, genes in MARKERS.items():
        present = [g for g in genes if g in logcpm.index]
        out[name] = logcpm.loc[present].mean(axis=0)
    return pd.DataFrame(out)


def main() -> None:
    print("Tissue marker scores ...")
    tis = marker_scores(DATA / "tissue_counts.tsv.gz")
    pred = pd.read_csv(PRED).set_index("sample_id")
    tis = tis.join(pred, how="inner")
    tis["correct"] = np.where(tis["pred_domainscaled"] == tis["true"], "correct", "wrong")

    print("\n=== GTEx: mean marker score (log-CPM) by tissue, correct vs wrong ===")
    t = tis.groupby(["gtex_tissue", "correct"])[list(MARKERS)].mean().round(2)
    t["n"] = tis.groupby(["gtex_tissue", "correct"]).size()
    print(t.to_string())

    print("\n=== GTEx wrong predictions: where did they go? ===")
    wrong = tis[tis["correct"] == "wrong"]
    print(wrong.groupby(["gtex_tissue", "pred_domainscaled"])[list(MARKERS)].mean().round(2)
          .assign(n=wrong.groupby(["gtex_tissue", "pred_domainscaled"]).size()).to_string())

    print("\nOrganoid marker scores (reads the big matrix, takes a few minutes) ...")
    org = marker_scores(DATA / "organoid_counts.tsv.gz")
    meta = pd.read_csv(DATA / "metadata" / "samples.csv").set_index("gsm")
    org = org.join(meta[["tissue_class"]], how="inner")
    print("\n=== Organoids: mean marker score by class ===")
    o = org.groupby("tissue_class")[list(MARKERS)].mean().round(2)
    o["n"] = org.groupby("tissue_class").size()
    print(o.to_string())

    RES.mkdir(exist_ok=True)
    t.to_csv(RES / "diagnose_tissue_markers.csv")
    o.to_csv(RES / "diagnose_organoid_markers.csv")
    print(f"\nSaved to {RES}/diagnose_*.csv")


if __name__ == "__main__":
    main()
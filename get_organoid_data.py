#!/usr/bin/env python3
"""
Expand the organoid dataset: OrganoidDB sample list -> GEO count matrices.

Step 1 (fast, no big downloads) - see what is available per tissue:
    python get_organoid_data.py list

Step 2 - download count matrices and merge them:
    python get_organoid_data.py download
    python get_organoid_data.py download --min-samples 30   # skip tiny classes

Optional: put the GSM ids you already have in have_gsm.txt (one per line)
and they will be skipped.

Requires: pip install pandas requests
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ODB_URL = "http://www.inbirg.com/organoid_db/download/all_org_samples"
GEO_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts"
    "&acc={gse}&format=file&file={gse}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
)

OUT = Path("data")
RAW = OUT / "raw_counts"
META = OUT / "metadata"
ODB_FILE = META / "organoiddb_all_samples.csv"
CANDIDATES = META / "candidates.csv"
HAVE_FILE = Path("have_gsm.txt")

SPECIES = "Homo sapiens"
PLATFORM = "RNA-Seq"  # bulk only; OrganoidDB labels single-cell as "scRNA-Seq"

# OrganoidDB tissue labels are fine-grained. Merge the ones that should be one
# class for your classifier. Edit this after looking at the output of `list`.
TISSUE_GROUPS = {
    "Brain": [
        "Cerebral cortex", "Midbrain", "Forebrain", "Hypothalamic arcuate",
        "Ventral brain", "Dorsal brain", "Telencephalon", "Choroid plexus",
        "Striatum", "Cerebrum", "Cerebellum", "Hindbrain", "Forebrain ventral",
        "Forebrain dorsal", "Neocortex", "Medial ganglionic eminence",
    ],
    "Colon": [
        "Colorectum", "Large intestine", "Ascending colon", "Proximal colon",
        "Distal colon", "Rectum", "Sigmoid colon", "Rectosigmoid mucosa",
        "Colorectal mucosa",
    ],
    "Small intestine": ["Ileum", "Jejunum", "Duodenum"],
    "Retina": ["Retina cup"],
    "Liver": ["Fetal liver"],
    "Bile duct": ["Intrahepatic bile duct", "Extrahepatic bile duct", "Biliary tract"],
    "Kidney": ["Kidney tubules", "Glomerulus", "Proximal tubules", "Mesonephros", "Metanephros"],
    "Lung": [
        "Alveolus", "Bronchus", "Airway", "Trachea", "Embryonic lung",
        "Fetal lung", "Fetal lung bud tip",
    ],
    "Breast": ["Mammary gland", "Mammary epithelium"],
    "Stomach": ["Fetal stomach", "Pediatric stomach", "Pylorus", "Gastric corpus", "Cardia"],
    "Endometrium": ["Uterine gland", "Uterus", "Endometrium gland"],
    "Cervix": ["Ectocervix", "Endocervix"],
}
TISSUE_MAP = {old: new for new, olds in TISSUE_GROUPS.items() for old in olds}
# Left unmerged on purpose: "Intestine" (does not say small or large) and
# developmental labels (Embryoid body, Neural tube, Foregut, ...).

# Rough flags to help you decide what to keep. Nothing is removed automatically.
CANCER_WORDS = r"cancer|tumou?r|carcinoma|adenoma|\bCRC\b|\bPDO|metasta|glioblastoma"
MODIFIED_WORDS = r"knock|\bKO\b|CRISPR|shRNA|mutant|infected|treated with"


def fetch_sample_list() -> pd.DataFrame:
    META.mkdir(parents=True, exist_ok=True)
    if not ODB_FILE.exists():
        print("Downloading OrganoidDB sample list ...")
        r = requests.get(ODB_URL, timeout=120)
        r.raise_for_status()
        ODB_FILE.write_bytes(r.content)
    df = pd.read_csv(ODB_FILE, encoding="latin-1", dtype=str)
    return df.rename(
        columns={
            "accession": "gse",
            "case_sample": "gsm",
            "X.Sample_characteristics_ch1": "characteristics",
        }
    )


def cmd_list(_args) -> None:
    df = fetch_sample_list()
    print(f"OrganoidDB total rows: {len(df)}")

    df = df[(df["species"] == SPECIES) & (df["platform"] == PLATFORM)].copy()
    df = df[df["gse"].str.startswith("GSE", na=False)]  # ArrayExpress ids need another route
    df = df.drop_duplicates("gsm")
    print(f"Human bulk RNA-seq samples on GEO: {len(df)}")

    if HAVE_FILE.exists():
        have = {x.strip() for x in HAVE_FILE.read_text().split() if x.strip()}
        before = len(df)
        df = df[~df["gsm"].isin(have)]
        print(f"Already downloaded (have_gsm.txt): {before - len(df)} skipped")

    df["tissue_class"] = df["tissue"].replace(TISSUE_MAP)
    text = df["characteristics"].fillna("")
    df["flag_cancer"] = text.str.contains(CANCER_WORDS, case=False, regex=True)
    df["flag_modified"] = text.str.contains(MODIFIED_WORDS, case=False, regex=True)

    summary = (
        df.groupby("tissue_class")
        .agg(
            samples=("gsm", "count"),
            studies=("gse", "nunique"),
            cancer=("flag_cancer", "sum"),
            modified=("flag_modified", "sum"),
        )
        .sort_values("samples", ascending=False)
    )
    pd.set_option("display.max_rows", None)
    print("\nNew samples available per tissue:\n")
    print(summary)

    df.to_csv(CANDIDATES, index=False)
    summary.to_csv(META / "tissue_summary.csv")
    print(f"\nSaved {len(df)} candidate samples -> {CANDIDATES}")


def download_gse(gse: str) -> Path | None:
    """Download the NCBI-generated raw count matrix for one series."""
    path = RAW / f"{gse}_raw_counts.tsv.gz"
    if path.exists():
        return path
    for attempt in range(3):
        try:
            r = requests.get(GEO_URL.format(gse=gse), timeout=180)
            if r.status_code == 200 and r.content[:2] == b"\x1f\x8b":  # real gzip
                path.write_bytes(r.content)
                return path
            return None  # NCBI has no generated counts for this series
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None


def cmd_download(args) -> None:
    if not CANDIDATES.exists():
        raise SystemExit("Run `python get_organoid_data.py list` first.")
    RAW.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CANDIDATES, dtype=str)

    if args.min_samples:
        sizes = df["tissue_class"].value_counts()
        df = df[df["tissue_class"].isin(sizes[sizes >= args.min_samples].index)]

    series = sorted(df["gse"].unique())
    wanted = set(df["gsm"])
    print(f"{len(df)} samples in {len(series)} GEO series")

    matrices, failed = [], []
    for i, gse in enumerate(series, 1):
        path = download_gse(gse)
        if path is None:
            failed.append(gse)
            print(f"[{i}/{len(series)}] {gse}: no NCBI counts")
            continue
        m = pd.read_csv(path, sep="\t", index_col=0)
        m = m[[c for c in m.columns if c in wanted]]
        print(f"[{i}/{len(series)}] {gse}: {m.shape[1]} samples")
        if m.shape[1]:
            matrices.append(m)
        time.sleep(0.4)  # be polite to NCBI

    if not matrices:
        raise SystemExit("Nothing downloaded.")

    counts = pd.concat(matrices, axis=1, join="inner")
    counts = counts.loc[:, ~counts.columns.duplicated()]
    counts.index.name = "GeneID"  # Entrez gene ids
    counts.to_csv(OUT / "organoid_counts.tsv.gz", sep="\t")

    got = df[df["gsm"].isin(counts.columns)]
    got.to_csv(META / "samples.csv", index=False)
    (META / "failed_gse.txt").write_text("\n".join(failed))

    print(f"\nMatrix: {counts.shape[0]} genes x {counts.shape[1]} samples")
    print(f"Series without NCBI counts: {len(failed)} -> {META / 'failed_gse.txt'}")
    print("\nSamples per tissue in the final matrix:")
    print(got["tissue_class"].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    d = sub.add_parser("download")
    d.add_argument("--min-samples", type=int, default=0, help="drop tissues with fewer samples")
    d.set_defaults(func=cmd_download)
    args = p.parse_args()
    args.func(args)
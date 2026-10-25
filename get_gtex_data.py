#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import requests

BASE = "https://storage.googleapis.com/adult-gtex/bulk-gex"
URLS = {
    "v10": BASE + "/v10/rna-seq/counts-by-tissue/gene_reads_v10_{slug}.gct.gz",
    "v8": BASE + "/v8/rna-seq/counts-by-tissue/gene_reads_2017-06-05_v8_{slug}.gct.gz",
}

# tissue class (same names as the organoid matrix) -> GTEx tissue files
GTEX_TISSUES = {
    "Brain": ["brain_cortex", "brain_frontal_cortex_ba9", "brain_hippocampus", "brain_substantia_nigra"],
    "Colon": ["colon_transverse", "colon_sigmoid"],
    "Small intestine": ["small_intestine_terminal_ileum"],
    "Liver": ["liver"],
    "Kidney": ["kidney_cortex"],
    "Lung": ["lung"],
    "Prostate": ["prostate"],
    "Breast": ["breast_mammary_tissue"],
    "Pancreas": ["pancreas"],
    "Endometrium": ["uterus"],
    "Stomach": ["stomach"],
    # GTEx has no retina, and "Intestine" is too vague to match.
}

OUT = Path("data")
RAW = OUT / "gtex_raw"
META = OUT / "metadata"
ANNOT = Path("Human.GRCh38.p13.annot.tsv.gz")
ORGANOID = OUT / "organoid_counts.tsv.gz"


def download(slug: str, version: str) -> Path | None:
    path = RAW / f"{version}_{slug}.gct.gz"
    if path.exists():
        return path
    url = URLS[version].format(slug=slug)
    try:
        r = requests.get(url, timeout=600)
    except requests.RequestException as e:
        print(f"   network error: {e}")
        return None
    if r.status_code != 200 or r.content[:2] != b"\x1f\x8b":
        print(f"   not found (HTTP {r.status_code}): {url}")
        return None
    path.write_bytes(r.content)
    return path


def read_gct(path: Path) -> pd.DataFrame:
    """Rows = Ensembl gene ids (version stripped), columns = GTEx sample ids."""
    df = pd.read_csv(path, sep="\t", skiprows=2)
    df = df.drop(columns=[c for c in ("id", "Description") if c in df.columns])
    df["Name"] = df["Name"].str.split(".").str[0]
    return df.groupby("Name").sum()


def ensembl_to_entrez() -> pd.Series:
    if not ANNOT.exists():
        raise SystemExit(f"Missing {ANNOT} - put the NCBI annotation file in this folder.")
    a = pd.read_csv(ANNOT, sep="\t", dtype=str)
    if "EnsemblGeneID" not in a.columns:
        raise SystemExit(f"No EnsemblGeneID column in {ANNOT}. Columns: {list(a.columns)}")
    a = a.dropna(subset=["EnsemblGeneID"]).drop_duplicates("EnsemblGeneID")
    return a.set_index("EnsemblGeneID")["GeneID"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--max-per-tissue", type=int, default=150, help="random cap per tissue class")
    p.add_argument("--version", choices=list(URLS), default="v10")
    args = p.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    mapping = ensembl_to_entrez()

    matrices, rows = [], []
    for tissue, slugs in GTEX_TISSUES.items():
        parts = []
        for slug in slugs:
            print(f"{tissue}: {slug}")
            path = download(slug, args.version)
            if path is None:
                continue
            m = read_gct(path)
            parts.append((slug, m))
            print(f"   {m.shape[1]} samples")
        if not parts:
            continue
        # same number from each sub-tissue, up to the cap
        per_part = max(1, args.max_per_tissue // len(parts))
        for slug, m in parts:
            if m.shape[1] > per_part:
                m = m.sample(n=per_part, axis=1, random_state=0)
            matrices.append(m)
            rows += [{"sample_id": s, "tissue_class": tissue, "gtex_tissue": slug, "type": "Tissue"}
                     for s in m.columns]

    if not matrices:
        raise SystemExit("Nothing downloaded. Try: python get_gtex_data.py --version v8")

    counts = pd.concat(matrices, axis=1, join="inner")

    # Ensembl -> Entrez; several Ensembl ids on one Entrez id are summed
    counts = counts[counts.index.isin(mapping.index)]
    counts.index = mapping.loc[counts.index].astype(int).values
    counts = counts.groupby(level=0).sum()
    counts.index.name = "GeneID"

    if ORGANOID.exists():
        organoid_genes = pd.read_csv(ORGANOID, sep="\t", usecols=[0]).iloc[:, 0]
        shared = counts.index.intersection(organoid_genes)
        print(f"\nGenes: GTEx {len(counts)}, organoid {len(organoid_genes)}, shared {len(shared)}")
        counts = counts.loc[shared]

    counts.to_csv(OUT / "tissue_counts.tsv.gz", sep="\t")
    meta = pd.DataFrame(rows)
    meta.to_csv(META / "tissue_samples.csv", index=False)

    print(f"Matrix: {counts.shape[0]} genes x {counts.shape[1]} samples -> {OUT / 'tissue_counts.tsv.gz'}")
    print("\nSamples per tissue:")
    print(meta["tissue_class"].value_counts())


if __name__ == "__main__":
    main()
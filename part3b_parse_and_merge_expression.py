from pathlib import Path
import pandas as pd
import gzip
import re


# Paths

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
SUPP_DIR = BASE_DIR / "geo_supplementary"

OUT_EXPR = BASE_DIR / "expression_matrix_supplementary.csv"
OUT_SAMPLE_INFO = BASE_DIR / "sample_info_supplementary.csv"


# Helpers

def open_text(path: Path):
    # If the file ends with .gz, open as gzip text; otherwise open as normal text
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def guess_sep(header_line: str) -> str:
    # If the header has more tabs than commas, assume TSV; else CSV
    return "\t" if header_line.count("\t") > header_line.count(",") else ","

def extract_gse(fname: str) -> str:
    # Extract the first "GSE####" substring from the filename
    m = re.search(r"(GSE\d+)", fname)
    return m.group(1) if m else "UNKNOWN"

def read_expression_file(path: Path) -> pd.DataFrame:
    """
    Reads a gene expression table:
      - first column = gene identifier (Symbol/Ensembl/etc.)
      - remaining columns = samples
    Fixes:
      - duplicated gene IDs are aggregated by mean
      - expression values are coerced to numeric
    Returns:
      DataFrame indexed by gene, columns = sample names
    """
    # Read first line to guess separator
    with open_text(path) as f:
        header = f.readline().strip()
    sep = guess_sep(header)

    # Load file
    df = pd.read_csv(
        path,
        sep=sep,
        compression="infer",
        comment="#",
        low_memory=False
    )

    if df.shape[1] < 2:
        return pd.DataFrame()

    # First column is the gene id
    df = df.rename(columns={df.columns[0]: "gene"})
    df["gene"] = df["gene"].astype(str).str.strip()
    df = df[df["gene"].notna()]

    # Convert all expression columns to numeric
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Aggregate duplicated genes by mean
    df = (
        df.groupby("gene", as_index=False)
          .mean(numeric_only=True)
          .set_index("gene")
    )

    return df


# Main

def main():
    expr_tables = []
    sample_rows = []

    files = sorted(SUPP_DIR.glob("*.txt.gz"))
    print("Expression files found:", len(files))

    for f in files:
        gse = extract_gse(f.name)
        print("Reading:", f.name)

        df = read_expression_file(f)
        if df.empty:
            print("  -> empty, skipped")
            continue

        # 🔑 FIX: handle duplicated sample columns inside a file
        # (some files repeat the same sample name more than once)
        if df.columns.duplicated().any():
            df = df.groupby(df.columns, axis=1).mean()

        # Prefix sample columns with GSE so different studies won't clash
        df = df.rename(columns={c: f"{gse}::{c}" for c in df.columns})

        # Record sample info
        for c in df.columns:
            sample_rows.append({
                "sample_id": c,
                "GSE": gse,
                "source_file": f.name
            })

        expr_tables.append(df)
        print("  -> genes:", df.shape[0], "samples:", df.shape[1])

    if not expr_tables:
        raise RuntimeError("No expression tables could be parsed.")

    # Concatenate across studies (union of genes; samples stacked as columns)
    combined = pd.concat(expr_tables, axis=1)

    print("Final expression matrix shape:", combined.shape)

    combined.to_csv(OUT_EXPR)
    pd.DataFrame(sample_rows).drop_duplicates().to_csv(OUT_SAMPLE_INFO, index=False)

    print("Saved:", OUT_EXPR)
    print("Saved:", OUT_SAMPLE_INFO)

if __name__ == "__main__":
    main()

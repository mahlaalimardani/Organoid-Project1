from pathlib import Path
import pandas as pd
import gzip


# 1) Paths


BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
SERIES_MATRIX_DIR = BASE_DIR / "geo_series_matrix"

OUT_EXPR = BASE_DIR / "expression_matrix_series_matrix.csv"
OUT_SAMPLE_MAP = BASE_DIR / "gsm_to_gse_from_series_matrix.csv"

# 2) Helper: read GEO Series Matrix expression table


def read_series_matrix_expression(path: Path) -> pd.DataFrame:
    """
    Reads a GEO Series Matrix file and returns:
      DataFrame with rows=genes, cols=GSM samples (expression values as strings/numbers)
    """
    if path.suffix.lower() == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        fh = open(path, "r", encoding="utf-8", errors="replace")

    data_started = False
    rows = []

    for line in fh:
        if line.startswith("!series_matrix_table_begin"):
            data_started = True
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if data_started:
            rows.append(line.rstrip("\n").split("\t"))

    fh.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={0: "gene"}).set_index("gene")

    # Try to convert values to numeric (non-numeric becomes NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


# 3) Main: combine all matrices


def main():
    matrix_files = sorted(SERIES_MATRIX_DIR.glob("GSE*_series_matrix*.txt*"))

    print("Series Matrix files found:", len(matrix_files))
    if not matrix_files:
        raise RuntimeError("No Series Matrix files found. Check geo_series_matrix folder.")

    all_expr = []
    sample_map_rows = []

    for f in matrix_files:
        gse = f.name.split("_series_matrix")[0]  # e.g., GSE106821
        print("Reading:", f.name)

        expr = read_series_matrix_expression(f)

        if expr.empty or expr.shape[1] == 0:
            print("  -> empty expression table, skipped")
            continue

        # Record which GSM columns belong to which GSE
        for gsm in expr.columns:
            sample_map_rows.append({"GSE": gse, "GSM": gsm})

        # Prefix columns with GSE to avoid accidental duplicates across studies
        expr = expr.rename(columns={c: f"{gse}::{c}" for c in expr.columns})

        all_expr.append(expr)
        print("  -> genes:", expr.shape[0], "samples:", expr.shape[1])

    if not all_expr:
        raise RuntimeError("All Series Matrix files were empty or unreadable.")

    # Concatenate by columns (same gene names align automatically; union of genes is kept)
    combined = pd.concat(all_expr, axis=1)

    print("Final combined matrix shape (genes, samples):", combined.shape)

    # Save expression matrix
    combined.to_csv(OUT_EXPR)
    print("Saved expression matrix:", OUT_EXPR)

    # Save GSM-to-GSE mapping (without the GSE:: prefix)
    sample_map = pd.DataFrame(sample_map_rows).drop_duplicates()
    sample_map.to_csv(OUT_SAMPLE_MAP, index=False)
    print("Saved GSM->GSE map:", OUT_SAMPLE_MAP)

if __name__ == "__main__":
    main()

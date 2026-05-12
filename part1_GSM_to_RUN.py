import subprocess
import sys
import pandas as pd
from io import StringIO
from pathlib import Path


# Configuration

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
IN_FILE = BASE_DIR / "run_manifest_human_bulk.csv"
OUT_FILE = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"


# Helpers

def run_cmd(cmd: list[str]) -> str:
    """Run a command and return stdout. Raise a clear error on failure."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(map(str, cmd))}\n\n"
            f"STDERR:\n{p.stderr}"
        )
    return p.stdout

def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Detect the GSM and RUN columns in pysradb output.

    pysradb output varies by version/source. GSM may appear as:
      - sample_accession
      - experiment_alias
      - experiment_accession
      - gsm

    RUN may appear as:
      - run_accession
      - srr
      - run
    """
    cols = {c.lower(): c for c in df.columns}

    gsm_col = (
        cols.get("sample_accession")
        or cols.get("experiment_alias")
        or cols.get("experiment_accession")
        or cols.get("gsm")
    )

    run_col = (
        cols.get("run_accession")
        or cols.get("srr")
        or cols.get("run")
    )

    if gsm_col is None or run_col is None:
        raise RuntimeError(
            f"Unexpected columns from pysradb output: {list(df.columns)}"
        )

    return gsm_col, run_col

def gsm_to_run_map(gsms: list[str], batch_size: int = 150) -> pd.DataFrame:
    """
    Map GSM -> RUN using pysradb 'gsm-to-srr' in batches.
    Returns a DataFrame with columns: GSM, RUN
    """
    all_parts = []

    for i in range(0, len(gsms), batch_size):
        batch = gsms[i:i + batch_size]
        batch_no = i // batch_size + 1
        print(f"[INFO] GSM->RUN batch {batch_no} ({len(batch)} GSMs)")

        # Run: python -m pysradb gsm-to-srr GSM1 GSM2 ...
        out = run_cmd([sys.executable, "-m", "pysradb", "gsm-to-srr"] + batch)

        df = pd.read_csv(StringIO(out), sep="\t")
        gsm_col, run_col = detect_columns(df)

        df = df.rename(columns={gsm_col: "GSM", run_col: "RUN"})
        df["GSM"] = df["GSM"].astype(str).str.strip()
        df["RUN"] = df["RUN"].astype(str).str.strip()

        all_parts.append(df[["GSM", "RUN"]].drop_duplicates())

    mapping = pd.concat(all_parts, ignore_index=True).drop_duplicates()
    return mapping

# Main

def main():
    print("Interpreter:", sys.executable)

    meta = pd.read_csv(IN_FILE)
    meta["GSM"] = meta["GSM"].astype(str).str.strip()

    missing = meta[meta["RUN"].isna()].copy()
    print("Rows missing RUN:", len(missing))

    gsms = sorted(missing["GSM"].unique())
    print("Unique GSM needing mapping:", len(gsms))

    if len(gsms) == 0:
        meta.to_csv(OUT_FILE, index=False)
        print("[DONE] Nothing to map. Saved:", OUT_FILE)
        return

    mapping = gsm_to_run_map(gsms, batch_size=150)

    # Sanity checks
    print("Mapped rows:", len(mapping))
    print("Unique GSM mapped:", mapping["GSM"].nunique())
    print("Unique RUN mapped:", mapping["RUN"].nunique())

    # Merge mapping back
    merged = meta.merge(mapping, on="GSM", how="left", suffixes=("", "_mapped"))
    merged["RUN"] = merged["RUN"].fillna(merged["RUN_mapped"])
    merged = merged.drop(columns=["RUN_mapped"])

    still_missing = merged["RUN"].isna().sum()
    print("After mapping, still missing RUN:", still_missing)

    if still_missing > 0:
        print("\nExamples still missing (first 20):")
        print(merged.loc[merged["RUN"].isna(), ["GSE", "GSM", "type", "tissue"]].head(20))

    merged.to_csv(OUT_FILE, index=False)
    print("\nSaved:", OUT_FILE)

if __name__ == "__main__":
    main()

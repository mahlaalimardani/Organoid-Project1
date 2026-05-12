import re
import sys
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path
import os

# 1) Paths

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
IN_FILE = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"
OUT_FILE = BASE_DIR / "run_manifest_with_geo_gsm.csv"


# 2) Helper: run pysradb command

def run_cmd(cmd: list[str]) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(map(str, cmd))}\n\n"
            f"STDERR:\n{p.stderr}"
        )
    return p.stdout

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    return None


# 3) Map RUN -> GSM in batches

def run_to_gsm_map(runs: list[str], batch_size: int = 150) -> pd.DataFrame:
    all_parts = []

    for i in range(0, len(runs), batch_size):
        batch = runs[i:i + batch_size]
        print(f"[INFO] RUN->GSM batch {i//batch_size + 1} ({len(batch)} runs)")

        # pysradb command (works for SRR/ERR/DRR)
        tmp_out = Path("tmp_srr_to_gsm.tsv")

        # Ask pysradb to write TSV output to a file (avoids console printing)
        cmd = [sys.executable, "-m", "pysradb", "srr-to-gsm"] + batch + ["-o", str(tmp_out)]
        run_cmd(cmd)

        # Read the TSV file produced by pysradb
        df = pd.read_csv(tmp_out, sep="\t")
        tmp_out.unlink(missing_ok=True)

        run_col = pick_col(df, ["run_accession", "srr", "run"])
        gsm_col = pick_col(df, ["gsm", "sample_accession", "experiment_alias", "experiment_accession"])

        if run_col is None or gsm_col is None:
            raise RuntimeError(f"Unexpected pysradb columns: {list(df.columns)}")

        df = df.rename(columns={run_col: "RUN", gsm_col: "GSM_GEO"})
        df["RUN"] = df["RUN"].astype(str).str.strip()
        df["GSM_GEO"] = df["GSM_GEO"].astype(str).str.strip()

        all_parts.append(df[["RUN", "GSM_GEO"]].drop_duplicates())

    mapping = pd.concat(all_parts, ignore_index=True).drop_duplicates()
    return mapping

# ----------------------------
# 4) Main
# ----------------------------
def main():
    meta = pd.read_csv(IN_FILE)
    meta["RUN"] = meta["RUN"].astype(str).str.strip()

    runs = sorted(meta["RUN"].dropna().unique())
    print("Unique RUNs:", len(runs))

    mapping = run_to_gsm_map(runs, batch_size=150)

    print("Mapped RUNs:", mapping["RUN"].nunique())
    print("Mapped GSMs:", mapping["GSM_GEO"].nunique())

    merged = meta.merge(mapping, on="RUN", how="left")

    missing = merged["GSM_GEO"].isna().sum()
    print("Rows missing GSM_GEO after mapping:", missing)

    merged.to_csv(OUT_FILE, index=False)
    print("Saved:", OUT_FILE)

if __name__ == "__main__":
    main()

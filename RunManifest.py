import re
import pandas as pd
from io import StringIO
from pathlib import Path

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

IN_FILE = BASE_DIR / "manifest_human_bulk_rnaseq.csv"
OUT_FILE = BASE_DIR / "run_manifest_human_bulk.csv"

df = pd.read_csv(IN_FILE)

# In your data, the column is called "GSM" but it actually contains run IDs like ERR...
df["GSM"] = df["GSM"].astype(str).str.strip()

# Detect run accessions (SRR/ERR/DRR)
run_pat = re.compile(r"^(SRR|ERR|DRR)\d+$", re.IGNORECASE)

df["RUN"] = df["GSM"].where(df["GSM"].str.match(run_pat), pd.NA)

# Report what we have
n_total = len(df)
n_run = df["RUN"].notna().sum()
n_missing = n_total - n_run

print("=== PART 2 (RUN MANIFEST) ===")
print("Total rows:", n_total)
print("Rows with RUN (SRR/ERR/DRR):", n_run)
print("Rows missing RUN:", n_missing)

if n_missing > 0:
    print("\nExamples missing RUN (first 10):")
    print(df.loc[df["RUN"].isna(), ["GSE", "GSM", "type", "tissue"]].head(10))

# Save run-level manifest
keep_cols = ["GSE", "GSM", "RUN", "type", "tissue", "studyid", "platform", "species"]
df[keep_cols].to_csv(OUT_FILE, index=False)

print("\nSaved:", OUT_FILE)
print(df[keep_cols].head(10))

from pathlib import Path
import pandas as pd

# 1) Configure paths
DATA_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
INPUT_FILE = DATA_DIR / "general_tissue_organoid.csv"
OUTPUT_FILE = DATA_DIR / "manifest_human_bulk_rnaseq.csv"

# 2) Load metadata
df = pd.read_csv(INPUT_FILE)

# 3) Basic cleaning (safe for mixed formatting)
df["species"] = df["species"].astype(str).str.strip().str.lower()
df["platform"] = df["platform"].astype(str).str.strip().str.lower()
df["type"] = df["type"].astype(str).str.strip()  # keep original case for labels like Tissue/Organoid

# 4) Filter: human + bulk RNA-Seq
manifest = df[
    (df["species"] == "homo_sapiens") &
    (df["platform"] == "rna-seq")
].copy()

# 5) Keep only necessary columns (avoid surprises later)
keep_cols = ["GSE", "GSM", "tissue", "type", "platform", "species", "studyid"]
manifest = manifest[keep_cols]

# 6) Sort for readability
manifest = manifest.sort_values(["GSE", "type", "tissue", "GSM"]).reset_index(drop=True)

# 7) Save
manifest.to_csv(OUTPUT_FILE, index=False)

# 8) Print sanity checks
print("=== PART 1 DONE ===")
print("Input file:", INPUT_FILE)
print("Output file:", OUTPUT_FILE)
print("Rows:", len(manifest))
print("Unique GSE:", manifest["GSE"].nunique())
print("Type distribution:")
print(manifest["type"].value_counts(dropna=False))

# Optional: show first rows
print("\nPreview:")
print(manifest.head(10))

from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
MANIFEST = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"
OUT = BASE_DIR / "tissue_gse_list.txt"

df = pd.read_csv(MANIFEST)
df["type"] = df["type"].astype(str).str.strip()
df["GSE"] = df["GSE"].astype(str).str.strip()

tissue_gses = sorted(df.loc[df["type"].str.lower() == "tissue", "GSE"].dropna().unique())
print("Tissue GSE count:", len(tissue_gses))

with open(OUT, "w", encoding="utf-8") as f:
    for gse in tissue_gses:
        f.write(gse + "\n")

print("Saved:", OUT)
print("First 10:", tissue_gses[:10])

from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

IN_FILE = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"
OUT_SUMMARY = BASE_DIR / "summary_2a_usable_counts.csv"

df = pd.read_csv(IN_FILE)

# Normalize basic fields
for c in ["RUN", "type", "tissue", "GSE", "GSM"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "nan", "None", "NA"]), c] = pd.NA

# Define usable criteria for 2(a)
df["usable"] = (
    df["RUN"].notna() &
    df["type"].notna() &
    df["tissue"].notna()
)

# Print key counts
print("=== QUESTION 2(a): USABLE FILES ===")
print("Total rows:", len(df))
print("With RUN:", df["RUN"].notna().sum())
print("With type:", df["type"].notna().sum())
print("With tissue:", df["tissue"].notna().sum())
print("USABLE (RUN+type+tissue):", df["usable"].sum())

print("\nUsable by type:")
print(df.loc[df["usable"], "type"].value_counts(dropna=False))

print("\nTop tissues (usable):")
print(df.loc[df["usable"], "tissue"].value_counts().head(20))

# Save a compact summary table
summary = pd.DataFrame([
    {"stage": "total_rows", "n": len(df)},
    {"stage": "with_RUN", "n": int(df["RUN"].notna().sum())},
    {"stage": "with_type", "n": int(df["type"].notna().sum())},
    {"stage": "with_tissue", "n": int(df["tissue"].notna().sum())},
    {"stage": "usable_RUN+type+tissue", "n": int(df["usable"].sum())},
])

summary.to_csv(OUT_SUMMARY, index=False)
print("\nSaved summary:", OUT_SUMMARY)

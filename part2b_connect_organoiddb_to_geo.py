from pathlib import Path
import pandas as pd

# 1) Define project folder (where your CSV files are located)
BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

# 2) Define input files
#    GEO manifest produced earlier (contains your 491 rows)
GEO_FILE = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"

#    OrganoidDB annotations file (note: must be read with cp1252 encoding)
ORG_FILE = BASE_DIR / "all_organoid_samples.csv"

# 3) Define output files
OUT_MERGED = BASE_DIR / "merged_geo_organoiddb.csv"
OUT_REPORT = BASE_DIR / "report_2b_organoiddb_match.csv"


# 4) Load GEO manifest (UTF-8 is fine in your GEO outputs)
geo = pd.read_csv(GEO_FILE)

# 5) Load OrganoidDB table using cp1252 encoding (your file is not UTF-8)
org = pd.read_csv(ORG_FILE, encoding="cp1252")

# 6) Basic sanity prints so you see what got loaded
print("GEO rows:", len(geo))
print("OrganoidDB rows:", len(org))


# 7) Clean/normalize the join column on both sides
#    - convert to string
#    - strip whitespace
#    - convert empty strings / 'nan' to NA
geo["studyid"] = geo["studyid"].astype(str).str.strip()
org["studyid"] = org["studyid"].astype(str).str.strip()

geo.loc[geo["studyid"].isin(["", "nan", "None", "NA"]), "studyid"] = pd.NA
org.loc[org["studyid"].isin(["", "nan", "None", "NA"]), "studyid"] = pd.NA

# 8) Drop OrganoidDB rows without a studyid (they cannot match)
org = org[org["studyid"].notna()].copy()


# 9) Reduce OrganoidDB columns to the ones you want to attach to GEO
#    (You can keep all columns, but selecting is safer/cleaner.)
org_keep = [
    "studyid",
    "accession",
    "case_sample",
    "tissue",
    "platform",
    "species",
    "Series_pubmed_id",
    "X.Sample_characteristics_ch1",
]
org_small = org[org_keep].copy()


# 10) Merge using a LEFT JOIN
#     - keeps ALL GEO rows (491)
#     - attaches OrganoidDB data where studyid matches
merged = geo.merge(org_small, on="studyid", how="left", suffixes=("_geo", "_orgdb"))

print("Merged rows:", len(merged))


# 11) Create a match flag: did this GEO row find an OrganoidDB record?
#     Here we use 'accession' as a match signal (it exists only if merge succeeded)
merged["matched_organoiddb"] = merged["accession"].notna()

total = len(merged)
matched = int(merged["matched_organoiddb"].sum())
unmatched = total - matched

print("Total GEO rows:", total)
print("Matched rows:", matched)
print("Unmatched rows:", unmatched)
print("Match rate:", matched / total if total else 0)


# 12) Save a small report table for your manuscript/proposal
report = pd.DataFrame([
    {"metric": "total_geo_rows", "value": total},
    {"metric": "matched_rows", "value": matched},
    {"metric": "unmatched_rows", "value": unmatched},
    {"metric": "match_rate", "value": matched / total if total else 0},
    {"metric": "join_key", "value": "studyid"},
])
report.to_csv(OUT_REPORT, index=False)


# 13) Save merged dataset for downstream steps
merged.to_csv(OUT_MERGED, index=False)

print("Saved merged file:", OUT_MERGED)
print("Saved report:", OUT_REPORT)

# 14) Optional: show a preview of matched rows
print("\nPreview of matched rows (first 5):")
print(merged.loc[merged["matched_organoiddb"]].head(5))

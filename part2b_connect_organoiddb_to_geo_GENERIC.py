from pathlib import Path
import pandas as pd


# 1) Paths (edit only filenames if needed)


# Project folder where your CSV files live
BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

# Your GEO manifest (already curated in Part 1/2/3)
# Use the most complete one:
GEO_MANIFEST = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"

# OrganoidDB annotations file:
# Change this to the OrganoidDB file you have (examples below)
# - "all_organoid_samples.csv"
# - "organoiddb_samples.tsv"
ORGANOID_DB = BASE_DIR / "all_organoid_samples.csv"

# Output merged file
OUT_MERGED = BASE_DIR / "merged_geo_organoiddb.csv"

# Output small matching report
OUT_REPORT = BASE_DIR / "report_2b_match_stats.csv"



# 2) Helper functions


def read_table(path: Path) -> pd.DataFrame:
    """
    Read CSV or TSV automatically based on file extension.
    """
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def normalize_str_series(s: pd.Series) -> pd.Series:
    """
    Normalize text for safer matching:
    - convert to string
    - strip spaces
    - remove common 'nan/none' strings
    """
    s = s.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA})
    return s


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name found in df from a candidate list.
    Matching is case-insensitive.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ----------------------------
# 3) Load GEO and OrganoidDB tables

geo = read_table(GEO_MANIFEST)
org = read_table(ORGANOID_DB)

print("Loaded GEO rows:", len(geo))
print("Loaded OrganoidDB rows:", len(org))


# ----------------------------
# 4) Detect a join key (studyid is best; GSE is second-best)


# GEO side keys we might have
geo_study_col = pick_first_existing_column(geo, ["studyid", "study_id", "organoid_db_studyid"])
geo_gse_col = pick_first_existing_column(geo, ["GSE", "gse", "gse_accession", "series"])

# OrganoidDB side keys we might have
org_study_col = pick_first_existing_column(org, ["studyid", "study_id", "study", "dataset_id"])
org_gse_col = pick_first_existing_column(org, ["GSE", "gse", "gse_accession", "geo_series", "series"])

# Choose join strategy
join_mode = None

if geo_study_col and org_study_col:
    join_mode = "studyid"
elif geo_gse_col and org_gse_col:
    join_mode = "GSE"
else:
    raise RuntimeError(
        "No compatible join key found.\n"
        f"GEO columns: {list(geo.columns)}\n"
        f"OrganoidDB columns: {list(org.columns)}\n"
        "Expected a shared key like studyid or GSE."
    )

print("Join mode selected:", join_mode)


# ----------------------------
# 5) Normalize join columns and rename to a common name


if join_mode == "studyid":
    geo = geo.rename(columns={geo_study_col: "JOIN_KEY"})
    org = org.rename(columns={org_study_col: "JOIN_KEY"})
else:
    geo = geo.rename(columns={geo_gse_col: "JOIN_KEY"})
    org = org.rename(columns={org_gse_col: "JOIN_KEY"})

# Normalize join keys (strip, clean)
geo["JOIN_KEY"] = normalize_str_series(geo["JOIN_KEY"])
org["JOIN_KEY"] = normalize_str_series(org["JOIN_KEY"])

# Drop OrganoidDB rows without a join key (cannot match anything)
org = org[org["JOIN_KEY"].notna()].copy()


# ----------------------------
# 6) Merge (left join keeps ALL GEO rows)


# "left" keeps all GEO rows and attaches OrganoidDB data where possible
merged = geo.merge(org, on="JOIN_KEY", how="left", suffixes=("_geo", "_orgdb"))

print("Merged rows:", len(merged))


# ----------------------------
# 7) Create a match indicator and report statistics


# Count whether a row matched any OrganoidDB entry (at least one OrganoidDB column filled)
# We use "any non-null besides JOIN_KEY" as a match signal
org_cols = [c for c in merged.columns if c.endswith("_orgdb")]
merged["matched_organoiddb"] = merged[org_cols].notna().any(axis=1) if org_cols else False

total_geo = len(merged)
matched = int(merged["matched_organoiddb"].sum())
unmatched = total_geo - matched

print("Total GEO rows:", total_geo)
print("Matched with OrganoidDB:", matched)
print("Unmatched:", unmatched)

# Save a small report
report = pd.DataFrame([
    {"metric": "total_geo_rows", "value": total_geo},
    {"metric": "matched_rows", "value": matched},
    {"metric": "unmatched_rows", "value": unmatched},
    {"metric": "match_rate", "value": matched / total_geo if total_geo else 0},
    {"metric": "join_mode", "value": join_mode},
])

report.to_csv(OUT_REPORT, index=False)


# ----------------------------
# 8) Save merged table

merged.to_csv(OUT_MERGED, index=False)

print("Saved merged file:", OUT_MERGED)
print("Saved report:", OUT_REPORT)

# Optional preview
print("\nPreview (first 5 rows):")
print(merged.head(5))

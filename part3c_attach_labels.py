from pathlib import Path
import pandas as pd


# Paths

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

MANIFEST = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"
SAMPLE_INFO = BASE_DIR / "sample_info_supplementary.csv"

OUT_LABELS = BASE_DIR / "ml_sample_labels.csv"
OUT_REPORT = BASE_DIR / "report_3c_label_coverage.csv"


# Main

def main():
    # Load sample list produced in Step 3(b)
    s = pd.read_csv(SAMPLE_INFO)

    # Extract GSE from sample_id like "GSE155498::SAMPLE_NAME"
    s["GSE"] = s["sample_id"].astype(str).str.split("::").str[0].str.strip()

    # Load your project manifest
    m = pd.read_csv(MANIFEST)
    m["GSE"] = m["GSE"].astype(str).str.strip()

    # Keep only label columns (deduplicate at GSE level)
    # If a GSE appears multiple times in manifest, we keep one row per GSE for labels
    label_cols = ["GSE", "type", "tissue"]
    m_lab = (
        m[label_cols]
        .dropna(subset=["GSE"])
        .drop_duplicates(subset=["GSE"])
        .copy()
    )

    # Merge labels onto each sample by GSE
    merged = s.merge(m_lab, on="GSE", how="left")

    # Report coverage
    total = len(merged)
    has_type = merged["type"].notna().sum()
    has_tissue = merged["tissue"].notna().sum()
    fully_labeled = merged[["type", "tissue"]].notna().all(axis=1).sum()

    report = pd.DataFrame([
        {"metric": "total_samples", "value": total},
        {"metric": "samples_with_type", "value": has_type},
        {"metric": "samples_with_tissue", "value": has_tissue},
        {"metric": "samples_with_type_and_tissue", "value": fully_labeled},
        {"metric": "type_coverage", "value": has_type / total if total else 0},
        {"metric": "tissue_coverage", "value": has_tissue / total if total else 0},
        {"metric": "full_label_coverage", "value": fully_labeled / total if total else 0},
    ])

    merged.to_csv(OUT_LABELS, index=False)
    report.to_csv(OUT_REPORT, index=False)

    print("Saved labels:", OUT_LABELS)
    print("Saved report:", OUT_REPORT)
    print(report)

    # Optional: show how many samples per class
    print("\nType distribution:")
    print(merged["type"].value_counts(dropna=False))

    print("\nTissue distribution (top 15):")
    print(merged["tissue"].value_counts(dropna=False).head(15))

if __name__ == "__main__":
    main()

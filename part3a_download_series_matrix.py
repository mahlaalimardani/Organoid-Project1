from pathlib import Path
import pandas as pd
import requests
import time
import re


# 1) Configuration


# Project folder where your files are located
BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")

# Manifest file containing your GSE list
MANIFEST_FILE = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"

# Output folder for downloaded Series Matrix files
OUT_DIR = BASE_DIR / "geo_series_matrix"
OUT_DIR.mkdir(exist_ok=True)

# NCBI GEO FTP base URL for Series Matrix files
# GEO structure: .../geo/series/GSEnnnxxx/GSEnnnnnn/matrix/...
GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

# Networking settings (safe defaults)
TIMEOUT = (20, 120)     # (connect_timeout, read_timeout) in seconds
SLEEP_BETWEEN = 0.5     # small delay between downloads (server-friendly)



# 2) Helpers


def gse_series_folder(gse: str) -> str:
    """
    Convert a GSE accession into its GEO 'series' folder prefix.

    Example:
      GSE12345 -> GSE12nnn
      GSE99999 -> GSE99nnn
    """
    num = int(gse.replace("GSE", ""))
    prefix = (num // 1000) * 1000
    return f"GSE{prefix//1000:02d}nnn".replace("nnn", "nnn") if prefix < 100000 else f"GSE{prefix}nnn"


def gse_parent_dir(gse: str) -> str:
    """
    Build the correct GEO parent directory for a given GSE.

    Example:
      GSE12345 -> GSE12nnn/GSE12345
      GSE9876  -> GSE9nnn/GSE9876
    """
    num = int(gse.replace("GSE", ""))
    parent = f"GSE{(num // 1000)}nnn"
    return f"{parent}/{gse}"


def series_matrix_url(gse: str) -> str:
    """
    Construct the URL for the gzipped Series Matrix file.
    """
    parent = gse_parent_dir(gse)
    return f"{GEO_FTP_BASE}/{parent}/matrix/{gse}_series_matrix.txt.gz"


def download(url: str, out_path: Path) -> bool:
    """
    Download a URL to out_path.
    Returns True on success, False on failure.
    """
    try:
        r = requests.get(url, stream=True, timeout=TIMEOUT)
        if r.status_code != 200:
            return False

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        tmp.replace(out_path)
        return True

    except Exception:
        return False



# 3) Main logic


def main():
    # Load manifest and extract unique GSE IDs
    df = pd.read_csv(MANIFEST_FILE)
    # Read all accessions from the manifest (some are GEO GSE, some are E-MTAB)
    all_accessions = df["GSE"].dropna().astype(str).str.strip().unique().tolist()

    # Keep only true GEO Series accessions: GSE + digits
    gse_pat = re.compile(r"^GSE\d+$", re.IGNORECASE)
    gses = sorted([x.upper() for x in all_accessions if gse_pat.match(x)])

    # Collect non-GSE accessions (e.g., E-MTAB-xxxxx) for reporting
    non_gse = sorted([x for x in all_accessions if not gse_pat.match(x)])

    print("Total unique accessions:", len(all_accessions))
    print("GEO GSE accessions:", len(gses))
    print("Non-GSE accessions:", len(non_gse))
    if non_gse:
        print("Examples of non-GSE:", non_gse[:10])

    print("Unique GSEs to download:", len(gses))
    print("Output folder:", OUT_DIR)

    # Track results
    ok = 0
    fail = 0

    for gse in gses:
        url = series_matrix_url(gse)
        out_file = OUT_DIR / f"{gse}_series_matrix.txt.gz"

        # Skip if already downloaded
        if out_file.exists() and out_file.stat().st_size > 0:
            print("[SKIP]", gse, "(already downloaded)")
            ok += 1
            continue

        print("[GET ]", gse, url)

        success = download(url, out_file)
        if success:
            print("[OK  ]", gse)
            ok += 1
        else:
            print("[FAIL]", gse, "(not found or download error)")
            fail += 1

        time.sleep(SLEEP_BETWEEN)

    print("\n=== DONE ===")
    print("Success:", ok)
    print("Failed :", fail)
    print("Folder :", OUT_DIR)


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import requests
import re
import time

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
MANIFEST_FILE = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"
OUT_DIR = BASE_DIR / "geo_supplementary"
OUT_DIR.mkdir(exist_ok=True)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"
TIMEOUT = (20, 120)
SLEEP = 0.3

KEEP_PATTERNS = [
    r"count", r"tpm", r"fpkm", r"expression", r"matrix"
]

OK_EXT = (".txt", ".tsv", ".csv", ".gz", ".zip")

def is_gse(x: str) -> bool:
    return bool(re.match(r"^GSE\d+$", x.strip(), re.I))

def gse_parent_dir(gse: str) -> str:
    num = int(gse.replace("GSE", ""))
    return f"GSE{(num // 1000)}nnn/{gse}"

def list_files(gse: str) -> list[str]:
    url = f"{GEO_FTP}/{gse_parent_dir(gse)}/suppl/"
    r = requests.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return []
    files = re.findall(r'href="([^"]+)"', r.text)
    files = [f for f in files if not f.endswith("/") and not f.startswith("?")]
    return sorted(set(files))

def good_file(name: str) -> bool:
    n = name.lower()
    if "raw" in n:
        return False
    if not any(n.endswith(ext) for ext in OK_EXT):
        return False
    return any(re.search(p, n) for p in KEEP_PATTERNS)

def download(url: str, out_path: Path) -> bool:
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

def main():
    df = pd.read_csv(MANIFEST_FILE)
    gses = sorted({g.upper() for g in df["GSE"].dropna().astype(str) if is_gse(g)})

    print("GSEs:", len(gses))
    rows = []

    for gse in gses:
        files = list_files(gse)
        chosen = [f for f in files if good_file(f)]

        if not chosen:
            print("[SKIP]", gse, "(no non-RAW expression files)")
            rows.append({"GSE": gse, "downloaded": ""})
            continue

        for fname in chosen:
            url = f"{GEO_FTP}/{gse_parent_dir(gse)}/suppl/{fname}"
            out_path = OUT_DIR / fname

            if out_path.exists() and out_path.stat().st_size > 0:
                continue

            ok = download(url, out_path)
            print("[DL]" if ok else "[FAIL]", gse, fname)
            time.sleep(SLEEP)

        rows.append({"GSE": gse, "downloaded": ";".join(chosen)})

    pd.DataFrame(rows).to_csv(OUT_DIR / "supp_download_report_noRAW.csv", index=False)
    print("Saved report:", OUT_DIR / "supp_download_report_noRAW.csv")

if __name__ == "__main__":
    main()

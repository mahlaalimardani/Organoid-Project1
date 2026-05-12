from pathlib import Path
import requests
import time
import re

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
GSE_LIST = BASE_DIR / "tissue_gse_list.txt"

OUTDIR = BASE_DIR / "geo_supplementary_tissue"
OUTDIR.mkdir(exist_ok=True)

REPORT = OUTDIR / "tissue_supp_download_report.csv"

SLEEP = 1.0
TIMEOUT = 60

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

def gse_parent_dir(gse: str) -> str:
    num = int(gse.replace("GSE", ""))
    base = (num // 1000) * 1000
    return f"GSE{base:03d}nnn"

def list_supp_files(gse: str) -> list[str]:
    parent = gse_parent_dir(gse)
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{parent}/{gse}/suppl/"
    r = SESSION.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return []
    # crude parse of ftp directory listing html
    return re.findall(r'href="([^"]+)"', r.text)

def is_good_processed(name: str) -> bool:
    n = name.lower()
    if n.endswith("/"):
        return False
    if "raw" in n:
        return False
    # keep common processed formats
    ok_ext = (".txt.gz", ".tsv.gz", ".csv.gz", ".txt", ".tsv", ".csv")
    return n.endswith(ok_ext) and ("count" in n or "fpkm" in n or "tpm" in n or "matrix" in n or "expr" in n)

def download(url: str, out_path: Path) -> bool:
    try:
        with SESSION.get(url, stream=True, timeout=TIMEOUT) as r:
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
    gses = [x.strip() for x in open(GSE_LIST, encoding="utf-8").read().splitlines() if x.strip()]
    print("Tissue GSEs:", len(gses))

    # Keep only GEO accessions (GSE). E-MTAB belongs to ArrayExpress/ENA and needs a different downloader.
    gses = [g for g in gses if g.startswith("GSE")]
    print("GSE-only Tissue GSEs:", len(gses))
    rows = []
    for gse in gses:
        files = list_supp_files(gse)
        good = [fn for fn in files if is_good_processed(fn)]

        if not good:
            print("[SKIP]", gse, "(no non-RAW expression files)")
            rows.append({"GSE": gse, "status": "SKIP", "n_files": 0})
            continue

        parent = gse_parent_dir(gse)
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{parent}/{gse}/suppl/"

        n_ok = 0
        for fn in good:
            url = base_url + fn
            out_path = OUTDIR / fn
            if out_path.exists() and out_path.stat().st_size > 0:
                n_ok += 1
                continue

            ok = download(url, out_path)
            print("[DL]" if ok else "[FAIL]", gse, fn)
            if ok:
                n_ok += 1
            time.sleep(SLEEP)

        rows.append({"GSE": gse, "status": "OK" if n_ok else "FAIL", "n_files": n_ok})

    # save report
    import pandas as pd
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print("Saved report:", REPORT)
    print("Output folder:", OUTDIR)

if __name__ == "__main__":
    main()

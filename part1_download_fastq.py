import pandas as pd
import requests
from pathlib import Path

BASE_DIR = Path(r"C:\Users\pishgaman\Desktop\Biology ISI\biology")
MANIFEST = BASE_DIR / "run_manifest_human_bulk_with_runs.csv"
OUTDIR = BASE_DIR / "fastq"
LOGDIR = BASE_DIR / "logs"
OUTDIR.mkdir(exist_ok=True)
LOGDIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

def ena_fastq_urls(run_id: str) -> list[str]:
    run_id = run_id.strip()
    api = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={run_id}&result=read_run&fields=fastq_ftp&format=tsv"
    )
    r = SESSION.get(api, timeout=60)
    r.raise_for_status()

    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        return []

    header = lines[0].split("\t")
    values = lines[1].split("\t")
    d = dict(zip(header, values))
    fastq_ftp = d.get("fastq_ftp", "").strip()
    if not fastq_ftp:
        return []

    paths = [p.strip() for p in fastq_ftp.split(";") if p.strip()]

    urls = []
    for p in paths:
        if p.startswith("http"):
            urls.append(p)
        elif p.startswith("ftp://"):
            urls.append(p.replace("ftp://", "https://"))
        elif p.startswith("ftp.sra.ebi.ac.uk/"):
            urls.append("https://" + p)
        else:
            urls.append("https://ftp.sra.ebi.ac.uk/" + p.lstrip("/"))
    return urls

def download_file(url: str, out_path: Path, log_file: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with SESSION.get(url, stream=True, timeout=120) as r:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"URL: {url}\nSTATUS: {r.status_code}\n")
        r.raise_for_status()

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)

def main():
    df = pd.read_csv(MANIFEST)
    runs = sorted(df["RUN"].dropna().astype(str).str.strip().unique())
    print("Total RUNs:", len(runs))
    print("Output:", OUTDIR)

    for run_id in runs:
        log_file = LOGDIR / f"{run_id}_ena.txt"

        try:
            urls = ena_fastq_urls(run_id)
            if not urls:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("No ENA fastq_ftp found.\n")
                print(f"[SKIP] {run_id} (no ENA URLs)")
                continue

            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(urls) + "\n\n")

            for url in urls:
                fname = url.split("/")[-1]
                out_path = OUTDIR / fname

                if out_path.exists() and out_path.stat().st_size > 0:
                    continue

                print(f"[DL] {run_id} -> {fname}")
                download_file(url, out_path, log_file)

            print(f"[OK] {run_id}")

        except Exception as e:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {repr(e)}\n")
            print(f"[FAIL] {run_id} (see {log_file})")

if __name__ == "__main__":
    main()





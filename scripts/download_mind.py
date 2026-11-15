"""Download MIND small (train + dev) into data/raw/.

MIND is for research use only (Microsoft Research License). If both mirrors fail,
download MINDsmall_train.zip and MINDsmall_dev.zip by hand from https://msnews.github.io/
and unzip them into data/raw/train and data/raw/dev.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MIRRORS = [
    "https://mind201910small.blob.core.windows.net/release/{name}",
    "https://recodatasets.z20.web.core.windows.net/newsrec/{name}",
]
FILES = {"train": "MINDsmall_train.zip", "dev": "MINDsmall_dev.zip"}


def fetch(name: str, dest: Path) -> bool:
    for mirror in MIRRORS:
        url = mirror.format(name=name)
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
    return False


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for split, name in FILES.items():
        out_dir = RAW / split
        if (out_dir / "behaviors.tsv").exists():
            print(f"{split}: already there")
            continue
        zip_path = RAW / name
        if not zip_path.exists() and not fetch(name, zip_path):
            sys.exit(f"Could not download {name}. See the docstring for the manual download.")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        print(f"{split}: extracted to {out_dir}")


if __name__ == "__main__":
    main()

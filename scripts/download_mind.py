"""Download MIND small (train + dev) into data/raw/.

MIND is for research use only (Microsoft Research License). The Azure links of the MIND site are closed
since 2025 (HTTP 409), so the files come from the Hugging Face mirror Recommenders/MIND, pinned to one
commit and checked against the SHA-256 of the files. If every mirror fails, download MINDsmall_train.zip
and MINDsmall_dev.zip by hand and unzip them into data/raw/train and data/raw/dev.
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MIRRORS = [
    "https://huggingface.co/datasets/Recommenders/MIND/resolve/2115068f457b577f93a268fffe22373e45727d89/{name}",
    "https://mind201910small.blob.core.windows.net/release/{name}",
    "https://recodatasets.z20.web.core.windows.net/newsrec/{name}",
]
FILES = {"train": "MINDsmall_train.zip", "dev": "MINDsmall_dev.zip"}
SHA256 = {"MINDsmall_train.zip": "6ef97a271580b98ccfc4301ada55cc639423cb0576a78b8dcfcf74a4dbcc3194",
          "MINDsmall_dev.zip": "d6ce515dcaa6b6d47ddf0a326eebc8a31b84735ae410285c9882ca2a06eec669"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> bool:
    for mirror in MIRRORS:
        url = mirror.format(name=name)
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest)
            if sha256(dest) == SHA256[name]:
                return True
            print("  wrong checksum, file removed")
            dest.unlink()
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

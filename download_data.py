"""
download_data.py
================
STEP 1 of the IPL Analytics project: fetch the raw data.

We use Cricsheet (https://cricsheet.org) -- a brilliant free resource that
publishes ball-by-ball data for every IPL match as CSV. The download is a
single ZIP that contains TWO files per match:

    <match_id>.csv        -> every delivery (ball) bowled in that match
    <match_id>_info.csv   -> match metadata (teams, venue, toss, result, ...)

Running this script downloads that ZIP and extracts it into data/raw/.
It is safe to run repeatedly: it skips the download if the data is already
there, unless you pass --force.

    python download_data.py
    python download_data.py --force   # re-download from scratch

Data credit: Cricsheet (https://cricsheet.org). Please keep the attribution.
"""
from __future__ import annotations

import argparse
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

# Path(__file__) is THIS file. .resolve().parent is the folder it lives in,
# which we treat as the project root. Building paths this way means the script
# works no matter which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context that can actually verify HTTPS certificates.

    macOS Python installed from python.org does NOT use the system keychain,
    so HTTPS downloads fail with 'CERTIFICATE_VERIFY_FAILED'. The portable fix
    is to verify against the CA bundle shipped by the `certifi` package.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # Fall back to the system default (works on most non-macOS setups).
        return ssl.create_default_context()


def already_downloaded() -> bool:
    """True if it looks like we've already extracted the data.

    A successful extraction leaves well over a thousand CSV files, so a simple
    count is a reliable enough check for our purposes.
    """
    return RAW_DIR.exists() and len(list(RAW_DIR.glob("*.csv"))) > 100


def download_and_extract() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IPL data from {DATA_URL} ...")
    # Some servers reject Python's default user-agent, so we set a polite one.
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "ipl-analytics-learning-project/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120, context=_ssl_context()) as response:
        raw_bytes = response.read()
    print(f"  downloaded {len(raw_bytes) / 1_000_000:.1f} MB")

    # The ZIP lives only in memory (io.BytesIO) -- no need to save it to disk.
    print(f"Extracting into {RAW_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        zf.extractall(RAW_DIR)
    n = len(list(RAW_DIR.glob("*.csv")))
    print(f"  extracted {n} CSV files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw IPL data from Cricsheet.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if data already exists."
    )
    args = parser.parse_args()

    if already_downloaded() and not args.force:
        n = len(list(RAW_DIR.glob("*.csv")))
        print(f"Data already present in {RAW_DIR} ({n} files).")
        print("Nothing to do. Use --force to re-download.")
        return

    download_and_extract()
    print("Done. Next: `python lesson1_build_dataset.py`")


if __name__ == "__main__":
    main()

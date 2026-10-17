"""Acquire pinned FineWeb-Edu Parquet files; no tokenization or model work."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parent
PRINT_LOCK = threading.Lock()


def report(message):
    with PRINT_LOCK:
        print(message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(item, plan):
    target = ROOT / "raw" / Path(item["path"]).name
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists():
        if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            raise RuntimeError(f"Existing raw file fails checksum; left unchanged: {target}")
        report(f"REUSED verified {target.name}")
        return target

    base_url = (
        f"https://huggingface.co/datasets/{plan['dataset']}/resolve/"
        f"{plan['revision']}/{item['path']}"
    )
    last_report = 0.0
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item["bytes"]:
            break
        if offset > item["bytes"]:
            raise RuntimeError(f"Partial file exceeds expected size: {partial}")
        try:
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(
                base_url + f"?download=true&attempt={attempt}",
                headers=headers, stream=True, timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    report(f"RESTART {target.name}: server did not honor resume")
                    offset = 0
                if offset:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Incorrect Content-Range for {target.name}")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset > item["bytes"]:
                            raise RuntimeError(f"Download exceeds expected size: {target.name}")
                        now = time.monotonic()
                        if now - last_report >= 30:
                            report(f"DOWNLOAD {target.name}: {offset:,}/{item['bytes']:,} bytes")
                            last_report = now
            if partial.stat().st_size == item["bytes"]:
                break
            raise RuntimeError(f"Incomplete response for {target.name}")
        except (requests.RequestException, OSError, RuntimeError) as error:
            report(f"RETRY {target.name} attempt {attempt}: {type(error).__name__}: {str(error)[:180]}")
            if attempt == 8:
                raise
            time.sleep(min(2 * attempt, 10))

    if not partial.exists() or partial.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Wrong final size: {partial}")
    actual_sha = sha256(partial)
    if actual_sha != item["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {partial}; raw file not published")
    parquet = pq.ParquetFile(partial)
    required = {"text", "id", "url", "language", "language_score", "token_count"}
    if not required.issubset(parquet.schema_arrow.names):
        raise RuntimeError(f"Unexpected Parquet schema: {partial}")
    partial.rename(target)
    report(f"VERIFIED {target.name}: {item['bytes']:,} bytes, {parquet.metadata.num_rows:,} documents, SHA-256 matches publisher")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = json.loads((ROOT / "acquisition_plan.json").read_text(encoding="utf-8"))
    items = plan["shards"][:args.count]
    (ROOT / "raw").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(acquire, item, plan) for item in items]
        for future in futures:
            future.result()
    report(f"COMPLETE: {len(items)} pinned source files verified")


if __name__ == "__main__":
    main()

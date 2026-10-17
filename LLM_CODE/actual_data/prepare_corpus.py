"""Clean, deduplicate, split and verify plain text. Never generates token IDs."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation")
POLICY = {
    "version": "fineweb_edu_125m_text_v1",
    "target_train_source_reported_gpt2_tokens": 3_000_000_000,
    "minimum_validation_source_reported_gpt2_tokens": 10_000_000,
    "language": "en",
    "minimum_language_score": 0.8,
    "minimum_rounded_educational_score": 3,
    "minimum_characters": 200,
    "minimum_whitespace_words": 60,
    "maximum_characters": 120_000,
    "maximum_repeated_line_character_fraction": 0.35,
    "minimum_character_retention": 0.85,
    "validation_host_hash_buckets": "10 of 1000; BLAKE2b, person=data125m-split",
    "deduplication": "Global whitespace-normalized NFC SHA-256 and canonical URL",
    "near_duplicate_policy": "Upstream FineWeb filtering; no additional local fuzzy deduplication",
    "document_separator": "\n\n",
    "tokenization_performed": False,
}
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\u200b]")
HORIZONTAL = re.compile(r"[ \t]{2,}|\t")
EMPTY_LINES = re.compile(r"\n{3,}")
LONG_RUN = re.compile(r"(.)\1{199,}")
HTML_START = re.compile(r"^\s*(?:<!doctype\s+html|<html\b)", re.I)
BOILERPLATE = {
    "accept all cookies", "accept cookies", "reject all cookies", "cookie settings",
    "manage cookies", "manage cookie preferences", "skip to main content",
    "skip to content", "all rights reserved", "all rights reserved.",
    "advertisement", "advertisements", "enable javascript to continue",
}
ERROR_TITLES = {
    "403 forbidden", "404 not found", "access denied", "service unavailable",
    "checking your browser", "just a moment...", "verify you are human",
}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_url(url):
    try:
        parts = urlsplit(url or "")
        host = (parts.hostname or "").lower().removeprefix("www.")
        pairs = parse_qsl(parts.query, keep_blank_values=True, max_num_fields=100)
        pairs = [(key, value) for key, value in pairs
                 if not key.lower().startswith("utm_")
                 and key.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}]
        query = urlencode(sorted(pairs))
        key = host + (parts.path.rstrip("/") or "/")
        return host, key + ("?" + query if query else "")
    except (ValueError, TypeError):
        return "", ""


def split_for_host(host):
    digest = hashlib.blake2b(host.encode("utf-8"), digest_size=8, person=b"data125m-split").digest()
    return "validation" if int.from_bytes(digest, "big") % 1000 < 10 else "train"


def clean_text(original):
    if not isinstance(original, str) or not original.strip():
        return None, "empty_text"
    if len(original) > POLICY["maximum_characters"]:
        return None, "overlong_document"
    if HTML_START.search(original):
        return None, "unextracted_html"
    if "<|endoftext|>" in original:
        return None, "reserved_token_literal"
    if original.count("\ufffd") > max(1, len(original) // 1000):
        return None, "encoding_noise"
    text = unicodedata.normalize("NFC", original.replace("\r\n", "\n").replace("\r", "\n"))
    text = CONTROL_RE.sub("", text)
    lines = [HORIZONTAL.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line.casefold() not in BOILERPLATE]
    text = EMPTY_LINES.sub("\n\n", "\n".join(lines)).strip()
    if len(text) < POLICY["minimum_characters"] or len(text.split()) < POLICY["minimum_whitespace_words"]:
        return None, "too_short"
    if len(text) / len(original) < POLICY["minimum_character_retention"]:
        return None, "mostly_boilerplate"
    if text.split("\n", 1)[0].casefold() in ERROR_TITLES:
        return None, "error_or_challenge_page"
    if LONG_RUN.search(text):
        return None, "repeated_character_noise"
    seen_lines = set()
    repeated = 0
    for line in text.split("\n"):
        if len(line) < 20:
            continue
        if line in seen_lines:
            repeated += len(line)
        seen_lines.add(line)
    if repeated / len(text) > POLICY["maximum_repeated_line_character_fraction"]:
        return None, "repeated_line_noise"
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return None, "invalid_unicode"
    return text, None


def initial_state():
    return {
        "next_shard": 0, "next_row": 0, "scanned_documents": 0,
        "rejected": {}, "source_files": [],
        "splits": {split: {"documents": 0, "bytes": 0, "index_bytes": 0,
                           "characters": 0, "original_characters": 0,
                           "source_reported_gpt2_tokens_before_local_cleaning": 0}
                   for split in SPLITS},
    }


def budget_reached(state):
    return (state["splits"]["train"]["source_reported_gpt2_tokens_before_local_cleaning"] >=
            POLICY["target_train_source_reported_gpt2_tokens"] and
            state["splits"]["validation"]["source_reported_gpt2_tokens_before_local_cleaning"] >=
            POLICY["minimum_validation_source_reported_gpt2_tokens"])


def prepare():
    if (ROOT / "manifest.json").exists():
        verify()
        return
    plan = json.loads((ROOT / "acquisition_plan.json").read_text(encoding="utf-8"))
    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    connection = sqlite3.connect(work / "dedup.sqlite3")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-1048576")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE IF NOT EXISTS seen (content BLOB PRIMARY KEY, url BLOB UNIQUE, split TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    stored_policy = connection.execute("SELECT value FROM metadata WHERE key='policy'").fetchone()
    policy_json = json.dumps(POLICY, sort_keys=True)
    if stored_policy and stored_policy[0] != policy_json:
        raise RuntimeError("Cleaning policy changed during a resumable build")
    connection.execute("INSERT OR IGNORE INTO metadata VALUES ('policy', ?)", (policy_json,))
    stored = connection.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
    state = json.loads(stored[0]) if stored else initial_state()
    handles = {}
    for split in SPLITS:
        for kind, suffix, key in (("text", ".txt.partial", "bytes"), ("index", ".documents.jsonl.partial", "index_bytes")):
            path = work / (split + suffix)
            # Recover publication interrupted after an atomic rename.
            final = ROOT / (split + (".txt" if kind == "text" else ".documents.jsonl"))
            if not path.exists() and final.exists():
                final.rename(path)
            handle = path.open("r+b" if path.exists() else "w+b")
            handle.truncate(state["splits"][split][key])
            handle.seek(state["splits"][split][key])
            handles[(split, kind)] = handle

    def checkpoint():
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('state', ?)", (json.dumps(state),))
        connection.commit()
        atomic_json(ROOT / "progress.json", state)

    checkpoint()
    last_checkpoint = time.monotonic()
    columns = ["text", "id", "url", "dump", "date", "language", "language_score", "token_count", "int_score"]
    try:
        while state["next_shard"] < len(plan["shards"]) and not budget_reached(state):
            item = plan["shards"][state["next_shard"]]
            raw_path = ROOT / "raw" / Path(item["path"]).name
            if not raw_path.exists():
                print(f"WAITING for verified source {raw_path.name}", flush=True)
                time.sleep(30)
                continue
            if not any(entry["path"] == item["path"] for entry in state["source_files"]):
                if raw_path.stat().st_size != item["bytes"] or file_hash(raw_path) != item["sha256"]:
                    raise RuntimeError(f"Source checksum failed: {raw_path}")
                state["source_files"].append(item)
            parquet = pq.ParquetFile(raw_path)
            available_columns = [name for name in columns if name in parquet.schema_arrow.names]
            required_columns = set(columns) - {"date", "dump"}
            if not required_columns.issubset(available_columns):
                raise RuntimeError(f"Source is missing required columns: {raw_path}")
            row_number = 0
            print(f"PROCESSING {raw_path.name}, resume row {state['next_row']:,}", flush=True)
            for batch in parquet.iter_batches(batch_size=256, columns=available_columns):
                batch_start = row_number
                row_number += batch.num_rows
                if row_number <= state["next_row"]:
                    continue
                for offset, row in enumerate(batch.to_pylist()):
                    position = batch_start + offset
                    if position < state["next_row"]:
                        continue
                    if budget_reached(state):
                        break
                    state["next_row"] = position + 1
                    state["scanned_documents"] += 1
                    reason = None
                    if row["language"] != "en" or float(row["language_score"] or 0) < POLICY["minimum_language_score"]:
                        reason = "language_quality"
                    elif int(row["int_score"] or 0) < POLICY["minimum_rounded_educational_score"]:
                        reason = "educational_quality"
                    elif int(row["token_count"] or 0) <= 0:
                        reason = "invalid_source_token_count"
                    else:
                        text, reason = clean_text(row["text"])
                    host, url_key = canonical_url(row["url"])
                    if not reason and not host:
                        reason = "missing_source_hostname"
                    if not reason:
                        payload = text.encode("utf-8")
                        content_key = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
                        url_hash = hashlib.sha256(url_key.encode("utf-8")).digest()
                        split = split_for_host(host)
                        cursor = connection.execute("INSERT OR IGNORE INTO seen VALUES (?, ?, ?)", (content_key, url_hash, split))
                        if cursor.rowcount == 0:
                            reason = "duplicate_content_or_url"
                    if reason:
                        state["rejected"][reason] = state["rejected"].get(reason, 0) + 1
                    else:
                        stats = state["splits"][split]
                        text_handle = handles[(split, "text")]
                        if stats["documents"]:
                            text_handle.write(b"\n\n")
                            stats["bytes"] += 2
                        start = stats["bytes"]
                        text_handle.write(payload)
                        stats["bytes"] += len(payload)
                        record = {
                            "id": row["id"], "source": plan["dataset"], "revision": plan["revision"],
                            "source_shard": item["path"], "source_row": position,
                            "url": row["url"], "hostname": host, "dump": row.get("dump"), "date": row.get("date"),
                            "byte_start": start, "byte_end": stats["bytes"],
                            "text_sha256": hashlib.sha256(payload).hexdigest(),
                            "normalized_content_sha256": content_key.hex(),
                            "source_reported_gpt2_tokens_before_local_cleaning": int(row["token_count"]),
                        }
                        index_payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        handles[(split, "index")].write(index_payload)
                        stats["index_bytes"] += len(index_payload)
                        stats["documents"] += 1
                        stats["characters"] += len(text)
                        stats["original_characters"] += len(row["text"])
                        stats["source_reported_gpt2_tokens_before_local_cleaning"] += int(row["token_count"])
                    if state["scanned_documents"] % 5000 == 0 or time.monotonic() - last_checkpoint > 30:
                        checkpoint()
                        last_checkpoint = time.monotonic()
                        print(f"CLEANED scanned={state['scanned_documents']:,} train_docs={state['splits']['train']['documents']:,} train_source_tokens={state['splits']['train']['source_reported_gpt2_tokens_before_local_cleaning']:,}", flush=True)
                if budget_reached(state):
                    break
            if not budget_reached(state):
                state["next_shard"] += 1
                state["next_row"] = 0
            checkpoint()
        if not budget_reached(state):
            raise RuntimeError("Available source shards exhausted before the corpus budget was met")
        checkpoint()
    finally:
        for handle in handles.values():
            handle.close()
        connection.close()

    for split in SPLITS:
        (work / (split + ".txt.partial")).replace(ROOT / (split + ".txt"))
        (work / (split + ".documents.jsonl.partial")).replace(ROOT / (split + ".documents.jsonl"))
    manifest = {
        "status": "built_pending_verification", "dataset": plan["dataset"], "revision": plan["revision"],
        "policy": POLICY, "source_files": state["source_files"], "splits": state["splits"],
        "scanned_documents": state["scanned_documents"], "rejected_documents": state["rejected"],
        "token_count_note": "Counts are publisher-provided GPT-2 lengths of retained source documents before local cleaning. Cleaned text was NOT tokenized; these are size estimates, not measured final token counts.",
        "files": {},
    }
    for split in SPLITS:
        for suffix in (".txt", ".documents.jsonl"):
            path = ROOT / (split + suffix)
            manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    atomic_json(ROOT / "manifest.json", manifest)
    verify()


def verify():
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    database = sqlite3.connect(f"file:{ROOT / 'work' / 'dedup.sqlite3'}?mode=ro", uri=True)
    totals = Counter()
    for filename, expected in manifest["files"].items():
        path = ROOT / filename
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise RuntimeError(f"Output checksum mismatch: {filename}")
        print(f"CHECKSUM PASS {filename}", flush=True)
    for split in SPLITS:
        document_count = 0
        source_tokens = 0
        characters = 0
        with (ROOT / (split + ".txt")).open("rb") as text_file, (ROOT / (split + ".documents.jsonl")).open("r", encoding="utf-8") as index:
            for line in index:
                record = json.loads(line)
                if document_count and text_file.read(2) != b"\n\n":
                    raise RuntimeError("Incorrect document separator")
                if text_file.tell() != record["byte_start"]:
                    raise RuntimeError("Incorrect document offset")
                payload = text_file.read(record["byte_end"] - record["byte_start"])
                text = payload.decode("utf-8", errors="strict")
                if not text or hashlib.sha256(payload).hexdigest() != record["text_sha256"]:
                    raise RuntimeError("Invalid document payload")
                digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
                if digest.hex() != record["normalized_content_sha256"]:
                    raise RuntimeError("Incorrect normalized document fingerprint")
                stored_split = database.execute("SELECT split FROM seen WHERE content=?", (digest,)).fetchone()
                if stored_split != (split,) or split_for_host(record["hostname"]) != split:
                    raise RuntimeError("Duplicate contamination or incorrect holdout split")
                if "<|endoftext|>" in text or clean_text(text)[0] != text:
                    raise RuntimeError("Cleaning policy is not stable for an output document")
                source_tokens += record["source_reported_gpt2_tokens_before_local_cleaning"]
                characters += len(text)
                document_count += 1
            if text_file.read(1):
                raise RuntimeError("Unindexed trailing text")
        expected = manifest["splits"][split]
        if (document_count, source_tokens, characters) != (expected["documents"], expected["source_reported_gpt2_tokens_before_local_cleaning"], expected["characters"]):
            raise RuntimeError(f"Split totals do not match: {split}")
        totals[split] = document_count
        print(f"CONTENT PASS {split}: {document_count:,} documents, UTF-8, offsets, fingerprints and host split", flush=True)
    stored_counts = dict(database.execute("SELECT split, COUNT(*) FROM seen GROUP BY split"))
    database.close()
    if stored_counts != dict(totals):
        raise RuntimeError("Deduplication database and output counts disagree")
    if manifest["splits"]["train"]["source_reported_gpt2_tokens_before_local_cleaning"] < POLICY["target_train_source_reported_gpt2_tokens"]:
        raise RuntimeError("Training data size target was not reached")
    manifest["status"] = "ready"
    manifest["verification"] = {
        "output_checksums": "PASS", "all_documents_utf8_and_nonempty": "PASS",
        "all_document_offsets_and_fingerprints": "PASS", "global_content_and_url_deduplication": "PASS",
        "train_validation_host_separation": "PASS", "cleaning_idempotence": "PASS",
        "source_reported_training_size_target": "PASS", "tokenization_performed": False,
    }
    atomic_json(manifest_path, manifest)
    print("CORPUS READY: train.txt and validation.txt verified; no tokenization performed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (ROOT / "work").mkdir(exist_ok=True)
    with (ROOT / "work" / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify() if args.verify else prepare()


if __name__ == "__main__":
    main()

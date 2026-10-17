"""Export the cleaned pilot as UTF-8 text for a later preprocessing pipeline.

Uses only the standard library. Does not tokenize, train a tokenizer, generate
embeddings/positions/labels, pack sequences, download data, or run OCR.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True

DATA_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = DATA_ROOT / "training_text/pilot_v1"
SOURCES = ("wikipedia", "fineweb", "code", "stackexchange",
           "gutenberg", "arxiv", "scanned_docs")
SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    result = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def immutable_snapshot():
    paths = set()
    for directory in ("raw", "standardized", "cleaned", "clean_pilot",
                      "standardize_pilot"):
        paths.update(p for p in (DATA_ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(DATA_ROOT)): file_hash(p) for p in sorted(paths)}


def read_corpus():
    checkpoint = json.loads(
        (DATA_ROOT / "cleaned/pilot_manifest.json").read_text(encoding="utf-8")
    )
    require(checkpoint["status"] == "complete", "The cleaned checkpoint is incomplete.")
    documents, seen_ids = [], set()
    for source in SOURCES:
        path = DATA_ROOT / "cleaned" / source / "documents.jsonl"
        require(file_hash(path) == checkpoint["output_jsonl_checksums"][source],
                f"{source}: cleaned checksum differs from the validated checkpoint.")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                doc = json.loads(line)
                require(set(doc) == {"id", "source", "text", "metadata"},
                        "Unexpected document schema.")
                require(isinstance(doc["id"], str) and doc["id"].strip()
                        and doc["id"] not in seen_ids, "Invalid/duplicate document ID.")
                require(doc["source"] == source, "Incorrect document source.")
                require(isinstance(doc["text"], str) and doc["text"].strip(),
                        f"{doc['id']}: text is empty.")
                require(isinstance(doc["metadata"], dict), "Invalid metadata.")
                json.dumps(doc, ensure_ascii=False, allow_nan=False).encode("utf-8")
                seen_ids.add(doc["id"])
                documents.append(doc)
    require(len(documents) == checkpoint["total_output_documents"],
            "Cleaned corpus count differs from its checkpoint.")
    return documents


def deduplicate(documents):
    kept, duplicates, seen = [], [], {}
    for doc in documents:
        fingerprint = sha256(doc["text"].encode("utf-8")).hexdigest()
        if fingerprint in seen:
            original = seen[fingerprint]
            require(original["text"] == doc["text"], "Unexpected hash collision.")
            duplicates.append({
                "id": doc["id"], "source": doc["source"],
                "kept_id": original["id"], "metadata": doc["metadata"],
                "reason": "byte_identical_cleaned_text",
            })
        else:
            seen[fingerprint] = doc
            kept.append(doc)
    return kept, duplicates


def split_corpus(documents):
    grouped = defaultdict(list)
    for doc in documents:
        grouped[doc["source"]].append(doc)
    result = {"train": [], "validation": []}
    for source in SOURCES:
        ranked = sorted(grouped[source], key=lambda d: (
            sha256(f"{SEED}:validation:{d['id']}".encode()).hexdigest(), d["id"]
        ))
        # Keep entire books/articles together. Small source sets retain at
        # least one validation document when they contain two or more records.
        holdout = max(1, len(ranked) // 10) if len(ranked) >= 2 else 0
        result["validation"].extend(ranked[:holdout])
        result["train"].extend(ranked[holdout:])
    for split, docs in result.items():
        docs.sort(key=lambda d: (
            sha256(f"{SEED}:{split}:{d['id']}".encode()).hexdigest(), d["id"]
        ))
    require(result["train"] and result["validation"], "Both splits must be nonempty.")
    return result


def review_flags(documents):
    result = []
    for doc in documents:
        flags = []
        information = doc["metadata"].get("cleaning", {})
        if (doc["source"] == "wikipedia"
                and information.get("percentage_removed", 0) > 75):
            flags.append("large_wikipedia_template_table_reference_reduction")
        if doc["source"] == "scanned_docs":
            flags.append("existing_uncorrected_ocr_text")
        if flags:
            result.append({"id": doc["id"], "source": doc["source"],
                           "flags": flags, "characters": len(doc["text"])})
    return result


def export_split(destination, split, documents):
    bytes_written = characters_written = 0
    with (destination / f"{split}.txt").open("wb") as text_file, (
        destination / f"{split}.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as jsonl_file, (
        destination / f"{split}_document_index.jsonl"
    ).open("w", encoding="utf-8", newline="\n") as index_file:
        for number, doc in enumerate(documents):
            if number:
                text_file.write(b"\n\n")
                bytes_written += 2
                characters_written += 2
            payload = doc["text"].encode("utf-8")
            entry = {
                "id": doc["id"], "source": doc["source"], "split": split,
                "record_index": number, "byte_start": bytes_written,
                "byte_end": bytes_written + len(payload),
                "character_start": characters_written,
                "character_end": characters_written + len(doc["text"]),
            }
            text_file.write(payload)
            jsonl_file.write(json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n")
            index_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            bytes_written += len(payload)
            characters_written += len(doc["text"])
    return {
        "documents": len(documents),
        "document_characters": sum(len(doc["text"]) for doc in documents),
        "text_file_bytes": bytes_written,
        "text_file_characters_including_separators": characters_written,
        "source_documents": dict(sorted(Counter(doc["source"] for doc in documents).items())),
    }


def verify(destination, documents):
    kept, duplicates = deduplicate(documents)
    expected = split_corpus(kept)
    seen_ids, split_texts = set(), {}
    for split, docs in expected.items():
        payload = (destination / f"{split}.txt").read_bytes()
        text = payload.decode("utf-8")
        records = [json.loads(line) for line in (
            destination / f"{split}.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        indices = [json.loads(line) for line in (
            destination / f"{split}_document_index.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        require(records == docs, f"{split}: text, IDs, ordering, or metadata changed.")
        require(payload == b"\n\n".join(doc["text"].encode("utf-8") for doc in docs),
                f"{split}: plain-text export differs from source documents.")
        require(len(indices) == len(docs), "Incorrect document index count.")
        for number, (doc, entry) in enumerate(zip(docs, indices)):
            require(doc["id"] not in seen_ids, "Document occurs in both splits.")
            seen_ids.add(doc["id"])
            require(entry["id"] == doc["id"] and entry["source"] == doc["source"]
                    and entry["split"] == split and entry["record_index"] == number,
                    "Document index identity mismatch.")
            require(payload[entry["byte_start"]:entry["byte_end"]].decode("utf-8")
                    == doc["text"], "Byte boundaries do not recover the exact document.")
            require(text[entry["character_start"]:entry["character_end"]] == doc["text"],
                    "Character boundaries do not recover the exact document.")
        split_texts[split] = {doc["text"] for doc in docs}
    require(seen_ids == {doc["id"] for doc in kept}, "Prepared ID set mismatch.")
    require(not split_texts["train"] & split_texts["validation"],
            "Exact duplicate text occurs in both splits.")
    require(seen_ids | {doc["id"] for doc in duplicates}
            == {doc["id"] for doc in documents}, "Input provenance is incomplete.")
    return {
        "valid_utf8": True, "all_jsonl_valid": True,
        "all_source_text_preserved_exactly": True, "all_metadata_preserved": True,
        "all_document_boundaries_verified": True, "unique_prepared_ids": len(seen_ids),
        "train_validation_id_overlap": 0, "train_validation_exact_text_overlap": 0,
        "no_empty_documents": True,
    }


def build(destination, documents, checksums):
    destination.mkdir()
    kept, duplicates = deduplicate(documents)
    splits = split_corpus(kept)
    stats = {split: export_split(destination, split, docs)
             for split, docs in splits.items()}
    flags = review_flags(kept)
    write_json(destination / "review_flags.json", flags)
    write_json(destination / "preparation_report.json", {
        "input_documents": len(documents), "kept_documents": len(kept),
        "exact_duplicates_removed": len(duplicates), "duplicate_provenance": duplicates,
        "flags_are_not_filters": True, "flagged_documents": len(flags),
        "eligibility_checks": "Cleaned checksums, UTF-8, schema, unique IDs, nonempty text.",
        "quality_policy": "Retain valid pilot text; record known OCR/large-cleaning-loss flags.",
        "split_policy": "Seed 42; whole documents; source-stratified ~10% holdout; minimum one per source with 2+ records.",
        "notes": "Small-source holdout ratios are higher than 10%; validation is a pilot smoke test.",
    })
    readme = f"""# Pilot text ready for preprocessing

The input is the completed cleaned 410-document pilot. These are text files,
not token IDs. No tokenizer, embeddings, positional encodings, sequence packing,
labels, EOS/BOS tokens, or model were generated.

## Files to use

- train.txt: UTF-8 training text, {stats['train']['documents']} complete documents.
- validation.txt: held-out UTF-8 text, {stats['validation']['documents']} documents.
- train.jsonl / validation.jsonl: preferred when processing each document
  separately. Every record retains id, source, text, and all existing metadata.
- *_document_index.jsonl: byte and Unicode-character offsets into each TXT file;
  endpoints are exclusive. Offsets recover original document text exactly.
- manifest.json: counts, checksums, preservation checks, and rebuild verification.
- review_flags.json / preparation_report.json: provenance and retained review flags.

Use the text field of each JSONL record as input to your tiktoken step. If you
prefer a single string, read train.txt as UTF-8. TXT documents are separated by
two newline characters, without injected titles, IDs, source labels, or special
tokens. Because documents can contain blank lines, use the JSONL or index files
when you need exact document boundaries. The pipeline can add any intended
document boundary tokens later.

Training and validation contain disjoint whole documents. No book or article
was divided between them. The split is deterministic and source-stratified;
small source sets have at least one held-out document. Exact duplicates are
checked before splitting; this pilot had {len(duplicates)}.

## Preservation and scope

All exported document text, Unicode, code indentation, spaces, line endings,
and metadata are preserved exactly from the cleaned checkpoint. Raw,
standardized, and cleaned data remain unchanged. No further prose rewriting,
spelling correction, language filtering, or near-duplicate filtering is applied.
The existing OCR noise and large Wikipedia template/table losses are flagged;
they are not silently repaired or filtered.

This is a small pilot for building and testing your preprocessing/model data
pipeline. It is not a full-scale corpus for pretraining a 1B-parameter model.
It contains literature, web/encyclopedic text, technical discussions, code
(including Markdown), scientific text, and existing OCR text.

## Reproduce / verify

From /data/vansh/LLM_FROM_SCRATCH/LLM_CODE:

    python data/prepare_text/prepare_training_text.py

The script uses only Python's standard library. A new export is built twice in
temporary directories, checked for identical bytes, and published after both
builds pass validation. Rerunning verifies an existing export without overwriting
it. Future corpus versions should be exported into a new versioned directory.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    validation = verify(destination, documents)
    manifest = {
        "version": "pilot_training_text_v1", "status": "ready",
        "input_documents": len(documents), "prepared_documents": len(kept),
        "exact_duplicates_removed": len(duplicates),
        "document_characters": sum(len(doc["text"]) for doc in kept),
        "splits": stats, "encoding": "UTF-8", "document_separator": "\n\n",
        "review_flagged_documents": len(flags), "validation": validation,
        "tokenization_performed": False, "tokenizer_training_performed": False,
        "embeddings_created": False, "positional_encodings_created": False,
        "new_dataset_downloads": 0, "ocr_executed": False,
        "earlier_checkpoints_unchanged": True, "input_file_checksums": checksums,
        "artifact_checksums": {
            path.name: file_hash(path) for path in sorted(destination.iterdir())
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def main():
    before = immutable_snapshot()
    documents = read_corpus()
    if OUTPUT.exists():
        manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        require(before == manifest["input_file_checksums"],
                "Earlier checkpoints have changed since the export.")
        validation = verify(OUTPUT, documents)
        for name, checksum in manifest["artifact_checksums"].items():
            require(file_hash(OUTPUT / name) == checksum, f"Export file changed: {name}")
        print("Existing prepared text verified without overwriting it.")
        print(json.dumps(validation, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix=".prepare_text_", dir=DATA_ROOT) as temporary:
        first, second = Path(temporary) / "first", Path(temporary) / "second"
        manifest = build(first, documents, before)
        build(second, documents, before)
        require({p.name: file_hash(p) for p in first.iterdir()}
                == {p.name: file_hash(p) for p in second.iterdir()},
                "Independent text exports are not byte-identical.")
        require(immutable_snapshot() == before, "Earlier checkpoints changed.")
        manifest["independent_rebuild_byte_identical"] = True
        write_json(first / "manifest.json", manifest)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        first.replace(OUTPUT)
    print("PREPROCESSING TEXT: READY")
    print(f"Train: {manifest['splits']['train']['documents']} documents, "
          f"{manifest['splits']['train']['document_characters']:,} characters")
    print(f"Validation: {manifest['splits']['validation']['documents']} documents, "
          f"{manifest['splits']['validation']['document_characters']:,} characters")
    print("UTF-8, text/metadata fidelity, boundaries, split isolation, "
          "independent rebuild, and unchanged checkpoints: PASS")
    print("Tokenization / embeddings / positional encodings: NOT PERFORMED")
    print(OUTPUT)


if __name__ == "__main__":
    main()

"""I/O, validation, diagnostics, and opt-in helpers for pilot cleaning."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re


DATA_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("wikipedia", "fineweb", "code", "stackexchange", "gutenberg",
           "arxiv", "scanned_docs")
SCHEMA_KEYS = {"id", "source", "text", "metadata"}
# Keep ordinary line endings and tabs. Never delete Unicode format characters,
# accents, mathematical symbols, or replacement glyphs based on appearance.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_immutable(data_root=DATA_ROOT):
    paths = []
    for directory in ("raw", "standardized"):
        paths.extend(p for p in (data_root / directory).rglob("*") if p.is_file())
    paths.extend(data_root / name for name in (
        "clean_pilot/clean_wikipedia_one.py",
        "cleaned/wikipedia/sample_001_cleaned.json",
        "cleaned/wikipedia/sample_001_comparison.txt",
    ))
    return {str(p.relative_to(data_root)): file_sha256(p) for p in sorted(paths)}


def strict_json(text):
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON numeric constant: {value}")
    return json.loads(text, parse_constant=invalid_constant)


def validate_schema(document, source):
    if not isinstance(document, dict) or set(document) != SCHEMA_KEYS:
        raise ValueError("Expected exactly id, source, text, and metadata.")
    if not isinstance(document["id"], str) or not document["id"].strip():
        raise ValueError("Document ID must be a non-empty string.")
    if document["source"] != source:
        raise ValueError(f"Expected source {source!r}.")
    if not isinstance(document["text"], str):
        raise ValueError("Text must be a string.")
    if not isinstance(document["metadata"], dict):
        raise ValueError("Metadata must be an object.")
    json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_jsonl(path, source):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                document = strict_json(line)
                validate_schema(document, source)
            except (ValueError, TypeError, UnicodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            yield document


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 indent=2) + "\n")


def remove_controls(text):
    return CONTROL_RE.sub("", text), len(CONTROL_RE.findall(text))


def normalize_prose(text, collapse_horizontal=False, preserve_indentation=True):
    """Opt-in whitespace cleanup; NEVER called by the code cleaner."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.rstrip(" \t")
        if not line.strip(" \t"):
            line = ""
        elif collapse_horizontal:
            if preserve_indentation:
                indent = re.match(r"[ \t]*", line).group()
                line = indent + re.sub(r"[ \t]{3,}", " ", line[len(indent):])
            else:
                line = re.sub(r"[ \t]{3,}", " ", line)
        lines.append(line)
    # Three newline characters represent two blank lines.
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines)).strip("\n")


def normalize_markdown(text):
    """Keep fenced/indented code and inline-code spacing; trim prose only."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    output = []
    fence = None
    blank_lines = 0
    for line in text.split("\n"):
        marker = re.match(r"^ {0,3}(\x60{3,}|~{3,})", line)
        protected = fence is not None or bool(marker) or line.startswith(
            ("    ", "\t")
        )
        if protected:
            output.append(line)
            blank_lines = 0
        else:
            line = line.rstrip(" \t")
            if not line:
                blank_lines += 1
                if blank_lines <= 2:
                    output.append("")
            else:
                blank_lines = 0
                output.append(line)
        if marker:
            marks = marker.group(1)
            if fence is None:
                fence = (marks[0], len(marks))
            elif (marks[0] == fence[0] and len(marks) >= fence[1]
                  and not line[marker.end():].strip()):
                fence = None
    return "\n".join(output).strip("\n")


def validate_pair(original, cleaned, source):
    validate_schema(cleaned, source)
    if not cleaned["text"].strip():
        raise ValueError("Cleaning produced empty text; document not discarded silently.")
    if (cleaned["id"] != original["id"]
            or cleaned["source"] != original["source"]):
        raise ValueError("Cleaning changed document identity.")
    preserved = {k: v for k, v in cleaned["metadata"].items() if k != "cleaning"}
    if preserved != original["metadata"]:
        raise ValueError("Cleaning changed existing metadata.")
    information = cleaned["metadata"].get("cleaning", {})
    if (information.get("original_characters") != len(original["text"])
            or information.get("cleaned_characters") != len(cleaned["text"])
            or information.get("characters_removed")
            != len(original["text"]) - len(cleaned["text"])):
        raise ValueError("Cleaning character statistics do not match text.")


def document_summary(original, cleaned):
    old, new = len(original["text"]), len(cleaned["text"])
    return {
        "id": original["id"],
        "original_characters": old,
        "cleaned_characters": new,
        "characters_removed": old - new,
        "percentage_removed": (old - new) / old * 100 if old else 0.0,
    }


def process_source(source, version, cleaner, errors, data_root=DATA_ROOT):
    input_path = data_root / "standardized" / source / "documents.jsonl"
    output_dir = data_root / "cleaned" / source
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "documents.jsonl"
    temporary = final_path.with_name("documents.jsonl.tmp")
    before_hash = file_sha256(input_path)
    summaries, reviews = [], []
    counters = Counter()
    input_ids = set()
    failures = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for original in read_jsonl(input_path, source):
                identifier = original["id"]
                try:
                    if identifier in input_ids:
                        raise ValueError("Duplicate input ID.")
                    input_ids.add(identifier)
                    if "cleaning" in original["metadata"]:
                        raise ValueError("Input already contains cleaning metadata.")
                    text, measured, diagnostics = cleaner(original["text"])
                    cleaned = deepcopy(original)
                    cleaned["text"] = text
                    summary = document_summary(original, cleaned)
                    cleaned["metadata"]["cleaning"] = {
                        "version": version,
                        "stage": version,
                        **{k: v for k, v in summary.items() if k != "id"},
                        "counters": measured,
                        "diagnostics": diagnostics,
                    }
                    validate_pair(original, cleaned, source)
                    handle.write(json.dumps(cleaned, ensure_ascii=False,
                                            allow_nan=False) + "\n")
                    summaries.append(summary)
                    counters.update(measured)
                    reviews.append((summary, original["text"], text,
                                    measured, diagnostics))
                    reviews.sort(key=lambda item: (
                        -item[0]["percentage_removed"], item[0]["id"]
                    ))
                    # Enough for all small sources and the top three losses.
                    reviews = reviews[:5]
                except Exception as error:
                    failures += 1
                    errors.append({
                        "source": source, "id": identifier,
                        "error_type": type(error).__name__, "message": str(error),
                    })
                    if failures >= 3:
                        raise RuntimeError(
                            f"{source}: repeated errors; stop this source."
                        ) from error
        if failures:
            raise RuntimeError(f"{source}: {failures} failures; final file not replaced.")
        if not summaries:
            raise ValueError(f"{source}: no input documents.")
        # Re-read every temporary line and compare with its immutable input.
        originals = list(read_jsonl(input_path, source))
        saved = list(read_jsonl(temporary, source))
        if len(originals) != len(saved) or len(saved) != len(input_ids):
            raise ValueError("Input/output counts or ID uniqueness disagree.")
        for original, cleaned in zip(originals, saved):
            validate_pair(original, cleaned, source)
        if file_sha256(input_path) != before_hash:
            raise ValueError("Standardized source changed during cleaning.")
        temporary.replace(final_path)
    except Exception as error:
        if not errors or errors[-1]["source"] != source:
            errors.append({"source": source, "id": None,
                           "error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        temporary.unlink(missing_ok=True)

    old = sum(item["original_characters"] for item in summaries)
    new = sum(item["cleaned_characters"] for item in summaries)
    lengths = [item["cleaned_characters"] for item in summaries]
    selected = reviews if len(summaries) <= 5 else reviews[:3]
    manifest = {
        "source": source, "cleaning_version": version,
        "input_file": str(input_path.relative_to(data_root)),
        "input_sha256_before": before_hash,
        "input_sha256_after": file_sha256(input_path),
        "standardized_input_unchanged": True,
        "input_documents": len(originals), "output_documents": len(saved),
        "failures": failures, "original_characters": old,
        "cleaned_characters": new, "characters_removed": old - new,
        "percentage_removed": (old - new) / old * 100 if old else 0.0,
        "min_characters": min(lengths), "max_characters": max(lengths),
        "mean_characters": new / len(lengths),
        "documents_with_zero_reduction": sum(
            item["characters_removed"] == 0 for item in summaries
        ),
        **{f"documents_over_{threshold}_percent_reduction": sum(
            item["percentage_removed"] > threshold for item in summaries
        ) for threshold in (25, 50, 75)},
        "source_counters": dict(sorted(counters.items())),
        "highest_reductions": [item[0] for item in selected],
        "output_sha256": file_sha256(final_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    report = [
        f"SOURCE: {source}",
        "Highest percentage reductions; all documents shown for sources of 5 or fewer.",
        "Large losses are review flags, not quality filters. No documents are filtered.",
    ]
    for summary, original, cleaned, measured, diagnostics in selected:
        report.extend([
            "\n===== DOCUMENT =====", f"ID: {summary['id']}",
            f"Original chars: {summary['original_characters']}",
            f"Cleaned chars: {summary['cleaned_characters']}",
            f"Percentage removed: {summary['percentage_removed']:.2f}%",
            "Measured counters: " + json.dumps(measured, sort_keys=True),
            "Diagnostics: " + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
            "\n===== ORIGINAL (first 850 characters) =====\n", original[:850],
            "\n===== CLEANED (first 850 characters) =====\n", cleaned[:850],
        ])
        if source in {"gutenberg", "arxiv", "scanned_docs"}:
            report.extend(["\n===== CLEANED END (last 250 characters) =====\n",
                           cleaned[-250:]])
    atomic_write(data_root / "cleaned/comparisons" / f"{source}.txt",
                 "\n".join(report) + "\n")
    return manifest

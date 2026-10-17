"""Run source-specific cleaning and validate the entire immutable pilot."""

import json
import sys

# Avoid adding caches beside the protected successful pilot script.
sys.dont_write_bytecode = True

import clean_arxiv
import clean_code
import clean_fineweb
import clean_gutenberg
import clean_scanned_docs
import clean_stackexchange
import clean_wikipedia
from common import (
    DATA_ROOT, SOURCES, atomic_write, file_sha256, read_jsonl,
    snapshot_immutable, strict_json, validate_pair, write_json,
)


CLEANERS = (
    clean_wikipedia, clean_fineweb, clean_code, clean_stackexchange,
    clean_gutenberg, clean_arxiv, clean_scanned_docs,
)
LABELS = dict(zip(SOURCES, (
    "Wikipedia", "FineWeb-Edu", "Code", "StackExchange",
    "Gutenberg", "arXiv", "Scanned Docs",
)))


def validate_all(data_root=DATA_ROOT):
    input_ids, output_ids = set(), set()
    input_count = output_count = 0
    counts = {}
    wikipedia_sample = None
    for source in SOURCES:
        originals = list(read_jsonl(
            data_root / "standardized" / source / "documents.jsonl", source
        ))
        cleaned = list(read_jsonl(
            data_root / "cleaned" / source / "documents.jsonl", source
        ))
        if len(originals) != len(cleaned):
            raise ValueError(f"{source}: input/output count mismatch.")
        for original, output in zip(originals, cleaned):
            validate_pair(original, output, source)
            if original["id"] in input_ids or output["id"] in output_ids:
                raise ValueError(f"Duplicate corpus ID: {original['id']}")
            input_ids.add(original["id"])
            output_ids.add(output["id"])
            if output["id"] == "wikipedia:1644311:1352244912":
                wikipedia_sample = output
        counts[source] = {"input": len(originals), "output": len(cleaned)}
        input_count += len(originals)
        output_count += len(cleaned)
    if input_ids != output_ids:
        raise ValueError("Standardized and cleaned ID sets differ.")
    sample = strict_json(
        (data_root / "cleaned/wikipedia/sample_001_cleaned.json").read_text(
            encoding="utf-8"
        )
    )
    if wikipedia_sample is None or any(
        sample[key] != wikipedia_sample[key] for key in ("id", "source", "text")
    ):
        raise ValueError("Wikipedia cleaning regression failed.")
    for key, value in sample["metadata"].items():
        if key != "cleaning" and wikipedia_sample["metadata"].get(key) != value:
            raise ValueError(f"Wikipedia regression metadata changed: {key}")
    for key in ("stage", "original_characters", "cleaned_characters",
                "characters_removed"):
        if (sample["metadata"]["cleaning"][key]
                != wikipedia_sample["metadata"]["cleaning"][key]):
            raise ValueError(f"Wikipedia regression cleaning field changed: {key}")
    return {
        "input_documents": input_count,
        "output_documents": output_count,
        "unique_cleaned_ids": len(output_ids),
        "source_counts": counts,
        "id_set_matches_standardized": True,
        "all_ids_unique": True,
        "all_jsonl_valid": True,
        "all_schemas_valid": True,
        "existing_metadata_preserved": True,
        "no_empty_documents": True,
        "wikipedia_regression": True,
    }


def output_checksums(data_root=DATA_ROOT):
    return {
        source: file_sha256(
            data_root / "cleaned" / source / "documents.jsonl"
        ) for source in SOURCES
    }


def print_tree(root, prefix=""):
    children = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for index, child in enumerate(children):
        last = index == len(children) - 1
        print(prefix + ("└── " if last else "├── ") + child.name
              + ("/" if child.is_dir() else ""))
        if child.is_dir():
            print_tree(child, prefix + ("    " if last else "│   "))


def print_report(manifest):
    print("=" * 60)
    print("PILOT CLEANING — COMPLETE REPORT")
    print("=" * 60)
    for source in SOURCES:
        stats = manifest["sources"][source]
        print(f"\n{LABELS[source]}")
        for label, key in (
            ("Input", "input_documents"), ("Output", "output_documents"),
            ("Failures", "failures"), ("Original chars", "original_characters"),
            ("Cleaned chars", "cleaned_characters"),
        ):
            print(f"{label}: {stats[key]}")
        print(f"Removed: {stats['percentage_removed']:.2f}%")
        if source == "wikipedia":
            print("Regression test: PASS")
        if source == "scanned_docs":
            print("OCR executed by us: NO")
    print("\n" + "-" * 60)
    for label, key in (
        ("Total standardized docs", "total_input_documents"),
        ("Total cleaned docs", "total_output_documents"),
        ("Failures", "total_failures"),
        ("Original characters", "total_original_characters"),
        ("Cleaned characters", "total_cleaned_characters"),
        ("Characters removed", "total_removed_characters"),
    ):
        print(f"{label}: {manifest[key]}")
    print(f"Overall reduction: {manifest['overall_reduction_percentage']:.2f}%")
    print(f"\nUnique cleaned IDs: {manifest['unique_cleaned_ids']} / "
          f"{manifest['total_output_documents']}")
    for label, key in (
        ("ID set matches standardized corpus", "id_set_matches_standardized"),
        ("Wikipedia regression test", "wikipedia_regression"),
        ("Standardized inputs unchanged", "standardized_inputs_unchanged"),
        ("Raw files unchanged", "raw_files_unchanged"),
        ("Successful pilot artifacts unchanged", "protected_pilot_unchanged"),
        ("All JSONL valid", "all_jsonl_valid"),
        ("All schemas valid", "all_schemas_valid"),
        ("Repeat-run reproducibility", "repeat_run_reproducibility"),
    ):
        value = manifest["validation"][key]
        result = "NOT CHECKED (run again)" if value is None else (
            "PASS" if value else "FAIL"
        )
        print(f"{label}: {result}")
    print("\n" + "=" * 60)
    print("CLEANING: " + ("COMPLETE" if manifest["status"] == "complete"
                          else "REPEAT RUN PENDING"))
    print("=" * 60 + "\n")
    print(str(DATA_ROOT / "cleaned") + "/")
    print_tree(DATA_ROOT / "cleaned")
    print("\nCleaning code created (existing one-document script unchanged):")
    for name in ("common.py", *(f"clean_{s}.py" for s in SOURCES),
                 "run_all_cleaning.py"):
        print(DATA_ROOT / "clean_pilot" / name)


def main():
    before = snapshot_immutable()
    previous = (
        output_checksums() if all(
            (DATA_ROOT / "cleaned" / source / "documents.jsonl").is_file()
            for source in SOURCES
        ) else None
    )
    errors, manifests = [], {}
    try:
        for cleaner in CLEANERS:
            manifests[cleaner.SOURCE] = cleaner.run(errors)
        validation = validate_all()
        after = snapshot_immutable()
        changed = [name for name in before.keys() | after.keys()
                   if before.get(name) != after.get(name)]
        if changed:
            raise ValueError("Immutable files changed: " + ", ".join(sorted(changed)))
        current = output_checksums()
        repeat = None if previous is None else previous == current
        if repeat is False:
            raise ValueError("Repeat-run output JSONL files are not byte-identical.")
        old = sum(m["original_characters"] for m in manifests.values())
        new = sum(m["cleaned_characters"] for m in manifests.values())
        manifest = {
            "status": "complete" if repeat else "repeat_pending",
            "total_input_documents": validation["input_documents"],
            "total_output_documents": validation["output_documents"],
            "total_failures": 0,
            "total_original_characters": old,
            "total_cleaned_characters": new,
            "total_removed_characters": old - new,
            "overall_reduction_percentage": (old - new) / old * 100 if old else 0.0,
            "unique_cleaned_ids": validation["unique_cleaned_ids"],
            "cleaning_versions": {c.SOURCE: c.VERSION for c in CLEANERS},
            "sources": manifests,
            "validation": {
                **validation,
                "standardized_inputs_unchanged": True,
                "raw_files_unchanged": True,
                "protected_pilot_unchanged": True,
                "repeat_run_reproducibility": repeat,
            },
            "immutable_checksums_before": before,
            "immutable_checksums_after": after,
            "output_jsonl_checksums": current,
            "repeat_run_previous_jsonl_checksums": previous,
            "ocr_executed_by_us": False,
        }
        write_json(DATA_ROOT / "cleaned/pilot_manifest.json", manifest)
        print_report(manifest)
    except Exception as error:
        if not errors:
            errors.append({"source": "pipeline", "id": None,
                           "error_type": type(error).__name__, "message": str(error)})
        write_json(DATA_ROOT / "cleaned/pilot_manifest.json", {
            "status": "failed", "completed_sources": manifests, "errors": errors,
        })
        print(f"CLEANING: FAILED\n{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        atomic_write(DATA_ROOT / "cleaned/errors/errors.jsonl", "".join(
            json.dumps(error, ensure_ascii=False, allow_nan=False) + "\n"
            for error in errors
        ))


if __name__ == "__main__":
    main()

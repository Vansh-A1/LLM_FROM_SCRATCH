"""Standardize one Wikipedia revision while preserving its raw Wikitext."""

import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]


def select_document(batch):
    query = batch.get("query") if isinstance(batch, dict) else None
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, list):
        raise ValueError("Expected the inspected Wikipedia structure: query.pages is a list.")

    for page in pages:
        if not isinstance(page, dict) or page.get("pageid") is None:
            continue
        title = page.get("title")
        revisions = page.get("revisions")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(revisions, list):
            continue
        for revision in revisions:
            if not isinstance(revision, dict):
                continue
            slots = revision.get("slots")
            main = slots.get("main") if isinstance(slots, dict) else None
            text = main.get("content") if isinstance(main, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) <= 500:
                continue
            revision_id = revision.get("revid")
            if revision_id is None:
                raise ValueError("The first usable article has no revision ID.")
            return {
                "id": f"wikipedia:{page['pageid']}:{revision_id}",
                "source": "wikipedia",
                "text": text,
                "metadata": {
                    "title": title,
                    "page_id": page["pageid"],
                    "revision_id": revision_id,
                    "revision_timestamp": revision.get("timestamp"),
                    "language": "en",
                    "content_format": main.get("contentformat"),
                    "content_model": main.get("contentmodel"),
                },
            }
    raise ValueError("No article with a page ID, title, revision, and content over 500 characters.")


def main():
    raw_dir = DATA_DIR / "raw/wikipedia"
    input_path = raw_dir / "articles_000.json"
    if not input_path.exists():
        batches = sorted(raw_dir.glob("articles_*.json"))
        if not batches:
            raise FileNotFoundError("No raw Wikipedia article batch is available.")
        input_path = batches[0]

    raw_bytes = input_path.read_bytes()
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    document = select_document(json.loads(raw_bytes.decode("utf-8")))
    output_path = DATA_DIR / "standardized/wikipedia/sample_001.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    metadata = saved.get("metadata", {})
    checks = {
        "JSON valid": isinstance(saved, dict),
        "ID present": saved.get("id") == document["id"] and bool(saved.get("id")),
        "Source correct": saved.get("source") == "wikipedia",
        "Text present": isinstance(saved.get("text"), str) and bool(saved["text"]),
        "Metadata present": bool(metadata.get("title"))
        and metadata.get("page_id") is not None
        and metadata.get("revision_id") is not None,
        "Raw content preserved": saved == document,
        "Raw input unchanged": hashlib.sha256(input_path.read_bytes()).hexdigest() == raw_hash,
    }

    print("WIKIPEDIA STANDARDIZATION TEST\n")
    print(f"Input:\n{input_path.relative_to(DATA_DIR.parent)}\n")
    print(f"Selected article:\nTitle: {metadata.get('title')}")
    print(f"Page ID: {metadata.get('page_id')}\nRevision ID: {metadata.get('revision_id')}\n")
    print(f"Raw content characters: {len(saved.get('text', ''))}\n")
    print(f"Output:\n{output_path.relative_to(DATA_DIR.parent)}\n\nChecks:")
    for label, passed in checks.items():
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError("RESULT: FAIL")
    print("\nRESULT: PASS\n\nText preview (first 500 characters):")
    print(saved["text"][:500])


if __name__ == "__main__":
    main()

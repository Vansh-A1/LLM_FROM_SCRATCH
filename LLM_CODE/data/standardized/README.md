# Standardized pilot corpus

Each line in a source's `documents.jsonl` is one UTF-8 document with exactly four fields:

```json
{"id":"stable source-prefixed ID","source":"source name","text":"extracted content","metadata":{}}
```

Metadata identifies the original raw file, record index when applicable, extraction method, and useful source fields. Native IDs are used when available; missing native IDs use deterministic SHA256 IDs. Wikipedia IDs identify both the page and revision. JSON serialization preserves Unicode.

- Wikipedia preserves the MediaWiki revision's raw Wikitext.
- FineWeb, Code, and Stack Exchange preserve their actual JSONL `text` strings and non-text metadata. Code whitespace is unchanged.
- Gutenberg TXT preserves the UTF-8 file text and line endings. HTML extracts visible text while omitting structural tags, script, and CSS content. EPUB resolves its container and OPF package and reads HTML content in spine order, excluding navigation-only content.
- arXiv extracts the existing embedded PDF text in page order.
- Scanned documents extract only their existing archive OCR text layers. No OCR runs; essentially empty layers are logged for future OCR.

Text is **not cleaned**. Wikitext, Gutenberg headers and footers, boilerplate, references, and parser-produced PDF layout imperfections remain. Structural HTML block boundaries and PDF/EPUB document boundaries supply text separators; there is no literary-text or whitespace cleanup.

Raw files are read only. The runner hashes every raw input before and after extraction, checks unique IDs and every output line, compares the Wikipedia regression sample, and writes source manifests plus `pilot_manifest.json`. The existing `sample_001.json` and `standardize_wikipedia_one.py` remain unchanged and are excluded from corpus counts.

Run from the project root:

```bash
python data/standardize_pilot/run_all_standardization.py
```

The extractors use Python's standard library. PDFs use PyMuPDF if already installed; otherwise they use the existing local Poppler `pdftotext` utility, with `pdfinfo` checking page counts when available. No new dependencies were installed.

Wikipedia arrays, JSONL records, and output lines are processed incrementally. At most one document's text is assembled at a time; duplicate-ID checks use a temporary disk-backed SQLite database. Filenames and record/spine order are deterministic. Output documents are reproducible from unchanged inputs; manifest timestamps identify each run.

JSONL files are published atomically after successful source completion. Individual failures are recorded in `errors/errors.jsonl` and successful documents are retained. Five consecutive failures of the same kind abort that source and preserve any previously published JSONL, requiring diagnosis before rerunning. A failed run must not be used as a complete corpus. Generated JSONL and temporary output files are ignored by Git; code, this README, manifests, and the original sample can be versioned.

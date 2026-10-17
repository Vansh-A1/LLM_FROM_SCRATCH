# Cleaned pilot corpus

This layer is derived from the 410 standardized pilot documents. Raw and
standardized files are immutable checkpoints. IDs, sources, and all existing
metadata are preserved; only text and an added metadata.cleaning object differ.
The successful Edward Backhouse sample, its comparison, and its original
one-document cleaner are preserved unchanged.

## Source-specific policies

- Wikipedia: reuse the tested mwparserfromhell renderer. Remove templates,
  references, media/category metadata, and complete tables; retain prose,
  visible links, headings, and lists. Remove observed orphan formatting and
  table markers. Template/table-heavy articles can lose most of their source
  characters. They remain in the corpus and are flagged for review.
- FineWeb-Edu: remove invalid controls, trailing whitespace, excessive interior
  spacing, and excessive blank lines. Keep questionable navigation/boilerplate.
- Code: remove invalid controls only. Preserve Markdown, HTML/code examples,
  licenses, comments, indentation, tabs, spaces, line endings, and blank lines.
- Stack Exchange: the inspected records are already extracted text/Markdown.
  Preserve discussions, URLs, inline code, and fenced/indented blocks. The HTML
  snippet in the pilot is explicitly an explanatory example, so its markup is
  useful content and remains. Trim prose trailing spaces and excessive blank lines.
- Gutenberg: inspect all three books and keep the text between their unambiguous
  official START/END markers. Validate distribution metadata before trimming
  the header/footer. Preserve titles, authors, prefaces, publication-era content,
  chapter/contents structure, literary lines, and indentation.
- arXiv: remove invalid controls and excess blank/trailing whitespace. Retain
  citations, reference sections, tables, equations, line order, and interior
  spacing. Repeated lines are reported for inspection, never deleted by frequency.
  Page headers, broken columns, and hyphenation may remain.
- Scanned documents: the standardized text came from existing OCR layers.
  No OCR is run. Only conservative control/whitespace cleanup is applied.
  No spelling guesses, historical spelling corrections, or noise filtering.

These policies do not perform quality filtering, language filtering,
deduplication, train/validation splits, tokenizer work, tokenization, packing,
sharding, or training.

## Reproduce

Use Python 3 with mwparserfromhell 0.7.2, already installed for the successful
one-document test. No new dependencies are required by this full cleaning stage.
All other modules use Python's standard library.

From /data/vansh/LLM_FROM_SCRATCH/LLM_CODE, run
python data/clean_pilot/run_all_cleaning.py twice. The second run compares each
output JSONL checksum with the preceding run and reports byte-identical
reproducibility. The first run reports repeat-run verification as pending.

Each source writes documents.jsonl.tmp, validates every record, then atomically
promotes it to documents.jsonl. A failed source keeps its previous final output.
Failures are logged in errors/errors.jsonl and stop promotion; three repeated
document errors stop that source immediately.

Per-source manifests record character counts, loss thresholds, extrema, mean
length, measured removal counters, and the highest-reduction document IDs.
The global pilot_manifest.json records count/schema/ID integrity, Wikipedia
regression, checksums for all raw/standardized checkpoints and protected pilot
artifacts, and repeat-run verification. Checksums are for file integrity and
reproducibility; document text is not hashed or compared for deduplication.

## Review

comparisons/*.txt contain short before/after previews for the three largest
percentage losses per large source, and every document in the smaller sources.
They also include measured diagnostics and short endings for books/PDF-derived
texts. Loss thresholds and OCR noise examples are review signals, not deletion
rules. Empty cleaned text is an error, never a silent omission.

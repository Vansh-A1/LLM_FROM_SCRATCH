# Dataset Sources

This file records the origin of every raw dataset used in the LLM pretraining corpus.

Raw dataset files are stored under `data/raw/` and are not committed to Git.

The purpose of this file is to make dataset acquisition reproducible and clearly document where each source came from.

---

## FineWeb-Edu

Source:
Hugging Face FineWeb-Edu

Purpose:
General and educational web text.

Pilot target:
5,000 documents.

Raw storage:
`data/raw/fineweb/`

Expected raw format:
Parquet / Hugging Face dataset files.

Status:
Not downloaded yet.

---

## Wikipedia

Source:
Official Wikimedia / English Wikipedia.

Purpose:
Encyclopedic and factual text.

Pilot target:
2,000 articles.

Initial test:
Approximately 100 articles before downloading the full pilot sample.

Raw storage:
`data/raw/wikipedia/`

Expected full-scale raw format:
Compressed Wikimedia XML dump files such as `.xml.bz2`.

Status:
Not downloaded yet.

---

## Code

Source:
Stack V2 Edu or another approved openly licensed source-code corpus.

Purpose:
Programming language and source-code data.

Pilot target:
2,000 source files.

Raw storage:
`data/raw/code/`

Expected format:
Parquet or source dataset format.

Status:
Not downloaded yet.

---

## Stack Exchange

Source:
Stack Exchange dataset.

Purpose:
Question-answer and explanatory text.

Pilot target:
1,000 documents.

Raw storage:
`data/raw/stackexchange/`

Status:
Not downloaded yet.

---

## Project Gutenberg

Source:
Project Gutenberg.

Purpose:
Books and long-form text.

Pilot target:
20 books.

Raw storage:
`data/raw/gutenberg/`

Expected formats:
- TXT
- HTML
- EPUB

Status:
Not downloaded yet.

---

## arXiv

Source:
arXiv.

Purpose:
Scientific and technical papers.

Pilot target:
100 papers.

Raw storage:
`data/raw/arxiv/`

Expected format:
PDF.

Status:
Not downloaded yet.

---

## Scanned Documents

Source:
Public-domain scanned document collections.

Purpose:
Test ingestion of image-based documents and later OCR handling.

Pilot target:
Approximately 200 pages.

Raw storage:
`data/raw/scanned_docs/`

Expected format:
Scanned PDF.

Status:
Not downloaded yet.

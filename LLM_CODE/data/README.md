# Data storage

This directory currently contains only the acquisition/storage layer of the LLM data project. The same source directories can hold the small pilot corpus and, later, the full pretraining corpus.

- `raw/` is immutable. Never manually edit downloaded raw dataset files or modify them in place.
- Each data source has its own directory. Keep downloaded datasets exactly as received, in their original formats.
- Store metadata separately in the corresponding source directory under `metadata/`.
- `manifests/pilot.json` defines the small corpus currently being built. Its values are initial pilot targets and may change later.
- `manifests/sources.json` records what each source is and which raw formats are expected.
- Keep normal text-based arXiv PDFs in `raw/arxiv/` and image-based/scanned PDFs in `raw/scanned_docs/` for future OCR experiments.

Processing, cleaning, extraction, OCR, deduplication, tokenization, sharding, and training will be added later. Do not create their code, directories, or converted data as part of this setup.

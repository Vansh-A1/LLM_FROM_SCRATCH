# Pilot text ready for preprocessing

The input is the completed cleaned 410-document pilot. These are text files,
not token IDs. No tokenizer, embeddings, positional encodings, sequence packing,
labels, EOS/BOS tokens, or model were generated.

## Files to use

- train.txt: UTF-8 training text, 367 complete documents.
- validation.txt: held-out UTF-8 text, 43 documents.
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
checked before splitting; this pilot had 0.

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

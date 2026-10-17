# English pretraining text for the 125M model

Use `train.txt` for training and `validation.txt` only for held-out evaluation.
The export is complete only when `manifest.json` has `"status": "ready"`.
During acquisition, `progress.json` records committed preparation progress.

The completed selection contains:

| Split | Documents | UTF-8 bytes | Source-reported GPT-2 tokens before local cleanup |
| --- | ---: | ---: | ---: |
| Training | 2,961,683 | 13,921,892,743 | 3,000,000,844 |
| Validation | 28,324 | 131,673,680 | 28,225,002 |

These are plain text files, not token-ID files. The source counts are used only
to budget corpus size. Final readiness still requires the manifest status above.

Both outputs are strict UTF-8 plain text. Whole documents are separated by two
newline characters. There are no injected source labels, token IDs, BOS/EOS
markers, embeddings, or positional encodings. Blank lines can occur within a
document; exact document boundaries and original URLs are recorded in
`train.documents.jsonl` and `validation.documents.jsonl` using exclusive byte
offsets. Read the text in chunks when using a large corpus.

## Size and intended use

The target is at least **3 billion source-reported GPT-2 tokens in the retained
training documents**, plus at least 10 million in validation. This provides
margin over the approximately 2.5-billion-token planning reference for a 125M
model. The final measured document counts and byte sizes are in `manifest.json`.

**No local tokenization is performed.** The size budget sums the publisher's
existing `token_count` metadata before local whitespace/boilerplate cleanup;
it is an estimate of final token volume, not a measured count of the cleaned
text. The manifest separately records original and cleaned character totals.
Model performance depends on training and evaluation; the corpus size alone
does not guarantee a particular capability.

This is an English educational/general-web corpus, not an instruction/chat
dataset or a dedicated source-code training mixture. Preparation writes only
inside `actual_data`. Earlier pilot data is unchanged, and these scripts do not
rewrite existing model/pipeline files.

## Source and attribution

Source: [HuggingFaceFW/FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
configuration `sample-10BT`, pinned revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

Credit: Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, Thomas Wolf,
and the FineWeb dataset contributors. See the saved original dataset card in
`source_docs/FineWeb-Edu.dataset-card.md` and the
[FineWeb research paper](https://arxiv.org/abs/2406.17557).

The publisher identifies the dataset license as
[Open Data Commons Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/)
and also references [Common Crawl's terms](https://commoncrawl.org/terms-of-use).
Individual source webpages retain their applicable rights; this is not a claim
that all underlying articles are public domain. Original URLs, crawl identifiers,
source rows and file checksums are retained for provenance.

## Preparation

- Verify downloaded Parquet files against the publisher's SHA-256 and size.
- Keep English documents with language confidence at least 0.8 and the source's
  rounded educational score at least 3.
- Normalize line endings and Unicode NFC; remove invalid control characters,
  byte-order marks and zero-width spaces; normalize horizontal whitespace.
- Remove a small documented set of exact standalone cookie/navigation/ad lines.
  Preserve paragraph breaks, facts, punctuation and document bodies.
- Reject empty/short documents, unextracted HTML, obvious error pages, excessive
  repeated lines/characters, severe encoding noise, overlong documents, or
  documents dominated by removed boilerplate. Do not truncate long articles.
- Reject literal GPT-2 end-of-text markers so source text cannot inject them.
- Deduplicate globally using whitespace-normalized content fingerprints and
  canonical URLs, including repeat crawls of the same URL.
- Assign every normalized website hostname deterministically to one split;
  approximately 1% of hostname hash buckets are held out for validation.
  Entire documents remain in one split. Exact/normalized duplicates cannot
  cross splits. This does not claim exhaustive semantic or fuzzy deduplication.

Source files are immutable under `raw/`. Resumable build state and the global
deduplication database live under `work/`. Large datasets are excluded from Git
by the local `.gitignore`.

## Verify or reproduce

The only helper scripts are for this acquisition/cleaning task:

    python3 -B download_sources.py --count 5 --workers 2
    python3 -B prepare_corpus.py
    python3 -B prepare_corpus.py --verify

The acquisition plan lists all source shards. If filtering requires more than
five shards, increase `--count`; existing valid downloads are verified and reused.
Preparation resumes committed state. After completion, rerunning preparation
verifies the existing export instead of rebuilding it.

Verification scans every document, validates UTF-8, offsets and checksums, checks
deduplication records and split assignment, and confirms the size budget. Existing
project/data checksums are stored in `protected_checksums.json` for preservation
verification. No model, training, tokenizer or embedding code is created here.

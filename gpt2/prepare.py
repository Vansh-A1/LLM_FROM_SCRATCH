#!/usr/bin/env python3
# Portions adapted from nanoGPT, MIT License.
# Copyright (c) 2022 Andrej Karpathy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Prepare the *full* nanoGPT OpenWebText corpus, never a demonstration dataset.

Reference: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
The old unnamespaced ``openwebtext`` dataset is now Skylion007/openwebtext.
Its original publisher migrated the source to Parquet; the pinned official
revision below works with datasets 4.x without executing remote Python code.
No alternative corpus or tokenizer is used. Document order, train_test_split,
encode_ordinary, and appended EOT follow nanoGPT. Execution-order optimization:
tokenize the source sequentially BEFORE applying the reference random split.
The stateless per-document tokenizer preserves row count/order, so this commutes
with the reference split and produces identical final binary bytes. Both split
index hashes are checked against a reference split of the original text rows on
every full preparation. This avoids random reads across ~40 GB of source text. Before randomized final
concatenation, the token cache is sequentially read into the OS page cache only
when Linux MemAvailable can fit it with an 8 GiB margin. This read-only I/O
optimization changes no tokens or split order; --no-cache-warm disables it.
Arrow's intermediate IDs are
stored as uint16 to reduce disk usage; this does not change any token IDs.

    python -m pip install 'numpy>=1.26' 'datasets==4.8.4' 'tiktoken==0.14.0' tqdm
    python prepare.py --check-only
    python prepare.py
    python prepare.py --verify-only
    python prepare.py --check-tokenization-order

Allow about 120 GiB of free space when cache and output share a filesystem.
This includes the ~24 GB download, ~40 GB Arrow text, tokenized cache, binaries,
and working headroom. On resume, allocated bytes in this pinned corpus's cache
are credited once, retaining at least 16 GiB cache working space plus 24 GiB
for output. Unrelated caches, sparse holes and symlink duplicates earn no credit.
Preparation can take hours. --check-only never downloads.
All writable Hugging Face/tokenizer caches live under --cache-dir. A complete
manifest and both fully scanned binaries are published by one directory rename.
Training must require manifest.json with complete=true and matching file sizes.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import sys
import tempfile
import uuid

DATASET = "Skylion007/openwebtext"
# The original publisher's complete, script-free corpus; never follow moving main.
DATASET_REVISION = "b4325f019c648b1641a1784748667e8b74e5e064"
REFERENCE = "https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py"
VOCAB_SIZE, EOT, SPLIT_SEED, VALIDATION_FRACTION = 50257, 50256, 2357, 0.0005
EXPECTED_DOCUMENTS = {"train": 8009762, "val": 4007}
REFERENCE_TOKENS = {"train": 9035582198, "val": 4434897}
GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parent
# Capture executing source at process startup; later edits cannot relabel a run.
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ENC = None


def tokenize_document(example):
    """Exactly nanoGPT's tokenizer operation, with one EOT per document."""
    global ENC
    if ENC is None:
        import tiktoken
        ENC = tiktoken.get_encoding("gpt2")
        if ENC.n_vocab != VOCAB_SIZE or ENC.eot_token != EOT:
            raise ValueError("Unexpected GPT-2 tokenizer vocabulary or EOT")
    ids = ENC.encode_ordinary(example["text"])
    ids.append(ENC.eot_token)
    return {"ids": ids, "len": len(ids)}


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "datasets", "tiktoken", "pyarrow", "huggingface_hub", "tqdm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def existing_parent(path):
    while not path.exists():
        path = path.parent
    return path


def known_cache_allocation(cache_dir, device):
    """Credit only allocated blocks belonging to the pinned source/cache revision.

    Hub snapshots contain symlinks into blobs: follow only the exact 80 expected
    source files, require targets inside a known Hub blob store, and count
    each (device,inode) once. For Arrow caches, visit only this exact dataset,
    configuration and revision (including its interrupted builder directory),
    never following symlink directories/files. Sparse holes are not allocated
    disk space. This also works with a shared --cache-dir without crediting other
    corpora or other OpenWebText revisions.
    """
    cache_dir = cache_dir.resolve()
    seen, allocated, files = set(), 0, 0

    def credit(path, allowed_root, allow_symlink=False):
        nonlocal allocated, files
        try:
            if path.is_symlink() and not allow_symlink:
                return
            target = path.resolve(strict=True)
            if not target.is_relative_to(allowed_root) or not target.is_relative_to(cache_dir):
                return
            info = target.stat()
            identity = (info.st_dev, info.st_ino)
            if not stat.S_ISREG(info.st_mode) or info.st_dev != device or identity in seen:
                return
            seen.add(identity)
            allocated += info.st_blocks * 512
            files += 1
        except (FileNotFoundError, OSError):
            # A concurrently writing preparation may rename a cache file.
            return

    hub_repo = cache_dir / "hub" / "datasets--Skylion007--openwebtext"
    snapshot = hub_repo / "snapshots" / DATASET_REVISION
    # Hub 1.x uses per-repository blobs; Hub 2.x also uses a global,
    # content-addressed blobs/<prefix>/<hash> store. Only blobs referenced by
    # THIS exact snapshot's expected source filenames qualify in either store.
    blob_roots = ((hub_repo / "blobs").resolve(),
                  (cache_dir / "hub" / "blobs").resolve())
    for index in range(80):
        path = snapshot / "plain_text" / f"train-{index:05d}-of-00080.parquet"
        if path.is_symlink():
            target = path.resolve()
            permitted = next((root for root in blob_roots
                              if root.is_relative_to(cache_dir) and target.is_relative_to(root)), None)
            if permitted is not None:
                credit(path, permitted, allow_symlink=True)
        else:
            # Non-symlink snapshot copies are real additional storage too.
            credit(path, snapshot.resolve())

    arrow_parent = (cache_dir / "datasets" / "Skylion007___openwebtext" /
                    "plain_text" / "0.0.0")
    roots = []
    for suffix in ("", ".incomplete"):
        directory = arrow_parent / (DATASET_REVISION + suffix)
        if not directory.is_dir() or directory.is_symlink():
            continue
        allowed = directory.resolve()
        if not allowed.is_relative_to(cache_dir):
            continue
        roots.append(str(directory))
        for root, directories, filenames in os.walk(directory, followlinks=False):
            directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
            for name in filenames:
                if name.endswith(".arrow") or name in ("dataset_info.json", "state.json"):
                    credit(Path(root) / name, allowed)
    return {"allocated_bytes": allocated, "files_counted": files,
            "source_snapshot": str(snapshot), "arrow_directories": roots,
            "policy": "pinned source/revision only; st_blocks*512; inode deduplication; same filesystem"}


def check_disk(cache_dir, data_dir):
    """Aggregate incremental requirements per filesystem with safe cache credit."""
    cache_device = existing_parent(cache_dir).stat().st_dev
    credit = known_cache_allocation(cache_dir, cache_device)
    fresh_cache_budget, working_margin = 96 * GIB, 16 * GIB
    cache_need = max(working_margin, fresh_cache_budget - credit["allocated_bytes"])
    requirements = {}
    # Cache credit never removes the 16 GiB working margin. Output always keeps
    # 24 GiB for final binaries and atomic publication, even if files exist.
    for label, path, need in (("cache", cache_dir, cache_need),
                              ("output", data_dir, 24 * GIB)):
        parent = existing_parent(path)
        key = parent.stat().st_dev
        entry = requirements.setdefault(key, {"paths": [], "required_bytes": 0,
                                               "free_bytes": shutil.disk_usage(parent).free})
        entry["paths"].append({"role": label, "path": str(path), "required_bytes": need})
        entry["required_bytes"] += need
        if label == "cache":
            entry["cache_fresh_budget_bytes"] = fresh_cache_budget
            entry["cache_working_margin_bytes"] = working_margin
            entry["cache_credit"] = credit
    result = list(requirements.values())
    for entry in result:
        entry["enough_space"] = entry["free_bytes"] >= entry["required_bytes"]
    return result


def preflight(args):
    installed = versions()
    missing = [name for name, version in installed.items() if version is None]
    disk = check_disk(args.cache_dir, args.data_dir)
    errors = []
    if missing:
        errors.append("Missing dependencies: " + ", ".join(missing))
    if installed["datasets"] and int(installed["datasets"].split(".")[0]) < 3:
        errors.append("Use datasets>=3 (tested with 4.8.4) for the pinned Parquet source")
    if not all(entry["enough_space"] for entry in disk):
        errors.append("Insufficient free disk for the incremental requirements shown above "
                      "(pinned-cache credit applied; at least 16 GiB cache + 24 GiB output retained). "
                      "Use --cache-dir/--data-dir on a larger filesystem")
    return {"dataset": DATASET, "revision": DATASET_REVISION,
            "expected_documents": sum(EXPECTED_DOCUMENTS.values()),
            "versions": installed, "disk": disk, "errors": errors,
            "ready_for_preparation": not errors,
            "download_performed": False}


def configure_caches(cache_dir):
    # Set before importing datasets/huggingface_hub; an explicit local cache keeps
    # a large corpus out of an unexpectedly small $HOME filesystem.
    for variable, subdirectory in (("HF_HUB_CACHE", "hub"),
                                   ("HF_DATASETS_CACHE", "datasets"),
                                   ("HF_XET_CACHE", "xet"),
                                   ("HF_MODULES_CACHE", "modules"),
                                   ("TIKTOKEN_CACHE_DIR", "tiktoken")):
        path = cache_dir / subdirectory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    tmp = cache_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = str(tmp)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def preparation_lock(data_dir):
    # flock is released by the OS even after a crash; the empty lock file stays.
    import fcntl
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    path = data_dir.parent / ("." + data_dir.name + ".prepare.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation is active for {data_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def scan_binary(path, documents=None):
    """Scan *every* token and byte in bounded chunks, not just a sample."""
    import numpy as np
    size = path.stat().st_size
    if size <= 0 or size % 2:
        raise ValueError(f"Invalid uint16 binary size: {path}: {size}")
    digest, count, eot_count = hashlib.sha256(), 0, 0
    minimum, maximum = VOCAB_SIZE, -1
    with path.open("rb") as handle:
        while raw := handle.read(16 * 1024 * 1024):
            digest.update(raw)
            ids = np.frombuffer(raw, dtype="<u2")
            minimum = min(minimum, int(ids.min()))
            maximum = max(maximum, int(ids.max()))
            if maximum >= VOCAB_SIZE:
                raise ValueError(f"Out-of-vocabulary token {maximum} in {path}")
            eot_count += int(np.count_nonzero(ids == EOT))
            count += ids.size
    if count != size // 2:
        raise ValueError(f"Token count/byte count mismatch in {path}")
    if documents is not None and eot_count != documents:
        raise ValueError(f"{path}: {eot_count} EOTs != {documents} documents")
    return {"filename": path.name, "tokens": int(count), "bytes": size,
            "sha256": digest.hexdigest(), "min_token": minimum,
            "max_token": maximum, "eot_count": eot_count}


def verify_dataset(data_dir):
    path = data_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No completed dataset manifest: {path}. Run prepare.py first.")
    manifest = json.loads(path.read_text())
    expected = {"schema_version": 1, "complete": True, "dataset": DATASET,
                "tokenizer": "gpt2", "vocab_size": VOCAB_SIZE,
                "dtype": "uint16", "byte_order": "little", "eot_token": EOT}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {key}: expected {value!r}, got {manifest.get(key)!r}")
    if manifest.get("split") != {"validation_fraction": VALIDATION_FRACTION,
                                 "seed": SPLIT_SEED, "shuffle": True}:
        raise ValueError("Dataset split configuration does not match nanoGPT")
    if manifest.get("source", {}).get("revision") != DATASET_REVISION:
        raise ValueError("Dataset source revision does not match this preparation script")
    hashes = []
    for split, docs in EXPECTED_DOCUMENTS.items():
        recorded = manifest["files"][split]
        if recorded["filename"] != f"{split}.bin" or recorded["documents"] != docs:
            raise ValueError(f"Invalid {split} file metadata")
        actual = scan_binary(data_dir / f"{split}.bin", documents=docs)
        for key, value in actual.items():
            if recorded.get(key) != value:
                raise ValueError(f"{split} {key} mismatch: {recorded.get(key)!r} != {value!r}")
        check_token_count(split, actual["tokens"])
        hashes.append(actual["sha256"])
        print(f"Verified {split}: {actual['tokens']:,} tokens, sha256={actual['sha256']}", flush=True)
    if hashes[0] == hashes[1]:
        raise ValueError("Training and validation binaries are identical")
    return manifest


def check_token_count(split, count):
    reference = REFERENCE_TOKENS[split]
    tolerance = 0.02 if split == "train" else 0.25
    if abs(count / reference - 1.0) > tolerance:
        raise ValueError(f"Implausible {split} token count {count:,}; nanoGPT reports {reference:,}. "
                         "Investigate source/order/tokenizer before training.")
    if count != reference:
        print(f"NOTE: {split} has {count:,} tokens; historical nanoGPT reference: {reference:,}. "
              "The official source revision and exact split-index hashes are recorded.", flush=True)


def validate_split_indices(split_dataset):
    """Check source-row disjointness and completeness, including duplicate indices."""
    import numpy as np
    total = sum(EXPECTED_DOCUMENTS.values())
    seen = np.zeros(total, dtype=np.bool_)
    metadata = {}
    for split, docs in EXPECTED_DOCUMENTS.items():
        part = split_dataset[split]
        if len(part) != docs or part._indices is None:
            raise ValueError(f"Unexpected {split} split length or missing index mapping")
        indices = np.asarray(part._indices.column(0).to_numpy(), dtype=np.int64)
        if len(indices) != docs or int(indices.min()) < 0 or int(indices.max()) >= total:
            raise ValueError("Invalid document indices")
        if np.unique(indices).size != docs or bool(seen[indices].any()):
            raise ValueError("Duplicate or overlapping train/validation document indices")
        seen[indices] = True
        metadata[split] = {"documents": docs,
                           "source_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                           "dataset_fingerprint": part._fingerprint}
    if not bool(seen.all()):
        raise ValueError("Some source documents are absent from the split")
    return metadata


def check_tokenization_order():
    """Prove commutation on an in-memory fixture; never write production data."""
    import numpy as np
    from datasets import Dataset, Features, Sequence, Value
    documents = [f"Document {i}: plain text and literal <|endoftext|>.\n" for i in range(10000)]
    source = Dataset.from_dict({"text": documents, "source_index": list(range(len(documents)))})
    features = Features({"source_index": Value("int64"), "ids": Sequence(Value("uint16")),
                         "len": Value("int64")})
    options = dict(test_size=VALIDATION_FRACTION, seed=SPLIT_SEED, shuffle=True, keep_in_memory=True)
    reference = source.train_test_split(**options).map(
        tokenize_document, remove_columns=["text"], features=features, keep_in_memory=True)
    sequential = source.map(tokenize_document, remove_columns=["text"], features=features,
                            keep_in_memory=True).train_test_split(**options)
    report = {}
    for split in ("train", "test"):
        left = np.concatenate([np.asarray(ids, dtype="<u2") for ids in reference[split]["ids"]])
        right = np.concatenate([np.asarray(ids, dtype="<u2") for ids in sequential[split]["ids"]])
        left_indices = np.asarray(reference[split]["source_index"], dtype="<i8")
        right_indices = np.asarray(sequential[split]["source_index"], dtype="<i8")
        if not np.array_equal(left, right) or not np.array_equal(left_indices, right_indices):
            raise AssertionError("Tokenize/split execution order changed tokens or original row order")
        if int(np.count_nonzero(left == EOT)) != len(reference[split]):
            raise AssertionError("Expected exactly one appended EOT per document")
        report["val" if split == "test" else split] = {
            "documents": len(reference[split]), "tokens": int(left.size),
            "token_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
            "source_indices_sha256": hashlib.sha256(left_indices.tobytes()).hexdigest()}
    print(json.dumps({"check": "tokenize-before-split equals split-before-tokenize", "passed": True,
                      "fixture": "synthetic, in-memory only; no production dataset written",
                      "split": report}, indent=2))


def available_memory_bytes():
    """Read Linux's reclaim-aware memory estimate; skip warmup on other hosts."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "MemAvailable:" and len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def warm_token_cache(cache_files, enabled=True):
    """Read source token Arrow files sequentially, with an explicit RAM guard.

    The final reference split visits documents in random order. If the complete
    token cache fits comfortably in available RAM, these bounded sequential
    reads let subsequent random accesses use the OS page cache. No file content,
    tokens, document order or split indices are modified. The ordinary writer is
    always the same, including when this optional optimization is skipped.
    """
    report = {"status": "skipped", "bytes_read": 0, "file_count": 0,
              "margin_bytes": 8 * GIB}

    def skip(reason):
        report["reason"] = reason
        print(f"Token cache warmup skipped: {reason}", flush=True)
        return report

    if not enabled:
        return skip("disabled by --no-cache-warm")
    available = available_memory_bytes()
    report["available_memory_bytes"] = available
    if available is None:
        return skip("Linux MemAvailable is unavailable; no portable RAM assumption made")
    files, seen = [], set()
    try:
        for entry in cache_files:
            path = Path(entry["filename"])
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                return skip(f"token cache entry is not a regular file: {path}")
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                seen.add(identity)
                files.append((path, info.st_size))
    except (OSError, KeyError, TypeError) as exc:
        return skip(f"could not inspect token cache: {exc}")
    total = sum(size for _, size in files)
    report.update(file_count=len(files), cache_bytes=total,
                  required_memory_bytes=total + report["margin_bytes"])
    if not files or total == 0:
        return skip("no materialized token cache files")
    if available < report["required_memory_bytes"]:
        return skip(f"need {report['required_memory_bytes'] / GIB:.2f} GiB available "
                    f"including margin; have {available / GIB:.2f} GiB")
    print(f"Warming {len(files)} token-cache files ({total / GIB:.2f} GiB) by read-only "
          f"sequential I/O; available RAM {available / GIB:.2f} GiB, margin 8 GiB.", flush=True)
    buffer = bytearray(8 * 1024 * 1024)
    try:
        for index, (path, _) in enumerate(files, start=1):
            with path.open("rb", buffering=0) as handle:
                while count := handle.readinto(buffer):
                    report["bytes_read"] += count
            print(f"Token cache warmup: {index}/{len(files)} files, "
                  f"{report['bytes_read'] / GIB:.2f}/{total / GIB:.2f} GiB read.", flush=True)
    except OSError as exc:
        return skip(f"optional sequential read failed after {report['bytes_read']} bytes: {exc}")
    if report["bytes_read"] != total:
        return skip("token cache size changed while warming; ordinary writer will verify output")
    report["status"] = "completed"
    print("Token cache warmup complete; token bytes and split order are unchanged.", flush=True)
    return report


def write_split(dset, path, split):
    import numpy as np
    from tqdm.auto import tqdm
    tokens = sum(int(np.sum(batch["len"], dtype=np.uint64))
                 for batch in dset.select_columns(["len"]).iter(batch_size=65536))
    check_token_count(split, tokens)
    if shutil.disk_usage(path.parent).free < tokens * 2 + 2 * GIB:
        raise OSError(f"Not enough free space to write {path}: need {tokens * 2:,} bytes + headroom")
    arr = np.memmap(path, dtype="<u2", mode="w+", shape=(tokens,))
    cursor = 0
    # nanoGPT uses 1024 contiguous shards. Smaller inner batches bound RAM
    # without changing concatenation order or document boundaries.
    shards = min(1024, len(dset))
    try:
        for index in tqdm(range(shards), desc=f"writing {split}.bin"):
            shard = dset.shard(num_shards=shards, index=index, contiguous=True)
            for batch in shard.with_format("numpy").iter(batch_size=256):
                ids = np.concatenate(batch["ids"])
                if ids.ndim != 1 or ids.size == 0 or ids.min() < 0 or ids.max() >= VOCAB_SIZE:
                    raise ValueError("Invalid token IDs before uint16 conversion")
                if cursor + len(ids) > tokens:
                    raise ValueError("More tokens than lengths advertise")
                arr[cursor:cursor + len(ids)] = ids
                cursor += len(ids)
        if cursor != tokens:
            raise ValueError(f"Wrote {cursor} tokens but expected {tokens}")
        arr.flush()
    finally:
        del arr
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    info = scan_binary(path, documents=len(dset))
    info["documents"] = len(dset)
    if info["tokens"] != tokens:
        raise ValueError("Written file length changed during verification")
    return info


def prepare(args):
    if args.data_dir.is_symlink():
        raise ValueError("Refusing symlink dataset destination")
    if (args.data_dir / "manifest.json").exists():
        print("Dataset already exists; verifying it instead of overwriting.", flush=True)
        verify_dataset(args.data_dir)
        return
    report = preflight(args)
    print(json.dumps(report, indent=2), flush=True)
    if report["errors"]:
        raise RuntimeError("Preparation preflight failed: " + "; ".join(report["errors"]))
    configure_caches(args.cache_dir)
    from datasets import Features, Sequence, Value, load_dataset
    from huggingface_hub import HfApi

    with preparation_lock(args.data_dir):
        if args.data_dir.exists() and any(args.data_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty {args.data_dir}. "
                               "Use --verify-only for a completed dataset.")
        print(f"Loading complete {DATASET}@{DATASET_REVISION}; all 80 source shards are required.", flush=True)
        # Pin and validate the upstream snapshot before corpus download.
        source_info = HfApi().dataset_info(DATASET, revision=DATASET_REVISION, files_metadata=True)
        source_files = sorted(item.rfilename for item in source_info.siblings
                              if item.rfilename.endswith(".parquet"))
        expected_files = [f"plain_text/train-{i:05d}-of-00080.parquet" for i in range(80)]
        if source_info.sha != DATASET_REVISION or source_files != expected_files:
            raise ValueError("Pinned OpenWebText source does not contain the expected 80 shards")
        dataset = load_dataset(DATASET, "plain_text", revision=DATASET_REVISION,
                               cache_dir=str(args.cache_dir / "datasets"),
                               num_proc=args.num_proc, split="train")
        if len(dataset) != sum(EXPECTED_DOCUMENTS.values()) or dataset.column_names != ["text"]:
            raise ValueError(f"Unexpected full corpus: {len(dataset):,} rows, columns={dataset.column_names}")
        # Preserve the exact original source-row split as an independent oracle.
        reference_split = dataset.train_test_split(test_size=VALIDATION_FRACTION,
                                                  seed=SPLIT_SEED, shuffle=True)
        reference_split["val"] = reference_split.pop("test")
        reference_index_info = validate_split_indices(reference_split)
        # encode_ordinary(text)+EOT is stateless and preserves row count/order:
        # map(source)[split_indices] == map(source[split_indices]). Sequential
        # source reads avoid random I/O over the ~40 GB text dataset. This only
        # changes execution order, never the tokenization or resulting binaries.
        all_tokenized = dataset.map(tokenize_document, remove_columns=["text"],
                                    features=Features({"ids": Sequence(Value("uint16")),
                                                       "len": Value("int64")}),
                                    desc="tokenizing OpenWebText sequentially", num_proc=args.num_proc)
        tokenized = all_tokenized.train_test_split(test_size=VALIDATION_FRACTION,
                                                  seed=SPLIT_SEED, shuffle=True)
        tokenized["val"] = tokenized.pop("test")
        index_info = validate_split_indices(tokenized)
        for split in ("train", "val"):
            if index_info[split]["source_indices_sha256"] != reference_index_info[split]["source_indices_sha256"]:
                raise ValueError("Sequential-tokenization split differs from original reference row indices")
            index_info[split]["reference_text_split_fingerprint"] = reference_index_info[split]["dataset_fingerprint"]
        print("Verified identical original and tokenized split-index hashes for train and val.", flush=True)
        cache_warming = warm_token_cache(all_tokenized.cache_files, enabled=not args.no_cache_warm)
        stage = args.data_dir.parent / f".{args.data_dir.name}.incomplete-{uuid.uuid4().hex}"
        stage.mkdir()
        published = False
        try:
            files = {split: write_split(tokenized[split], stage / f"{split}.bin", split)
                     for split in ("train", "val")}
            if files["train"]["sha256"] == files["val"]["sha256"]:
                raise ValueError("Training and validation are identical")
            source_lfs = {}
            for item in source_info.siblings:
                if item.rfilename in source_files:
                    lfs = item.lfs
                    source_lfs[item.rfilename] = {"size": item.size,
                        "sha256": getattr(lfs, "sha256", None) if lfs else None}
            manifest = {"schema_version": 1, "complete": True,
                        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "dataset": DATASET, "tokenizer": "gpt2", "vocab_size": VOCAB_SIZE,
                        "dtype": "uint16", "byte_order": "little", "eot_token": EOT,
                        "tokenization": "tiktoken.encode_ordinary(text), append EOT per document",
                        "split": {"validation_fraction": VALIDATION_FRACTION,
                                  "seed": SPLIT_SEED, "shuffle": True},
                        "split_verification": index_info, "files": files,
                        "execution_order": "sequential tokenization before identical reference split",
                        "split_index_equivalence_verified": True,
                        "cache_warming": cache_warming,
                        "source": {"repository": DATASET, "revision": DATASET_REVISION,
                                   "format": "official Parquet migration", "files": source_lfs,
                                   "dataset_fingerprint": dataset._fingerprint},
                        "versions": versions(), "reference": REFERENCE,
                        "prepare_source_sha256": SOURCE_SHA256}
            with (stage / "manifest.json").open("w") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            fsync_directory(stage)
            # POSIX atomically replaces an existing EMPTY destination directory.
            # A trainer can see either the old empty dir or all verified outputs.
            os.replace(stage, args.data_dir)
            published = True
            fsync_directory(args.data_dir.parent)
            print(f"COMPLETE: published verified data to {args.data_dir}", flush=True)
            for split, info in files.items():
                print(f"{split}: {info['documents']:,} documents; {info['tokens']:,} tokens; "
                      f"sha256={info['sha256']}", flush=True)
        except BaseException:
            if published:
                print(f"Dataset was published to {args.data_dir}, but final durability/logging failed; "
                      "run --verify-only before training.", file=sys.stderr)
            else:
                print(f"Preparation did not complete. Unpublished staging files, if any: {stage}. "
                      "No completion manifest was published to the training directory.", file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "openwebtext")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "openwebtext")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--no-cache-warm", action="store_true",
                        help="disable optional read-only token cache prewarming before final concatenation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="check dependencies/disk, no corpus download")
    mode.add_argument("--verify-only", action="store_true", help="scan all completed tokens and verify SHA-256")
    mode.add_argument("--check-tokenization-order", action="store_true",
                      help="prove byte/index equivalence on an in-memory mathematical fixture")
    args = parser.parse_args()
    args.data_dir, args.cache_dir = args.data_dir.resolve(), args.cache_dir.resolve()
    if args.num_proc < 1:
        parser.error("--num-proc must be positive")
    if args.data_dir == args.cache_dir or args.data_dir in args.cache_dir.parents:
        parser.error("--cache-dir must not be inside --data-dir (publication must be atomic)")
    if args.check_only:
        report = preflight(args)
        print(json.dumps(report, indent=2))
        return 0 if report["ready_for_preparation"] else 1
    try:
        if args.check_tokenization_order:
            check_tokenization_order()
        elif args.verify_only:
            verify_dataset(args.data_dir)
        else:
            prepare(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

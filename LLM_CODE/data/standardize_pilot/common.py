"""Shared streaming I/O, validation, provenance, and extraction utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]
SOURCES = ('wikipedia', 'fineweb', 'code', 'stackexchange', 'gutenberg', 'arxiv', 'scanned_docs')


class ExtractionError(Exception):
    def __init__(self, error_type, message, *, requires_future_ocr=False):
        super().__init__(message)
        self.error_type = error_type
        self.requires_future_ocr = requires_future_ocr


class SourceAborted(Exception):
    def __init__(self, message, statistics):
        super().__init__(message)
        self.statistics = statistics


@dataclass
class RawRecord:
    path: Path
    index: int | None
    value: object = None
    error: ExtractionError | None = None


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path, data_dir):
    return Path(path).relative_to(data_dir).as_posix()


def reject_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def json_loads(value):
    return json.loads(value, parse_constant=reject_constant)


def serialize(value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    try:
        encoded.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ExtractionError('invalid_unicode', 'Record contains an unpaired Unicode surrogate and cannot be serialized as UTF-8') from exc
    return encoded


@contextmanager
def atomic_text_writer(path):
    """Publish only a fully written file, keeping an existing final file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_text_writer(path) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


class StreamingJSON:
    """Read one JSON value at a time; never materialize the whole pages array."""
    def __init__(self, handle, chunk_size=65536):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ''
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder(parse_constant=reject_constant)

    def fill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.handle.read(self.chunk_size)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ''
            self.fill()

    def consume(self, expected):
        if self.peek() != expected:
            raise ValueError(f'Expected JSON {expected!r}, found {self.peek()!r}')
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()
                continue
            # A number at a chunk boundary may not be complete yet.
            if end == len(self.buffer) and not self.eof:
                self.fill()
                continue
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and end < len(self.buffer) and self.buffer[end] in 'eE.+-0123456789'):
                if self.eof:
                    raise ValueError('Invalid JSON number')
                self.fill()
                continue
            self.position = end
            return value

    def skip_value(self):
        token = self.peek()
        if token == '{':
            self.consume('{')
            if self.peek() != '}':
                while True:
                    if not isinstance(self.value(), str):
                        raise ValueError('JSON object key is not a string')
                    self.consume(':')
                    self.skip_value()
                    if self.peek() != ',':
                        break
                    self.consume(',')
            self.consume('}')
        elif token == '[':
            self.consume('[')
            if self.peek() != ']':
                while True:
                    self.skip_value()
                    if self.peek() != ',':
                        break
                    self.consume(',')
            self.consume(']')
        else:
            self.value()

    def member(self, wanted):
        self.consume('{')
        while self.peek() != '}':
            key = self.value()
            if not isinstance(key, str):
                raise ValueError('JSON object key is not a string')
            self.consume(':')
            if key == wanted:
                return
            self.skip_value()
            if self.peek() != ',':
                break
            self.consume(',')
        raise ValueError(f'Missing JSON member {wanted!r}')

    def finish_object(self):
        while self.peek() == ',':
            self.consume(',')
            if not isinstance(self.value(), str):
                raise ValueError('JSON object key is not a string')
            self.consume(':')
            self.skip_value()
        self.consume('}')


def iter_json_array(path, prefix=()):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        stream = StreamingJSON(handle)
        for key in prefix:
            stream.member(key)
        stream.consume('[')
        if stream.peek() != ']':
            while True:
                yield stream.value()
                if stream.peek() != ',':
                    break
                stream.consume(',')
        stream.consume(']')
        for _ in prefix:
            stream.finish_object()
        if stream.peek():
            raise ValueError('Unexpected content after the JSON document')


def jsonl_records(paths):
    for path in paths:
        with path.open('rb') as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    value = json_loads(line.decode('utf-8'))
                except UnicodeDecodeError as exc:
                    yield RawRecord(path, index, error=ExtractionError('invalid_utf8', str(exc)))
                except (ValueError, TypeError) as exc:
                    yield RawRecord(path, index, error=ExtractionError('invalid_json', str(exc)))
                else:
                    yield RawRecord(path, index, value)


def trace(item, data_dir, method):
    return {'raw_file': relative_path(item.path, data_dir), 'raw_record_index': item.index,
            'extraction_method': method}


def stable_id(source, record, text):
    native = record.get('id')
    if isinstance(native, str) and native.strip():
        return f'{source}:{native}'
    if isinstance(native, int) and not isinstance(native, bool):
        return f'{source}:{native}'
    nested = record.get('metadata')
    url = record.get('url')
    if url is None and isinstance(nested, dict):
        url = nested.get('url')
    digest = hashlib.sha256(serialize([source, url, text]).encode('utf-8')).hexdigest()
    return f'{source}:sha256:{digest}'


def jsonl_document(item, data_dir, source, method, text_field='text'):
    record = item.value
    if not isinstance(record, dict):
        raise ExtractionError('invalid_record', 'Expected a JSON object')
    text = record.get(text_field)
    if not isinstance(text, str):
        raise ExtractionError('missing_text', f'Missing string field {text_field!r}')
    metadata = trace(item, data_dir, method)
    metadata['source_metadata'] = {key: value for key, value in record.items() if key != text_field}
    return {'id': stable_id(source, record, text), 'source': source, 'text': text, 'metadata': metadata}


def validate_document(document, expected_source=None):
    if not isinstance(document, dict) or set(document) != {'id', 'source', 'text', 'metadata'}:
        raise ExtractionError('invalid_schema', 'Exactly id, source, text, metadata are required')
    if not isinstance(document['id'], str) or not document['id'].strip():
        raise ExtractionError('invalid_id', 'ID must be a non-empty string')
    if not isinstance(document['source'], str) or document['source'] not in SOURCES or (expected_source and document['source'] != expected_source):
        raise ExtractionError('invalid_source', 'Unexpected document source')
    if not isinstance(document['text'], str):
        raise ExtractionError('invalid_text', 'Text must be a string')
    if not document['text'].strip():
        raise ExtractionError('empty_text', 'Extraction produced empty or whitespace-only text')
    if not isinstance(document['metadata'], dict):
        raise ExtractionError('invalid_metadata', 'Metadata must be a JSON object')
    serialize(document)


class IdRegistry:
    """Disk-backed uniqueness checks keep ID memory use bounded."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE ids (id TEXT PRIMARY KEY, raw_file TEXT, record_index INTEGER)')

    def add(self, document_id, raw_file, index):
        try:
            self.connection.execute('INSERT INTO ids VALUES (?, ?, ?)', (document_id, raw_file, index))
        except sqlite3.IntegrityError:
            original = self.connection.execute('SELECT raw_file, record_index FROM ids WHERE id=?', (document_id,)).fetchone()
            raise ExtractionError('duplicate_id', f'Duplicate ID {document_id!r}; first occurrence: {original}') from None

    def count(self):
        return self.connection.execute('SELECT count(*) FROM ids').fetchone()[0]

    def close(self):
        self.connection.close()


class ErrorLog:
    def __init__(self, handle):
        self.handle = handle
        self.count = 0

    def record(self, source, item, error, data_dir):
        entry = {'source': source, 'raw_file': relative_path(item.path, data_dir),
                 'raw_record_index': item.index,
                 'error_type': getattr(error, 'error_type', type(error).__name__),
                 'message': str(error)[:1200]}
        if getattr(error, 'requires_future_ocr', False):
            entry['requires_future_ocr'] = True
        self.handle.write(serialize(entry) + '\n')
        self.handle.flush()
        self.count += 1
        print(f"ERROR {source} {entry['raw_file']} record {item.index}: {entry['error_type']}: {entry['message']}")
        return entry['error_type']


def empty_statistics(source, paths, data_dir):
    return {'source': source, 'input_documents': 0, 'output_documents': 0, 'failed_documents': 0,
            'total_characters': 0, 'min_characters': None, 'max_characters': None,
            'mean_characters': 0, 'raw_files': [relative_path(path, data_dir) for path in paths],
            'extraction_methods': [], 'timestamp': timestamp(), 'status': 'COMPLETE'}


def standardize_source(module, data_dir, errors, ids):
    source = module.SOURCE
    statistics = empty_statistics(source, module.input_files(data_dir), data_dir)
    output_dir = data_dir / 'standardized' / source
    previous_error, repeated = None, 0
    methods = set()
    try:
        with atomic_text_writer(output_dir / 'documents.jsonl') as output:
            for item in module.records(data_dir):
                statistics['input_documents'] += 1
                try:
                    if item.error:
                        raise item.error
                    document = module.extract(item, data_dir)
                    validate_document(document, source)
                    encoded = serialize(document)
                    ids.add(document['id'], relative_path(item.path, data_dir), item.index)
                except (MemoryError, OSError):
                    raise
                except Exception as exc:
                    statistics['failed_documents'] += 1
                    error_type = errors.record(source, item, exc, data_dir)
                    repeated = repeated + 1 if error_type == previous_error else 1
                    previous_error = error_type
                    if repeated >= 5:
                        raise SourceAborted(f'Five consecutive {error_type} failures; fix this extractor before rerunning.', statistics) from exc
                    continue
                output.write(encoded + '\n')
                length = len(document['text'])
                statistics['output_documents'] += 1
                statistics['total_characters'] += length
                statistics['min_characters'] = length if statistics['min_characters'] is None else min(statistics['min_characters'], length)
                statistics['max_characters'] = length if statistics['max_characters'] is None else max(statistics['max_characters'], length)
                methods.add(document['metadata']['extraction_method'])
                previous_error, repeated = None, 0
    except Exception as exc:
        statistics['status'] = 'ABORTED'
        statistics['abort_message'] = str(exc)
        # Atomic publication did not happen: these attempted outputs are not final.
        statistics['attempted_output_documents'] = statistics['output_documents']
        statistics['output_documents'] = 0
        statistics['total_characters'] = 0
        statistics['min_characters'] = statistics['max_characters'] = None
        write_json(output_dir / 'manifest.json', statistics)
        raise SourceAborted(str(exc), statistics) from exc
    statistics['mean_characters'] = statistics['total_characters'] / statistics['output_documents'] if statistics['output_documents'] else 0
    statistics['extraction_methods'] = sorted(methods)
    if statistics['failed_documents']:
        statistics['status'] = 'FAILED'
    write_json(output_dir / 'manifest.json', statistics)
    return statistics


def acquisition_records(data_dir, source):
    path = data_dir / 'metadata' / source / 'pilot_files.json'
    lookup = {}
    for record in iter_json_array(path):
        if not isinstance(record, dict) or not isinstance(record.get('local_file'), str):
            raise ValueError(f'Invalid acquisition metadata in {path}')
        name = record['local_file']
        if name in lookup:
            raise ValueError(f'Duplicate acquisition filename {name}')
        lookup[name] = record
    for path in sorted((data_dir / 'raw' / source).glob('*.pdf')):
        if path.name not in lookup:
            yield RawRecord(path, None, error=ExtractionError('missing_acquisition_metadata', 'PDF has no matching acquisition record'))
        else:
            yield RawRecord(path, None, lookup[path.name])


def extract_pdf_text(path):
    """Read the existing text layer only; OCR is never invoked."""
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(path) as document:
            if not document.page_count:
                raise ExtractionError('invalid_pdf', 'PDF contains no pages')
            return '\n\n'.join(page.get_text() for page in document), document.page_count, 'pymupdf'
    binary = shutil.which('pdftotext')
    if not binary:
        raise ExtractionError('missing_pdf_dependency', 'Install PyMuPDF or provide the local pdftotext utility')
    result = subprocess.run([binary, '-enc', 'UTF-8', '-eol', 'unix', str(path), '-'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ExtractionError('pdf_extraction_failed', result.stderr.decode('utf-8', errors='replace')[:1000])
    pages = result.stdout.decode('utf-8').split('\f')
    if pages and pages[-1] == '':
        pages.pop()  # Poppler's final page-boundary sentinel, not book content.
    info = shutil.which('pdfinfo')
    if info:
        info_result = subprocess.run([info, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env={**os.environ, 'LC_ALL': 'C'}, check=False)
        match = re.search(r'^Pages:\s*(\d+)\s*$', info_result.stdout.decode('utf-8', errors='replace'), re.MULTILINE)
        if info_result.returncode or not match or int(match.group(1)) != len(pages):
            raise ExtractionError('pdf_page_count_mismatch', 'PDF structure and extracted page boundaries disagree')
    if not pages:
        raise ExtractionError('invalid_pdf', 'PDF contains no pages')
    return '\n\n'.join(pages), len(pages), 'pdftotext'

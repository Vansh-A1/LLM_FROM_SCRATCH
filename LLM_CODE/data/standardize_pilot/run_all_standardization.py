"""Reproduce the standardized pilot from existing, read-only raw inputs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import standardize_arxiv
import standardize_code
import standardize_fineweb
import standardize_gutenberg
import standardize_scanned_docs
import standardize_stackexchange
import standardize_wikipedia
from common import (DATA_DIR, ErrorLog, ExtractionError, IdRegistry, RawRecord, SourceAborted,
                    atomic_text_writer, json_loads, relative_path, sha256_file,
                    standardize_source, timestamp, validate_document, write_json)

MODULES = (standardize_wikipedia, standardize_fineweb, standardize_code,
           standardize_stackexchange, standardize_gutenberg, standardize_arxiv,
           standardize_scanned_docs)
LABELS = {'wikipedia': 'Wikipedia', 'fineweb': 'FineWeb-Edu', 'code': 'Code',
          'stackexchange': 'StackExchange', 'gutenberg': 'Gutenberg', 'arxiv': 'arXiv',
          'scanned_docs': 'Scanned Docs'}
EXPECTED = {'wikipedia': 100, 'fineweb': 100, 'code': 100, 'stackexchange': 100,
            'gutenberg': 3, 'arxiv': 5, 'scanned_docs': 2}


def smoke_test(data_dir, summaries, scratch, raw_files):
    registry = IdRegistry(Path(scratch) / 'validation_ids.sqlite')
    sample = json_loads((data_dir / 'standardized/wikipedia/sample_001.json').read_text(encoding='utf-8'))
    fields = ('title', 'page_id', 'revision_id', 'revision_timestamp', 'content_format', 'content_model')
    result = {'all_jsonl_valid': True, 'all_document_schemas_valid': True,
              'manifest_statistics_match': True, 'traceability_valid': True,
              'ids_unique': True, 'wikipedia_regression': False, 'validation_errors': [],
              'output_documents': 0, 'total_characters': 0, 'sources': {}, 'previews': {}}
    regression_matches = 0
    try:
        for module in MODULES:
            source = module.SOURCE
            path = data_dir / 'standardized' / source / 'documents.jsonl'
            counts = {'output_documents': 0, 'total_characters': 0, 'min_characters': None, 'max_characters': None}
            fallback_preview = None
            if not path.is_file():
                result['all_jsonl_valid'] = False
                result['validation_errors'].append(f'{source}: output JSONL is missing')
                result['sources'][source] = counts
                continue
            with path.open(encoding='utf-8', newline='') as handle:
                for line_index, line in enumerate(handle):
                    counts['output_documents'] += 1
                    try:
                        document = json_loads(line)
                    except ValueError as exc:
                        result['all_jsonl_valid'] = False
                        result['validation_errors'].append(f'{source} line {line_index}: {exc}')
                        continue
                    try:
                        validate_document(document, source)
                    except ExtractionError as exc:
                        result['all_document_schemas_valid'] = False
                        result['validation_errors'].append(f'{source} line {line_index}: {exc}')
                        continue
                    metadata = document['metadata']
                    raw_file = metadata.get('raw_file')
                    record_index = metadata.get('raw_record_index')
                    if (raw_file not in raw_files or 'raw_record_index' not in metadata
                            or not isinstance(metadata.get('extraction_method'), str)
                            or not metadata['extraction_method']
                            or (record_index is not None and (not isinstance(record_index, int) or isinstance(record_index, bool)))):
                        result['traceability_valid'] = False
                        result['validation_errors'].append(f'{source} line {line_index}: invalid raw traceability')
                    try:
                        registry.add(document['id'], raw_file, record_index)
                    except ExtractionError as exc:
                        result['ids_unique'] = False
                        result['validation_errors'].append(str(exc))
                    length = len(document['text'])
                    counts['total_characters'] += length
                    counts['min_characters'] = length if counts['min_characters'] is None else min(counts['min_characters'], length)
                    counts['max_characters'] = length if counts['max_characters'] is None else max(counts['max_characters'], length)
                    if fallback_preview is None:
                        fallback_preview = document['text'][:min(400, max(0, length - 1))]
                    if source not in result['previews'] and length >= 500:
                        result['previews'][source] = document['text'][:400]
                    if source == 'wikipedia' and document['id'] == sample['id']:
                        regression_matches += 1
                        result['wikipedia_regression'] = (
                            document['source'] == sample['source']
                            and document['text'] == sample['text']
                            and all(metadata.get(field) == sample['metadata'].get(field) for field in fields)
                        )
            counts['mean_characters'] = counts['total_characters'] / counts['output_documents'] if counts['output_documents'] else 0
            if source not in result['previews'] and fallback_preview is not None:
                result['previews'][source] = fallback_preview
            if any(counts[key] != summaries[source].get(key) for key in counts):
                result['manifest_statistics_match'] = False
                result['validation_errors'].append(f'{source}: output statistics disagree with its manifest')
            result['sources'][source] = counts
            result['output_documents'] += counts['output_documents']
            result['total_characters'] += counts['total_characters']
        result['unique_ids'] = registry.count()
        result['ids_unique'] = result['ids_unique'] and registry.count() == result['output_documents']
        result['wikipedia_regression'] = result['wikipedia_regression'] and regression_matches == 1
    finally:
        registry.close()
    return result


def checksum_report(paths, before, data_dir):
    rows = []
    for path in paths:
        key = relative_path(path, data_dir)
        after = sha256_file(path) if path.is_file() else None
        rows.append({'file': key, 'sha256_before': before[key], 'sha256_after': after,
                     'unchanged': after == before[key]})
    return rows


def print_report(manifest, previews, data_dir):
    print('\n' + '=' * 50 + '\nPILOT EXTRACTION + STANDARDIZATION\n' + '=' * 50)
    for source, stats in manifest['sources'].items():
        print(f"\n{LABELS[source]}\nInput: {stats['input_documents']}\nOutput: {stats['output_documents']}\nFailures: {stats['failed_documents']}\nCharacters: {stats['total_characters']}")
        print(f"Minimum chars: {stats['min_characters']}\nMaximum chars: {stats['max_characters']}\nMean chars: {stats['mean_characters']:.2f}")
        if source == 'wikipedia':
            print('Regression test:', 'PASS' if manifest['validation']['wikipedia_regression'] else 'FAIL')
        if source == 'scanned_docs':
            print('OCR executed by us: NO')
    print('\n' + '-' * 50)
    print(f"Total inputs: {manifest['input_documents']}\nTotal standardized documents: {manifest['output_documents']}\nTotal failures: {manifest['failed_documents']}\nUnique IDs: {manifest['unique_ids']} / {manifest['output_documents']}\nTotal characters: {manifest['total_characters']}")
    for label, passed in [
        ('Raw files unchanged', manifest['raw_checksum_verification']['passed']),
        ('Existing Wikipedia test files unchanged', manifest['protected_files_verification']['passed']),
        ('All JSONL valid', manifest['validation']['all_jsonl_valid']),
        ('All document schemas valid', manifest['validation']['all_document_schemas_valid']),
        ('Manifest statistics match', manifest['validation']['manifest_statistics_match']),
        ('Traceability valid', manifest['validation']['traceability_valid']),
    ]:
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    for discrepancy in manifest['input_count_discrepancies']:
        print('COUNT DISCREPANCY:', discrepancy)
    for error in manifest['validation']['validation_errors']:
        print('VALIDATION ERROR:', error)
    print('\n' + '=' * 50 + f"\nEXTRACTION + STANDARDIZATION: {manifest['status']}\n" + '=' * 50)
    for source, preview in previews.items():
        print(f'\n{LABELS[source]} preview (one document, at most 400 characters):\n{preview}')
    print('\nStandardized directory tree:')
    directory = data_dir / 'standardized'
    print('data/standardized/')
    def tree(path, prefix=''):
        children = sorted(child for child in path.iterdir() if child.name != '__pycache__')
        for index, child in enumerate(children):
            last = index == len(children) - 1
            print(prefix + ('└── ' if last else '├── ') + child.name + ('/' if child.is_dir() else ''))
            if child.is_dir():
                tree(child, prefix + ('    ' if last else '│   '))
    tree(directory)


def run(data_dir=DATA_DIR):
    data_dir = Path(data_dir).resolve()
    files_by_source = {module.SOURCE: module.input_files(data_dir) for module in MODULES}
    for source, paths in files_by_source.items():
        if not paths:
            raise FileNotFoundError(f'{source}: no supported raw pilot inputs found')
    raw_paths = sorted({path for paths in files_by_source.values() for path in paths})
    raw_before = {relative_path(path, data_dir): sha256_file(path) for path in raw_paths}
    protected = [data_dir / 'standardize_pilot/standardize_wikipedia_one.py',
                 data_dir / 'standardized/wikipedia/sample_001.json']
    metadata_paths = [data_dir / 'metadata' / source / 'pilot_files.json' for source in ('arxiv', 'scanned_docs')]
    protected_before = {relative_path(path, data_dir): sha256_file(path) for path in protected}
    metadata_before = {relative_path(path, data_dir): sha256_file(path) for path in metadata_paths}
    summaries = {}
    with tempfile.TemporaryDirectory(prefix='llm-standardization-', dir='/tmp') as scratch:
        registry = IdRegistry(Path(scratch) / 'writer_ids.sqlite')
        try:
            with atomic_text_writer(data_dir / 'standardized/errors/errors.jsonl') as error_handle:
                errors = ErrorLog(error_handle)
                for module in MODULES:
                    print(f'Standardizing {LABELS[module.SOURCE]}...', flush=True)
                    try:
                        summaries[module.SOURCE] = standardize_source(module, data_dir, errors, registry)
                    except SourceAborted as exc:
                        summaries[module.SOURCE] = exc.statistics
                        item = RawRecord(files_by_source[module.SOURCE][0], None)
                        errors.record(module.SOURCE, item, ExtractionError('source_aborted', str(exc)), data_dir)
                error_count = errors.count
        finally:
            registry.close()
        validation = smoke_test(data_dir, summaries, scratch, raw_before)
    previews = validation.pop('previews')
    raw_rows = checksum_report(raw_paths, raw_before, data_dir)
    protected_rows = checksum_report(protected, protected_before, data_dir)
    metadata_rows = checksum_report(metadata_paths, metadata_before, data_dir)
    input_set_unchanged = all(module.input_files(data_dir) == files_by_source[module.SOURCE] for module in MODULES)
    total_inputs = sum(stats['input_documents'] for stats in summaries.values())
    total_failures = sum(stats['failed_documents'] for stats in summaries.values())
    discrepancies = [{'source': source, 'expected': EXPECTED[source], 'actual': stats['input_documents']}
                     for source, stats in summaries.items() if stats['input_documents'] != EXPECTED[source]]
    checks_pass = all(validation[key] for key in ('all_jsonl_valid', 'all_document_schemas_valid',
                       'manifest_statistics_match', 'traceability_valid', 'ids_unique', 'wikipedia_regression'))
    complete = (checks_pass and not error_count and not total_failures
                and validation['output_documents'] == total_inputs
                and all(stats['status'] == 'COMPLETE' for stats in summaries.values())
                and input_set_unchanged and all(row['unchanged'] for row in raw_rows + protected_rows + metadata_rows))
    manifest = {'timestamp': timestamp(), 'status': 'COMPLETE' if complete else 'FAILED',
                'schema_fields': ['id', 'source', 'text', 'metadata'], 'sources': summaries,
                'input_documents': total_inputs, 'output_documents': validation['output_documents'],
                'failed_documents': total_failures, 'error_records': error_count,
                'unique_ids': validation['unique_ids'], 'total_characters': validation['total_characters'],
                'input_count_discrepancies': discrepancies, 'validation': validation,
                'raw_checksum_verification': {'passed': input_set_unchanged and all(row['unchanged'] for row in raw_rows),
                                              'input_file_set_unchanged': input_set_unchanged, 'files': raw_rows},
                'protected_files_verification': {'passed': all(row['unchanged'] for row in protected_rows), 'files': protected_rows},
                'acquisition_metadata_verification': {'passed': all(row['unchanged'] for row in metadata_rows), 'files': metadata_rows},
                'ocr_executed': False}
    write_json(data_dir / 'standardized/pilot_manifest.json', manifest)
    print_report(manifest, previews, data_dir)
    return manifest


def main():
    try:
        manifest = run()
    except Exception as exc:
        print(f'EXTRACTION + STANDARDIZATION: FAILED\n{type(exc).__name__}: {exc}')
        return 1
    return 0 if manifest['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())

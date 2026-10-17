"""Preserve the actual Stack V2 Edu text field, including all whitespace."""
from common import jsonl_document, jsonl_records

SOURCE = 'code'


def input_files(data_dir):
    return sorted((data_dir / 'raw/code').rglob('*.jsonl'))


def records(data_dir):
    return jsonl_records(input_files(data_dir))


def extract(item, data_dir):
    return jsonl_document(item, data_dir, SOURCE, 'jsonl_code_text')

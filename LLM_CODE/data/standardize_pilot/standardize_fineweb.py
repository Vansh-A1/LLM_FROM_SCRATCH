"""Preserve FineWeb-Edu JSONL text and its non-text source fields."""
from common import jsonl_document, jsonl_records

SOURCE = 'fineweb'


def input_files(data_dir):
    return sorted((data_dir / 'raw/fineweb').rglob('*.jsonl'))


def records(data_dir):
    return jsonl_records(input_files(data_dir))


def extract(item, data_dir):
    return jsonl_document(item, data_dir, SOURCE, 'jsonl_text_field')

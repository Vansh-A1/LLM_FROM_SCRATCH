"""Keep the existing combined Stack Exchange text as one document."""
from common import jsonl_document, jsonl_records

SOURCE = 'stackexchange'


def input_files(data_dir):
    return sorted((data_dir / 'raw/stackexchange').rglob('*.jsonl'))


def records(data_dir):
    return jsonl_records(input_files(data_dir))


def extract(item, data_dir):
    return jsonl_document(item, data_dir, SOURCE, 'jsonl_text_field')

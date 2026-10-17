"""Read the scans' existing text layers; never perform OCR."""
from common import ExtractionError, acquisition_records, extract_pdf_text, trace

SOURCE = 'scanned_docs'


def input_files(data_dir):
    return sorted((data_dir / 'raw/scanned_docs').glob('*.pdf'))


def records(data_dir):
    return acquisition_records(data_dir, SOURCE)


def extract(item, data_dir):
    acquired = item.value
    text, pages, parser = extract_pdf_text(item.path)
    if sum(character.isalnum() for character in text) < 50:
        raise ExtractionError('requires_future_ocr', 'The existing PDF text layer is essentially empty; OCR was not run.', requires_future_ocr=True)
    metadata = trace(item, data_dir, 'existing_pdf_ocr_text_layer')
    metadata.update(title=acquired.get('title'), source=acquired.get('source'),
                    source_url=acquired.get('source_url'), pages=pages, pdf_parser=parser,
                    requires_future_ocr=False)
    if acquired.get('archive_identifier'):
        metadata['archive_identifier'] = acquired['archive_identifier']
    return {'id': f'scanned_docs:{item.path.stem}', 'source': SOURCE, 'text': text, 'metadata': metadata}

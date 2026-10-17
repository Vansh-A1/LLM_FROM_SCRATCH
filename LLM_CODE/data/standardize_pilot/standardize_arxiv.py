"""Extract digital PDF text in page order, using acquisition metadata."""
from common import ExtractionError, acquisition_records, extract_pdf_text, trace

SOURCE = 'arxiv'


def input_files(data_dir):
    return sorted((data_dir / 'raw/arxiv').glob('*.pdf'))


def records(data_dir):
    return acquisition_records(data_dir, SOURCE)


def extract(item, data_dir):
    acquired = item.value
    arxiv_id = acquired.get('arxiv_id')
    if not isinstance(arxiv_id, str) or not arxiv_id:
        raise ExtractionError('missing_arxiv_id', 'Acquisition metadata has no arXiv ID')
    text, pages, parser = extract_pdf_text(item.path)
    metadata = trace(item, data_dir, 'pdf_embedded_text')
    metadata.update(title=acquired.get('title'), arxiv_id=arxiv_id, category=acquired.get('category'),
                    source_url=acquired.get('source_url'), pages=pages, pdf_parser=parser)
    if acquired.get('version') is not None:
        metadata['version'] = acquired['version']
    return {'id': f'arxiv:{arxiv_id}', 'source': SOURCE, 'text': text, 'metadata': metadata}

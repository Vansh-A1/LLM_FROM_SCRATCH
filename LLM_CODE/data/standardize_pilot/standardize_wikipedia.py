"""Stream actual MediaWiki page records, preserving revision Wikitext."""

from common import ExtractionError, RawRecord, iter_json_array, trace

SOURCE = 'wikipedia'


def input_files(data_dir):
    return sorted((data_dir / 'raw/wikipedia').glob('articles_*.json'))


def records(data_dir):
    for path in input_files(data_dir):
        for index, page in enumerate(iter_json_array(path, ('query', 'pages'))):
            yield RawRecord(path, index, page)


def extract(item, data_dir):
    page = item.value
    if not isinstance(page, dict) or page.get('pageid') is None:
        raise ExtractionError('missing_page_id', 'Page has no pageid')
    if not isinstance(page.get('title'), str) or not page['title']:
        raise ExtractionError('missing_title', 'Page has no title')
    revisions = page.get('revisions')
    if not isinstance(revisions, list) or not revisions:
        raise ExtractionError('missing_revision', 'Page has no revisions')
    revision = revisions[0]
    if not isinstance(revision, dict) or revision.get('revid') is None:
        raise ExtractionError('missing_revision_id', 'Revision has no revid')
    slots = revision.get('slots')
    main = slots.get('main') if isinstance(slots, dict) else None
    text = main.get('content') if isinstance(main, dict) else None
    if not isinstance(text, str):
        raise ExtractionError('missing_revision_content', 'Main revision slot has no string content')
    metadata = trace(item, data_dir, 'mediawiki_revision_content')
    metadata.update(title=page['title'], page_id=page['pageid'], revision_id=revision['revid'],
                    revision_timestamp=revision.get('timestamp'), language='en',
                    content_format=main.get('contentformat'), content_model=main.get('contentmodel'))
    return {'id': f"wikipedia:{page['pageid']}:{revision['revid']}", 'source': SOURCE,
            'text': text, 'metadata': metadata}

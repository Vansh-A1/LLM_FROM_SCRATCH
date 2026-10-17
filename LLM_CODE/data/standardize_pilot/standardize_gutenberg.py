"""Extract TXT, visible HTML text, and EPUB spine text without book cleanup."""

import posixpath
import zipfile
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from common import ExtractionError, RawRecord, trace

SOURCE = 'gutenberg'
BOOKS = {
    'pride_and_prejudice/pride_and_prejudice.txt': (1342, 'Pride and Prejudice', 'txt'),
    'frankenstein/frankenstein.html': (84, 'Frankenstein', 'html'),
    'sherlock_holmes/sherlock_holmes.epub': (1661, 'The Adventures of Sherlock Holmes', 'epub'),
}


class VisibleTextParser(HTMLParser):
    BLOCKS = {'body', 'p', 'div', 'section', 'article', 'header', 'footer', 'h1', 'h2',
              'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre', 'li', 'ul', 'ol', 'table', 'tr', 'dl', 'dt', 'dd'}
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, ignore_navigation=False):
        super().__init__(convert_charrefs=True)
        self.ignore_navigation = ignore_navigation
        self.stack = []
        self.parts = []

    def hidden(self):
        return bool(self.stack and self.stack[-1][1])

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        hidden_tags = {'head', 'script', 'style', 'template'}
        if self.ignore_navigation:
            hidden_tags.add('nav')
        style = ''.join((attributes.get('style') or '').lower().split())
        hidden = self.hidden() or tag in hidden_tags or 'hidden' in attributes or 'display:none' in style or 'visibility:hidden' in style
        if not hidden and (tag in self.BLOCKS or tag in {'br', 'hr'}):
            self.parts.append('\n')
        if tag not in self.VOID:
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        hidden = self.hidden()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                hidden = self.stack[index][1]
                del self.stack[index:]
                break
        if not hidden and tag in self.BLOCKS:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden():
            self.parts.append(data)


def visible_text(markup, *, ignore_navigation=False):
    parser = VisibleTextParser(ignore_navigation)
    parser.feed(markup)
    parser.close()
    return ''.join(parser.parts)


def epub_spine_text(path):
    with zipfile.ZipFile(path) as archive:
        container = ElementTree.fromstring(archive.read('META-INF/container.xml'))
        rootfiles = [node for node in container.iter() if node.tag.rsplit('}', 1)[-1] == 'rootfile']
        if not rootfiles:
            raise ExtractionError('invalid_epub', 'No OPF rootfile in container.xml')
        preferred = next((node for node in rootfiles if node.get('media-type') == 'application/oebps-package+xml'), rootfiles[0])
        package_path = preferred.get('full-path')
        if not package_path:
            raise ExtractionError('invalid_epub', 'OPF rootfile has no full-path')
        package = ElementTree.fromstring(archive.read(package_path))
        manifest = next((node for node in package if node.tag.rsplit('}', 1)[-1] == 'manifest'), None)
        spine = next((node for node in package if node.tag.rsplit('}', 1)[-1] == 'spine'), None)
        if manifest is None or spine is None:
            raise ExtractionError('invalid_epub', 'OPF manifest or spine is missing')
        items = {node.get('id'): node for node in manifest}
        chunks, spine_files = [], []
        for reference in spine:
            item_id = reference.get('idref')
            if item_id not in items:
                raise ExtractionError('invalid_epub', f'Unknown spine item {item_id!r}')
            item = items[item_id]
            if item.get('media-type') not in {'application/xhtml+xml', 'text/html'}:
                continue
            if 'nav' in (item.get('properties') or '').split():
                continue
            href = urlsplit(item.get('href') or '')
            if href.scheme or href.netloc or not href.path:
                raise ExtractionError('invalid_epub', 'Spine item must be an internal content document')
            member = posixpath.normpath(posixpath.join(posixpath.dirname(package_path), unquote(href.path)))
            if member.startswith('../') or member.startswith('/'):
                raise ExtractionError('invalid_epub', 'Invalid archive content path')
            with archive.open(member) as content:
                markup = content.read().decode('utf-8')
            extracted = visible_text(markup, ignore_navigation=True)
            if extracted.strip():
                chunks.append(extracted)
            spine_files.append(member)
        if not chunks:
            raise ExtractionError('empty_epub', 'No readable text in the publication spine')
        return '\n\n'.join(chunks), spine_files


def input_files(data_dir):
    return sorted(path for path in (data_dir / 'raw/gutenberg').rglob('*')
                  if path.is_file() and path.suffix.lower() in {'.txt', '.html', '.epub'})


def records(data_dir):
    raw = data_dir / 'raw/gutenberg'
    for path in input_files(data_dir):
        book = BOOKS.get(path.relative_to(raw).as_posix())
        if book is None:
            yield RawRecord(path, None, error=ExtractionError('unknown_book_metadata', 'No verified Gutenberg ID for this book'))
        else:
            yield RawRecord(path, None, book)


def extract(item, data_dir):
    book_id, title, book_format = item.value
    spine_files = None
    if book_format == 'txt':
        with item.path.open(encoding='utf-8', newline='') as handle:
            text = handle.read()
        method = 'plain_text_utf8'
    elif book_format == 'html':
        with item.path.open(encoding='utf-8', newline='') as handle:
            text = visible_text(handle.read())
        method = 'html_visible_text'
    else:
        text, spine_files = epub_spine_text(item.path)
        method = 'epub_spine_text'
    metadata = trace(item, data_dir, method)
    metadata.update(title=title, gutenberg_id=book_id, format=book_format)
    if spine_files is not None:
        metadata['spine_files'] = spine_files
    return {'id': f'gutenberg:{book_id}', 'source': SOURCE, 'text': text, 'metadata': metadata}

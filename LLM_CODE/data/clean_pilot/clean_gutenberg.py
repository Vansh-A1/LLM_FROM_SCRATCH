"""Trim only confidently identified Gutenberg distribution boundaries."""

import re

from common import normalize_prose, process_source, remove_controls


SOURCE = "gutenberg"
VERSION = "gutenberg_v1"
START = re.compile(
    r"(?im)^[ \t]*\*{3}[ \t]*START OF (?:THE|THIS) PROJECT GUTENBERG "
    r"EBOOK[^\n]*?\*{3}[ \t]*\r?$"
)
END = re.compile(
    r"(?im)^[ \t]*\*{3}[ \t]*END OF (?:THE|THIS) PROJECT GUTENBERG "
    r"EBOOK[^\n]*?\*{3}[ \t]*\r?$"
)


def clean(text):
    starts, ends = list(START.finditer(text)), list(END.finditer(text))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise ValueError("Ambiguous/missing Gutenberg boundaries; inspect rather than guess.")
    start, end = starts[0], ends[0]
    header, footer = text[:start.start()], text[end.end():]
    if header.strip() and not (
        "Project Gutenberg" in header and "This eBook is for the use" in header
    ):
        raise ValueError("Text before START is not clearly distribution metadata.")
    if footer.strip() and not (
        "Project Gutenberg" in footer
        and ("LICENSE" in footer or "Updated editions" in footer)
    ):
        raise ValueError("Text after END is not clearly distribution metadata.")
    body, controls = remove_controls(text[start.end():end.start()])
    cleaned = normalize_prose(body)
    return cleaned, {
        "headers_removed": int(bool(header.strip())),
        "footers_removed": int(bool(footer.strip())),
        "boundary_markers_removed": 2,
        "distribution_header_characters_removed": len(header),
        "distribution_footer_characters_removed": len(footer),
        "boundary_marker_characters_removed": len(start.group()) + len(end.group()),
        "control_characters_removed": controls,
    }, {
        "start_marker": start.group().strip(),
        "end_marker": end.group().strip(),
        "body_start_offset": start.end(),
        "body_end_offset": end.start(),
        "whitespace_policy": "Keep body lines and indentation; trim trailing spaces and excessive blank lines.",
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

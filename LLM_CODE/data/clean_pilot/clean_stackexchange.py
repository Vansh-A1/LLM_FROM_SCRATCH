"""Conservative cleanup of the inspected plain-text/Markdown discussions."""

import re

from common import normalize_markdown, process_source, remove_controls


SOURCE = "stackexchange"
VERSION = "stackexchange_v1"


def clean(text):
    text, controls = remove_controls(text)
    # In this pilot the div/span tags are explicitly introduced as an HTML code
    # snippet, and '<and list the tags1>' is prose notation. They are information,
    # not leftover page markup. Preserve them and all explanatory URLs.
    cleaned = normalize_markdown(text)
    return cleaned, {"control_characters_removed": controls}, {
        "format": "already_extracted_text_and_markdown",
        "html_policy": "Preserve literal explanatory HTML examples.",
        "literal_html_tag_candidates_preserved": len(
            re.findall(r"</?(?:div|span)\b[^>]*>", cleaned, re.I)
        ),
        "code_policy": "Keep fenced/indented blocks and inline code unchanged.",
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

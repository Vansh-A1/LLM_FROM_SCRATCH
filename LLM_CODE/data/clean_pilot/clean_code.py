"""Preserve code, Markdown, licenses, comments, and all valid whitespace."""

from common import process_source, remove_controls


SOURCE = "code"
VERSION = "code_v1"


def clean(text):
    cleaned, controls = remove_controls(text)
    return cleaned, {"control_characters_removed": controls}, {
        "whitespace_policy": "Preserve line endings, tabs, indentation, spaces, and blank lines.",
        "large_reduction_review_required": (
            (len(text) - len(cleaned)) / len(text) > 0.25 if text else False
        ),
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

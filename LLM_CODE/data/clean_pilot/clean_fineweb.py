"""Light cleanup of already-extracted FineWeb prose; no boilerplate guessing."""

from common import normalize_prose, process_source, remove_controls


SOURCE = "fineweb"
VERSION = "fineweb_v1"


def clean(text):
    text, controls = remove_controls(text)
    cleaned = normalize_prose(text, collapse_horizontal=True)
    return cleaned, {"control_characters_removed": controls}, {
        "boilerplate_policy": "Retain questionable navigation and web text.",
        "whitespace_policy": "Trim trailing spaces; collapse 3+ interior spaces and 3+ blank lines.",
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

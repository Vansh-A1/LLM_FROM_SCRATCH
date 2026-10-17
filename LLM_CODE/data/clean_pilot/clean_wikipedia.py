"""Reuse the protected pilot renderer, with measured malformed-markup fixes."""

import re
import mwparserfromhell

from common import process_source, remove_controls
from clean_wikipedia_one import clean_wikitext, remaining_markup


SOURCE = "wikipedia"
VERSION = "wikipedia_wikitext_v1"


def clean(text):
    text, controls = remove_controls(text)
    cleaned, counters = clean_wikitext(text)
    # Observed corpus artifacts: unmatched quotes, nowiki-wrapped list markers,
    # and a lone row separator outside a valid table. Never delete their prose.
    # An italic closing pair followed by the possessive apostrophe can leave
    # three quotes before 's' in malformed input. Retain that prose apostrophe.
    cleaned, possessives = re.subn(
        r"(?<=\w)('{3,})(?=s\b)",
        lambda match: "'" if len(match.group()) % 2 else "",
        cleaned,
    )
    cleaned, quotes = re.subn(r"'{2,}", "", cleaned)
    cleaned, lists = re.subn(r"(?m)^[ \t]*[*#;:]+[ \t]+", "", cleaned)
    cleaned, rows = re.subn(r"(?m)^[ \t]*(?:\|-|\|\})[ \t]*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    counters.update({
        "control_characters_removed": controls,
        "unmatched_formatting_runs_removed": quotes + possessives,
        "residual_list_markers_removed": lists,
        "orphan_table_markers_removed": rows,
    })
    residual = remaining_markup(cleaned)
    if any(residual.values()):
        raise ValueError(f"Wikipedia markup requires diagnosis: {residual}")
    return cleaned, counters, {
        "parser": "mwparserfromhell",
        "parser_version": mwparserfromhell.__version__,
        "remaining_markup": residual,
        "table_policy": "Remove complete parser-recognized tables; retain other prose.",
        "counter_scope": (
            "Directly discarded nodes; nested templates counted inside discarded "
            "templates. Nodes hidden inside discarded references/tables are not "
            "counted again."
        ),
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

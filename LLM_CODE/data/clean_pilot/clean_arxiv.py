"""Preserve PDF-derived equations, references, citations, and line structure."""

from collections import Counter

from common import normalize_prose, process_source, remove_controls


SOURCE = "arxiv"
VERSION = "arxiv_v1"


def clean(text):
    text, controls = remove_controls(text)
    # Page boundaries were flattened at extraction. Repeated scientific terms
    # and figure labels are meaningful; do not infer a header from frequency.
    cleaned = normalize_prose(text)
    lines = Counter(line.strip() for line in cleaned.split("\n")
                    if 5 <= len(line.strip()) <= 70)
    repeated = [{"line": line, "occurrences": count}
                for line, count in sorted(lines.items(),
                                          key=lambda item: (-item[1], item[0]))
                if count >= 3][:5]
    return cleaned, {
        "control_characters_removed": controls,
        "repeated_headers_or_footers_removed": 0,
    }, {
        "repeated_short_lines_retained_for_review": repeated,
        "page_artifact_policy": "No inferred page/header/footer deletions.",
        "equation_policy": "No repairs, line joins, or guessed hyphenation.",
        "whitespace_policy": "Preserve nonblank line order, indentation, and interior spacing.",
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

"""Clean existing OCR-layer text without OCR, spelling guesses, or filtering."""

from common import normalize_prose, process_source, remove_controls


SOURCE = "scanned_docs"
VERSION = "scanned_docs_v1"


def clean(text):
    text, controls = remove_controls(text)
    cleaned = normalize_prose(text)
    nonblank = [line for line in cleaned.split("\n") if line.strip()]
    questionable = [line for line in nonblank
                    if not any(character.isalpha() for character in line)]
    return cleaned, {
        "control_characters_removed": controls,
        "ocr_executions": 0,
    }, {
        "ocr_executed_by_us": False,
        "ocr_policy": "Existing OCR only; retain uncorrected spelling and noisy text.",
        "nonblank_lines": len(nonblank),
        "lines_without_letters_retained": len(questionable),
        "lines_without_letters_examples": [line[:100] for line in questionable[:5]],
        "replacement_characters_retained": cleaned.count("\ufffd"),
        "whitespace_policy": "Preserve nonblank lines; trim trailing spaces and excessive blank lines.",
    }


def run(errors, data_root=None):
    kwargs = {} if data_root is None else {"data_root": data_root}
    return process_source(SOURCE, VERSION, clean, errors, **kwargs)

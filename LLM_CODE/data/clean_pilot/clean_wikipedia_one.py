"""Conservatively clean only the fixed Edward Backhouse Wikipedia sample.

Run with Python 3 and mwparserfromhell installed. No raw or standardized file
is written. Templates, references, media, metadata links, and tables are dropped;
visible prose is rendered recursively from parser nodes, without rewriting.
"""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile

import mwparserfromhell
from mwparserfromhell import nodes


DATA_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = DATA_ROOT / "standardized/wikipedia/sample_001.json"
OUTPUT_DIR = DATA_ROOT / "cleaned/wikipedia"
OUTPUT_PATH = OUTPUT_DIR / "sample_001_cleaned.json"
COMPARISON_PATH = OUTPUT_DIR / "sample_001_comparison.txt"
EXPECTED_ID = "wikipedia:1644311:1352244912"
EXPECTED_TITLE = "Edward Backhouse"
SCHEMA_KEYS = {"id", "source", "text", "metadata"}

# Recognized metadata prefixes only; an arbitrary colon in a title is not a
# reason to delete a link. Escaped or labeled language links remain visible.
LANGUAGE_PREFIXES = {
    "ar", "bg", "ca", "cs", "da", "de", "el", "en", "eo", "es", "et",
    "fa", "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko",
    "la", "lt", "lv", "nl", "no", "pl", "pt", "ro", "ru", "simple",
    "sk", "sl", "sr", "sv", "ta", "th", "tr", "uk", "ur", "vi", "zh",
}
INTERWIKI_PREFIXES = {
    "commons", "meta", "mw", "species", "w", "wikt", "wiktionary",
    "b", "wikibooks", "n", "wikinews", "q", "wikiquote",
    "s", "wikisource", "v", "wikiversity", "voy", "wikivoyage",
}
BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "pre",
    "ul", "ol", "li", "dl", "dt", "dd",
}


class WikitextRenderer:
    """Render visible text; use the parser for nested markup boundaries."""

    def __init__(self):
        self.removals = Counter({
            "templates_removed": 0,
            "template_arguments_removed": 0,
            "references_removed": 0,
            "reference_lists_removed": 0,
            "comments_removed": 0,
            "media_links_removed": 0,
            "category_links_removed": 0,
            "interwiki_links_removed": 0,
            "tables_removed": 0,
            "unlabeled_external_links_removed": 0,
            "nonvisible_tags_removed": 0,
        })

    def render(self, code):
        parts = []
        for node in code.nodes:
            if isinstance(node, nodes.Text):
                parts.append(str(node))
            elif isinstance(node, nodes.Template):
                # Count nested templates too, but never render their contents.
                nested = str(node)
                self.removals["templates_removed"] += len(
                    mwparserfromhell.parse(nested).filter_templates()
                )
            elif isinstance(node, nodes.Argument):
                self.removals["template_arguments_removed"] += 1
            elif isinstance(node, nodes.Comment):
                self.removals["comments_removed"] += 1
            elif isinstance(node, nodes.Wikilink):
                title = str(node.title).strip()
                namespace = title.lstrip(":").partition(":")[0].casefold()
                if namespace in {"file", "image", "media"} and ":" in title:
                    self.removals["media_links_removed"] += 1
                elif namespace == "category" and ":" in title:
                    self.removals["category_links_removed"] += 1
                elif (
                    ":" in title
                    and namespace in LANGUAGE_PREFIXES | INTERWIKI_PREFIXES
                    and not title.startswith(":")
                    and node.text is None
                ):
                    self.removals["interwiki_links_removed"] += 1
                else:
                    visible = node.text if node.text is not None else node.title
                    rendered = self.render(visible)
                    if title.startswith(":") and node.text is None:
                        rendered = rendered.lstrip(":")
                    parts.append(rendered)
            elif isinstance(node, nodes.ExternalLink):
                if node.title is not None and str(node.title).strip():
                    parts.append(self.render(node.title))
                else:
                    self.removals["unlabeled_external_links_removed"] += 1
            elif isinstance(node, nodes.Heading):
                parts.append("\n\n" + self.render(node.title).strip() + "\n\n")
            elif isinstance(node, nodes.HTMLEntity):
                parts.append(node.normalize())
            elif isinstance(node, nodes.Tag):
                tag = str(node.tag).strip().casefold()
                if tag == "ref":
                    self.removals["references_removed"] += 1
                elif tag == "references":
                    self.removals["reference_lists_removed"] += 1
                elif tag == "table":
                    # Table cells cannot reliably be rendered as natural prose.
                    self.removals["tables_removed"] += 1
                    parts.append("\n\n")
                elif tag in {"script", "style", "gallery", "imagemap"}:
                    self.removals["nonvisible_tags_removed"] += 1
                elif tag in {"br", "hr"}:
                    parts.append("\n")
                elif node.wiki_markup in {"*", "#", ";", ":"}:
                    # The parser emits list markers as self-closing tags.
                    # Keep a definition separator when ':' occurs after a term.
                    current_line = "".join(parts).rsplit("\n", 1)[-1]
                    if node.wiki_markup == ":" and current_line.strip():
                        parts.append(": ")
                else:
                    visible = self.render(node.contents)
                    if tag in BLOCK_TAGS:
                        parts.append("\n" + visible + "\n")
                    else:
                        # Includes bold/italic tags and small/span wrappers.
                        parts.append(visible)
            else:
                # Unexpected parser nodes must not silently lose useful prose.
                raise ValueError(f"Unsupported parser node: {type(node).__name__}")
        return "".join(parts)


def clean_wikitext(text):
    """Return readable text and removal diagnostics; perform no I/O."""
    code = mwparserfromhell.parse(text)
    renderer = WikitextRenderer()
    rendered = renderer.render(code)
    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip()
             for line in rendered.split("\n")]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned, dict(renderer.removals)


def remaining_markup(text):
    return {
        "Templates remaining": text.count("{{"),
        "Wikilink markup remaining": text.count("[[") + text.count("]]"),
        "Ref markup remaining": len(re.findall(r"<\s*/?\s*ref\b", text, re.I)),
        "Bold/italic markup remaining": len(re.findall(r"'{2,}", text)),
        "Heading markup remaining": len(re.findall(
            r"^\s*={2,}.*?={2,}\s*$", text, re.M
        )),
        "Table markup remaining": len(re.findall(
            r"^\s*(?:\{\||\|\}|\|-|\|\+)", text, re.M
        )),
        "HTML tags remaining": len(re.findall(
            r"</?[A-Za-z][^>\n]*>", text
        )),
        "List markup remaining": len(re.findall(
            r"^\s*[*#;:]+(?:\s|$)", text, re.M
        )),
    }


def valid_schema(document):
    return (
        set(document) == SCHEMA_KEYS
        and isinstance(document["id"], str)
        and document["source"] == "wikipedia"
        and isinstance(document["text"], str)
        and isinstance(document["metadata"], dict)
    )


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def print_report(original, output, checks, markup):
    cleaning = output["metadata"]["cleaning"]
    print("=" * 50)
    print("WIKIPEDIA CLEANING — ONE DOCUMENT TEST")
    print("=" * 50)
    print(f"\nArticle:\n{EXPECTED_TITLE}\n\nID:\n{output['id']}\n")
    for label, key in (
        ("Original characters", "original_characters"),
        ("Cleaned characters", "cleaned_characters"),
        ("Characters removed", "characters_removed"),
    ):
        print(f"{label}: {cleaning[key]}")
    print(f"Percentage removed: {cleaning['percentage_removed']:.2f}%\n")
    for label, count in markup.items():
        print(f"{label}: {count}")
    print("\nRequired phrase checks:")
    for label, passed in checks.items():
        print(f"{label}: {'PASS' if passed else 'FAIL'}")
    print(f"\nInput SHA256 before: {cleaning['input_sha256']}")
    print(f"Input SHA256 after:  {sha256(INPUT_PATH.read_bytes()).hexdigest()}")
    print(f"Parser: mwparserfromhell {mwparserfromhell.__version__}")
    print("Removal diagnostics: " + json.dumps(cleaning["removals"]))
    if cleaning["percentage_removed"] >= 50:
        print("Large reduction: manually inspect the entire cleaned text.")
    print(f"\nRESULT: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("\n" + "=" * 50)
    print("CLEANED PREVIEW (first 1000 characters)\n")
    print(output["text"][:1000])


def main():
    original_bytes = INPUT_PATH.read_bytes()
    before_hash = sha256(original_bytes).hexdigest()
    original = json.loads(original_bytes.decode("utf-8"))
    if not valid_schema(original):
        raise ValueError("The standardized sample has an unexpected schema.")
    if (original["id"] != EXPECTED_ID
            or original["metadata"].get("title") != EXPECTED_TITLE):
        raise ValueError("Refusing to clean a different Wikipedia document.")
    if "cleaning" in original["metadata"]:
        raise ValueError("The immutable input is already marked as cleaned.")

    cleaned, removals = clean_wikitext(original["text"])
    original_characters = len(original["text"])
    removed = original_characters - len(cleaned)
    output = deepcopy(original)
    output["text"] = cleaned
    output["metadata"]["cleaning"] = {
        "stage": "wikipedia_wikitext_v1",
        "parser": "mwparserfromhell",
        "parser_version": mwparserfromhell.__version__,
        "original_characters": original_characters,
        "cleaned_characters": len(cleaned),
        "characters_removed": removed,
        "percentage_removed": round(removed / original_characters * 100, 4),
        "input_sha256": before_hash,
        "removals": removals,
    }
    markup = remaining_markup(cleaned)
    checks = {
        "Edward Backhouse": EXPECTED_TITLE in cleaned,
        "Quaker": ("Quaker" not in original["text"] or "Quaker" in cleaned),
        "Non-empty text": bool(cleaned.strip()),
        "Meaningful prose remains": (
            "was a Quaker philanthropist and writer on church history."
            in cleaned
        ),
        "Paragraphs and newlines preserved": len(cleaned.splitlines()) > 1,
        "Markup removed": not any(markup.values()),
        "JSON schema valid": valid_schema(output),
        "ID preserved": output["id"] == original["id"],
        "Source preserved": output["source"] == original["source"],
        "Metadata preserved": (
            {key: value for key, value in output["metadata"].items()
             if key != "cleaning"} == original["metadata"]
        ),
        "Standardized input unchanged": (
            sha256(INPUT_PATH.read_bytes()).hexdigest() == before_hash
        ),
    }
    if not all(checks.values()):
        print_report(original, output, checks, markup)
        raise SystemExit("Validation failed; no cleaned artifacts written.")

    payload = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    atomic_write(OUTPUT_PATH, payload)
    atomic_write(
        COMPARISON_PATH,
        "===== ORIGINAL PREVIEW =====\n\n" + original["text"][:1200]
        + "\n\n===== CLEANED PREVIEW =====\n\n" + cleaned[:1200] + "\n",
    )
    checks["Saved JSON verified"] = json.loads(
        OUTPUT_PATH.read_text(encoding="utf-8")
    ) == output
    checks["Standardized input unchanged"] = (
        sha256(INPUT_PATH.read_bytes()).hexdigest() == before_hash
    )
    print_report(original, output, checks, markup)
    if not all(checks.values()):
        raise SystemExit("Post-write validation failed; inspect the report.")


if __name__ == "__main__":
    main()

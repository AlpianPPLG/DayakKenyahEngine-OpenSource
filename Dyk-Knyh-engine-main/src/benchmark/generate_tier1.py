"""
ILTE-Bench — Tier 1 Golden Set Generator
========================================

Generates the tier-1 (lexical) golden set directly from the ILTE dictionary.
Every usable dictionary entry becomes two test cases (id->dyk and dyk->id),
so this tier costs zero human effort and acts as a regression net for
dictionary lookups and fuzzy matching. It does NOT test context awareness —
that is what tier 2/3 (human-validated) are for.

Usage: python generate_tier1.py [--dict PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_DICT = BENCH_DIR.parent / "dict" / "dictionary_alt.json"
DEFAULT_OUT = BENCH_DIR / "golden_set" / "tier1_lexical.json"

WORD_RE = re.compile(r"^[a-z][a-z'\u00e0-\u00ff]*$")


def normalize(text: str) -> str:
    """NFC-normalize, strip, and lowercase a dictionary entry."""
    return unicodedata.normalize("NFC", str(text)).strip().lower()


def is_testable(key: str, value: str) -> bool:
    """Whether a dictionary pair is usable as a lexical test case."""
    if not key or not value:
        return False
    if key == value:  # self-identity: would score 1.0 without translation
        return False
    if key.isdigit() or value.isdigit():  # numeric pairs are not word lookups
        return False
    if "<" in key or "<" in value:  # markup noise
        return False
    return WORD_RE.match(key) is not None and WORD_RE.match(value) is not None


def build_entries(dictionary: dict[str, str]) -> list[dict]:
    """Convert the dictionary into tier-1 test cases (both directions)."""
    entries: list[dict] = []
    for raw_key, raw_value in dictionary.items():
        key, value = normalize(raw_key), normalize(raw_value)
        if not is_testable(key, value):
            continue
        entries.append(
            {
                "source": key,
                "reference": value,
                "source_lang": "id",
                "target_lang": "dyk",
                "entry": key,
                "tier": 1,
            }
        )
        entries.append(
            {
                "source": value,
                "reference": key,
                "source_lang": "dyk",
                "target_lang": "id",
                "entry": key,
                "tier": 1,
            }
        )
    # Stable, reproducible order.
    entries.sort(key=lambda e: (e["source_lang"], e["entry"], e["source"]))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ILTE tier-1 golden set")
    parser.add_argument("--dict", dest="dict_path", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.dict_path.exists():
        raise SystemExit(f"Dictionary not found: {args.dict_path}")

    with open(args.dict_path, "r", encoding="utf-8") as fh:
        dictionary = json.load(fh)

    entries = build_entries(dictionary)
    stats = {
        "dictionary_entries": len(dictionary),
        "numeric_skipped": sum(1 for k in dictionary if normalize(k).isdigit()),
        "identical_skipped": sum(
            1 for k, v in dictionary.items() if normalize(k) == normalize(v)
        ),
        "test_cases": len(entries),
        "id_to_dyk": sum(1 for e in entries if e["source_lang"] == "id"),
        "dyk_to_id": sum(1 for e in entries if e["source_lang"] == "dyk"),
    }

    print("Tier-1 generation summary")
    for name, value in stats.items():
        print(f"  {name.replace('_', ' '):20}: {value}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"tier": 1, "name": "lexical", "stats": stats}, "entries": entries}
    with open(args.out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()

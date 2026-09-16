"""
ILTE-Bench — Smoke Tests (dependency-free)
==========================================

Plain-assert tests so CI or a laptop without pytest can still verify the
benchmark core. Run from anywhere:

    python src/benchmark/tests/test_metrics.py

Covers:
* chrF++ edge cases (perfect, partial, disjoint, empty, multi-word)
* BLEU basic behavior
* exact_match_rate
* generate_tier1 entry invariants (against the committed golden set, if present)
* adapter contract for the always-available DUMMY engine
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent  # src/benchmark
sys.path.insert(0, str(BENCH_DIR))

from metrics import bleu_score, chrfpp_score, exact_match_rate  # noqa: E402
from adapters import create_adapter  # noqa: E402

GOLDEN_TIER1 = BENCH_DIR / "golden_set" / "tier1_lexical.json"


def test_chrfpp_perfect() -> None:
    assert chrfpp_score("inu", "inu") == 1.0
    assert chrfpp_score("ca' elas dua", "ca' elas dua") == 1.0


def test_chrfpp_edge_cases() -> None:
    # Empty sides.
    assert chrfpp_score("", "inu") == 0.0
    assert chrfpp_score("inu", "") == 0.0
    # Total mismatch must be strictly below a one-character-off match.
    total = chrfpp_score("bala", "saleng")
    partial = chrfpp_score("ino", "inu")
    assert 0.0 <= total < partial < 1.0
    # Order of arguments must not change the score.
    assert chrfpp_score("ino", "inu") == chrfpp_score("inu", "ino")


def test_bleu_basic() -> None:
    assert bleu_score("inu", "inu") == 1.0
    assert bleu_score("", "inu") == 0.0
    assert bleu_score("bala", "saleng") < bleu_score("inu", "inu")


def test_exact_match_rate() -> None:
    pairs = [("inu", "inu"), ("bala", "saleng")]
    assert exact_match_rate(pairs) == 0.5
    assert exact_match_rate([]) == 0.0
    # Normalization: case and surrounding whitespace.
    assert exact_match_rate([("  Inu ", "INU")]) == 1.0


def test_tier1_golden_set_invariants() -> None:
    if not GOLDEN_TIER1.exists():
        print("  (golden set not generated yet — skipping)")
        return
    payload = json.loads(GOLDEN_TIER1.read_text(encoding="utf-8"))
    entries = payload["entries"]
    assert payload["meta"]["tier"] == 1
    assert len(entries) > 0
    for e in entries:
        assert e["source"] and e["reference"]
        assert e["source"] != e["reference"]
        assert {e["source_lang"], e["target_lang"]} == {"id", "dyk"}
    # Every id->dyk case must have a mirrored dyk->id case.
    fwd = {(e["source"], e["reference"]) for e in entries if e["source_lang"] == "id"}
    rev = {(e["reference"], e["source"]) for e in entries if e["source_lang"] == "dyk"}
    assert fwd == rev


def test_dummy_adapter_contract() -> None:
    adapter = create_adapter("dummy")
    assert adapter.supports("id", "dyk") and adapter.supports("dyk", "id")
    assert not adapter.supports("dyk", "en")
    out = adapter.translate_text("apa", "id", "dyk")
    assert out == "apa"
    # Unsupported direction must raise, not silently copy.
    try:
        adapter.translate_text("apa", "dyk", "en")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported direction")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

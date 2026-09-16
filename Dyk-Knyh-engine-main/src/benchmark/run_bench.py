"""
ILTE-Bench — Benchmark Runner
=============================

Runs a tier golden set through selected engines and emits a leaderboard.

Usage examples:
    python run_bench.py --engine dummy
    python run_bench.py --engine dummy alt --direction id-dyk --limit 20
    python run_bench.py --engine dummy --all-directions --out results/
    python run_bench.py --engine ati --direction dyk-id --tier 1

Outputs:
    results/leaderboard_<timestamp>.md   human-readable leaderboard
    results/bench_<timestamp>.json       full per-pair detail (for future tiers)

Notes:
* The DUMMY adapter (copy source) is the floor; a real engine should beat it.
* Only tier 1 exists today. Tier 2/3 golden sets land in golden_set/ with the
  same entry schema and the runner picks them up automatically.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from adapters import DEFAULT_DICT, available_engines, create_adapter
from metrics import PairResult, chrfpp_score, bleu_score, summarize_pairs

BENCH_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = BENCH_DIR / "golden_set"
RESULTS_DIR = BENCH_DIR / "results"

DIRECTIONS = {
    "id-dyk": ("id", "dyk"),
    "dyk-id": ("dyk", "id"),
}


def load_tier(tier: int) -> list[dict]:
    """Load a tier golden set by tier number (tier 1 only today)."""
    path = GOLDEN_DIR / f"tier{tier}_lexical.json"
    if tier != 1 or not path.exists():
        raise SystemExit(
            f"Tier {tier} golden set not found (only tier 1 exists; "
            f"run generate_tier1.py)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["entries"]


def run_engine_on_cases(
    adapter, cases: list[dict], source_lang: str, target_lang: str
) -> list[PairResult]:
    """Translate every case with one adapter and score the outputs."""
    results: list[PairResult] = []
    for case in cases:
        start = time.perf_counter()
        error = None
        hypothesis = ""
        try:
            hypothesis = adapter.translate_text(case["source"], source_lang, target_lang)
        except Exception:
            error = traceback.format_exc(limit=2)
        latency_ms = (time.perf_counter() - start) * 1000.0

        results.append(
            PairResult(
                source=case["source"],
                reference=case["reference"],
                hypothesis="" if error else hypothesis,
                source_lang=source_lang,
                target_lang=target_lang,
                chrfpp=0.0 if error else chrfpp_score(hypothesis, case["reference"]),
                bleu=0.0 if error else bleu_score(hypothesis, case["reference"]),
                latency_ms=latency_ms,
                error=error,
            )
        )
    return results


def run_engine(
    engine_key: str,
    entries: list[dict],
    directions: list[str],
    limit: int | None,
) -> list:
    """Run one engine across requested directions; skip unsupported ones."""
    results = []
    for direction in directions:
        source_lang, target_lang = DIRECTIONS[direction]

        cases = [
            e for e in entries
            if e["source_lang"] == source_lang and e["target_lang"] == target_lang
        ]
        if limit:
            cases = cases[:limit]
        if not cases:
            print(f"  [{engine_key}] no cases for {direction}, skipping")
            continue

        kwargs = {"source_lang": source_lang, "target_lang": target_lang}
        try:
            adapter = create_adapter(engine_key, dict_path=DEFAULT_DICT, **kwargs)
        except Exception:
            print(f"  [{engine_key}] adapter construction failed, skipping")
            traceback.print_exc(limit=2)
            continue

        # Probe capability without loading anything.
        if not adapter.supports(source_lang, target_lang):
            print(f"  [{engine_key}] does not support {direction}, skipping")
            continue

        try:
            load_time = adapter.ensure_loaded()
        except Exception:
            print(f"  [{engine_key}] engine failed to load for {direction}, skipping")
            traceback.print_exc(limit=2)
            continue

        print(
            f"  [{engine_key}] {direction}: translating {len(cases)} cases "
            f"(load {load_time:.1f}s)..."
        )
        pairs = run_engine_on_cases(adapter, cases, source_lang, target_lang)
        n_err = sum(1 for p in pairs if p.error)
        if n_err:
            print(f"  [{engine_key}] {direction}: {n_err} case errors")
        results.append((direction, pairs, load_time))

        # Free heavy models between directions to keep RAM/VRAM sane.
        adapter.engine = None
    return results


def write_leaderboard(
    engine_results: dict[str, list],
    tier: int,
    out_dir: Path,
    limit: int | None,
) -> Path:
    """Render the markdown leaderboard."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"leaderboard_{stamp}.md"

    lines = [
        "# ILTE-Bench Leaderboard",
        "",
        f"- Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"- Tier      : {tier}",
        f"- Cases     : {limit if limit else 'all'}",
        "",
        "chrF++ is the headline metric because DYK is low-resource and",
        "morphology-sensitive; see src/benchmark/metrics.py for rationale.",
        "",
        "Note for tier 1: cases are single words, so BLEU's +1 smoothing",
        "floors a total miss at 0.500. A BLEU of 0.5 here means completely",
        "wrong, not half right. Trust chrF++ for this tier.",
        "",
        "| Engine | Direction | Cases | Errors | chrF++ | BLEU | Exact | Avg ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for engine_key, dir_results in engine_results.items():
        for direction, pairs, _load_time in dir_results:
            summary = summarize_pairs(engine_key, pairs)
            d = summary.directions[0]
            lines.append(
                f"| {engine_key} | {direction} | {d.n_cases} | {d.n_errors} "
                f"| {d.chrfpp_avg:.3f} | {d.bleu_avg:.3f} | {d.exact_rate:.1%} "
                f"| {d.latency_ms_avg:.0f} |"
            )

    # Macro-average row per engine across directions.
    lines += ["", "## Macro average (all directions equally weighted)", ""]
    lines.append("| Engine | chrF++ | Exact |")
    lines.append("|---|---:|---:|")
    ranked = []
    for engine_key, dir_results in engine_results.items():
        all_pairs = [p for _d, ps, _t in dir_results for p in ps]
        if not all_pairs:
            continue
        summary = summarize_pairs(engine_key, all_pairs)
        ranked.append((engine_key, summary.chrfpp_macro, summary.exact_rate_overall))
    for engine_key, macro, exact in sorted(ranked, key=lambda r: -r[1]):
        lines.append(f"| {engine_key} | {macro:.3f} | {exact:.1%} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_detail_json(engine_results: dict[str, list], tier: int, out_dir: Path) -> Path:
    """Persist full per-pair detail for future analysis (tier 2/3, calibration)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench_{stamp}.json"

    payload = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "tier": tier,
        },
        "engines": {},
    }
    for engine_key, dir_results in engine_results.items():
        engine_block = {}
        for direction, pairs, load_time in dir_results:
            engine_block[direction] = {
                "load_time_s": round(load_time, 2),
                "pairs": [
                    {
                        "source": p.source,
                        "reference": p.reference,
                        "hypothesis": p.hypothesis,
                        "chrfpp": round(p.chrfpp, 4),
                        "bleu": round(p.bleu, 4),
                        "latency_ms": round(p.latency_ms, 1),
                        "error": p.error,
                    }
                    for p in pairs
                ],
            }
        payload["engines"][engine_key] = engine_block

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ILTE benchmark")
    parser.add_argument(
        "--engine", nargs="+", default=["dummy"],
        help=f"engines to run: {available_engines()}",
    )
    parser.add_argument("--tier", type=int, default=1, help="golden set tier")
    parser.add_argument(
        "--direction", nargs="*", default=["id-dyk"],
        choices=list(DIRECTIONS), help="translation direction(s) to test",
    )
    parser.add_argument("--all-directions", action="store_true",
                        help="run both id->dyk and dyk->id")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of cases per direction (smoke runs)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR,
                        help="output directory for results")
    args = parser.parse_args()

    directions = ["id-dyk", "dyk-id"] if args.all_directions else args.direction

    entries = load_tier(args.tier)
    print(f"Loaded tier {args.tier} golden set: {len(entries)} entries")
    print(f"Engines: {', '.join(args.engine)} | Directions: {', '.join(directions)}")

    engine_results: dict[str, list] = {}
    for engine_key in args.engine:
        print(f"Running engine: {engine_key}")
        try:
            engine_results[engine_key] = run_engine(
                engine_key, entries, directions, args.limit
            )
        except Exception:
            print(f"  [{engine_key}] unexpected failure, engine skipped")
            traceback.print_exc(limit=3)
            engine_results[engine_key] = []

    if not any(pairs for res in engine_results.values() for _d, pairs, _t in res):
        # Non-zero exit so CI treats "every engine failed" as a red run.
        print("\nNo results produced. Nothing to rank.")
        raise SystemExit(1)

    lb_path = write_leaderboard(engine_results, args.tier, args.out, args.limit)
    json_path = write_detail_json(engine_results, args.tier, args.out)
    print(f"\nLeaderboard written to: {lb_path}")
    print(f"Detail JSON written to: {json_path}")
    print("\nLeaderboard:")
    print(lb_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

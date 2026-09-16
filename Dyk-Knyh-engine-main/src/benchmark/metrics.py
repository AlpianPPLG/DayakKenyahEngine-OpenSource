"""
ILTE-Bench — Evaluation Metrics
===============================

Dependency-free evaluation metrics for the ILTE benchmark.

Why chrF++ as the headline metric?
----------------------------------
ILTE translates between Indonesian and Dayak Kenyah (DYK), a low-resource
language. For such languages:

* BLEU works on word tokens. DYK is likely morphologically rich, so one wrong
  affix marks the whole word as wrong -> BLEU collapses to near zero and
  stops being informative.
* chrF++ operates on character n-grams plus word n-grams ("++" adds the word
  level). It is robust to partial/morphological matches, needs no tokenization
  training data, and correlates better with human judgement on low-resource
  pairs (Popovic 2017). This matches how closely a DYK word is spelled.

We implement chrF++ from scratch (deterministic, no external deps); BLEU is
provided only as a secondary reference metric, also from scratch, following
Papineni et al. (2002) with smoothing so short sentences do not zero out.

All scores are in the 0..1 range unless noted otherwise.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# chrF++ (character n-gram F-score + word n-gram bonus)
# ---------------------------------------------------------------------------

def _ngrams(seq: list[str], n: int) -> Counter:
    """Counter of n-grams of length n over a pre-tokenized sequence."""
    if n <= 0 or len(seq) < n:
        return Counter()
    return Counter(
        tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)
    )


def _f_score(hyp_counter: Counter, ref_counter: Counter) -> float:
    """n-gram F-score with the chrF convention of beta=2 (recall-weighted)."""
    beta = 2.0
    if not hyp_counter or not ref_counter:
        return 0.0

    overlap = sum((hyp_counter & ref_counter).values())  # multiset intersection
    hyp_total = sum(hyp_counter.values())
    ref_total = sum(ref_counter.values())

    precision = overlap / hyp_total
    recall = overlap / ref_total
    if precision + recall == 0.0:
        return 0.0

    beta_sq = beta * beta
    return (
        (1 + beta_sq)
        * precision
        * recall
        / (beta_sq * precision + recall)
    )


def chrfpp_score(hypothesis: str, reference: str) -> float:
    """
    chrF++ — averaged F-scores (beta=2, recall-weighted) over character
    n-grams (1..6) plus word n-grams (1..2), the "+" in chrF++.

    Follows sacrebleu conventions (POP01-style character extraction) so
    scores are comparable with `sacrebleu -m chrf`.
    """
    hyp = hypothesis.strip()
    ref = reference.strip()
    if not hyp or not ref:
        return 1.0 if hyp == ref else 0.0

    def char_tokens(text: str) -> list[str]:
        # POP01: split on whitespace and split each token into characters.
        return [ch for tok in text.split() for ch in tok]

    def word_tokens(text: str) -> list[str]:
        return text.split()

    char_scores = []
    for n in range(1, 7):
        hyp_ng = _ngrams(char_tokens(hyp), n)
        ref_ng = _ngrams(char_tokens(ref), n)
        if not ref_ng:
            # Order longer than the reference is not applicable; skip it
            # (same convention as sacrebleu) instead of counting a zero.
            break
        char_scores.append(_f_score(hyp_ng, ref_ng))

    word_scores = []
    for n in range(1, 3):
        hyp_ng = _ngrams(word_tokens(hyp), n)
        ref_ng = _ngrams(word_tokens(ref), n)
        if not ref_ng:
            break
        word_scores.append(_f_score(hyp_ng, ref_ng))

    # Sacrebleu-style weighting: word n-grams contribute 2x total weight.
    avg_char = sum(char_scores) / len(char_scores)
    avg_word = sum(word_scores) / len(word_scores) if word_scores else 0.0
    return (avg_char + avg_word) / 2.0


# ---------------------------------------------------------------------------
# BLEU (secondary reference metric, with add-smoothing)
# ---------------------------------------------------------------------------

def bleu_score(hypothesis: str, reference: str, max_order: int = 4) -> float:
    """
    Single-pair BLEU with +1 smoothing. Reference metric only; chrF++ is
    the headline for DYK.
    """
    hyp_tokens = hypothesis.strip().split()
    ref_tokens = reference.strip().split()
    if not hyp_tokens or not ref_tokens:
        return 1.0 if hypothesis.strip() == reference.strip() else 0.0

    precisions: list[float] = []
    for n in range(1, max_order + 1):
        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        n_count = sum(hyp_ngrams.values())
        if n_count == 0:
            # Order longer than the hypothesis: no such n-grams exist, so
            # contribute nothing (filtered out of the geometric mean below).
            precisions.append(0.0)
            continue
        precisions.append(1.0 / (n_count + 1) if overlap == 0 else overlap / n_count)

    if all(p == 0 for p in precisions):
        return 0.0

    log_sum = sum(math.log(p) for p in precisions if p > 0)
    brevity = min(0.0, 1.0 - len(ref_tokens) / len(hyp_tokens))
    return math.exp(log_sum + brevity)


# ---------------------------------------------------------------------------
# Lexical hit-rate
# ---------------------------------------------------------------------------

def exact_match_rate(pairs: list[tuple[str, str]]) -> float:
    """Fraction of (hypothesis, reference) pairs that match after normalization."""
    if not pairs:
        return 0.0
    hits = sum(1 for h, r in pairs if h.strip().lower() == r.strip().lower())
    return hits / len(pairs)


# ---------------------------------------------------------------------------
# Per-engine aggregates
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """Scored outcome of a single test case."""
    source: str
    reference: str
    hypothesis: str
    source_lang: str
    target_lang: str
    chrfpp: float
    bleu: float
    latency_ms: float
    error: str | None = None


@dataclass
class DirectionSummary:
    """Aggregate scores for one translation direction of one engine."""
    source_lang: str
    target_lang: str
    n_cases: int = 0
    n_errors: int = 0
    chrfpp_avg: float = 0.0
    bleu_avg: float = 0.0
    exact_rate: float = 0.0
    latency_ms_avg: float = 0.0


@dataclass
class EngineSummary:
    """Full benchmark result for one engine."""
    engine: str
    directions: list[DirectionSummary] = field(default_factory=list)
    details: list[PairResult] = field(default_factory=list)

    @property
    def chrfpp_macro(self) -> float:
        """Macro-average of per-direction chrF++ (directions weighted equally)."""
        valid = [d for d in self.directions if d.n_cases > 0]
        if not valid:
            return 0.0
        return sum(d.chrfpp_avg for d in valid) / len(valid)

    @property
    def exact_rate_overall(self) -> float:
        if not self.details:
            return 0.0
        ok_pairs = [
            (p.hypothesis, p.reference) for p in self.details if p.error is None
        ]
        return exact_match_rate(ok_pairs)


def summarize_pairs(engine_name: str, pairs: list[PairResult]) -> EngineSummary:
    """Group pair results by direction and compute averages."""
    engine = EngineSummary(engine=engine_name, details=pairs)

    by_direction: dict[tuple[str, str], list[PairResult]] = {}
    for p in pairs:
        by_direction.setdefault((p.source_lang, p.target_lang), []).append(p)

    for (src, tgt), group in sorted(by_direction.items()):
        ok_group = [p for p in group if p.error is None]
        d = DirectionSummary(
            source_lang=src,
            target_lang=tgt,
            n_cases=len(group),
            n_errors=len(group) - len(ok_group),
        )
        if ok_group:
            d.chrfpp_avg = sum(p.chrfpp for p in ok_group) / len(ok_group)
            d.bleu_avg = sum(p.bleu for p in ok_group) / len(ok_group)
            d.latency_ms_avg = sum(p.latency_ms for p in ok_group) / len(ok_group)
            d.exact_rate = exact_match_rate([(p.hypothesis, p.reference) for p in ok_group])
        engine.directions.append(d)

    return engine

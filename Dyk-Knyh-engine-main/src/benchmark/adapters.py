"""
ILTE-Bench — Engine Adapters
============================

Wraps every ILTE engine behind one uniform interface so the benchmark runner
never touches engine internals:

    adapter = create_adapter("ATI", dict_path=..., source_lang="id", target_lang="dyk")
    adapter.ensure_loaded()                       # may download heavy models
    adapter.translate_text("apa", "id", "dyk")    # -> str

Design notes (based on how the engines are actually built):

* ALT/ADV load their Helsinki MT pipelines eagerly in __init__ and only
  support the dyk pivot (en->id or dyk->id->en). Calling them with
  (dyk, id) hits a code path that was never meant to be a reverse lookup,
  so the adapter advertises the honest capability matrix instead of
  pretending all four engines are interchangeable.
* ATI/ZS construct with (dict_path, source_lang, target_lang) and flip the
  dictionary internally for dyk->id. They are constructed per-direction.
* ATI/ZS are always created with lazy_load=True: models load on first use
  via @property, which lets us measure load time and skip model downloads
  in --quick mode.
* ALT/ADV print error tracebacks via print/logger inside translate_text on
  failure and return partial text; the adapter cannot see that, so
  translation success is judged by the metrics, not by exceptions.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make engine modules importable regardless of where the benchmark is run from.
ENGINES_DIR = Path(__file__).resolve().parent.parent / "engines"
if str(ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINES_DIR))

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_DICT = BENCH_DIR.parent / "dict" / "dictionary_alt.json"


@dataclass
class AdapterMeta:
    """Static capability facts about an engine."""
    key: str                    # short key used on the CLI ("alt", "dummy", ...)
    supported: frozenset[tuple[str, str]]  # (source, target) pairs it can really do


class BaseAdapter:
    """Uniform interface over an ILTE engine."""

    meta: AdapterMeta

    def __init__(self, dict_path: Path | str = DEFAULT_DICT) -> None:
        self.dict_path = Path(dict_path)
        self.load_time_s: float = 0.0
        self._loaded = False

    # -- interface ---------------------------------------------------------

    def _build(self) -> object:
        """Construct the underlying engine object. Must be implemented."""
        raise NotImplementedError

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Call the underlying engine's translate_text. Must be implemented."""
        raise NotImplementedError

    # -- shared logic ------------------------------------------------------

    def ensure_loaded(self) -> float:
        """Construct the engine once; returns wall-clock load time in seconds."""
        if self._loaded:
            return self.load_time_s
        start = time.perf_counter()
        self.engine = self._build()
        self.load_time_s = time.perf_counter() - start
        self._loaded = True
        return self.load_time_s

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if (source_lang, target_lang) not in self.meta.supported:
            raise ValueError(
                f"{self.meta.key.upper()} does not support "
                f"{source_lang}->{target_lang}; supported: "
                f"{sorted(self.meta.supported)}"
            )
        self.ensure_loaded()
        return self._translate(text, source_lang, target_lang)

    def supports(self, source_lang: str, target_lang: str) -> bool:
        return (source_lang, target_lang) in self.meta.supported


# ---------------------------------------------------------------------------
# DUMMY baseline (always available, zero dependencies)
# ---------------------------------------------------------------------------

class DummyAdapter(BaseAdapter):
    """
    Deterministic baseline: copies the source text.
    chrF++ of the copy baseline is the floor every real engine must beat.
    """

    def __init__(self, dict_path: Path | str = DEFAULT_DICT) -> None:
        super().__init__(dict_path)
        self.meta = AdapterMeta(
            key="dummy",
            supported=frozenset({("id", "dyk"), ("dyk", "id"), ("id", "en"), ("en", "id")}),
        )

    def _build(self) -> object:
        return None

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return text


# ---------------------------------------------------------------------------
# Real engines
# ---------------------------------------------------------------------------

class AltAdapter(BaseAdapter):
    """
    ILTE-ALT — dictionary + Levenshtein, Helsinki MT pipelines loaded eagerly.

    ALT's dyk->id path runs the raw DYK word list through the id->en MT model,
    which is a known design gap, so only id->dyk is advertised as supported.
    """

    def __init__(self, dict_path: Path | str = DEFAULT_DICT) -> None:
        super().__init__(dict_path)
        self.meta = AdapterMeta(
            key="alt",
            supported=frozenset({("id", "dyk")}),
        )

    def _build(self) -> object:
        from engine_ALT import IndigenousTranslator  # type: ignore[import-not-found]

        return IndigenousTranslator(str(self.dict_path))

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.engine.translate_text(text, source_lang, target_lang)


class AdvAdapter(BaseAdapter):
    """
    ILTE-ADV — BERT + SentenceTransformer context engine.
    Same capability matrix as ALT (dyk pivot only, id->dyk advertised).
    """

    def __init__(self, dict_path: Path | str = DEFAULT_DICT) -> None:
        super().__init__(dict_path)
        self.meta = AdapterMeta(
            key="adv",
            supported=frozenset({("id", "dyk")}),
        )

    def _build(self) -> object:
        from engine_ADV import OptimizedIndigenousTranslator  # type: ignore[import-not-found]

        return OptimizedIndigenousTranslator(str(self.dict_path))

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.engine.translate_text(text, source_lang, target_lang)


class _LazyDirectionalAdapter(BaseAdapter):
    """Shared plumbing for ATI/ZS: one engine instance per direction."""

    engine_module: str
    engine_class: str
    key: str

    def __init__(
        self,
        dict_path: Path | str = DEFAULT_DICT,
        source_lang: str = "id",
        target_lang: str = "dyk",
    ) -> None:
        super().__init__(dict_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.meta = AdapterMeta(
            key=self.key,
            supported=frozenset({(source_lang, target_lang)}),
        )

    def _build(self) -> object:
        module = __import__(self.engine_module, fromlist=[self.engine_class])
        cls = getattr(module, self.engine_class)
        return cls(
            str(self.dict_path),
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            lazy_load=True,  # heavy models load on first translate via @property
        )

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.engine.translate_text(text, source_lang, target_lang)


class AtiAdapter(_LazyDirectionalAdapter):
    """ILTE-ATI v3.0.0-Alpha.3 — iterative attention engine (id<->dyk, per-direction)."""

    engine_module = "engine_ATI"
    engine_class = "EnhancedILTETranslationEngine"
    key = "ati"


class ZsAdapter(_LazyDirectionalAdapter):
    """ILTE-ZS v2.1.2-Beta.3 — hybrid RBMT/FST/zero-shot engine (id<->dyk, per-direction)."""

    engine_module = "engine_ZS"
    engine_class = "EnhancedILTEZSTranslationEngine"
    key = "zs"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def create_adapter(key: str, dict_path: Path | str = DEFAULT_DICT, **kwargs) -> BaseAdapter:
    """Build an adapter by short key; ATI/ZS take source_lang/target_lang kwargs. Unknown keys raise ValueError."""
    key = key.strip().lower()
    if key == "dummy":
        return DummyAdapter(dict_path)
    if key == "alt":
        return AltAdapter(dict_path)
    if key == "adv":
        return AdvAdapter(dict_path)
    if key == "ati":
        return AtiAdapter(dict_path, **kwargs)
    if key == "zs":
        return ZsAdapter(dict_path, **kwargs)
    raise ValueError(
        f"Unknown engine key '{key}'. Available: dummy, alt, adv, ati, zs"
    )


def available_engines() -> list[str]:
    return ["dummy", "alt", "adv", "ati", "zs"]

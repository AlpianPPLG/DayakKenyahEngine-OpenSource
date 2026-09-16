# 🧪 ILTE-Bench — Framework Evaluasi ILTE (Fase 0)

Benchmark dependensi-nol untuk mengukur kualitas keempat engine ILTE secara
objektif. Semua klaim README ("ATI paling akurat", "ZS paling robust")
akhirnya bisa diuji dengan angka.

## Struktur

```
benchmark/
├── adapters.py            # Adapter seragam: DUMMY, ALT, ADV, ATI, ZS
├── metrics.py             # chrF++ (utama), BLEU (referensi), exact match
├── run_bench.py           # Runner CLI + leaderboard markdown + JSON detail
├── generate_tier1.py      # Generator golden set tier 1 dari kamus
├── golden_set/
│   └── tier1_lexical.json # 858 test case (429 id→dyk + 429 dyk→id)
├── results/               # Output leaderboard & detail (timestamped)
└── tests/test_metrics.py  # Test suite tanpa dependency
```

## Cara pakai cepat

```sh
# 1. Generate golden set tier 1 (sekali, ulangi saat kamus berubah)
python generate_tier1.py

# 2. Baseline DUMMY (copy source) — pastikan pipeline bekerja
python run_bench.py --engine dummy

# 3. Engine sungguhan, arah id→dyk (20 kasus pertama sebagai smoke test)
python run_bench.py --engine alt --limit 20

# 4. Dua arah sekaligus
python run_bench.py --engine zs --all-directions

# 5. Test suite
python tests/test_metrics.py
```

Untuk engine selain DUMMY, jalankan dari environment yang sudah memasang
dependensi `src/docs/requirements.txt` (torch, transformers, Sastrawi, dst).
Model Helsinki/SentenceTransformer akan diunduh otomatis saat pertama kali.

> ⚠️ **Pin `transformers<5`** (mis. `pip install "transformers<5"`).
> Transformers v5 menghapus task pipeline `translation_*` dan tokenizer
> Marian — semua engine ILTE ditulis untuk API v4 dan akan gagal dengan
> `ValueError: Unrecognized configuration class ... MarianConfig` di v5.
> ATI/ZS juga butuh `malaya` yang menarik TensorFlow (~600 MB).

## Metrik

| Metrik | Peran | Catatan |
|---|---|---|
| **chrF++** | Utama | Robust untuk bahasa low-resource & morfologi DYK; konvensi sacrebleu (POP01, order char 1–6 + kata 1–2, β=2) |
| BLEU | Referensi | +1 smoothing; cenderung collapse untuk kata tunggal |
| Exact match | Sanity | Persentase jawaban persis sama |
| Avg ms | Performa | Latensi per kasus; waktu load model tercatat di JSON detail |

DUMMY (copy source) adalah **floor**: engine sungguhan wajib mengalahkannya.

## Matriks kemampuan (jujur sesuai kode engine)

| Engine | id→dyk | dyk→id | Catatan |
|---|---|---|---|
| ALT | ✅ | ❌ | Jalur dyk→id melewati MT id→en — gap desain |
| ADV | ✅ | ❌ | Pivot yang sama dengan ALT |
| ATI | ✅ | ✅ | Instans per arah; lazy-load model |
| ZS | ✅ | ✅ | Instans per arah; lazy-load model; MBart zero-shot |

ATI/ZS selalu dibuat dengan `lazy_load=True`; model berat dimuat saat
terjemahan pertama sehingga waktu load terukur realistis.

## Rencana lanjutan

- **Tier 2** (`golden_set/tier2_sentences.json`): 50–100 kalimat hasil
  wawancara penutur asli. Format sama dengan tier 1 → runner otomatis.
- **Tier 3** (`tier3_discourse.json`): paragraf dongeng + referensi manusia.
- **Analisis kalibrasi confidence**: apakah skor confidence engine berkorelasi
  dengan kebenaran (Spearman/Brier)?
- **Profil RAM/VRAM** via psutil untuk leaderboard biaya.
- **CI**: jalankan `tests/test_metrics.py` + `run_bench.py --engine dummy`
  sebagai regresi.

## Skema entri golden set

```json
{
  "source": "apa",
  "reference": "inu",
  "source_lang": "id",
  "target_lang": "dyk",
  "entry": "apa",
  "tier": 1
}
```

Tier 2/3 cukup mengikuti skema ini; `run_bench.py` tinggal didaftarkan
path barunya di `load_tier()`.

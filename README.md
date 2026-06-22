# Language Family Trees from Information Theory
**Complexity Lab group mini-project — Jonathan Cowley & Nil Doğan**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opalsaints/language-family-trees/blob/main/LanguageTrees.ipynb)

Rediscover the language family tree from raw text using information theory: character-trigram
**Jensen–Shannon divergence** + **gzip compression distance (NCD)** → UPGMA dendrogram, on the
parallel **UDHR** corpus. The cross-script case is handled with **uroman** romanization.

## Files
| file | what |
|---|---|
| `LanguageTrees.ipynb` | **The deliverable** — self-contained Colab tutorial. Run top-to-bottom; reproduces every figure. |
| `langtree.py` | Reusable functions (cleaning, n-gram dists, JS divergence, NCD, entropy ladder, trees). |
| `tier1.py` | Tier 1 runner — Latin-script tree + Shannon validation. |
| `tier2.py` | Tier 2 runner — mixed-script failure → uroman fix. |
| `build_notebook.py` | Regenerates `LanguageTrees.ipynb`. |
| `figures/` | Saved dendrograms. |

## Run
- **Colab (recommended):** upload `LanguageTrees.ipynb`, Runtime → Run all. (Installs `nltk uroman scipy matplotlib`.)
- **Local:** `pip install nltk uroman scipy matplotlib` then `python tier1.py` / `python tier2.py`.

## Key results
- **Shannon validation:** English F1 ≈ 4.08 bits/char (Shannon 1951: 4.03). F2/F3 biased low on short
  UDHR text = the finite-sample *estimation wall*.
- **Tier 1 (Latin script):** trees recover **Romance**, **Germanic** (West vs North), **Slavic**,
  **Uralic**; JS and NCD agree (r = 0.87). English clusters with **Romance** — the Norman-borrowing signal.
- **Tier 2 (cross-script):** raw text clusters **by writing system** (Kazakh pairs with Russian by
  Cyrillic; Hebrew–Arabic maximally distant). After **uroman**: nearest-neighbour-matches-family
  **0.44 → 1.00**; Hebrew–Arabic and Turkish–Kazakh become mutual nearest neighbours. Semitic & Turkic recovered.
- uroman renders Hebrew/Arabic as vowelless consonant skeletons (`כל בני האדם` → `kl vny hadm`) —
  Nil's vowel-redundancy point, surfacing in the pipeline.

## For Tuesday's progress review (Greg)
- "You suggested language similarity via divergence on letter n-grams — here's where we are."
- Show: the Latin-script tree (families recovered + English↔Romance borrowing) and the cross-script
  raw→romanized contrast (the failure and the uroman fix).
- Ask: how far to push it — add the gold-tree (Glottolog/lang2vec) Robinson–Foulds score? more languages?
  the MI-vs-distance/criticality arm (Tier 3)?

## Honest limits
Letter statistics capture **orthographic/surface** similarity, which tracks but ≠ true genealogy
(borrowing, orthography). Recovery is **partial** (close relatives robust, deep ties shaky). Not a
substitute for cognate-based comparative linguistics.

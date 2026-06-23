# Language Family Trees from Information Theory
**Complexity Lab group mini-project — Jonathan Cowley & Nil Doğan**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opalsaints/language-family-trees/blob/main/LanguageTrees.ipynb)

Can we **rediscover the family tree of languages from raw text and information theory alone** — no
dictionaries, no grammar, no hand-built features — *and prove the information theory does real work a
trivial baseline can't?* We represent each language by its character-trigram distribution, measure
**Jensen–Shannon divergence** on the **parallel Bible corpus** (same verses, so any difference is
between languages, not topics), build a tree, and score it against the Glottolog family tree.

> **Honest origin story.** A first version was *tied* on a small sample by a dumb baseline ("which
> characters does a language use" — no frequencies). We rebuilt on a large many-family corpus, then ran
> a **second critical review** that found several headline numbers were scored against impossible
> endpoints. This version fixes that: every tree-vs-gold number is reported against its real **floor**
> and **random null**, every purity against its **chance floor**, plus a quartet distance, a valid
> bootstrap, and a battery of independent cross-checks. See `CRITICAL_REVIEW_23jun.md`.

## Key results (reported honestly)
- **The crux holds on the true scale.** On 57 content-controlled languages (raw), trigram-JS beats the
  dumb alphabet baseline by every honest metric: **GQD 0.35 vs 0.55**, rescaled-RF 0.78 vs 0.91, and
  family-NN purity **0.67 vs 0.44** — a margin of **+0.36 over the chance floor (0.31)**, not over 0.
  (normRF cannot reach 0 against a polytomous gold; its floor is ≈0.40 and the random-tree null ≈1.0.)
- **The cross-script proof = removing a degeneracy.** Raw cross-script JS saturates at exactly 1.0 (an
  undefined nearest neighbour). Romanizing to one alphabet (**uroman**, inventory 2616 → 26) removes
  that artifact; the dumb baseline can't use letters, yet trigram-JS family-NN **rises to 0.745** — so
  the n-gram *statistics* carry it. Most signal is frequency-level (unigrams ≈ trigrams).
- **An independent cognate phylogeny validates it.** A LexStat cognate-distance tree (descent-based, not
  surface form) is a near-bullseye vs the gold (**GQD 0.032**, family-NN 0.84) and **agrees with our
  text tree (Mantel r = 0.74, p = 0.0001)** and with ASJP (r = 0.92). Strongest cross-check in the project.
- **Agrees with the field standard (ASJP).** Mantel **r ≈ 0.78** (p = 0.0005); the raw orthographic arm
  (independent of ASJP's transcription) also agrees (r ≈ 0.50), and a **partial Mantel** shows the
  agreement survives removing coarse family blocks (r ≈ 0.70; within-IE r ≈ 0.86).
- **Not a biblical-register artifact (FLORES-200).** The identical pipeline on modern Wikipedia-register
  parallel text gives **identical family-NN** (raw 0.673, romanized 0.745) and the same ballpark normRF.
- **Holds at breadth.** 102 languages / 29 families: family-NN 0.57 (raw) / 0.63 (romanized) vs the
  baseline's 0.31 / 0.34 and a chance floor of 0.15; far below the ≈1.0 random null.
- **A negative result we keep.** Phonetic IPA transcription (epitran) *hurts* vs uroman — per-language
  phoneme inventories fragment the n-grams; a single shared alphabet helps n-gram recovery more.
- **Honest claim corrections.** "English↔Romance = borrowing" was a register/orthography artifact (on
  Bible text English clusters Germanic). "Hebrew–Arabic recovered" is *partly* a uroman vowelless-abjad
  artifact. The panel is **54% Indo-European**, so the headline is helped by class imbalance: IE-only
  purity ≈ 1.0 but non-IE-only ≈ 0.35 — we foreground **what the method recovers** (IE + within-family
  / genus structure), not universal family recovery.

## Run it
- **Colab (recommended):** click the badge — loads the latest `main`, clones the code + Bible corpus,
  runs top-to-bottom. Public repo, no auth. The heavy method arms display committed figures; their
  scripts (below) reproduce them.
- **Local:** `pip install -r requirements.txt`, then run any script (e.g. `python bible_poc.py`).

## Files
| file | what |
|---|---|
| `LanguageTrees.ipynb` | **The deliverable** — self-contained Colab notebook; honest scoring + all arms. |
| `langtree.py` | Core: cleaning, sparse JS (+Lidstone), baselines, NCD, UPGMA/NJ, RF + **rf_triple**, **GQD**, chance-floor + tie-aware purity, **Mantel/partial-Mantel**, bias-corrected entropy (Miller-Madow/Grassberger), **verse-block bootstrap** + branch support, perplexity distance, delta-score, metadata + **Glottolog** gold trees. |
| `biblecorpus.py` | Parallel-Bible loader (verse alignment, gold-label fixes, cached uroman, per-verse units). |
| `bible_poc.py` | The crux experiment, scored on the true scale. |
| `bible_romanize.py` | Cross-script romanization arm + D1 proof. |
| `flores_replicate.py` | **Modern-register replication on FLORES-200** (not a Bible artifact). |
| `entropy_support_demo.py` | Bias-corrected entropy ladder + per-clade bootstrap support (UPGMA + NJ). |
| `asjp_tree.py` | ASJP phonetic-wordlist cross-check + **Mantel** tests. |
| `cognate_arm.py` | **Cognate-based (LexStat) descent phylogeny** — the decisive validation. |
| `bible_ipa.py` | IPA (epitran) arm — the negative result. |
| `methods_compare.py` | Alternative distances (plug-in / Lidstone / perplexity / NCD), bias diagnostics. |
| `reticulation.py` | Treelikeness (δ-score) + NeighborNet (borrowing/areal signal). |
| `corpus_check.py` | GlotLID language-ID data-quality gate (corpus is clean). |
| `beast_arm.py` | BEAST2 Bayesian scaffolding (binary cognate matrix → NEXUS + XML). |
| `scale_up.py` | 102-language breadth run. |
| `controls.py` | Latinate-orthography + vowelless-abjad control tests. |
| `CRITICAL_REVIEW_23jun.md` | The 32-finding adversarial review + literature that drove this version. |

## Honest limits
Character statistics measure **orthographic/surface** similarity — it tracks genealogy but isn't
identical to it; non-IE-only recovery is near chance. The raw cross-script arm is degenerate (JS=1.0);
romanization *removes* that artifact rather than adding signal, and **uroman** is lossy (Semitic recovery
is part artifact). The 102-language breadth arm has a content confound (not verse-aligned). gzip-NCD is a
crude proxy. Read normRF against its floor (≈0.40) and null (≈1.0), and prefer **GQD** + the cognate/ASJP
cross-checks. This complements, not replaces, cognate-based comparative linguistics.

## References
Shannon 1951; Benedetto, Caglioti & Loreto 2002 + Goodman 2002; Cilibrasi & Vitányi 2005; Bentz et al.
2017; Gamallo et al. 2017; Greenhill 2011; Jäger 2018 (ASJP) & 2015 (LexStat); Saitou & Nei 1987 (NJ);
Hermjakob, May & Knight 2018 (uroman); Mortensen et al. 2018 (epitran); Felsenstein 1985 (bootstrap);
Miller 1955 / Grassberger 2003 (entropy bias); Pompei, Loreto & Tria 2011 (GQD); Holland et al. 2002
(δ-score); Mantel 1967; Kargaran et al. 2023 (GlotLID); Bouckaert et al. 2014 (BEAST2); Glottolog.

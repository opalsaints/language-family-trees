# Language Family Trees from Information Theory
**Complexity Lab group mini-project — Jonathan Cowley & Nil Doğan**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opalsaints/language-family-trees/blob/main/LanguageTrees.ipynb)

Can we **rediscover the family tree of languages from raw text and information theory alone** — no
dictionaries, no grammar, no hand-built features — *and prove the information theory does real work a
trivial baseline can't?* We represent each language by its character-trigram distribution, measure
**Jensen–Shannon divergence** between languages on the **parallel Bible corpus** (same verses, so any
difference is between languages, not topics), build a tree, and score it against the true Glottolog
family tree.

> **Honest origin story.** A first version was tied on a small sample by a *dumb baseline* (just "which
> characters does a language use" — no frequencies, no information theory). We took that seriously,
> rebuilt on a large many-family corpus, **always report the trivial baselines + a negative control**,
> and only claim the information theory works where it provably beats them.

## Key results
- **It beats the dumb baseline at scale.** On 57 content-controlled languages, trigram-JS nearest-
  neighbour **family purity 0.65 vs the alphabet baseline's 0.44**. (On the old toy set it merely
  *tied* — the scale + corpus are what make the information theory earn its keep.)
- **The cleanest proof (cross-script).** Romanizing every language into one alphabet with **uroman**
  collapses the character inventory **2638 → 26**, so the alphabet baseline *can't* work — yet trigram-JS
  **rises to 0.77** (margin **+0.30**). Cross-script families reconnect (Hebrew–Arabic JS 1.00→0.76,
  Telugu–Kannada 1.00→0.55, Cyrillic↔Latin Slavic 1.00→0.75).
- **Proper methods & nulls.** Neighbor-Joining beats UPGMA (normRF 0.71 vs 0.79); cophenetic
  correlation 0.91; a **random-tree null sits at ~0.97**, so our trees capture real structure;
  gzip-NCD corroborates. n-gram order **n=1…5 is flat** (n=3 was not cherry-picked); bootstrap CIs are
  tight.
- **Agrees with the field standard.** Our text tree vs the **ASJP** phonetic-wordlist tree: distance
  correlation **r = 0.78** — two completely different methods recover the same structure.
- **Holds at breadth.** 102 languages / 29 families: trigram-JS family purity 0.63 vs baseline 0.34.
- **Honest claim corrections.** The old "English↔Romance = Norman borrowing" was a UDHR *legal-register*
  artifact — on Bible text English correctly clusters Germanic. "Hebrew–Arabic recovered" is *partly* a
  vowelless-abjad + uroman artifact (stripping vowels from Romance pulls it toward Hebrew), though the
  Semitic pair stays genuinely closer than chance.

## Run it
- **Colab (recommended):** click the badge — it loads the latest from `main`, clones the code + Bible
  corpus, and runs top-to-bottom. The repo is public, so no GitHub auth is needed.
- **Local:** `pip install numpy scipy matplotlib dendropy uroman` then run any of the scripts below.

## Files
| file | what |
|---|---|
| `LanguageTrees.ipynb` | **The deliverable** — self-contained Colab notebook; reproduces the core results + figures. |
| `langtree.py` | Core library: cleaning, sparse JS, baselines, gzip-NCD (fixed), UPGMA, Neighbor-Joining, corrected RF, cophenetic, random-tree null, bootstrap CIs, metadata gold tree. |
| `biblecorpus.py` | Parallel-Bible loader (verse alignment, cached uroman romanization). |
| `bible_poc.py` | The crux experiment: trigram-JS vs alphabet/unigram/shuffle baselines. |
| `bible_romanize.py` | Cross-script romanization arm + the D1 proof figure. |
| `asjp_tree.py` | ASJP phonetic-wordlist cross-check. |
| `robustness.py` | n-gram order sweep + bootstrap confidence intervals. |
| `scale_up.py` | 102-language breadth run. |
| `controls.py` | Latinate-orthography and vowelless-abjad control tests. |
| `figures/` | All saved figures. |
| `audit_findings_22jun.json` | The adversarial self-audit that drove this rebuild. |

## Honest limits
Character statistics measure **orthographic/surface** similarity — it tracks genealogy but isn't
identical to it. Bible text is translated (mild shared translationese) and its Hebrew/Arabic is a
classical register; **uroman** is lossy and non-neutral (abjads → vowelless skeletons). Read RF against
the ~0.97 random null, and alongside cophenetic correlation and family purity. This is a complement to,
not a replacement for, cognate-based comparative linguistics.

## References
Shannon 1951; Benedetto, Caglioti & Loreto 2002 (*Language Trees and Zipping*) + Goodman 2002 comment;
Cilibrasi & Vitányi 2005 (NCD); Bentz et al. 2017 (entropy across the Bible corpus); Gamallo et al.
2017 (character-n-gram language distance); Greenhill 2011; Jäger 2018 (ASJP); Saitou & Nei 1987 (NJ);
Hermjakob, May & Knight 2018 (uroman); Paninski 2003.

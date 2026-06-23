All key claims are verified against ground truth: `Altaic` vs `Altaic(?)` typo confirmed, Korean=`Altaic(?)`, Japanese=`Japonic`, Icelandic Script=`Ethiopic` (data error), README hard-codes "~0.97" null, bootstrap resamples multinomial at full N, rf_corrected docstring claims "0 = no conflicting splits". I have enough verified ground truth. Now I'll write the report.

# Adversarial + Literature Synthesis: Language-Tree Mini-Project Decision Report

*Prepared for Jonathan + Nil — 23 Jun 2026. Verified against the live repo where load-bearing (metadata typos, README null line, `rf_corrected` docstring, `nn_purity` tie-break, bootstrap resampling all confirmed by direct read).*

---

## 1. Executive summary

- **Two headline numbers are scored against impossible endpoints.** normRF is reported on an implied `[0,1]` scale where 0 = perfect, but on *this* polytomous gold a genealogically-perfect tree floors at **normRF ≈ 0.40** and the random null is actually **≈1.00, not the 0.97 the README hard-codes**. So NJ 0.71 / UPGMA 0.79 are mid-window, not "70–80% wrong." **Highest-value fix: report the triple (observed, floor, null), or switch to Generalized Quartet Distance (GQD).**
- **Family-purity 0.65/0.77 is benchmarked against the wrong floor.** Random-NN chance is **0.308** (because 31/57 = 54% of the panel is one "Indo-European" bucket), never 0. The honest lift is ~0.34, not "0.65 vs 0.44." Split out, IE-only purity is 0.93 but **non-IE purity is ~0.39 — barely above the 0.31 floor.** The method largely separates the dominant block, not genealogy.
- **The plug-in trigram-JS distance is upward-biased, and the bias scales with K (trigram-vocab size) and N (text length), which correlate with script/morphology, which correlate with family.** Confirmed: distances correlate r≈0.40 with max(K) and r≈0.37 with |ΔK|. A measurable fraction of the recovered tree is a vocab-size/length artifact along the very axis being tested. No smoothing, no bias correction — **below the field's minimum bar** (Gamallo 2017; Bentz 2017; Arora/Meister/Cotterell 2022).
- **The raw cross-script arm is a degeneracy, not a finding.** Every single-script language (Greek, Telugu, Korean, Thai…) sits at **JS = 1.000 to all 56 others** (923/1596 pairs ≥0.999); `nn_purity`'s `min()` then silently returns the first-listed language ("Amharic magnet"). ~9/57 nearest neighbours are pure list-order tie-breaks. Romanization mostly *removes this degeneracy* rather than *adding signal*.
- **The bootstrap is invalid** — it resamples each language's tokens multinomially from its *own plug-in distribution at full N*, so every replicate reproduces the same bias; CIs are mis-centered (around the bias) and ~27% too narrow. "Purity 0.77, 95% CI…" overstates precision.
- **The gold standard is hand-typed and internally inconsistent** (verified): Turkish=`Altaic`, Korean=`Altaic(?)` (literal typo splits the one relationship it meant to assert), Japanese=`Japonic`, Icelandic Script=`Ethiopic` (data error); 6 singleton families **cap purity at 0.895**. Both RF and purity score against an unreliable target.
- **Reliability machinery describes the wrong tree.** NJ is the headline topology, but cophenetic 0.91, bootstrap CIs, and branch support are *all* computed on **UPGMA**. The confidence numbers do not correspond to the advertised tree.
- **Single highest-value fixes (in order):** (1) report RF as a triple / adopt GQD; (2) print the 0.31 purity chance floor and add genus-level purity; (3) fix the gold-label typos + flag singleton ceiling; (4) flag JS=1.0 ties as undefined NN; (5) widen/relabel the bootstrap claim. Items 2–5 are all quick wins doable before Friday.

---

## 2. Issues ranked by (severity × likelihood of distorting a headline)

### #1 — RF reported against an unreachable 0; floor ≈0.40, null ≈1.00 (not 0.97) — **CRITICAL**
- **What's wrong / mechanism.** `rf_corrected` uses dendropy `symmetric_difference` with denom = inferred bipartitions (54, binary) + gold bipartitions (23, polytomous). A binary tree *must* add ~31 splits the polytomous gold lacks; every one counts as a "difference." So normRF ≥ 31/77 = **0.403** even for a perfect, zero-conflict refinement of gold. The random null on the real gold is **p05=p50=p95=1.000** (min ≈0.974), yet the README, `robustness.py:40`, and `build_notebook.py:302` all hard-code "~0.97" (verified: README line 27).
- **Evidence.** Docstring literally says *"0 = no conflicting splits"* (verified line 440) — false. Probe: resolved gold vs gold → normRF 0.403. NJ 0.71 maps to (0.71−0.40)/(1.00−0.40) ≈ **0.52** of the achievable window.
- **New vs prior audit.** New. Prior audit recommended the sum-of-bipartitions denominator *as the fix*; the fix moved the floor but did not remove it.
- **Root cause (lit).** RF symmetric difference over a polytomous reference is the wrong metric class. Field standard for scoring a binary tree vs a polytomous Glottolog gold is **GQD** (Pompei, Loreto & Tria 2011), which scores 0 when nothing is contradicted.
- **Solutions.**
  - *(low)* Compute and report the **triple (observed, floor-from-resolved-gold, null-p50)** or rescale `(obs−floor)/(null−floor)`. Tradeoff: still RF's low resolution. Ref: project's own `random_tree_null` + `resolve_polytomies`.
  - *(med)* Adopt **GQD** via tqDist (Sand 2014) or R `Quartet`. Benchmark: Jäger 2025 reports GQD-vs-Glottolog of 0.048 (MSA) / 0.095 (PMI) / 0.227 (cognate classes) — gives your number an external scale. Tradeoff: C++/R bridge, changes every headline number.
  - *(low)* Fix the docstring + README "0.97" → "≈1.00" immediately.
- **Recommended:** Fix README/docstring + report the triple **now**; add GQD if time. Both, ideally.

### #2 — Family-purity benchmarked against wrong floor; dominated by 54%-IE bloc — **CRITICAL**
- **Mechanism.** `nn_purity` (verified lines 395–402) counts a hit if a language's NN is *any* same-Family language, at the coarsest rank where IE = 31/57. A Hindi→Russian or Greek→English hit counts despite a 5000-yr split. Random-NN floor = Σ c(c−1)/[n(n−1)] = **0.308**. IE-only 0.93 vs non-IE-only **0.39** (≈ chance).
- **Evidence.** README "0.65 vs 0.44" with no chance line (verified line 20). Genus-level purity drops to ~0.60 (< the advertised 0.65) against a genus chance of only 0.067.
- **New vs prior.** New on the 57-lang crux (prior flagged it only for the retired 9-lang Tier-2).
- **Root cause (lit).** Purity is *not* chance-corrected and inflates under class imbalance (sklearn clustering docs; Hubert & Arabie 1985).
- **Solutions.**
  - *(low)* **Print the 0.31 chance floor** next to every purity number; report **IE-only vs non-IE-only** separately. Quick win.
  - *(low)* Add **genus-level purity** (Germanic/Romance/Slavic…) — this is the honest "does it recover families" test. Quick win; genus column already in rows.
  - *(med)* Replace headline with **Adjusted Rand Index / B-cubed** (chance-corrected). Tradeoff: less intuitive, numbers lower.
- **Recommended:** Do all three of the low/med fixes; foreground genus-level + ARI, keep family-purity only with the floor printed.

### #3 — Plug-in trigram-JS bias scales with K and N (differential along family axis) — **CRITICAL**
- **Mechanism.** `js_div_counts` uses raw MLE freqs, no smoothing. Plug-in entropy is biased ≈−(K−1)/2N (Miller-Madow/Roulston 1999); JS = H(mix)−mean(H) does **not** cancel this, leaving JS biased *upward* by O(K/N). Large-K (CJK/agglutinative) or short-text languages get larger distances to *everyone*. Confirmed r=0.40 with max(K), r=0.37 with |ΔK|; two i.i.d. draws from the *same* distribution score JS 0.62 (a=26) → 0.85 (a=80).
- **New vs prior.** New (quantified bias-vs-K/N; prior covered only UDHR truncation).
- **Root cause (lit).** No smoothing + plug-in MLE on a sparse high-dim space. Field standard: smoothed char-n-gram models + symmetrized **perplexity/cross-entropy** (Gamallo 2017), or **NSB/James-Stein-shrinkage/Chao-Shen** estimators (Bentz 2017; Hausser-Strimmer 2009; Arora 2022).
- **Solutions.**
  - *(low)* **Kneser-Ney or Lidstone smoothing** of distributions before JS; lifts to field minimum. Tradeoff: hyperparameter.
  - *(med)* **James-Stein shrinkage** (R `entropy`, or ~10 lines numpy) applied to the *distance*, not just the entropy ladder. Tradeoff: shrinks margins (honest).
  - *(med)* Switch distance to **symmetrized perplexity** (Gamallo). Tradeoff: needs held-out split, not a true metric without symmetrization.
  - *(low, free diagnostic)* **Report unigram-JS alongside trigram-JS** and the residual corr(distance, K) after any fix.
- **Recommended:** Ship the unigram-vs-trigram ablation + report corr-with-K now (free, honest); add shrinkage if time. A full perplexity rewrite is future work.

### #4 — Raw cross-script JS = 1.000 saturation → arbitrary tie-broken NN ("Amharic magnet") — **CRITICAL**
- **Mechanism.** Disjoint scripts share ~only the space trigram → JS saturates at 1.0. `min()` returns the first index → NN dictated by list order. ~9/57 raw NN are ties; reversing list order swings purity 0.667→0.649.
- **New vs prior.** New at full-Bible scale (prior assumed corpus swap mooted it).
- **Root cause (lit).** Using orthography as the representation; the field never computes genealogy from glyphs (Greenhill 2011; Jäger 2018).
- **Solutions.**
  - *(low)* **Detect ties; refuse to assign a NN when min-distance is shared** (flag as undefined). Removes a class of fake findings instantly. Quick win.
  - *(low)* Frame romanization as **removing a degeneracy**, not adding signal.
  - *(med)* Add an **epitran→IPA arm** (Mortensen 2018) as the principled cross-script representation; restores vowels (unlike uroman). Tradeoff: ~60-lang coverage, lossy.
- **Recommended:** Tie-flagging + honest reframing now; epitran arm as high-value-if-time.

### #5 — Invalid bootstrap → falsely tight, mis-centered CIs — **MAJOR**
- **Mechanism.** `bootstrap_ci`/`branch_support` resample tokens multinomially from each language's own plug-in distribution at full N (verified `tots.append(int(v.sum()))` line 416/623). Every replicate reproduces the bias; multinomial ignores trigram-overlap dependence. Probe: bootstrap CI [0.632,0.642] centered *above* the point estimate 0.567 and never near truth 0; sd ~27% too small.
- **Root cause (lit).** Resampling a *fitted distribution* is not the Felsenstein (1985) bootstrap, which resamples *data units*.
- **Solutions.**
  - *(med)* **Sentence/verse block bootstrap** (resample aligned units, re-clean, re-count, re-infer). Felsenstein 1985. Tradeoff: slower, wider CIs (correct).
  - *(low)* At minimum, **stop calling the current interval a "95% CI"** — relabel as "Monte-Carlo wobble of a biased statistic."
- **Recommended:** Relabel now; block bootstrap if time.

### #6 — Hand-typed gold is inconsistent (Altaic typo, Icelandic script error, singleton ceiling) — **MAJOR**
- **Evidence (verified).** Turkish=`Altaic`, Korean=`Altaic(?)`, Japanese=`Japonic`, Icelandic Script=`Ethiopic`. 6 singleton families → purity ceiling **0.895**.
- **Root cause (lit).** Gold should be built programmatically from **Glottolog** (`pyglottolog.newick_tree`), not hand-typed strings; URIEL/lang2vec flattens hierarchy so is unsuitable for topology.
- **Solutions.**
  - *(low)* **Collapse `Altaic`/`Altaic(?)`, fix Icelandic script, report the 0.895 ceiling.** Quick win.
  - *(med)* **Build gold from Glottolog** + RF/GQD sensitivity over 2–3 resolutions. Tradeoff: still polytomous (→ needs GQD).
- **Recommended:** String fixes + ceiling disclosure now; Glottolog gold is high-value-if-time.

### #7 — Reliability numbers computed on UPGMA while NJ is the headline — **MAJOR**
- **Mechanism.** `cophenetic_corr` (368), `bootstrap_ci` (430), `branch_support` (590,637), and plotted dendrograms use `upgma()`; NJ is only called for the headline RF. The 0.91 cophenetic and all supports describe the worse, clock-assuming tree.
- **Root cause (lit).** UPGMA assumes a molecular clock (ultrametric); orthographic change is non-clocklike. NJ assumes additivity, which JS also violates (Atteson 1997) — negative branches are clamped (`max(.,0)`), hiding the violation.
- **Solutions.**
  - *(low)* **Move cophenetic/support onto the NJ tree**; report NJ's **negative-branch fraction** as an additivity diagnostic. Quick win.
  - *(low)* Soften "NJ is the proper/assumption-free method" → "trades the clock for additivity, which JS also violates."
- **Recommended:** Both low-effort fixes now.

### #8 — ASJP r=0.78 uses invalid Pearson p; doesn't isolate fine structure; not independent — **MAJOR**
- **Mechanism.** `asjp_tree.py:116-118` uses `np.corrcoef` over C(k,2) non-independent pairs (naive p=1.1e-67 meaningless). Probe: two matrices sharing *only* 5-block coarse structure already correlate r=0.71. And the text arm correlated is the **romanized** one — both inputs are transliterations, so it's not orthography-vs-phonology.
- **Root cause (lit).** Matrix-vs-matrix needs **Mantel** (Mantel 1967); coarse-block agreement masquerades as fine-structure agreement (needs **partial Mantel** controlling a script/major-family block matrix). Harmon & Glor 2010 caveat: Mantel has low power.
- **Solutions.**
  - *(low)* **Mantel permutation test** (`skbio.stats.distance.mantel`) + **partial Mantel** on the block matrix. Quick win. Also correlate the **raw** (orthographic) arm vs ASJP for genuine independence.
- **Recommended:** Mantel + partial Mantel now; scope claim to "romanized text vs ASJP agree, strongly within IE."

### #9 — Semitic "recovery" is substantially a uroman vowelless-skeleton artifact — **MAJOR**
- **Mechanism.** uroman renders abjads as near-vowelless skeletons (Hebrew vowel-fraction 0.10, Arabic 0.21 vs Spanish 0.46, Greek 0.50). JS is dominated by high-frequency chars (vowels); two low-vowel strings collapse onto a shared consonant space → manufactured similarity on exactly the claimed pair.
- **Root cause (lit).** uroman is a transliterator, not a G2P transcriber. epitran (Mortensen 2018) reconstructs vowels per language.
- **Solutions.** *(low)* Keep the `controls.py` devowel control and **state the Semitic reconnection is partly artifactual**; *(med)* re-run Hebrew/Arabic through epitran. Recommended: honest caveat now; epitran future.

### #10 — Breadth (102-lang) arm has content confound (align=False, OT vs NT) — **MAJOR**
- **Mechanism.** `scale_up.py` `load(align=False)` → `sorted(verse_ids)[:4000]`; full Bibles start `b.1CH` (OT), NT-only start `b.1CO` (NT). Full-vs-NT correlates with IE-vs-non-IE, so the method partly separates *content*, not family.
- **Solutions.** *(low)* Run breadth with **align=True**, or caveat the 0.63 as not a clean replication of the controlled 57-lang result. Recommended: caveat now, re-run if time.

### #11 — FLORES too small for large-inventory scripts; Serbian script mismatch; budget-before-romanization — **MAJOR**
- **Mechanism.** ~2009 sentences → Chinese K/N=0.77, Korean K/N=0.37 (near-pure-noise). Serbian is Latin in Bible, Cyrillic in FLORES (breaks the same-representation control). char_cap=30000 set on *raw* text; uroman expands unevenly (Korean→65k, 2.19× imbalance) → finite-sample bias credited to the "constant-alphabet" effect.
- **Solutions.** *(low)* **Report K/N per language and drop/flag K/N > ~0.3**; fix Serbian script; apply char_cap *after* romanization. *(low)* **GlotLID** (Kargaran 2023) line-level language check. Recommended: K/N flag + Serbian fix + post-romanization cap.

### #12 — n=3 unjustified; "stable across n" actually shows trigrams add ~nothing over unigrams — **MINOR**
- **Mechanism.** unigram-JS normRF 0.870 = trigram normRF 0.870; purity 0.632 vs 0.649. If unigrams match trigrams, higher-order info-theoretic content isn't driving the result, and n=3 is the worst case for plug-in bias.
- **Solutions.** *(low)* **Choose n by held-out perplexity**, report the full RF-vs-n curve with unigram shown, and reframe honestly ("frequency-level signal suffices"). Quick win.

### #13 — NCD panel-averaged self-floor clamps close-relative signal to 0 — **MINOR**
- **Mechanism.** `ncd_matrix_fixed` subtracts one global mean floor then `max(0, d−floor)`; any genuinely close pair below the floor collapses to exactly 0, degrading the NCD fine structure the cross-check relies on.
- **Solutions.** *(low)* **Per-pair self-floor**, no global clamp; treat NCD as direction-only corroboration (Goodman 2002 caveat). Quick win.

### #14 — NJ on non-additive JS clamps negative branches — **MINOR**
- Covered under #7. Report the negative-branch fraction; don't call NJ assumption-free.

---

## 3. Why specific numbers may be coming out wrong

| Reported number | Most likely cause |
|---|---|
| **Raw cross-script JS = 1.000** | Disjoint scripts share ~only the space trigram → JS saturates at its max. Not a measured distance; it's a ceiling. (#4) |
| **"Amharic magnet" raw NN** | `min()` first-wins tie-break over 56 tied-at-1.0 languages → NN = first in list order. Pure artifact. (#4) |
| **Family purity 0.65 (raw) / 0.77 (romanized)** | (a) 0.31 chance floor from 54%-IE imbalance + (b) script/areal proximity packing IE together, not genealogy. Honest lift ~0.34; non-IE alone ≈0.39. Romanized 0.77 also inflated by 2.19× uneven data budget. (#2, #11) |
| **Alphabet baseline 0.44 / 0.47 / 0.34** | Same coarse-Family inflation; only ~0.13 above the 0.31 floor — so the "+0.30 margin" is over the baseline, not over chance. (#2) |
| **NJ 0.71 vs UPGMA 0.79** | Both on a scale whose true floor is 0.40 and ceiling ≈1.00. On [0.40,1.00] they're 0.52 vs 0.65 of the window — NJ's *relative* lead is larger, but neither is "close to truth." (#1) |
| **Random null "0.97"** | Stale (old 14-taxon/2(n−3) era). Real null on the 57-taxa polytomous gold is **p50 = 1.000**. README/robustness/build_notebook contradict the live function. (#1) |
| **Cophenetic 0.91** | Measures UPGMA's fit to its *own* distances — irrelevant to the headline NJ tree. (#7) |
| **ASJP r = 0.78** | Largely coarse-block (script/major-family) agreement (block-only matrices already give r≈0.71); naive p invalid; both arms romanized so not independent. Within-IE r=0.855 is the real signal. (#8) |
| **Bootstrap "95% CI"** | Resamples the plug-in distribution at full N → mis-centered on the bias, ~27% too narrow. (#5) |
| **Breadth 0.63 vs 0.34** | Confounded with OT-vs-NT content (align=False) that correlates with IE-vs-non-IE. (#10) |
| **English F₁ ≈ 4.05 bits** | Plug-in MLE is downward-biased by ≈(K−1)/2N; bias-correction (Miller-Madow/Grassberger) is the right move (already noted as in progress). |

---

## 4. Prioritized action list ("everything we should do")

### MUST-FIX-BEFORE-FRIDAY (all quick wins, mostly text/reporting)
1. **[QW] Fix the RF scale story:** correct README/`robustness.py`/`build_notebook.py` "0.97" → "≈1.00"; fix the `rf_corrected` docstring ("0 = no conflicting splits" is false); **report the triple (observed, floor≈0.40, null≈1.00)** for every RF number. (#1)
2. **[QW] Print the 0.308 purity chance floor** next to every purity figure; **add genus-level purity** and **IE-only vs non-IE-only split**. (#2)
3. **[QW] Fix the gold:** collapse `Altaic`/`Altaic(?)`, fix Icelandic Script `Ethiopic`→`Latin`, **state the 0.895 singleton ceiling**. (#6)
4. **[QW] Flag JS=1.0 ties as undefined NN** (don't report "Greek→Amharic"); reframe romanization as *removing a degeneracy*. (#4)
5. **[QW] Relabel the bootstrap** — it is not a 95% CI; describe it honestly. (#5)
6. **[QW] Move cophenetic + branch support onto the NJ tree**; report NJ negative-branch fraction; soften "assumption-free" language. (#7)
7. **[QW] Mantel + partial Mantel** for ASJP r (replace naive Pearson p); scope the claim to "romanized text vs ASJP, within-IE." (#8)
8. **[QW] Unigram-vs-trigram ablation table + corr(distance, K)** reported; reframe n-gram contribution honestly. (#3, #12)
9. **[QW] Caveat the Semitic reconnection** as partly a uroman vowelless-skeleton artifact (cite your own devowel control). (#9)
10. **[QW] Caveat the breadth arm** (content confound) and **report FLORES K/N per language**, flagging K/N>0.3. (#10, #11)

### HIGH-VALUE-IF-TIME
- **Adopt GQD** (tqDist or R `Quartet`) as primary tree-vs-gold metric; benchmark against Jäger 2025 (0.048/0.095/0.227). (#1)
- **James-Stein shrinkage** (or Kneser-Ney smoothing) on the trigram distributions, applied to the *distance*. (#3)
- **Build gold from Glottolog** (`pyglottolog`) + RF/GQD sensitivity over resolutions. (#6)
- **Block (sentence/verse) bootstrap** on the NJ tree (Felsenstein). (#5)
- **epitran→IPA arm** as the principled cross-script representation; re-test Semitic/Telugu-Kannada. (#4, #9)
- **Re-run breadth with align=True**; fix Serbian script + post-romanization char_cap; **GlotLID** purity gate. (#10, #11)
- **Per-pair NCD self-floor** (drop global clamp). (#13)

### FUTURE-WORK
- Symmetrized **perplexity/cross-entropy** distance (Gamallo 2017) as the field-standard replacement for raw JS.
- **Cognate-aware arm** (LingPy LexStat / Rama string kernels / ASJP PMI) — the only arm that targets descent rather than surface form.
- **NeighborNet/SplitsTree + delta-score** to *visualize* borrowing/areal (non-tree) signal instead of forcing a tree.
- Character/cognate **Bayesian inference** (BEAST2/MrBayes) — the genuine field gold standard; likely beyond a course project.
- Orthographic-depth confound test (correlate K vs an external transparency ranking).

---

## 5. What is already solid — KEEP and foreground (don't over-correct)

- **The core engineering is correct** (independently re-probed in the adversarial pass): `clean()`'s Turkish casefold + U+0307 fix works with zero collateral damage; `js_div_counts` matches scipy `jensenshannon` to 1e-16 and handles disjoint/identical/empty correctly; `gold_newick_from_rows` nests Family→Genus→Subgenus correctly; `nj_newick` emits valid parseable Newick (54 bipartitions); `_random_binary_newick` is ~uniform over topologies. **Do not re-chase these.**
- **The verse-aligned, content-controlled 57-language design is genuinely good** and is the project's strongest methodological move — it controls content far better than the UDHR/byte-truncation prior work. Keep it as the backbone (just disclose it's NT-only-for-everyone, and that "Hebrew" is a Modern-Hebrew translation of the Greek NT).
- **The romanized arm's qualitative reconnections are a real and interesting result** — Hebrew–Arabic 1.00→0.76, Telugu–Kannada 1.00→0.55, Cyrillic↔Latin Slavic 1.00→0.75 — *provided* you reframe it as "removing the script degeneracy" rather than "adding genealogical signal," and caveat the Semitic case.
- **NJ's relative advantage over UPGMA survives rescaling** (0.52 vs 0.65 of the achievable window) — the *direction* of that comparison is defensible once you state the assumptions.
- **Within-IE structure is the strongest, most defensible claim** (IE-only purity 0.93, IE-only ASJP r=0.855). Lead with what the method *does* recover (the IE block and within-IE genus structure) rather than overclaiming general family recovery.
- **Honesty already in the README** (orthographic-not-genealogical caveat, uroman lossiness, Bible register, gzip-NCD crudeness) is good and ahead of where many course projects sit — the fixes above mostly make the *numbers* match the caveats you've already written.

**Bottom line:** No result is fabricated, but several headline *numbers are on the wrong scale or against the wrong baseline*, and ~half the recovered structure on the full panel is plausibly script/length/imbalance artifact. The ten Friday quick-wins are almost entirely reporting/relabeling and will convert "overclaimed" into "defensible" without new modeling. The single most important one is **reporting RF and purity against their true floors/nulls (0.40 / 1.00 / 0.31)** instead of an implied 0.
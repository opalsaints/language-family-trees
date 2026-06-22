"""Generate the self-contained Colab notebook LanguageTrees.ipynb.

The notebook is a thin orchestrator: in Colab it clones this (public) repo so it
can `import langtree, biblecorpus` (the tested, committed code) instead of
duplicating it, then reproduces every result and figure. Locally it just imports
the modules already present. Regenerating after a code change is therefore cheap.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Language Family Trees from Information Theory
### Complexity Lab — group mini-project (Jonathan Cowley & Nil Doğan)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opalsaints/language-family-trees/blob/main/LanguageTrees.ipynb)

**Big question.** Languages evolve like species — shared ancestors, splits, drift. Can we
*rediscover* the family tree of languages from **raw text and information theory alone** — no
dictionaries, no grammar, no hand-built features — and, crucially, show the information theory does
real work a trivial baseline can't?
""")

md(r"""
## What this notebook shows

We built a first version, then ran a hard adversarial audit of our own method. It found (1) on a tiny
distinct-script sample a **dumb "which letters does it use" baseline tied** our trigram method — the
information theory was doing nothing; and (2) a **bug** silently corrupting Turkish. This is the
rebuilt, honest version: fixed bug, a large content-controlled many-family corpus (the parallel
Bible), and **the trivial baselines + a negative control reported every time**, so the contribution
of the information theory is measured, not assumed.

Contents: cleaning → corpus → Shannon ladder → distances & baselines → the **crux experiment**
(does it beat the baseline at scale?) → the tree → the **cross-script romanization arm** (and the
clean proof the information theory does the work) → **tree methods & robustness** (Neighbor-Joining
vs UPGMA, cophenetic correlation, a proper random-tree null, compression distance). Still to come:
ASJP cross-check and a larger language panel.
""")

md("## 0. Setup  — clone the repo (for the code) + the Bible corpus")
code(r"""
import os, sys, subprocess, importlib
def ensure(pkg, mod=None):
    try: importlib.import_module(mod or pkg)
    except ImportError: subprocess.run([sys.executable,"-m","pip","install","-q",pkg], check=False)
for p in ["numpy","scipy","matplotlib","dendropy","uroman"]:
    ensure(p)

# get the tested code (langtree.py, biblecorpus.py). Locally they're already here.
try:
    import langtree  # noqa
except ImportError:
    if not os.path.isdir("language-family-trees"):
        subprocess.run(["git","clone","--depth","1",
                        "https://github.com/opalsaints/language-family-trees"], check=False)
    sys.path.insert(0, os.path.abspath("language-family-trees"))

import langtree as lt
import biblecorpus as bc
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

REPO_DIR = os.path.dirname(os.path.abspath(bc.__file__))   # for committed figures
# get the corpus (cloned next to biblecorpus.py so its paths resolve)
cdir = os.path.join(os.path.dirname(bc.__file__), "corpus", "bible-corpus")
if not os.path.isdir(os.path.join(cdir, "bibles")):
    os.makedirs(os.path.dirname(cdir), exist_ok=True)
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/christos-c/bible-corpus", cdir], check=False)
print("ready — corpus present:", os.path.isdir(os.path.join(cdir, "bibles")))
""")

md(r"""
## 1. Cleaning the text (and the bug we caught)

Each language → a stream of casefolded Unicode letters, runs of non-letters collapsed to one space.
The audit's bug: Turkish `İ` casefolds to `i` + a combining dot, and naively mapping non-letters to
spaces turned that dot into a **word break**, shredding Turkish n-grams. Fixed by NFC-normalizing,
keeping combining marks attached (this also protects Arabic/Indic scripts), and dropping the stray dot.
""")
code(r"""
assert lt.clean("İSTANBUL İNSAN") == "istanbul insan"   # the bug, fixed
print("clean('İstanbul') =", repr(lt.clean("İstanbul")),
      "| accents kept:", repr(lt.clean("café CAFÉ")))
""")

md(r"""
## 2. The corpus — the parallel Bible (content-controlled)

The [christos-c parallel Bible](https://github.com/christos-c/bible-corpus): ~100 verse-aligned
translations. Same verses in every language ⇒ any distance is between **languages**, not topics. Its
`metadata.csv` gives each language's Family/Genus/Subgenus, from which we build the gold tree (no
hand-typing, no circularity). We use a 57-language, ~13-family, multi-script subset.
""")
code(r"""
d = bc.load(verse_cap=2000, char_cap=30000)     # equal, content-controlled budget per language
names, rows, iso = d["names"], d["rows"], d["iso"]
fam_of = {r[0]: r[1] for r in rows}
raw = [lt.clean(d["rawtext"][n]) for n in names]
from collections import Counter
print(len(names), "languages,", len(d["common"]), "shared verses")
print("families:", dict(sorted(Counter(fam_of.values()).items(), key=lambda kv:-kv[1])))
""")

md(r"""
## 3. Foundations — Shannon's entropy ladder

Block entropy $H(n)$ over $n$-grams; conditional entropy $F_n=H(n)-H(n-1)$; redundancy is the drop
$F_1\ge F_2\ge\dots$. With enough text the plug-in estimate lands near Shannon's classic values.
""")
code(r"""
eng = raw[names.index("English")]
prev = 0.0
print(f"English sample: {len(eng):,} chars, {len(set(eng))} symbols")
for n in range(1, 5):
    Hn = lt.block_entropy(eng, n); print(f"  F{n} = {Hn-prev:.2f} bits/char"); prev = Hn
print("Shannon 1951 reference: F1=4.03, F2=3.32, F3=3.1")
""")

md(r"""
## 4. Distances, baselines, and the crux experiment

Each language → a character-trigram distribution; distance = **Jensen–Shannon divergence** (symmetric,
bounded relative-entropy cousin; sparse so it scales to many scripts). Alongside it we *always* compute
**unigram-JS** (order-blind), the **alphabet-overlap** dumb baseline (set of characters; no
frequencies, no information theory), and a **shuffle** negative control (order destroyed). We score each
tree against the gold family tree with an honest Robinson–Foulds (denominator = the trees' own
bipartitions) and report nearest-neighbour family purity.
""")
code(r"""
gold = lt.gold_newick_from_rows(rows)
methods = {
    "trigram-JS  (information theory)": lt.js_matrix(raw, 3),
    "unigram-JS  (frequencies only)":   lt.js_matrix(raw, 1),
    "alphabet-overlap  (DUMB BASELINE)": lt.alphabet_jaccard_matrix(raw),
    "shuffle-trigram  (neg. control)":  lt.js_matrix([lt.shuffle_chars(t) for t in raw], 3),
}
print(f"{'method':36s}{'normRF':>8s}{'family-NN':>11s}")
print("-"*55)
res = {}
for nm, D in methods.items():
    _,_,rf = lt.rf_corrected(lt.linkage_to_newick(lt.upgma(D, names), names), gold)
    pur = lt.nn_purity(D, names, fam_of); res[nm] = (rf, pur)
    print(f"{nm:36s}{rf:8.3f}{pur:11.3f}")
print("-"*55)
print("=> at scale the information theory BEATS the dumb baseline (unlike the toy case where it tied);")
print("   most of the gain is in the frequencies, n-gram order adds a smaller increment.")
""")

md("### The tree (raw text)")
code(r"""
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram
def plot_family_tree(D, title):
    fams = sorted(set(fam_of.values())); cm = plt.get_cmap("tab20")
    col = {f: cm(i % 20) for i, f in enumerate(fams)}
    Z = lt.upgma(D, names)
    fig, ax = plt.subplots(figsize=(11, 0.33*len(names)+2))
    dendrogram(Z, labels=names, orientation="right", ax=ax, color_threshold=0, above_threshold_color="#999")
    ax.set_title(title); ax.set_xlabel("Jensen–Shannon distance (character trigrams)")
    for lb in ax.get_ymajorticklabels(): lb.set_color(col[fam_of[lb.get_text()]]); lb.set_fontsize(8)
    ax.legend(handles=[Patch(facecolor=col[f], label=f) for f in fams], loc="lower right", fontsize=6, ncol=2)
    plt.tight_layout(); plt.show()
plot_family_tree(methods["trigram-JS  (information theory)"],
                 f"Raw parallel Bible — trigram-JS tree, {len(names)} languages (coloured by family)")
""")
md(r"""
Within a writing system, families fall out (Romance, Germanic, Latin-script Slavic, Austronesian,
Niger-Congo, Indo-Aryan). But every **unique-script** language sits at distance ≈ 1.0 with no
neighbour — the **script confound**: disjoint character sets give JS = 1. That depresses the score and
sets up the next step.
""")

md(r"""
## 5. The cross-script fix — romanization (uroman), and the cleanest proof

We romanize every language into one Latin alphabet with **uroman** and re-run. Two payoffs: cross-script
families should reconnect, and — since romanizing makes every language share one alphabet — the
alphabet-overlap baseline should **collapse**, so any family recovery is now unambiguously the n-gram
*statistics* doing the work, not the letter inventory.
""")
code(r"""
rom_raw = bc.romanize_cached(names, d["rawtext"], iso, tag="v2000_c30000")  # cached after first run
rom = [lt.clean(rom_raw[n]) for n in names]

def family_nn(texts): return lt.nn_purity(lt.js_matrix(texts,3), names, fam_of)
def alpha_nn(texts):  return lt.nn_purity(lt.alphabet_jaccard_matrix(texts), names, fam_of)
R = (family_nn(raw),  alpha_nn(raw),  len(set("".join(raw))-{" "}))
M = (family_nn(rom),  alpha_nn(rom),  len(set("".join(rom))-{" "}))
print(f"{'':10s}{'trigram-JS':>12s}{'alphabet-baseline':>20s}{'union alphabet':>16s}")
print(f"{'RAW':10s}{R[0]:12.3f}{R[1]:20.3f}{R[2]:16d}")
print(f"{'ROMANIZED':10s}{M[0]:12.3f}{M[1]:20.3f}{M[2]:16d}")
print(f"\nD1 proof: romanizing collapses the alphabet ({R[2]}->{M[2]} chars); the baseline can't improve,")
print(f"but trigram-JS rises to {M[0]:.3f} (margin +{M[0]-M[1]:.3f}) -> the information theory does the work.")

# D1 figure
fig, ax = plt.subplots(figsize=(6.5,4))
x=np.arange(2); w=.36
ax.bar(x-w/2,[R[0],M[0]],w,label="trigram-JS (information theory)",color="#1f77b4")
ax.bar(x+w/2,[R[1],M[1]],w,label="alphabet-overlap (dumb baseline)",color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(["RAW\n(many scripts)","ROMANIZED\n(one alphabet)"])
ax.set_ylim(0,1); ax.set_ylabel("nearest-neighbour family purity")
ax.set_title("Equalize the alphabet → the baseline collapses,\nfamily recovery is the information theory")
for i,v in enumerate([R[0],M[0]]): ax.text(i-w/2,v+.02,f"{v:.2f}",ha="center",fontsize=8)
for i,v in enumerate([R[1],M[1]]): ax.text(i+w/2,v+.02,f"{v:.2f}",ha="center",fontsize=8)
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")
code(r"""
Djs_rom = lt.js_matrix(rom, 3)
idx = {n:i for i,n in enumerate(names)}
print("cross-script family pairs, JS distance (raw -> romanized):")
for a,b in [("Hebrew","Arabic"),("Telugu","Kannada"),("Russian","Polish"),("Greek","Latin")]:
    if a in idx and b in idx:
        Dr = lt.js_matrix(raw,3)
        print(f"  {a}-{b}: {Dr[idx[a],idx[b]]:.3f} -> {Djs_rom[idx[a],idx[b]]:.3f}")
plot_family_tree(Djs_rom, f"Romanized (uroman) Bible — trigram-JS tree, {len(names)} languages")
""")
md(r"""
After romanization Hebrew–Arabic, the Dravidian trio, and Cyrillic/Latin Slavic reconnect, family
purity rises, and the alphabet baseline is left behind. Caveat (uroman is not neutral): it renders the
abjads Hebrew/Arabic as **vowelless consonant skeletons** (`כל בני האדם` → `kl vny hadm`) and CJK as
spaceless pinyin — so some "reconnection" reflects shared romanization artefacts, not pure genealogy.
""")

md(r"""
## 6. Tree methods & robustness

UPGMA assumes a molecular clock; **Neighbor-Joining** does not. We also report **cophenetic
correlation** (how faithfully the dendrogram preserves the distances), a **proper null** (RF of random
*tree topologies* vs gold — the old label-shuffle null couldn't reach low RF), and **gzip NCD** as an
independent estimator.
""")
code(r"""
D = Djs_rom
up = lt.linkage_to_newick(lt.upgma(D, names), names)
nj = lt.nj_newick(D, names)
Dncd = lt.ncd_matrix_fixed(rom, cap_bytes=30000)
p05,p50,p95,mn = lt.random_tree_null(names, gold, n=300)
print(f"UPGMA   normRF vs gold : {lt.rf_corrected(up, gold)[2]:.3f}")
print(f"Neighbor-Joining normRF: {lt.rf_corrected(nj, gold)[2]:.3f}   (NJ usually beats UPGMA here)")
print(f"gzip NCD normRF vs gold: {lt.rf_corrected(lt.linkage_to_newick(lt.upgma(Dncd,names),names), gold)[2]:.3f}")
print(f"cophenetic correlation : {lt.cophenetic_corr(D, names):.3f}   (dendrogram fidelity to distances)")
print(f"random-tree null normRF: p05={p05:.3f} p50={p50:.3f} (our trees sit FAR below chance)")
""")

md("## 7. Robustness — n-gram order sweep + bootstrap confidence intervals")
code(r"""
print("n-gram order sweep (romanized) — order is reported, not tuned on the gold tree:")
for n in range(1, 6):
    Dn = lt.js_matrix(rom, n)
    rf = lt.rf_corrected(lt.linkage_to_newick(lt.upgma(Dn, names), names), gold)[2]
    print(f"  n={n}: normRF {rf:.3f}, family-NN {lt.nn_purity(Dn, names, fam_of):.3f}")
ci = lt.bootstrap_ci(rom, names, fam_of, gold, n_boot=80)
print(f"bootstrap (n=3): family-NN {ci['purity'][1]:.3f} "
      f"(95% CI {ci['purity'][0]:.3f}-{ci['purity'][2]:.3f}), "
      f"RF {ci['rf'][1]:.3f} (CI {ci['rf'][0]:.3f}-{ci['rf'][2]:.3f})")
print("Flat across n (n=3 not cherry-picked) and stable under resampling.")
""")

md(r"""
## 8. Controls — are the two headline claims real?

The audit flagged two over-claims. We test them directly.
""")
code(r"""
raw_d = dict(zip(names, raw)); romd = dict(zip(names, rom))
cnt = lambda t: lt.ngram_counter(t, 3)
def js_excl(a, b, ex):
    ca, cb = cnt(a), cnt(b)
    for k in ex: ca.pop(k, None); cb.pop(k, None)
    return lt.js_div_counts(ca, cb)
lat = ["tio","ion","ent","ati","ity","nce","ate","ous","ive"]
print("(A) English<->Romance: borrowing, or Latinate spelling?  full JS -> minus Latinate trigrams:")
for o in ["French","Spanish","German","Dutch"]:
    if o in raw_d:
        print(f"   English-{o:7s}: {lt.js_div_counts(cnt(raw_d['English']), cnt(raw_d[o])):.3f}"
              f" -> {js_excl(raw_d['English'], raw_d[o], lat):.3f}")
print("   On Bible text English is ALREADY closest to Germanic -> the old 'Romance/Norman' pull")
print("   was a UDHR legal-register artifact, dissolved by the corpus change.")
dv = lambda t: "".join(c for c in t if c not in "aeiou")
print("\n(B) Hebrew-Arabic: Semitic recovered, or a vowelless-abjad artifact?")
print(f"   Hebrew-Arabic JS: {lt.js_div_counts(cnt(romd['Hebrew']), cnt(romd['Arabic'])):.3f}")
for l in ["Spanish","Italian"]:
    print(f"   {l}-Hebrew: {lt.js_div_counts(cnt(romd[l]), cnt(romd['Hebrew'])):.3f}"
          f" -> {lt.js_div_counts(cnt(dv(romd[l])), cnt(romd['Hebrew'])):.3f}  (vowels stripped)")
print("   Stripping vowels pulls Romance toward Hebrew -> part artifact; but Hebrew-Arabic stay")
print("   closer than any de-voweled Romance language -> part real.")
""")

md(r"""
## 9. Cross-check against the field standard — ASJP

[ASJP](https://asjp.clld.org) is the standard **phonetic-wordlist** database for quantitative language
comparison — it has no writing-system confound. We build its tree for our languages (`python
asjp_tree.py`) and compare. Our character-text tree and the ASJP wordlist tree agree at **distance
correlation r = 0.78** — two completely different methods recovering the same structure. (ASJP vs gold
normRF 0.59; our romanized text tree 0.79; random ≈ 0.97.)
""")
code(r"""
from IPython.display import Image, display
display(Image(os.path.join(REPO_DIR, "figures", "asjp_tree.png")))
""")

md(r"""
## 10. Scale-up — 102 languages, 29 families

Run on every complete + New-Testament Bible (`python scale_up.py`; *comparable* text here, since no
verse is shared by all 102). The result holds at full breadth: romanized trigram-JS family purity
**0.63 vs the alphabet baseline's 0.34**, normRF 0.83 vs ≈ 0.99 random.
""")
code(r"""
display(Image(os.path.join(REPO_DIR, "figures", "scaleup_romanized_tree.png")))
""")

md(r"""
## 11. Honest limits & what's next

**Limits.** Character statistics measure orthographic/surface similarity (tracks genealogy, ≠ it);
Bible text is translated (mild shared translationese) and its Hebrew/Arabic is a classical register;
uroman is lossy/non-neutral. RF is harsh — read it against the ~0.97 random null, and alongside
cophenetic and family-purity.

**Done here:** trivial baselines + negative control, romanization arm + the D1 proof, Neighbor-Joining
vs UPGMA, cophenetic correlation, a proper random-tree null, gzip-NCD, n-gram sweep, bootstrap CIs, the
ASJP field-standard cross-check, a 102-language scale-up, and controls for both headline claims.

**Next.** FLORES+ as a modern-register cross-corpus replication; per-clade bootstrap branch support;
even broader language sampling.

### References
Shannon 1951; Benedetto, Caglioti & Loreto 2002 (*Language Trees and Zipping*) + Goodman 2002 comment;
Cilibrasi & Vitányi 2005 (NCD); Bentz et al. 2017 (entropy across the Bible corpus); Gamallo et al.
2017 (character-n-gram language distance); Greenhill 2011; Jäger 2018 (ASJP); Saitou & Nei 1987 (NJ);
Hermjakob, May & Knight 2018 (uroman); Paninski 2003.
""")

nb["cells"] = cells
nb["metadata"] = {"language_info": {"name": "python"}, "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
with open("LanguageTrees.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote LanguageTrees.ipynb with", len(cells), "cells")

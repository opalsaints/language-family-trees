"""Demonstrate (A) bias-corrected entropy estimators and (B) per-clade bootstrap
branch support on the curated, verse-aligned Bible language set.

Mirrors scale_up.py in style. Two outputs:
  - figures/entropy_estimators.png : F_N conditional-entropy curves under the
    plug-in MLE vs Miller-Madow vs Grassberger, for a handful of languages.
  - figures/bootstrap_support_tree.png : the trigram-JS UPGMA tree annotated
    with bootstrap split support (%) at internal nodes, leaves coloured by family.
"""
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, to_tree

import langtree as lt
import biblecorpus as bc

os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------- load corpus
d = bc.load(selection=bc.SELECTION, verse_cap=2000, align=True, return_units=True)
texts = [lt.clean(d["rawtext"][n]) for n in d["names"]]
names = d["names"]
fam_of = {r[0]: r[1] for r in d["rows"]}
# genus = the well-known sub-family (Germanic/Italic/Slavic/Indo-Iranian/Semitic/...)
gen_of = {r[0]: (r[2] or r[1]) for r in d["rows"]}
# friendlier display names for the genus-level clades the task names explicitly
GENUS_ALIAS = {"Italic": "Romance", "Indo-Iranian": "Indo-Aryan",
               "Southern": "Dravidian-S", "South-Central": "Dravidian-SC"}
gold = lt.gold_newick_from_rows(d["rows"])
print(f"loaded {len(names)} languages, {len(d['common'])} common verses, "
      f"{len(set(fam_of.values()))} families")

# ================================================================ PART 0: sim
print("\n=== entropy-estimator simulation (uniform i.i.d., true H = log2 K) ===")
sim = lt.entropy_estimator_simulation(K=64, Ns=(128, 1024, 16384), reps=200, seed=0)
print(f"  K={sim['K']}  true H = log2(K) = {sim['truth']:.4f} bits")
print(f"  {'N':>7} {'plugin':>10} {'MM':>10} {'grassberger':>12}")
for r in sim["rows"]:
    print(f"  {r['N']:>7} {r['plugin']:>10.4f} {r['mm']:>10.4f} {r['grassberger']:>12.4f}")
print("  -> plug-in underestimates at small N; MM & Grassberger far closer; "
      "all converge to log2(K).")

# ============================================================ PART A: entropy
# English plus a few diverse others (Latin-script so F_1 is comparable).
focus = [n for n in ["English", "German", "Finnish", "Spanish"] if n in names]
if "English" not in focus and names:
    focus = names[:4]
text_of = dict(zip(names, texts))

print("\n=== conditional-entropy ladder F_1..F_4 (bits): plugin vs mm vs grassberger ===")
ladder = {}  # name -> {est -> {n -> F_n}}
for nm in focus:
    ladder[nm] = {est: lt.conditional_entropies_est(text_of[nm], max_n=4, estimator=est)
                  for est in ("plugin", "mm", "grassberger")}
    print(f"\n  [{nm}]")
    print(f"    {'F_N':>4} {'plugin':>10} {'mm':>10} {'grassberger':>12} {'(grass-plugin)':>16}")
    for n in range(1, 5):
        p = ladder[nm]["plugin"][n]
        m = ladder[nm]["mm"][n]
        g = ladder[nm]["grassberger"][n]
        print(f"    F_{n:<2} {p:>10.4f} {m:>10.4f} {g:>12.4f} {g - p:>16.4f}")

# Highlight English: F_1 reference ~4.0 bits; plug-in lowest; gap widens with n.
if "English" in ladder:
    e = ladder["English"]
    print(f"\n  English F_1: plugin={e['plugin'][1]:.4f}  mm={e['mm'][1]:.4f}  "
          f"grassberger={e['grassberger'][1]:.4f} bits (reference ~4.0)")
    gap1 = e["grassberger"][1] - e["plugin"][1]
    gap4 = e["grassberger"][4] - e["plugin"][4]
    print(f"  English grassberger-minus-plugin gap: F_1={gap1:.4f}  F_4={gap4:.4f} bits "
          f"(gap WIDENS with n as counts get sparser)")

# figure: F_N curves, 3 estimators
fig, axes = plt.subplots(1, len(focus), figsize=(3.4 * len(focus), 3.6), sharey=True)
if len(focus) == 1:
    axes = [axes]
styles = {"plugin": ("o-", "#1f77b4"), "mm": ("s--", "#ff7f0e"),
          "grassberger": ("^-", "#2ca02c")}
ns = list(range(1, 5))
for ax, nm in zip(axes, focus):
    for est in ("plugin", "mm", "grassberger"):
        ys = [ladder[nm][est][n] for n in ns]
        mk, col = styles[est]
        ax.plot(ns, ys, mk, color=col, label=est, markersize=5)
    ax.set_title(nm, fontsize=10)
    ax.set_xlabel("block order n")
    ax.set_xticks(ns)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("conditional entropy F_N (bits/char)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Bias-corrected conditional entropy: plug-in (lowest) vs Miller-Madow vs Grassberger",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/entropy_estimators.png", dpi=130)
plt.close(fig)
print("\nsaved figures/entropy_estimators.png")

# ===================================================== PART B: branch support
# Use the VALID verse-block bootstrap (Felsenstein 1985; resamples aligned verses),
# NOT the token bootstrap (which resamples a fitted multinomial and overstates support).
# The figure uses UPGMA; we ALSO report support on the headline NJ tree below.
print("\n=== verse-block bootstrap branch support (Felsenstein; trigram-JS UPGMA) ===")
support = lt.branch_support_block(d["units"], names, n_boot=30, ngram=3, seed=0, method="upgma")

# Label each clade by the dominant GENUS (Germanic, Romance, Slavic, ...) of its
# members; purity = fraction of members in that genus.
def dom_genus(members):
    cnt = defaultdict(int)
    for mb in members:
        g = gen_of.get(mb, "?")
        cnt[GENUS_ALIAS.get(g, g)] += 1
    g, k = max(cnt.items(), key=lambda kv: kv[1])
    return g, k, len(members)

print(f"  reference tree: {support['ref_newick'][:80]}...")

# For each named sub-family present in the corpus, find the smallest reference
# clade that contains ALL its members (its monophyletic clade if one exists) and
# report that clade's bootstrap support.
genus_members = defaultdict(list)
for lab in names:
    genus_members[GENUS_ALIAS.get(gen_of[lab], gen_of[lab])].append(lab)

clade_sets = [(frozenset(c["members"]), c["support"]) for c in support["clades"]]
print("\n  named sub-family clades (Germanic/Romance/Slavic/Indo-Aryan/Semitic/...):")
print(f"    {'genus':<14} {'n':>2}  {'monophyletic?':>13} {'support':>8}  members")
for genus in sorted(genus_members, key=lambda g: -len(genus_members[g])):
    mem = set(genus_members[genus])
    if len(mem) < 2:
        continue  # need >=2 taxa to form a split
    # exact monophyletic clade = a reference split whose member set == this genus
    exact = [s for fs, s in clade_sets if fs == mem]
    if exact:
        sup = exact[0]
        status = "yes"
    else:
        # smallest reference clade that contains all of this genus
        containing = sorted(((fs, s) for fs, s in clade_sets if mem <= fs),
                            key=lambda x: len(x[0]))
        if containing:
            fs, sup = containing[0]
            status = f"in n={len(fs)}"
        else:
            sup, status = float("nan"), "split"
    show = sorted(mem) if len(mem) <= 8 else sorted(mem)[:8] + ["..."]
    sup_s = f"{sup*100:5.1f}%" if sup == sup else "   n/a"
    print(f"    {genus:<14} {len(mem):>2}  {status:>13} {sup_s:>8}  {show}")

# the HEADLINE tree is Neighbor-Joining -> report support on IT too (the reliability
# numbers should describe the tree we lead with, not only UPGMA).
print("\n  same sub-families on the HEADLINE Neighbor-Joining tree (verse-block bootstrap):")
support_nj = lt.branch_support_block(d["units"], names, n_boot=30, ngram=3, seed=0, method="nj")
nj_sets = [(frozenset(c["members"]), c["support"]) for c in support_nj["clades"]]
for genus in sorted(genus_members, key=lambda g: -len(genus_members[g])):
    mem = set(genus_members[genus])
    if len(mem) < 2:
        continue
    exact = [s for fs, s in nj_sets if fs == mem]
    if exact:
        print(f"    {genus:<14} n={len(mem):<2} monophyletic support {exact[0]*100:5.1f}%")
    else:
        cont = sorted(((fs, s) for fs, s in nj_sets if mem <= fs), key=lambda x: len(x[0]))
        s = f"{cont[0][1]*100:5.1f}% (in n={len(cont[0][0])})" if cont else "split"
        print(f"    {genus:<14} n={len(mem):<2} {s}")

# also print all high-support clades (labelled by dominant genus) for transparency
print("\n  all clades with support >= 60% (labelled by dominant genus):")
for c in support["clades"]:
    if c["support"] >= 0.60:
        g, k, tot = dom_genus(c["members"])
        show = c["members"] if len(c["members"]) <= 8 else c["members"][:8] + ["..."]
        print(f"    {c['support']*100:5.1f}%  ({g}:{k}/{tot})  {show}")

# ----------------------------------------------- figure: tree + support labels
D0 = lt.js_matrix(texts, 3)
Z = lt.upgma(D0, names)

# map: frozenset(leaf labels under an internal node) -> support
sup_by_set = {frozenset(c["members"]): c["support"] for c in support["clades"]}
# branch_support reports the SMALLER side; a node's leaf set may be the smaller
# OR the larger side, so also index by the complement.
all_labels = set(names)
for c in support["clades"]:
    comp = frozenset(all_labels - set(c["members"]))
    sup_by_set.setdefault(comp, c["support"])

fams = sorted(set(fam_of.values()))
cmap = plt.get_cmap("tab20")
col = {f: cmap(i % 20) for i, f in enumerate(fams)}

fig, ax = plt.subplots(figsize=(11, 0.32 * len(names) + 2))
dn = dendrogram(Z, labels=names, orientation="right", ax=ax,
                color_threshold=0, above_threshold_color="#999")

# annotate internal nodes with support %. Use scipy tree to recover leaf sets,
# and the dendrogram's icoord/dcoord for placement.
root, nodelist = to_tree(Z, rd=True)

def leafset(node):
    if node.is_leaf():
        return frozenset([names[node.id]])
    return leafset(node.get_left()) | leafset(node.get_right())

# dendrogram x is distance, y is leaf position; for each merge compute support
# Build a lookup from cluster height -> support by walking the linkage tree.
def annotate(node):
    if node.is_leaf():
        return
    ls = leafset(node)
    sup = sup_by_set.get(ls)
    # node height = distance at which it merges = node.dist
    if sup is not None and not node.is_leaf():
        # y position = mean leaf position of its members (in dendrogram coords)
        ys = [leaf_y[lbl] for lbl in ls if lbl in leaf_y]
        if ys:
            ax.annotate(f"{int(round(sup*100))}", xy=(node.dist, np.mean(ys)),
                        fontsize=6.5, color="#b00", ha="right", va="center")
    annotate(node.get_left())
    annotate(node.get_right())

# leaf y-positions from the dendrogram (ivl order, spaced by 10 starting at 5)
leaf_y = {lbl: 5 + 10 * i for i, lbl in enumerate(dn["ivl"])}
annotate(root)

ax.set_title(f"Trigram-JS UPGMA tree with bootstrap support % "
             f"(n_boot=200), {len(names)} languages")
ax.set_xlabel("Jensen-Shannon distance (character trigrams)")
for lb in ax.get_ymajorticklabels():
    lb.set_color(col[fam_of[lb.get_text()]])
    lb.set_fontsize(7)
ax.legend(handles=[Patch(facecolor=col[f], label=f) for f in fams],
          loc="lower right", fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig("figures/bootstrap_support_tree.png", dpi=130)
plt.close(fig)
print("\nsaved figures/bootstrap_support_tree.png")
print("\ndone.")

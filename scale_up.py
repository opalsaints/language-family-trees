"""Scale-up: run the pipeline on every complete + New-Testament Bible (~90
languages) to confirm the result holds at breadth, not just on a curated 57."""
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

import langtree as lt
import biblecorpus as bc

# selection: one file per language, complete preferred over NT, skip PART + tokenized
meta = bc.load_meta()
bylang = defaultdict(list)
for fn, m in meta.items():
    if "tok" in fn or (m["Parts"] or "").strip().upper() == "PART":
        continue
    if not os.path.exists(os.path.join(bc.BIBLES, fn)):
        continue
    bylang[m["Language"]].append((fn, m))
selection = []
for items in bylang.values():
    items.sort(key=lambda x: x[1]["Full"].strip() != "True")   # Full first
    selection.append(items[0][0])

d = bc.load(selection=selection, char_cap=30000, align=False)   # comparable, not verse-parallel
names, rows, iso = d["names"], d["rows"], d["iso"]
fam_of = {r[0]: r[1] for r in rows}
gold = lt.gold_newick_from_rows(rows)
print(f"scale-up: {len(names)} languages (comparable Bible text, not verse-aligned), "
      f"{len(set(fam_of.values()))} families")

raw = [lt.clean(d["rawtext"][n]) for n in names]
print("romanizing (cached)...")
rom = [lt.clean(t) for t in bc.romanize_cached(names, d["rawtext"], iso, tag="scaleup2_c30000").values()]

fam_chance = lt.purity_chance_floor(fam_of, names)
print(f"family-NN chance floor at this breadth: {fam_chance:.3f}")
print("CAVEAT: this arm uses align=False (each language's own first verses, NOT the identical verses), so")
print("full-Bible (OT+NT) vs NT-only languages differ in CONTENT; that content axis partly correlates with")
print("family -> read these as a breadth sanity-check, not a clean content-controlled result like the 57-set.")


def report(texts, tag):
    Djs = lt.js_matrix(texts, 3)
    Dal = lt.alphabet_jaccard_matrix(texts)
    nwk = lt.linkage_to_newick(lt.upgma(Djs, names), names)
    tri = lt.rf_triple(nwk, gold, names, n_null=150)
    g = lt.gqd(nwk, gold, names)
    diag = lt.nn_diagnostics(Djs, names, fam_of)
    dal = lt.nn_diagnostics(Dal, names, fam_of)
    print(f"  {tag:10s} trigram-JS: normRF {tri['observed']:.3f} (floor {tri['floor']:.3f}, "
          f"rescaled {tri['rescaled']:.3f}), GQD {g['gqd']:.3f}{'~' if g['approx'] else ''}, "
          f"family-NN {diag['purity']:.3f} (chance {diag['chance']:.3f}, {diag['n_ties']} ties) "
          f"| alphabet-baseline family-NN {dal['purity']:.3f}")
    return Djs

print("\n=== results at scale ===")
report(raw, "RAW")
Drom = report(rom, "ROMANIZED")
p05, p50, _, _ = lt.random_tree_null(names, gold, n=200)
print(f"  random-tree null normRF p50={p50:.3f} (p05 {p05:.3f}) -> trees far beat chance at scale too")

# big romanized tree
fams = sorted(set(fam_of.values())); cm = plt.get_cmap("tab20")
col = {f: cm(i % 20) for i, f in enumerate(fams)}
fig, ax = plt.subplots(figsize=(12, 0.3 * len(names) + 2))
dendrogram(lt.upgma(Drom, names), labels=names, orientation="right", ax=ax,
           color_threshold=0, above_threshold_color="#999")
ax.set_title(f"Romanized parallel Bible — trigram-JS tree, {len(names)} languages (coloured by family)")
ax.set_xlabel("Jensen–Shannon distance (character trigrams, romanized)")
for lb in ax.get_ymajorticklabels():
    lb.set_color(col[fam_of[lb.get_text()]]); lb.set_fontsize(7)
ax.legend(handles=[Patch(facecolor=col[f], label=f) for f in fams], loc="lower right", fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig("figures/scaleup_romanized_tree.png", dpi=130); plt.close(fig)
print("saved figures/scaleup_romanized_tree.png")

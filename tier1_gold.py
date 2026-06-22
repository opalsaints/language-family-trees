"""
Quantitative evaluation: how close is our inferred tree to the TRUE language
family tree (Glottolog grouping)? Normalized Robinson-Foulds distance.
"""
import random
import langtree as lt
from tier1 import names, Djs, Dncd   # reuse Tier-1 distance matrices

# Gold tree from known Glottolog families (Indo-European nests Germanic+Romance
# +Slavic; Uralic and Turkic are separate). NOTE: English is Germanic here, so
# our "English-with-Romance" borrowing artefact will count as an error.
GOLD = ("((((Swedish,Danish),(German,Dutch,English)),"
        "((Spanish,Portuguese),(Italian,French)),(Polish,Czech)),"
        "(Finnish,Hungarian),Turkish);")

def tree_newick(D):
    return lt.linkage_to_newick(lt.upgma(D, names), names)

js_nwk, ncd_nwk = tree_newick(Djs), tree_newick(Dncd)

print("Robinson-Foulds vs the true family tree (0 = identical topology):")
for tag, nwk in [("JS divergence", js_nwk), ("gzip NCD", ncd_nwk)]:
    rf, mx, norm = lt.robinson_foulds(nwk, GOLD)
    print(f"  {tag:14s}: RF = {rf}/{mx}  (normalized {norm:.2f})")

# random baseline: shuffle leaf labels on the JS topology
random.seed(0)
base = []
for _ in range(500):
    shuf = names[:]; random.shuffle(shuf)
    nwk = lt.linkage_to_newick(lt.upgma(Djs, names), shuf)
    base.append(lt.robinson_foulds(nwk, GOLD)[2])
print(f"  random baseline: normalized RF = {sum(base)/len(base):.2f} (avg of 500 shuffles)")
print("\nLower than the random baseline => the tree captures real family structure.")

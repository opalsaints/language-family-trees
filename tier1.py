"""
Tier 1 — the guaranteed core.
Latin-script languages -> trigram Jensen-Shannon divergence + gzip NCD
-> UPGMA trees, validated against Shannon 1951 numbers and known families.
"""
import numpy as np
from nltk.corpus import udhr
import langtree as lt

# display name -> (NLTK fileid, family)
LANGS = {
    "English":    ("English-Latin1",              "Germanic"),
    "German":     ("German_Deutsch-Latin1",       "Germanic"),
    "Dutch":      ("Dutch_Nederlands-Latin1",     "Germanic"),
    "Swedish":    ("Swedish_Svenska-Latin1",      "Germanic"),
    "Danish":     ("Danish_Dansk-Latin1",         "Germanic"),
    "Spanish":    ("Spanish-Latin1",              "Romance"),
    "Italian":    ("Italian-Latin1",              "Romance"),
    "French":     ("French_Francais-Latin1",      "Romance"),
    "Portuguese": ("Portuguese_Portugues-Latin1", "Romance"),
    "Polish":     ("Polish-Latin2",               "Slavic"),
    "Czech":      ("Czech-UTF8",                   "Slavic"),
    "Finnish":    ("Finnish_Suomi-Latin1",        "Uralic"),
    "Hungarian":  ("Hungarian_Magyar-UTF8",       "Uralic"),
    "Turkish":    ("Turkish_Turkce-UTF8",         "Turkic"),
}
FAMILY_COLOR = {"Germanic": "#1f77b4", "Romance": "#d62728", "Slavic": "#2ca02c",
                "Uralic": "#9467bd", "Turkic": "#ff7f0e"}

names = list(LANGS)
colors = {n: FAMILY_COLOR[LANGS[n][1]] for n in names}

# ---- load + clean, equalize sample size (control finite-sample bias) ----
raw = {n: lt.clean(udhr.raw(LANGS[n][0])) for n in names}
m = min(len(t) for t in raw.values())
texts = [raw[n][:m] for n in names]
print(f"{len(names)} languages, each truncated to {m} chars")

# ---- validation 1: Shannon entropy ladder on English ----
F = lt.conditional_entropies(raw["English"], max_n=4)
print("\nShannon validation (English UDHR):")
print(f"  F1={F[1]:.2f}  F2={F[2]:.2f}  F3={F[3]:.2f} bits/char")
print("  (Shannon 1951 ref: F1=4.03, F2=3.32, F3=3.1)")

# ---- distances ----
Djs = lt.js_distance_matrix(texts, n=3)
Dncd = lt.ncd_matrix(texts)

# ---- trees ----
lt.plot_tree(Djs, names, "Tier 1 — trigram Jensen-Shannon divergence (Latin script)",
             "figures/tier1_js_tree.png", colors)
lt.plot_tree(Dncd, names, "Tier 1 — gzip Normalized Compression Distance (Latin script)",
             "figures/tier1_ncd_tree.png", colors)

# ---- validation 2: does it recover families? (5 clusters) ----
def report(D, tag):
    print(f"\n{tag}: 5-cluster assignment")
    for cid, members in sorted(lt.cluster_labels(D, names, 5).items()):
        fams = {LANGS[x][1] for x in members}
        flag = "OK" if len(fams) == 1 else "MIXED"
        print(f"  cluster {cid} [{flag}]: {', '.join(members)}")

report(Djs, "JS")
report(Dncd, "NCD")

# ---- method agreement (correlation of the two distance matrices) ----
iu = np.triu_indices(len(names), 1)
r = np.corrcoef(Djs[iu], Dncd[iu])[0, 1]
print(f"\nJS vs NCD distance correlation: r = {r:.3f}")
print("\nSaved: figures/tier1_js_tree.png, figures/tier1_ncd_tree.png")

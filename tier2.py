"""
Tier 2 — the cross-script twist.
Mix scripts (Latin, Cyrillic, Hebrew, Arabic, Greek). On RAW text the tree
clusters by WRITING SYSTEM, not language family (the instructive failure).
Then romanize everything with uroman and re-run: do the real families
(Semitic: Hebrew-Arabic; Turkic: Turkish-Kazakh) re-emerge?
"""
import numpy as np
from nltk.corpus import udhr
import uroman as ur
import langtree as lt

# name -> (fileid, family, script, uroman lcode)
LANGS = {
    "English":  ("English-Latin1",          "Indo-European", "Latin",    "eng"),
    "Spanish":  ("Spanish-Latin1",          "Indo-European", "Latin",    "spa"),
    "German":   ("German_Deutsch-Latin1",   "Indo-European", "Latin",    "deu"),
    "Turkish":  ("Turkish_Turkce-UTF8",     "Turkic",        "Latin",    "tur"),
    "Kazakh":   ("Kazakh-UTF8",             "Turkic",        "Cyrillic", "kaz"),
    "Russian":  ("Russian-UTF8",            "Indo-European", "Cyrillic", "rus"),
    "Greek":    ("Greek_Ellinika-UTF8",     "Indo-European", "Greek",    "ell"),
    "Hebrew":   ("Hebrew_Ivrit-UTF8",       "Semitic",       "Hebrew",   "heb"),
    "Arabic":   ("Arabic_Alarabia-Arabic",  "Semitic",       "Arabic",   "ara"),
}
SCRIPT_COLOR = {"Latin": "#1f77b4", "Cyrillic": "#2ca02c", "Greek": "#9467bd",
                "Hebrew": "#d62728", "Arabic": "#ff7f0e"}
FAMILY_COLOR = {"Indo-European": "#1f77b4", "Turkic": "#ff7f0e", "Semitic": "#d62728"}
names = list(LANGS)


def equalize(d):
    m = min(len(t) for t in d.values())
    return [d[n][:m] for n in names], m


# ---------- RAW (each language in its own script) ----------
raw = {n: lt.clean(udhr.raw(LANGS[n][0])) for n in names}
raw_texts, m1 = equalize(raw)
Draw = lt.js_distance_matrix(raw_texts, n=3)
lt.plot_tree(Draw, names, "Tier 2 RAW — trigram JS divergence (mixed scripts)",
             "figures/tier2_raw_tree.png",
             {n: SCRIPT_COLOR[LANGS[n][2]] for n in names})

# ---------- ROMANIZED (uroman -> common Latin space) ----------
uro = ur.Uroman()
rom = {n: lt.clean(uro.romanize_string(udhr.raw(LANGS[n][0]), lcode=LANGS[n][3]))
       for n in names}
rom_texts, m2 = equalize(rom)
Drom = lt.js_distance_matrix(rom_texts, n=3)
lt.plot_tree(Drom, names, "Tier 2 ROMANIZED (uroman) — trigram JS divergence",
             "figures/tier2_rom_tree.png",
             {n: FAMILY_COLOR[LANGS[n][1]] for n in names})

# ---------- analysis ----------
idx = {n: i for i, n in enumerate(names)}


def d(D, a, b):
    return D[idx[a], idx[b]]


def nearest(D, a):
    order = sorted((x for x in names if x != a), key=lambda x: d(D, a, x))
    return order[0]


def purity(D, key):  # fraction whose nearest neighbour shares the attribute
    f = {"family": 1, "script": 2}[key]
    hits = sum(LANGS[nearest(D, n)][f] == LANGS[n][f] for n in names)
    return hits / len(names)


print(f"RAW: equalized to {m1} chars | ROMANIZED: {m2} chars\n")
print("Nearest-neighbour by FAMILY (higher = recovers real relatedness):")
print(f"  raw       = {purity(Draw,'family'):.2f}")
print(f"  romanized = {purity(Drom,'family'):.2f}")
print("Nearest-neighbour by SCRIPT (higher = confounded by writing system):")
print(f"  raw       = {purity(Draw,'script'):.2f}")
print(f"  romanized = {purity(Drom,'script'):.2f}")

print("\nKey family pairs — JS distance (raw -> romanized):")
for a, b in [("Hebrew", "Arabic"), ("Turkish", "Kazakh")]:
    print(f"  {a}-{b}: {d(Draw,a,b):.3f} -> {d(Drom,a,b):.3f}")
print("\nWho each language's nearest neighbour is (raw -> romanized):")
for n in names:
    print(f"  {n:9s}: {nearest(Draw,n):9s} -> {nearest(Drom,n)}")
print("\nSaved: figures/tier2_raw_tree.png, figures/tier2_rom_tree.png")

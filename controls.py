"""Controls for two claims the audit flagged as overstated.

(A) "English clusters with Romance = Norman borrowing." Test whether it's really
    shared LATINATE ORTHOGRAPHY (-tion/-ion/-ent...): remove that trigram family
    and see if the English->Romance pull collapses.
(B) "Semitic (Hebrew-Arabic) recovered after romanization." Test whether it's a
    VOWELLESS-ABJAD artifact: uroman writes Hebrew/Arabic without vowels; strip
    vowels from a vowel-rich language and see if it drifts toward the Semitic pair.
"""
import langtree as lt
import biblecorpus as bc

d = bc.load(verse_cap=2000, char_cap=30000)
names = d["names"]
raw = {n: lt.clean(d["rawtext"][n]) for n in names}
rom = {n: lt.clean(t) for n, t in zip(names, bc.romanize_cached(names, d["rawtext"], d["iso"], tag="v2000_c30000").values())}

def cnt(t):
    return lt.ngram_counter(t, 3)

def js_excl(a, b, excl):
    ca, cb = cnt(a), cnt(b)
    for k in excl:
        ca.pop(k, None); cb.pop(k, None)
    return lt.js_div_counts(ca, cb)

print("=== (A) English<->Romance: borrowing, or Latinate orthography? ===")
EN = raw["English"]
latinate = ["tio", "ion", "ent", "ati", "ity", "nce", "ate", "ral", "ion", "ous", "ive"]
toks = [EN[i:i+3] for i in range(len(EN) - 2)]
rate = sum(t in ("tio", "ion") for t in toks) / len(toks)
print(f"  English 'tio'/'ion' trigram rate: {rate*100:.2f}% of all trigrams")
for other in ["French", "Spanish", "Italian", "German", "Dutch"]:
    full = lt.js_div_counts(cnt(EN), cnt(raw[other]))
    excl = js_excl(EN, raw[other], latinate)
    print(f"  English-{other:8s} JS: {full:.3f} -> {excl:.3f} (Δ {excl-full:+.3f}) after removing the Latinate trigram family")
print("  If removing -tion/-ion/-ent moves English AWAY from Romance more than from Germanic,")
print("  the 'borrowing' signal is largely shared Latinate spelling, not genealogy.")

print("\n=== (B) Hebrew-Arabic: Semitic recovered, or a vowelless-abjad artifact? ===")
def devowel(t):
    return "".join(c for c in t if c not in "aeiou")
HE, AR = rom["Hebrew"], rom["Arabic"]
print(f"  Hebrew-Arabic JS (romanized, both vowelless): {lt.js_div_counts(cnt(HE), cnt(AR)):.3f}")
for lang in ["Spanish", "Italian", "Latin"]:
    base = lt.js_div_counts(cnt(rom[lang]), cnt(HE))
    dv = lt.js_div_counts(cnt(devowel(rom[lang])), cnt(HE))
    print(f"  {lang}-Hebrew JS: {base:.3f} -> {dv:.3f} (Δ {dv-base:+.3f}) after stripping {lang}'s vowels")
print("  If stripping a vowel-rich language's vowels pulls it toward Hebrew, then part of the")
print("  Hebrew-Arabic 'reconnection' is shared vowellessness (abjad + uroman), not pure genealogy.")

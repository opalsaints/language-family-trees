"""Cognate-aware arm (ARM = DESCENT, not surface form).

This is the only arm that targets genuine COGNACY rather than surface string/
compression similarity. We pull the ASJP wordlists for our corpus languages
(exactly as asjp_tree.py does: languages.csv ISO639P3code -> doculect IDs;
forms.csv Language_ID / Parameter_ID / Form, ASJPcode strings), pick per ISO the
doculect with the most concepts (>=20), build a lingpy Wordlist, and run cognate
detection (LexStat with a trained scorer; SCA fallback if LexStat training is too
heavy). From the cognate CLASSES we compute a pairwise cognate DISTANCE
    d(a,b) = 1 - (shared cognate concepts) / (concepts both have)
-> UPGMA + NJ trees.

We then report, vs the Glottolog gold:
  * rf_triple (RF cannot credit a correct refinement) AND gqd (scores 0 for a
    correct refinement) -- both, per project rules
  * nn_diagnostics (family nearest-neighbour purity with chance floor + tie policy)
and AGREEMENT (Mantel r,p) between the cognate tree and
  (a) our romanized text tree (trigram Jensen-Shannon), and
  (b) the ASJP raw Levenshtein tree (mean normalized Levenshtein, asjp_tree.py).

If LexStat is what runs, this validates whether the surface-text tree matches a
genuine cognate-based phylogeny built by descent-aware sequence comparison.
"""
import csv
import logging
import os
import sys
import numpy as np

logging.disable(logging.INFO)        # quiet lingpy's per-concept INFO spam

import langtree as lt
import biblecorpus as bc

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(os.path.abspath(__file__))
ASJP = os.path.join(HERE, "corpus", "asjp", "cldf")

# ----- ASJP loading (mirrors asjp_tree.py) ----------------------------------

def lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la or not lb:
        return la or lb
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def ndist(a, b):
    m = max(len(a), len(b))
    return lev(a, b) / m if m else 0.0


def asjp_lev_matrix(wordlists):
    """ASJP raw Levenshtein tree input: mean normalized Lev over shared concepts."""
    m = len(wordlists)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            shared = set(wordlists[i]) & set(wordlists[j])
            d = np.mean([ndist(wordlists[i][c], wordlists[j][c]) for c in shared]) if shared else 1.0
            D[i, j] = D[j, i] = d
    return D


def asjp_tokenize(form):
    """Turn an ASJPcode string into per-symbol tokens (raw-Levenshtein arm only).

    ASJPcode is one-symbol-per-segment except for modifiers '~' (combine prev two
    into a digraph) and '$' (combine prev three). We collapse those so e.g.
    'tS~' -> one token 'tS'. Used for the ASJP raw-Levenshtein comparison arm.
    """
    raw = list(form)
    toks = []
    for ch in raw:
        if ch in '~$' and toks:
            # merge: '~' combines last 2 base symbols already emitted, '$' last 3.
            n = 2 if ch == '~' else 3
            merge = toks[-n:] if len(toks) >= n else toks[:]
            toks = toks[:len(toks) - len(merge)] + [''.join(merge)]
        elif ch in ' *"':       # spacing / stress markers: drop
            continue
        else:
            toks.append(ch)
    return [t for t in toks if t]


def load_asjp_for_corpus(verse_cap=2000, char_cap=30000, min_concepts=20):
    d = bc.load(verse_cap=verse_cap, char_cap=char_cap)
    names, rows, iso = d["names"], d["rows"], d["iso"]

    iso_want = {iso[n] for n in names if iso[n]}
    iso_to_ids = {}
    with open(os.path.join(ASJP, "languages.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r["ISO639P3code"]
            if code in iso_want:
                iso_to_ids.setdefault(code, []).append(r["ID"])
    candidate_ids = {i for ids in iso_to_ids.values() for i in ids}

    # forms[doculect][concept] = asjpcode Form (for raw-Levenshtein arm)
    # segs[doculect][concept]  = IPA token list from the Segments column (for lingpy)
    # Keep FIRST synonym per concept (consistent with asjp_tree.py).
    forms, segs = {}, {}
    with open(os.path.join(ASJP, "forms.csv"), encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        li, pi, fi = hdr.index("Language_ID"), hdr.index("Parameter_ID"), hdr.index("Form")
        si = hdr.index("Segments")
        for row in rd:
            if row[li] in candidate_ids:
                if row[pi] in forms.get(row[li], {}):
                    continue
                forms.setdefault(row[li], {})[row[pi]] = row[fi]
                segs.setdefault(row[li], {})[row[pi]] = row[si].split()

    keep, wl, wl_seg, krows, doculect_of = [], [], [], [], {}
    for n in names:
        ids = [i for i in iso_to_ids.get(iso[n], []) if i in forms]
        if not ids:
            continue
        best = max(ids, key=lambda i: len(forms[i]))
        if len(forms[best]) < min_concepts:
            continue
        keep.append(n)
        wl.append(forms[best])
        wl_seg.append(segs[best])
        doculect_of[n] = best
        krows.append(next(r for r in rows if r[0] == n))

    missing = [n for n in names if n not in keep]
    return d, names, rows, iso, keep, wl, wl_seg, krows, doculect_of, missing


# ----- lingpy cognate detection ---------------------------------------------

def build_lingpy_wordlist(keep, wl_seg):
    """keep: our labels; wl_seg[i]: {concept_id -> IPA token list} for keep[i]
    (from the ASJP Segments column, which lingpy's sound-class model understands).
    Returns a lingpy Wordlist (doculect=our label, concept=ASJP param id)."""
    from lingpy import Wordlist
    D = {0: ["doculect", "concept", "ipa", "tokens"]}
    idx = 1
    for lab, conceptmap in zip(keep, wl_seg):
        for concept, toks in conceptmap.items():
            toks = [t for t in toks if t]
            if not toks:
                continue
            D[idx] = [lab, str(concept), " ".join(toks), toks]
            idx += 1
    return Wordlist(D)


def run_cognates(lexwl, runs=1000, threshold=0.55, seed=0):
    """Try LexStat (trained scorer); fall back to SCA, then edit-distance.
    Returns (method_name, cognate_lookup) where cognate_lookup[(label,concept)]=class.

    LexStat's get_scorer derives the random correspondence distribution by
    shuffling (stdlib random / numpy). Seed both so the trained scorer -- and thus
    the cognate classes and every downstream number -- is reproducible."""
    import random as _random
    from lingpy import LexStat
    method = None
    lex = LexStat(lexwl, check=False)
    try:
        _random.seed(seed)
        np.random.seed(seed)
        lex.get_scorer(runs=runs)
        lex.cluster(method="lexstat", threshold=threshold, ref="cogid")
        method = "lexstat"
    except Exception as e:  # noqa: BLE001
        print(f"  [LexStat training failed: {type(e).__name__}: {e}] -> SCA fallback", flush=True)
        try:
            lex.cluster(method="sca", threshold=0.45, ref="cogid")
            method = "sca"
        except Exception as e2:  # noqa: BLE001
            print(f"  [SCA failed: {type(e2).__name__}: {e2}] -> edit-distance fallback", flush=True)
            lex.cluster(method="edit-dist", threshold=0.75, ref="cogid")
            method = "edit-dist"

    lookup = {}
    for i in lex:
        lab = lex[i, "doculect"]
        concept = lex[i, "concept"]
        cog = lex[i, "cogid"]
        # keep first cognate id per (lab,concept) (synonyms collapsed at load)
        lookup.setdefault((lab, concept), cog)
    return method, lookup


def cognate_distance_matrix(keep, lookup):
    """d(a,b) = 1 - (#concepts where a,b share a cognate class) / (#concepts both have).
    Cognate classes are GLOBAL across the wordlist, but two languages only count a
    concept as cognate if they were assigned the SAME class for that SAME concept."""
    # per-label: {concept -> class}
    bylab = {lab: {} for lab in keep}
    for (lab, concept), cog in lookup.items():
        if lab in bylab:
            bylab[lab][concept] = cog
    n = len(keep)
    D = np.zeros((n, n))
    npairs = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = bylab[keep[i]], bylab[keep[j]]
            shared = set(ca) & set(cb)
            if not shared:
                D[i, j] = D[j, i] = 1.0
                continue
            same = sum(1 for c in shared if ca[c] == cb[c])
            d = 1.0 - same / len(shared)
            D[i, j] = D[j, i] = d
            npairs[i, j] = npairs[j, i] = len(shared)
    return D, npairs


# ----- main -----------------------------------------------------------------

def main():
    print("=== Cognate-aware ARM (descent, via lingpy on ASJP wordlists) ===\n")
    (d, names, rows, iso, keep, wl, wl_seg, krows, doculect_of,
     missing) = load_asjp_for_corpus()
    fam_of = {r[0]: r[1] for r in rows}

    print(f"ASJP covers {len(keep)}/{len(names)} of our languages "
          f"(avg {np.mean([len(w) for w in wl]):.0f} concepts each)")
    if missing:
        print("  graceful-skip (not in ASJP / <20 concepts):", ", ".join(missing))

    # --- cognate detection ---
    print("\nBuilding lingpy Wordlist and running cognate detection ...", flush=True)
    lexwl = build_lingpy_wordlist(keep, wl_seg)
    print(f"  Wordlist: {lexwl.width} doculects x {lexwl.height} concepts, "
          f"{len(lexwl)} forms total", flush=True)
    method, lookup = run_cognates(lexwl, runs=1000, threshold=0.55)
    print(f"  cognate method used: {method}")
    n_classes = len({v for v in lookup.values()})
    print(f"  distinct cognate classes: {n_classes}")

    Dcog, npairs = cognate_distance_matrix(keep, lookup)
    tri = np.triu_indices(len(keep), 1)
    print(f"  mean shared concepts per pair: {npairs[tri].mean():.1f} "
          f"(min {npairs[tri].min()})")

    # --- trees ---
    gold = lt.gold_newick_from_rows(krows)
    cog_upgma = lt.linkage_to_newick(lt.upgma(Dcog, keep), keep)
    cog_nj = lt.nj_newick(Dcog, keep)

    # comparison arms on the SAME subset
    Dasjp = asjp_lev_matrix(wl)                                   # ASJP raw Levenshtein
    rom = bc.romanize_cached(names, d["rawtext"], iso, tag="v2000_c30000")
    rom_keep = [lt.clean(rom[n]) for n in keep]
    Dtext = lt.js_matrix(rom_keep, 3)                            # our text tree

    # --- tree-vs-gold: BOTH rf_triple and gqd ---
    print("\n=== cognate tree vs Glottolog gold (BOTH metrics) ===")
    for tag, nwk in [("UPGMA", cog_upgma), ("NJ", cog_nj)]:
        rt = lt.rf_triple(nwk, gold, keep)
        gq = lt.gqd(nwk, gold, keep)
        print(f"  [{tag}] rf_triple: observed={rt['observed']:.3f} "
              f"floor={rt['floor']:.3f} null_p50={rt['null_p50']:.3f} "
              f"rescaled={rt['rescaled']:.3f}")
        print(f"         gqd={gq['gqd']:.3f} "
              f"(resolved_in_gold={gq['resolved_in_gold']}, approx={gq['approx']})")

    # --- family NN diagnostics (chance floor + tie policy) ---
    print("\n=== family nearest-neighbour purity (cognate distances) ===")
    nd = lt.nn_diagnostics(Dcog, keep, fam_of)
    print(f"  purity(tie=first)={nd['purity']:.3f}  "
          f"purity(tie=miss)={nd['purity_tiemiss']:.3f}  "
          f"n_ties={nd['n_ties']}  n={nd['n']}  chance_floor={nd['chance']:.3f}")

    # --- agreement (Mantel) ---
    print("\n=== agreement (Mantel r, p) ===")
    r_text, p_text = lt.mantel(Dcog, Dtext, perms=9999)
    r_asjp, p_asjp = lt.mantel(Dcog, Dasjp, perms=9999)
    print(f"  cognate vs our text tree (trigram-JS) : r={r_text:.3f}  p={p_text:.4f}")
    print(f"  cognate vs ASJP raw Levenshtein tree  : r={r_asjp:.3f}  p={p_asjp:.4f}")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import dendrogram
    os.makedirs("figures", exist_ok=True)
    fams = sorted(set(fam_of[k] for k in keep))
    cm = plt.get_cmap("tab20")
    col = {f: cm(i % 20) for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(11, 0.33 * len(keep) + 2))
    dendrogram(lt.upgma(Dcog, keep), labels=keep, orientation="right", ax=ax,
               color_threshold=0, above_threshold_color="#999")
    ax.set_title(f"Cognate tree ({method} cognate classes, ASJP wordlists), "
                 f"{len(keep)} languages\n"
                 "descent-aware: 1 - shared-cognate proportion over shared concepts")
    ax.set_xlabel("cognate distance (1 - shared cognate / shared concepts)")
    for lb in ax.get_ymajorticklabels():
        lb.set_color(col[fam_of[lb.get_text()]])
        lb.set_fontsize(8)
    ax.legend(handles=[Patch(facecolor=col[f], label=f) for f in fams],
              loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig("figures/cognate_tree.png", dpi=130)
    plt.close(fig)
    print("\nsaved figures/cognate_tree.png")

    return dict(method=method, keep=keep, n_classes=n_classes,
                cog_upgma=cog_upgma, cog_nj=cog_nj)


if __name__ == "__main__":
    main()

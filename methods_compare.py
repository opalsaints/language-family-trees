#!/usr/bin/env python3
"""
methods_compare.py  --  ARM: alternative-distance comparison.

On the curated ROMANIZED Bible set (bc.load(verse_cap=2000, char_cap=30000),
uroman-romanized, tag="v2000_c30000"), compare four distance methods head-to-head
against the SAME family gold tree, each scored with:
  - rf_triple   (observed / floor / null_p50 / rescaled normalized RF)
  - gqd         (generalized quartet distance; 0 for a correct refinement)
  - nn_diagnostics (family nearest-neighbour purity + chance floor + ties)

Methods:
  (1) trigram-JS plug-in            js_matrix(n=3, alpha=0)
  (2) trigram-JS Lidstone-smoothed  js_matrix(n=3, alpha=0.3)
  (3) perplexity (Gamallo char-LM)  perplexity_distance_matrix(n=3, alpha=0.1)
  (4) gzip NCD per-pair-floor       ncd_matrix_v2()

Also reports, per method, the vocabulary-size bias:
  corr(distance, maxK)  and  corr(distance, |dK|)
where K = #distinct trigrams in a language and the correlation is Pearson over all
unique language pairs (maxK = max(K_i,K_j); |dK| = |K_i-K_j|).

Trees built with UPGMA (consistent with the project's main pipeline). Labels are
sanitized (underscore -> hyphen) BEFORE any tree work because dendropy (used inside
gqd) silently turns unquoted Newick underscores into spaces, which otherwise breaks
leaf-label matching for Farsi_Persian / Mandarin_Chinese.

Run:  /opt/miniconda3/bin/python3 methods_compare.py
"""
import os, sys, time
import numpy as np

# Run from this script's own directory (the repo) so relative corpus/figures paths
# resolve on any machine — NOT a hardcoded local path (that broke Colab/Snellius).
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import langtree as lt
import biblecorpus as bc

N_NULL = 300          # rf_triple random-tree null draws
TREE_METHOD = "upgma" # consistent with the main pipeline


def san(s):
    """dendropy reads unquoted Newick underscores as spaces -> sanitize everywhere."""
    return s.replace("_", "-")


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def offdiag_pairs(D, K):
    """Return (dist_vals, maxK_vals, absdK_vals) over all i<j pairs."""
    n = D.shape[0]
    dvals, maxk, adk = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            dvals.append(D[i, j])
            maxk.append(max(K[i], K[j]))
            adk.append(abs(K[i] - K[j]))
    return np.array(dvals), np.array(maxk), np.array(adk)


def score_method(name, D, names, gold, group, K):
    """Build UPGMA tree from D, score it, and compute the bias correlations."""
    skipped = []
    # tree + topology scores
    if TREE_METHOD == "upgma":
        nwk = lt.linkage_to_newick(lt.upgma(D, names), names)
    else:
        nwk = lt.nj_newick(D, names)
    rf = lt.rf_triple(nwk, gold, names, n_null=N_NULL)
    g = lt.gqd(nwk, gold, names)
    nn = lt.nn_diagnostics(D, names, group)
    # vocabulary-size bias
    dvals, maxk, adk = offdiag_pairs(D, K)
    r_maxK = pearson(dvals, maxk)
    r_absdK = pearson(dvals, adk)
    return {
        "name": name,
        "rf_obs": rf["observed"], "rf_floor": rf["floor"],
        "rf_null": rf["null_p50"], "rf_resc": rf["rescaled"],
        "gqd": g["gqd"], "gqd_approx": g["approx"],
        "purity": nn["purity"], "purity_tiemiss": nn["purity_tiemiss"],
        "n_ties": nn["n_ties"], "chance": nn["chance"],
        "r_maxK": r_maxK, "r_absdK": r_absdK,
        "skipped": skipped,
    }


def main():
    t_all = time.time()
    print("=" * 92)
    print("ARM  --  alternative-distance comparison (ROMANIZED curated set, UPGMA, same gold)")
    print("=" * 92)

    # ---- curated romanized set --------------------------------------------------
    d = bc.load(verse_cap=2000, char_cap=30000)
    names = [san(n) for n in d["names"]]
    rows = [(san(r[0]), r[1], r[2], r[3]) for r in d["rows"]]
    group = {san(r[0]): r[1] for r in d["rows"]}          # label -> family
    gold = lt.gold_newick_from_rows(rows)

    rom_raw = bc.romanize_cached(d["names"], d["rawtext"], d["iso"], tag="v2000_c30000")
    rom = {san(n): lt.clean(rom_raw[n]) for n in d["names"]}
    texts = [rom[n] for n in names]

    # K = number of DISTINCT trigrams per language (the vocabulary-size variable)
    K = [len(lt.ngram_counter(t, 3)) for t in texts]
    chance = lt.purity_chance_floor(group, names)

    print(f"languages          : {len(names)}")
    print(f"families            : {len(set(group.values()))}  "
          f"{sorted(set(group.values()))}")
    print(f"K (distinct 3-grams): min={min(K)}  median={int(np.median(K))}  max={max(K)}")
    print(f"NN chance floor      : {chance:.3f}")
    print(f"rf_triple null draws : {N_NULL}    tree method: {TREE_METHOD.upper()}")
    print("-" * 92)

    # ---- build the four distance matrices --------------------------------------
    methods = []

    t = time.time()
    D1 = lt.js_matrix(texts, n=3, alpha=0.0)
    print(f"[1] trigram-JS plug-in (alpha=0)        built in {time.time()-t:5.1f}s")
    methods.append(("trigram-JS plug-in (a=0)", D1))

    t = time.time()
    D2 = lt.js_matrix(texts, n=3, alpha=0.3)
    print(f"[2] trigram-JS Lidstone (alpha=0.3)     built in {time.time()-t:5.1f}s")
    methods.append(("trigram-JS Lidstone (a=0.3)", D2))

    t = time.time()
    D3 = lt.perplexity_distance_matrix(texts, n=3, alpha=0.1, cap=40000)
    print(f"[3] perplexity (Gamallo char-LM)        built in {time.time()-t:5.1f}s")
    methods.append(("perplexity char-LM (a=0.1)", D3))

    t = time.time()
    D4 = lt.ncd_matrix_v2(texts, cap_bytes=30000)   # equal BYTE budget (romanization expands length unevenly)
    print(f"[4] gzip NCD (per-pair floor)           built in {time.time()-t:5.1f}s")
    methods.append(("gzip NCD (per-pair floor)", D4))
    print("-" * 92)

    # ---- score every method -----------------------------------------------------
    results = []
    for nm, D in methods:
        results.append(score_method(nm, D, names, gold, group, K))

    # ---- comparison table -------------------------------------------------------
    hdr = (f"{'method':<28} {'RF_obs':>7} {'RF_flr':>7} {'RF_null':>7} "
           f"{'RF_resc':>7} | {'GQD':>6} | {'NN_pur':>7} {'chance':>6} {'ties':>4} | "
           f"{'r(d,maxK)':>9} {'r(d,|dK|)':>9}")
    print("\nCOMPARISON TABLE  (lower RF_resc / GQD = better; NN_pur above chance = better;")
    print("                   r(d,*) near 0 = less vocabulary-size bias)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<28} "
              f"{r['rf_obs']:7.3f} {r['rf_floor']:7.3f} {r['rf_null']:7.3f} "
              f"{r['rf_resc']:7.3f} | {r['gqd']:6.3f} | "
              f"{r['purity']:7.3f} {r['chance']:6.3f} {r['n_ties']:4d} | "
              f"{r['r_maxK']:9.3f} {r['r_absdK']:9.3f}")
    print("=" * len(hdr))
    approxes = [r["name"] for r in results if r["gqd_approx"]]
    if approxes:
        print(f"note: GQD computed by quartet sampling (approx=True) for: {approxes}")
    else:
        print("note: GQD computed exactly (approx=False) for all methods.")

    # ---- figure: grouped bars (rescaled-RF + GQD per method) --------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        labels = [r["name"].replace(" (", "\n(") for r in results]
        rf_resc = [r["rf_resc"] for r in results]
        gqd_v = [r["gqd"] for r in results]
        x = np.arange(len(results)); w = 0.38
        fig, ax = plt.subplots(figsize=(10, 5.2))
        b1 = ax.bar(x - w/2, rf_resc, w, label="rescaled RF (vs gold)", color="#3b6ea5")
        b2 = ax.bar(x + w/2, gqd_v, w, label="GQD (vs gold)", color="#c0504d")
        ax.set_ylabel("distance to gold  (lower = better)")
        ax.set_title("ARM: distance methods vs gold family tree (ROMANIZED set, UPGMA)")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(max(rf_resc), max(gqd_v)) * 1.25 + 0.02)
        ax.legend(loc="upper right", fontsize=9)
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        outpath = os.path.join("figures", "methods_compare.png")
        fig.savefig(outpath, dpi=140)
        print(f"\nsaved figure: {os.path.abspath(outpath)}")
    except Exception as e:
        print(f"\n[skip] figure not written: {type(e).__name__}: {e}")

    print(f"\ntotal wall time: {time.time()-t_all:.1f}s")
    return results


if __name__ == "__main__":
    main()

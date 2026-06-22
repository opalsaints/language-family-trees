"""
Language similarity trees from text via information theory.

Core reusable functions (used by both the Tier-1 / Tier-2 scripts and the
Colab notebook):
  - text cleaning (Unicode-letter, casefolded)
  - character n-gram probability distributions
  - Jensen-Shannon divergence distance matrix
  - gzip Normalized Compression Distance (NCD) matrix
  - Shannon block/conditional entropy (F_N) for validation
  - UPGMA tree + dendrogram

Complexity Lab group project (Jonathan + Nil).
"""
import gzip
import math
import re
import unicodedata
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------- text cleaning
def clean(text, keep_diacritics=True):
    """Casefold + NFC-normalize, keep Unicode letters (and combining marks attached
    to them), and collapse every run of non-letters to a single space.

    Combining marks (Unicode category 'M') are KEPT attached to their base letter,
    never turned into spaces — so we never inject spurious word breaks. This is
    correct for abjad/Indic scripts (Arabic harakat, Devanagari matras) and fixes
    the Turkish bug where 'İ'.casefold() == 'i' + U+0307 and the stray combining
    dot used to become a space ('İstanbul' -> 'i stanbul'). The U+0307 combining
    dot-above (the dotted-capital-I casefold artifact) is removed outright so the
    Turkish 'i' is clean.
    """
    text = unicodedata.normalize("NFC", text.casefold()).replace("̇", "")
    if not keep_diacritics:
        # strip combining marks (e.g. é -> e) — used for ASCII-fold experiments
        text = "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    out = "".join(ch if unicodedata.category(ch)[0] in "LM" else " " for ch in text)
    return re.sub(r"\s+", " ", out).strip()


# ----------------------------------------------------------------- n-gram dists
def ngram_counts(text, n=3):
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def ngram_dist(text, n=3, vocab=None):
    """Probability vector over `vocab` (a fixed ordered list of n-grams)."""
    c = ngram_counts(text, n)
    tot = sum(c.values())
    return np.array([c.get(g, 0) / tot for g in vocab])


def global_vocab(texts, n=3):
    v = set()
    for t in texts:
        for i in range(len(t) - n + 1):
            v.add(t[i:i + n])
    return sorted(v)


# --------------------------------------------------------------- JS divergence
def js_distance_matrix(texts, n=3):
    """Pairwise Jensen-Shannon distance (sqrt of JS divergence; a true metric)."""
    from scipy.spatial.distance import jensenshannon
    vocab = sorted({t[i:i + n] for t in texts for i in range(len(t) - n + 1)})
    P = np.array([ngram_dist(t, n, vocab) for t in texts])
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            d = jensenshannon(P[i], P[j], base=2)  # JS *distance* (metric)
            D[i, j] = D[j, i] = d
    return D


# --------------------------------------------------------------------- gzip NCD
def _clen(b):
    return len(gzip.compress(b, 9))


def ncd_matrix(texts):
    """Normalized Compression Distance via gzip (Li & Vitanyi / Cilibrasi)."""
    raw = [t.encode("utf-8") for t in texts]
    C = [_clen(b) for b in raw]
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            cxy = _clen(raw[i] + raw[j])
            d = (cxy - min(C[i], C[j])) / max(C[i], C[j])
            D[i, j] = D[j, i] = d
    np.fill_diagonal(D, 0.0)
    return D


# ----------------------------------------------- Shannon block / cond. entropy
def block_entropy(text, n):
    c = ngram_counts(text, n)
    tot = sum(c.values())
    return -sum((v / tot) * math.log2(v / tot) for v in c.values())


def conditional_entropies(text, max_n=4):
    """F_N = H(block_N) - H(block_{N-1}); F_1 = unigram entropy. Shannon 1951."""
    H = {0: 0.0}
    F = {}
    for n in range(1, max_n + 1):
        H[n] = block_entropy(text, n)
        F[n] = H[n] - H[n - 1]
    return F


# --------------------------------------------------------------------- tree viz
def upgma(D, labels):
    """UPGMA linkage from a square distance matrix."""
    return linkage(squareform(D, checks=False), method="average")


def plot_tree(D, labels, title, path, colors=None):
    import matplotlib.pyplot as plt
    Z = upgma(D, labels)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(labels) + 1.5))
    dendrogram(Z, labels=labels, orientation="right", ax=ax,
               color_threshold=0.0, above_threshold_color="#555")
    ax.set_title(title)
    if colors:
        for lbl in ax.get_ymajorticklabels():
            lbl.set_color(colors.get(lbl.get_text(), "black"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Z


def cluster_labels(D, labels, k):
    Z = upgma(D, labels)
    assign = fcluster(Z, t=k, criterion="maxclust")
    out = {}
    for lbl, a in zip(labels, assign):
        out.setdefault(int(a), []).append(lbl)
    return out


# ------------------------------------------------ gold-tree (Robinson-Foulds)
def linkage_to_newick(Z, labels):
    """Topology-only Newick string from a scipy linkage matrix."""
    from scipy.cluster.hierarchy import to_tree
    t = to_tree(Z, rd=False)

    def rec(node):
        if node.is_leaf():
            return labels[node.id]
        return f"({rec(node.get_left())},{rec(node.get_right())})"
    return rec(t) + ";"


def robinson_foulds(newick_a, newick_b):
    """Normalized symmetric difference (RF) between two unrooted topologies.
    Returns (rf, max_rf, normalized in [0,1]); 0 = identical topology."""
    import dendropy
    from dendropy.calculate import treecompare
    tns = dendropy.TaxonNamespace()
    ta = dendropy.Tree.get(data=newick_a, schema="newick", taxon_namespace=tns)
    tb = dendropy.Tree.get(data=newick_b, schema="newick", taxon_namespace=tns)
    ta.encode_bipartitions(); tb.encode_bipartitions()
    rf = treecompare.symmetric_difference(ta, tb)
    n = sum(1 for _ in ta.leaf_node_iter())
    max_rf = 2 * (n - 3)  # unrooted binary trees
    return rf, max_rf, (rf / max_rf if max_rf else 0.0)


# ===================================================================
# Hardened additions (Bible-scale, many-script): sparse distances,
# baselines, corrected RF, and metadata-derived gold trees.
# ===================================================================
import random as _random


def ngram_counter(text, n=3):
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def js_div_counts(ca, cb):
    """Jensen-Shannon DISTANCE (sqrt of JS divergence, base 2, in [0,1]) between
    two n-gram Counters. Sparse: only iterates the union of observed n-grams, so
    it scales to many scripts without a giant dense global-vocab matrix."""
    ta = sum(ca.values()); tb = sum(cb.values())
    if ta == 0 or tb == 0:
        return 1.0
    js = 0.0
    for k in set(ca) | set(cb):
        p = ca.get(k, 0) / ta
        q = cb.get(k, 0) / tb
        m = 0.5 * (p + q)
        if p > 0:
            js += 0.5 * p * math.log2(p / m)
        if q > 0:
            js += 0.5 * q * math.log2(q / m)
    return math.sqrt(max(js, 0.0))


def js_matrix(texts, n=3):
    """Pairwise JS-distance matrix over character n-grams (sparse, multi-script)."""
    cs = [ngram_counter(t, n) for t in texts]
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = js_div_counts(cs[i], cs[j])
    return D


def alphabet_jaccard_matrix(texts):
    """BASELINE: distance = 1 - |A∩B|/|A∪B| over the *set* of characters used
    (no frequencies, no order, no n-grams, no information theory). If trigram-JS
    cannot beat this, the information theory is adding nothing."""
    sets = [set(t) - {" "} for t in texts]
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            u = len(sets[i] | sets[j])
            d = 1.0 - (len(sets[i] & sets[j]) / u) if u else 0.0
            D[i, j] = D[j, i] = d
    return D


def shuffle_chars(text, seed=0):
    """NEGATIVE CONTROL: destroy all sequential order, keep the exact character
    inventory + unigram frequencies. Trees on shuffled text isolate how much
    structure comes from order (n-grams) vs inventory/frequency alone."""
    r = _random.Random(seed)
    chars = list(text)
    r.shuffle(chars)
    return "".join(chars)


def _safe(label):
    """Newick-safe label (letters/digits/underscore only, non-empty, unique-able)."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return s or "X"


def gold_newick_from_rows(rows):
    """Build a (polytomous) gold Newick from taxonomy rows.
    rows: iterable of (label, family, genus, subgenus). Nests
    Family -> Genus -> Subgenus -> language. Missing levels collapse upward.
    Labels are NOT sanitized here — pass already-safe labels."""
    from collections import defaultdict
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for label, fam, gen, sub in rows:
        tree[fam or "NA"][gen or "_"][sub or "_"].append(label)

    def join(children):
        children = [c for c in children if c]
        if len(children) == 1:
            return children[0]
        return "(" + ",".join(children) + ")"

    fam_nodes = []
    for fam, gens in tree.items():
        gen_nodes = []
        for gen, subs in gens.items():
            sub_nodes = [join(labels) for labels in subs.values()]
            gen_nodes.append(join(sub_nodes))
        fam_nodes.append(join(gen_nodes))
    return join(fam_nodes) + ";"


def _internal_bipartitions(tree):
    """Count non-trivial bipartitions (internal edges) of an unrooted dendropy tree."""
    tree.is_rooted = False
    tree.encode_bipartitions()
    n_leaves = sum(1 for _ in tree.leaf_node_iter())
    c = 0
    for nd in tree.preorder_node_iter():
        if nd.parent_node is None or nd.is_leaf():
            continue
        # an internal edge splits >=2 taxa from the rest on both sides
        nleaf = sum(1 for _ in nd.leaf_iter())
        if 2 <= nleaf <= n_leaves - 2:
            c += 1
    return c


def _gz(b):
    return len(gzip.compress(b, 9))


def ncd_matrix_fixed(texts, cap_bytes=None):
    """gzip Normalized Compression Distance with the audit fixes:
    symmetric (average of both concatenation orders), a rare separator byte at
    the seam, an equal byte budget (cap_bytes) so multi-byte scripts aren't
    penalised, and the self-compression floor subtracted. Crude Kolmogorov
    proxy — report comparatively, not as ground truth."""
    raw = [t.encode("utf-8") for t in texts]
    if cap_bytes:
        raw = [b[:cap_bytes] for b in raw]
    SEP = b"\x01"
    C = [_gz(b) for b in raw]
    floors = [((_gz(b + SEP + b) - cx) / cx if cx else 0.0) for b, cx in zip(raw, C)]
    floor = sum(floors) / len(floors)
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            cxy = 0.5 * (_gz(raw[i] + SEP + raw[j]) + _gz(raw[j] + SEP + raw[i]))
            d = (cxy - min(C[i], C[j])) / max(C[i], C[j])
            D[i, j] = D[j, i] = max(0.0, d - floor)
    return D


def nj_newick(D, labels):
    """Saitou-Nei Neighbor-Joining tree (unrooted, no molecular-clock
    assumption) -> Newick with branch lengths. Topology is what RF compares."""
    name = {i: labels[i] for i in range(len(labels))}
    active = list(range(len(labels)))
    dist = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            dist[(i, j)] = float(D[i][j])
    nxt = len(labels)

    def dget(a, b):
        return dist[(a, b)] if a < b else dist[(b, a)]

    while len(active) > 2:
        m = len(active)
        r = {a: sum(dget(a, b) for b in active if b != a) for a in active}
        best = bi = bj = None
        for ii in range(m):
            for jj in range(ii + 1, m):
                a, b = active[ii], active[jj]
                q = (m - 2) * dget(a, b) - r[a] - r[b]
                if best is None or q < best:
                    best, bi, bj = q, a, b
        a, b = bi, bj
        dab = dget(a, b)
        la = 0.5 * dab + (r[a] - r[b]) / (2 * (m - 2))
        lb = dab - la
        u = nxt; nxt += 1
        name[u] = f"({name[a]}:{max(la,0):.6f},{name[b]}:{max(lb,0):.6f})"
        for c in active:
            if c in (a, b):
                continue
            duc = 0.5 * (dget(a, c) + dget(b, c) - dab)
            x, y = (u, c) if u < c else (c, u)
            dist[(x, y)] = duc
        active.remove(a); active.remove(b); active.append(u)
    a, b = active
    return f"({name[a]}:{max(dget(a,b),0):.6f},{name[b]});"


def cophenetic_corr(D, labels):
    """Cophenetic correlation: how faithfully the UPGMA dendrogram preserves the
    original pairwise distances (1.0 = perfect). The standard tree-quality
    diagnostic, separate from RF (which compares topology to the gold tree)."""
    from scipy.cluster.hierarchy import cophenet
    Z = upgma(D, labels)
    coph, _ = cophenet(Z, squareform(D, checks=False))
    return coph


def _random_binary_newick(labels, rng):
    nodes = labels[:]
    rng.shuffle(nodes)
    while len(nodes) > 1:
        a = nodes.pop(); b = nodes.pop()
        nodes.insert(0, f"({a},{b})")
        rng.shuffle(nodes)
    return nodes[0] + ";"


def random_tree_null(labels, gold_newick, n=500, seed=0):
    """Proper null: RF of RANDOM binary tree TOPOLOGIES vs the gold tree (not the
    old label-shuffle-on-one-fixed-shape, which can't reach low RF). Returns
    (p05, p50, p95, min) of normalized RF over n random trees."""
    import random as _r
    rng = _r.Random(seed)
    vals = sorted(rf_corrected(_random_binary_newick(labels, rng), gold_newick)[2]
                  for _ in range(n))
    pick = lambda q: vals[min(len(vals) - 1, int(q * len(vals)))]
    return pick(0.05), pick(0.50), pick(0.95), vals[0]


def nn_purity(D, labels, group):
    """Fraction of items whose nearest neighbour shares its `group` (e.g. family).
    `group` maps label -> group key. Robust, intuitive complement to RF."""
    hits = 0
    for i, lab in enumerate(labels):
        j = min((k for k in range(len(labels)) if k != i), key=lambda k: D[i, k])
        hits += (group[labels[j]] == group[lab])
    return hits / len(labels)


def rf_corrected(newick_inferred, newick_gold):
    """Robinson-Foulds with an HONEST denominator = sum of the two trees' own
    non-trivial bipartitions (not 2(n-3)). Handles a multifurcating gold tree:
    a binary tree merely *refining* gold's polytomies is not over-penalised.
    Returns (rf, denom, normalized in [0,1]); 0 = no conflicting splits."""
    import dendropy
    from dendropy.calculate import treecompare
    tns = dendropy.TaxonNamespace()
    ti = dendropy.Tree.get(data=newick_inferred, schema="newick", taxon_namespace=tns)
    tg = dendropy.Tree.get(data=newick_gold, schema="newick", taxon_namespace=tns)
    ti.is_rooted = False; tg.is_rooted = False
    ti.encode_bipartitions(); tg.encode_bipartitions()
    rf = treecompare.symmetric_difference(ti, tg)
    denom = _internal_bipartitions(ti) + _internal_bipartitions(tg)
    return rf, denom, (rf / denom if denom else 0.0)

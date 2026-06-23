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


def js_div_counts(ca, cb, alpha=0.0):
    """Jensen-Shannon DISTANCE (sqrt of JS divergence, base 2, in [0,1]) between
    two n-gram Counters. Sparse: only iterates the union of observed n-grams, so
    it scales to many scripts without a giant dense global-vocab matrix.

    `alpha`>0 applies Lidstone (add-alpha) smoothing over the UNION of observed
    n-grams (K = |union|): p_k = (c_k + alpha)/(N + alpha*K). alpha=0 is the
    plug-in MLE (legacy, exact). Smoothing reduces the upward small-sample /
    large-vocabulary bias that makes plug-in JS grow with n-gram-vocabulary size
    (the differential bias the critical review flagged)."""
    ta = sum(ca.values()); tb = sum(cb.values())
    if ta == 0 or tb == 0:
        return 1.0
    union = set(ca) | set(cb)
    if alpha:
        K = len(union)
        da = ta + alpha * K
        db = tb + alpha * K
    else:
        da, db = ta, tb
    js = 0.0
    for k in union:
        p = (ca.get(k, 0) + alpha) / da
        q = (cb.get(k, 0) + alpha) / db
        m = 0.5 * (p + q)
        if p > 0:
            js += 0.5 * p * math.log2(p / m)
        if q > 0:
            js += 0.5 * q * math.log2(q / m)
    return math.sqrt(max(js, 0.0))


def js_matrix(texts, n=3, alpha=0.0):
    """Pairwise JS-distance matrix over character n-grams (sparse, multi-script).
    `alpha` passes Lidstone smoothing through to js_div_counts (0 = plug-in)."""
    cs = [ngram_counter(t, n) for t in texts]
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = js_div_counts(cs[i], cs[j], alpha)
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


def nn_purity(D, labels, group, tie="first", atol=1e-9):
    """Fraction of items whose nearest neighbour shares its `group` (e.g. family).
    `group` maps label -> group key. Robust, intuitive complement to RF.

    `tie` controls behaviour when the minimum distance is shared by >1 neighbour
    (e.g. raw cross-script JS saturates at exactly 1.0, so the nearest neighbour is
    a list-order artifact):
      'first' (default, legacy) — take the first such neighbour (back-compatible);
      'miss'  — a tied NN is genealogically undefined, count it as a non-hit;
      'drop'  — exclude tied items from the denominator entirely."""
    hits = 0
    counted = 0
    n = len(labels)
    for i in range(n):
        row = [(D[i, k], k) for k in range(n) if k != i]
        dmin = min(d for d, _ in row)
        cands = [k for d, k in row if d <= dmin + atol]
        is_tie = len(cands) > 1
        if is_tie and tie == "drop":
            continue
        counted += 1
        if is_tie and tie == "miss":
            continue
        if group[labels[cands[0]]] == group[labels[i]]:
            hits += 1
    return hits / counted if counted else 0.0


def bootstrap_ci(texts, names, group, gold_newick, n_boot=150, seed=0, ngram=3):
    """Token bootstrap: resample each language's n-gram multiset (multinomial on
    its empirical distribution) and recompute family-NN purity + normalized RF,
    giving 95% confidence intervals on the headline numbers. Returns
    {'purity': (lo, med, hi), 'rf': (lo, med, hi)}."""
    rng = np.random.default_rng(seed)
    base = [ngram_counter(t, ngram) for t in texts]
    keys = [list(c.keys()) for c in base]
    probs, tots = [], []
    for c in base:
        v = np.array(list(c.values()), dtype=float)
        tots.append(int(v.sum()))
        probs.append(v / v.sum() if v.sum() else v)
    purs, rfs = [], []
    for _ in range(n_boot):
        cs = []
        for ks, pr, tot in zip(keys, probs, tots):
            draw = rng.multinomial(tot, pr)
            cs.append({k: int(x) for k, x in zip(ks, draw) if x})
        m = len(cs)
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                D[i, j] = D[j, i] = js_div_counts(cs[i], cs[j])
        purs.append(nn_purity(D, names, group))
        rfs.append(rf_corrected(linkage_to_newick(upgma(D, names), names), gold_newick)[2])
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                     float(np.percentile(a, 97.5)))
    return {"purity": pct(purs), "rf": pct(rfs)}


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


# ===================================================================
# Bias-corrected block/conditional entropy estimators.
#
# The plug-in MLE (block_entropy above) is systematically biased LOW: it
# underestimates the true entropy, and the bias grows when the number of
# distinct n-grams K approaches the token count N (i.e. for larger n, where
# counts are sparse). Two standard corrections are provided. Both return BITS.
#   - Miller-Madow: cheap first-order bias correction, +(K-1)/(2N) nats.
#   - Grassberger (2003): a finite-sample estimator that is markedly less
#     biased than MM in the sparse regime.
# Verified by i.i.d. simulation (uniform on K symbols, true H = log2 K): see
# entropy_estimator_simulation() below.
# ===================================================================
_LN2 = math.log(2.0)


def block_entropy_mm(text, n):
    """Miller-Madow bias-corrected block-n entropy, in BITS.

    H_MM = H_plugin + (K-1)/(2N), where the (K-1)/(2N) term is in NATS, so it is
    divided by ln 2 to land in bits. K = number of DISTINCT observed n-grams,
    N = total n-gram token count. Always >= the plug-in value."""
    c = ngram_counter(text, n)
    N = sum(c.values())
    if N == 0:
        return 0.0
    K = len(c)
    H_plugin = -sum((v / N) * math.log2(v / N) for v in c.values())
    return H_plugin + (K - 1) / (2 * N * _LN2)


def _grassberger_G(ni):
    """Grassberger's G(n) = psi(n) + 0.5*(-1)^n*(psi((n+1)/2) - psi(n/2)),
    vectorised over an array of integer counts (psi = digamma)."""
    from scipy.special import psi
    ni = np.asarray(ni, dtype=float)
    safe = np.where(ni > 0, ni, 1.0)   # avoid psi(0) singularity / nan warning
    out = psi(safe) + 0.5 * ((-1.0) ** safe) * (psi((safe + 1) / 2.0) - psi(safe / 2.0))
    return np.where(ni > 0, out, 0.0)


def block_entropy_grassberger(text, n):
    """Grassberger (2003) finite-sample block-n entropy estimator, in BITS.

    In nats: H = ln(N) - (1/N) * sum_i n_i * G(n_i), with n_i the count of the
    i-th observed n-gram, N = sum_i n_i, and G as in _grassberger_G. Converted
    to bits via /ln 2. Less biased than Miller-Madow when counts are sparse."""
    c = ngram_counter(text, n)
    N = sum(c.values())
    if N == 0:
        return 0.0
    ni = np.array(list(c.values()), dtype=float)
    H_nats = math.log(N) - (1.0 / N) * float(np.sum(ni * _grassberger_G(ni)))
    return H_nats / _LN2


def conditional_entropies_est(text, max_n=4, estimator="plugin"):
    """Shannon conditional-entropy ladder F_N with a choice of block-entropy
    estimator. F_N = H(block_N) - H(block_{N-1}) for N>=2, F_1 = H(block_1).
    estimator in {'plugin','mm','grassberger'} selects block_entropy /
    block_entropy_mm / block_entropy_grassberger. Returns {n: F_n} in BITS."""
    fns = {"plugin": block_entropy,
           "mm": block_entropy_mm,
           "grassberger": block_entropy_grassberger}
    if estimator not in fns:
        raise ValueError(f"estimator must be one of {sorted(fns)}, got {estimator!r}")
    be = fns[estimator]
    H = {0: 0.0}
    F = {}
    for n in range(1, max_n + 1):
        H[n] = be(text, n)
        F[n] = H[n] - H[n - 1]
    return F


def entropy_estimator_simulation(K=64, Ns=(128, 1024, 16384), reps=200, seed=0):
    """SANITY CHECK / verification of the entropy estimators by simulation.

    Draw N i.i.d. samples uniformly from K symbols (so the TRUE entropy is
    exactly log2(K)). For each N, average the plug-in, Miller-Madow and
    Grassberger estimates over `reps` independent draws. Expected behaviour
    (and what this returns evidence for): at small N the plug-in UNDERestimates
    log2(K) while MM and Grassberger are markedly closer, and all three converge
    to log2(K) as N grows.

    Returns dict(truth=log2(K), rows=[{'N':N,'plugin':..,'mm':..,
    'grassberger':..}, ...]). The estimators operate on a 1-gram 'text' whose
    symbols are taken from a K-letter alphabet, exercising the very same code
    paths as block_entropy_*("...", 1)."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    # an alphabet of K distinct single-codepoint symbols (use a private-use block
    # so every symbol survives clean() concerns is irrelevant — we feed counts
    # directly via a synthetic text of length-1 tokens).
    alphabet = [chr(0xE000 + i) for i in range(K)]
    truth = math.log2(K)
    rows = []
    for N in Ns:
        pl = mm = gr = 0.0
        for _ in range(reps):
            idx = rng.integers(0, K, size=N)
            text = "".join(alphabet[i] for i in idx)
            pl += block_entropy(text, 1)
            mm += block_entropy_mm(text, 1)
            gr += block_entropy_grassberger(text, 1)
        rows.append({"N": N, "plugin": pl / reps,
                     "mm": mm / reps, "grassberger": gr / reps})
    return {"truth": truth, "K": K, "rows": rows}


# ===================================================================
# Per-clade bootstrap branch support for the UPGMA reference tree.
# ===================================================================
def branch_support(texts, names, group=None, n_boot=200, ngram=3, seed=0):
    """Non-parametric bootstrap branch support for the trigram-JS UPGMA tree.

    Reference tree = linkage_to_newick(upgma(js_matrix(texts, ngram), names)).
    For each of n_boot replicates, multinomial-resample every language's n-gram
    counts from its OWN empirical distribution (same total N per language),
    recompute the JS distance matrix and build a bootstrap UPGMA Newick. Support
    of each non-trivial internal bipartition of the REFERENCE tree = fraction of
    bootstrap trees that contain that exact split.

    Implementation note (a notorious bug): the reference tree and ALL bootstrap
    trees are parsed under ONE shared dendropy TaxonNamespace; splits are then
    comparable as integer bitmasks. Using separate namespaces silently yields
    all-zero support.

    Returns {'ref_newick': <labelled newick>,
             'clades': [{'members': [taxa on smaller side], 'support': float in [0,1]}, ...]}.
    `group` is accepted for API symmetry (e.g. colouring downstream) but does not
    affect the support computation."""
    import dendropy

    cs0 = [ngram_counter(t, ngram) for t in texts]
    D0 = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            D0[i, j] = D0[j, i] = js_div_counts(cs0[i], cs0[j])
    ref_newick = linkage_to_newick(upgma(D0, names), names)

    # one shared namespace for reference + every bootstrap tree
    tns = dendropy.TaxonNamespace()
    ref = dendropy.Tree.get(data=ref_newick, schema="newick", taxon_namespace=tns,
                            preserve_underscores=True)
    ref.is_rooted = False
    ref.encode_bipartitions()

    # collect reference non-trivial internal bipartitions. dendropy's
    # leafset_taxa(tns) gives the taxa on the leafset side of each split; the
    # other side is the complement over `names`.
    ref_splits = []  # (split_bitmask, members_on_smaller_side)
    seen = set()
    for bp in ref.bipartition_encoding:
        if bp.is_trivial():
            continue
        side_a = [tx.label for tx in bp.leafset_taxa(tns)]
        side_b = [lbl for lbl in names if lbl not in side_a]
        if len(side_a) < 2 and len(side_b) < 2:
            continue  # trivial after all
        smaller = side_a if len(side_a) <= len(side_b) else side_b
        key = frozenset(smaller)
        if key in seen:  # encode_bipartitions can list both edge sides
            continue
        seen.add(key)
        ref_splits.append((bp.split_bitmask, sorted(smaller)))

    # build empirical sampling distributions per language
    rng = np.random.default_rng(seed)
    keys = [list(c.keys()) for c in cs0]
    probs, tots = [], []
    for c in cs0:
        v = np.array(list(c.values()), dtype=float)
        tots.append(int(v.sum()))
        probs.append(v / v.sum() if v.sum() else v)

    counts = [0] * len(ref_splits)
    for _ in range(n_boot):
        cs = []
        for ks, pr, tot in zip(keys, probs, tots):
            draw = rng.multinomial(tot, pr)
            cs.append({k: int(x) for k, x in zip(ks, draw) if x})
        m = len(cs)
        Db = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                Db[i, j] = Db[j, i] = js_div_counts(cs[i], cs[j])
        bnewick = linkage_to_newick(upgma(Db, names), names)
        bt = dendropy.Tree.get(data=bnewick, schema="newick", taxon_namespace=tns,
                               preserve_underscores=True)
        bt.is_rooted = False
        bt.encode_bipartitions()
        boot_splits = {bp.split_bitmask for bp in bt.bipartition_encoding
                       if not bp.is_trivial()}
        for idx, (sb, _members) in enumerate(ref_splits):
            if sb in boot_splits:
                counts[idx] += 1

    clades = []
    for cnt, (sb, members) in zip(counts, ref_splits):
        sup = cnt / n_boot if n_boot else 0.0
        sup = min(1.0, max(0.0, sup))
        clades.append({"members": members, "support": sup})
    # largest (most inclusive) clades first for readability
    clades.sort(key=lambda c: (-len(c["members"]), c["members"]))
    return {"ref_newick": ref_newick, "clades": clades}


# ===================================================================
# Rigor v2 — honest reporting helpers + principled method arms.
# (Added after the 23 Jun critical review: report numbers against their true
#  floors/nulls, add chance-corrected and tie-aware purity, Mantel significance,
#  generalized quartet distance, a Glottolog gold tree, a valid site bootstrap,
#  smoothing-aware NCD, a perplexity distance, and a treelikeness diagnostic.)
# ===================================================================

# ---------- honest scaling for Robinson-Foulds ----------
def rf_triple(inferred_newick, gold_newick, labels, n_null=300, seed=0):
    """Report normalized RF against its TRUE endpoints, not an implied [0,1] with
    0 = perfect. Against a POLYTOMOUS gold even a genealogically-perfect (correctly
    refined) tree cannot reach 0 — it floors at the cost of binarizing the gold's
    polytomies. Returns {'observed','floor','null_p50','rescaled'} where
      floor    = normRF of a random binary RESOLUTION of the gold vs the gold,
      null_p50 = median normRF of random binary topologies vs the gold,
      rescaled = (observed-floor)/(null_p50-floor)  (0 = as good as a perfect
                 refinement of gold, 1 = no better than a random tree)."""
    import dendropy
    import random as _r
    obs = rf_corrected(inferred_newick, gold_newick)[2]
    floors = []
    for k in range(5):
        tns = dendropy.TaxonNamespace()
        g = dendropy.Tree.get(data=gold_newick, schema="newick", taxon_namespace=tns)
        g.resolve_polytomies(rng=_r.Random(seed + k))
        floors.append(rf_corrected(g.as_string(schema="newick"), gold_newick)[2])
    floor = float(np.median(floors))
    null_p50 = random_tree_null(labels, gold_newick, n=n_null, seed=seed)[1]
    span = null_p50 - floor
    return {"observed": obs, "floor": floor, "null_p50": null_p50,
            "rescaled": (obs - floor) / span if span > 1e-12 else float("nan")}


# ---------- chance-corrected + tie-aware nearest-neighbour purity ----------
def purity_chance_floor(group, labels=None):
    """Expected nearest-neighbour family purity under RANDOM assignment =
    sum_c n_c(n_c-1) / (n(n-1)). The honest floor a purity number must beat: it is
    > 0 (never 0) and rises with class imbalance (here ~0.31, since IE = 54%)."""
    from collections import Counter
    labs = list(labels) if labels is not None else list(group)
    counts = Counter(group[l] for l in labs)
    n = len(labs)
    if n < 2:
        return 0.0
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def nearest_neighbors(D, labels, atol=1e-9):
    """For each item: (nn_label, is_tie, dmin). is_tie=True when >1 neighbour shares
    the minimum distance (e.g. raw cross-script JS = 1.0), so the NN is undefined."""
    out = []
    n = len(labels)
    for i in range(n):
        row = [(D[i, k], k) for k in range(n) if k != i]
        dmin = min(d for d, _ in row)
        cands = [k for d, k in row if d <= dmin + atol]
        out.append((labels[cands[0]], len(cands) > 1, dmin))
    return out


def nn_diagnostics(D, labels, group):
    """Family-NN purity under both tie policies + the tie count + the chance floor,
    so the JS=1.0 saturation artefact and the imbalance baseline are both visible."""
    return {"purity": nn_purity(D, labels, group, tie="first"),
            "purity_tiemiss": nn_purity(D, labels, group, tie="miss"),
            "n_ties": sum(1 for _, t, _ in nearest_neighbors(D, labels) if t),
            "n": len(labels),
            "chance": purity_chance_floor(group, labels)}


# ---------- Mantel / partial-Mantel (valid significance for matrix correlation) ----------
def _triu(D):
    n = D.shape[0]
    return D[np.triu_indices(n, 1)]


def mantel(D1, D2, perms=9999, seed=0):
    """Mantel permutation test for two distance matrices (the valid replacement for
    a naive Pearson p over non-independent pairs). Returns (r, p)."""
    a, b = _triu(D1), _triu(D2)
    r0 = float(np.corrcoef(a, b)[0, 1])
    n = D1.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    ge = 1
    for _ in range(perms):
        p = rng.permutation(n)
        bp = D2[np.ix_(p, p)][iu]
        if abs(np.corrcoef(a, bp)[0, 1]) >= abs(r0):
            ge += 1
    return r0, ge / (perms + 1)


def partial_mantel(D1, D2, Dctrl, perms=9999, seed=0):
    """Partial Mantel: correlation of D1,D2 after regressing both on a control
    matrix Dctrl (e.g. a same-script / same-major-family block matrix), to test for
    FINE structure beyond coarse block agreement. Returns (r_partial, p)."""
    a, b, c = _triu(D1), _triu(D2), _triu(Dctrl)

    def resid(y, x):
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta

    ra, rb = resid(a, c), resid(b, c)
    r0 = float(np.corrcoef(ra, rb)[0, 1])
    n = D1.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    ge = 1
    for _ in range(perms):
        p = rng.permutation(n)
        bp = D2[np.ix_(p, p)][iu]
        rbp = resid(bp, c)
        if abs(np.corrcoef(ra, rbp)[0, 1]) >= abs(r0):
            ge += 1
    return r0, ge / (perms + 1)


# ---------- Generalized Quartet Distance (pure Python; 0 for a correct refinement) ----------
def _splits_as_index_sets(newick, labels):
    import dendropy
    tns = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(data=newick, schema="newick", taxon_namespace=tns,
                          preserve_underscores=True)
    t.is_rooted = False
    t.encode_bipartitions()
    idx = {lab: i for i, lab in enumerate(labels)}
    out = []
    for bp in t.bipartition_encoding:
        if bp.is_trivial():
            continue
        out.append(frozenset(idx[tx.label] for tx in bp.leafset_taxa(tns)))
    return out


def _quartet_topo(splits, q):
    """The 2|2 split a quartet q=(i,j,k,l) induces in a tree, canonicalised as the
    frozenset side containing min(q); None if the quartet is unresolved (star)."""
    qs = set(q)
    for S in splits:
        inter = S & qs
        if len(inter) == 2:
            other = qs - inter
            return frozenset(inter if min(inter) < min(other) else other)
    return None


def gqd(newick_inferred, newick_gold, labels, max_full=70, n_sample=400000, seed=0):
    """Generalized Quartet Distance of an inferred tree to a (possibly polytomous)
    gold (Pompei, Loreto & Tria 2011): fraction of the gold's RESOLVED quartets that
    the inferred tree resolves DIFFERENTLY. Unlike RF this scores 0 for a tree that
    merely refines the gold's polytomies. Exhaustive for n<=max_full taxa; above
    that, a random sample of n_sample quartets (reported as approximate)."""
    import itertools
    import random as _r
    Si = _splits_as_index_sets(newick_inferred, labels)
    Sg = _splits_as_index_sets(newick_gold, labels)
    n = len(labels)
    if n <= max_full:
        quartets = itertools.combinations(range(n), 4)
        approx = False
    else:
        rng = _r.Random(seed)
        quartets = (tuple(sorted(rng.sample(range(n), 4))) for _ in range(n_sample))
        approx = True
    diff = resolved = 0
    for q in quartets:
        tg = _quartet_topo(Sg, q)
        if tg is None:
            continue
        resolved += 1
        ti = _quartet_topo(Si, q)
        if ti is None or ti != tg:
            diff += 1
    return {"gqd": diff / resolved if resolved else 0.0,
            "resolved_in_gold": resolved, "approx": approx}


# ---------- Glottolog-built gold tree (programmatic, replaces hand-typed metadata) ----------
def gold_newick_from_glottolog(iso_codes, glottolog_dir, label_for=None):
    """Build a gold Newick for the given ISO 639-3 codes straight from Glottolog's
    genealogy via pyglottolog, instead of the hand-maintained metadata columns.
    `label_for` maps iso -> emitted leaf label (default the iso code). Returns
    (newick, matched_iso_set); ISO codes absent from Glottolog are skipped. Slow
    (scans all languoids once)."""
    from pyglottolog import Glottolog
    g = Glottolog(str(glottolog_dir))
    want = set(iso_codes)
    found = {}
    for l in g.languoids():
        if l.iso and l.iso in want and l.iso not in found:
            found[l.iso] = l
    label_for = label_for or {}

    root = {}
    for iso, lang in found.items():
        node = root
        for (_name, gid, _level) in lang.lineage:   # family -> ... (excludes the language)
            node = node.setdefault(gid, {})
        node.setdefault("__leaves__", []).append(_safe(label_for.get(iso, iso)))

    def to_newick(node):
        parts = list(node.get("__leaves__", []))
        for k, child in node.items():
            if k != "__leaves__":
                parts.append(to_newick(child))
        parts = [p for p in parts if p]
        return parts[0] if len(parts) == 1 else "(" + ",".join(parts) + ")"

    return to_newick(root) + ";", set(found)


# ---------- valid site/block bootstrap (Felsenstein 1985) ----------
def _build_tree_newick(texts, names, ngram, method, alpha=0.0):
    D = js_matrix(texts, ngram, alpha)
    if method == "nj":
        return nj_newick(D, names)
    return linkage_to_newick(upgma(D, names), names)


def _reference_splits(ref_newick, names):
    import dendropy
    tns = dendropy.TaxonNamespace()
    ref = dendropy.Tree.get(data=ref_newick, schema="newick", taxon_namespace=tns,
                            preserve_underscores=True)
    ref.is_rooted = False
    ref.encode_bipartitions()
    splits, seen = [], set()
    for bp in ref.bipartition_encoding:
        if bp.is_trivial():
            continue
        side_a = [tx.label for tx in bp.leafset_taxa(tns)]
        side_b = [l for l in names if l not in side_a]
        if len(side_a) < 2 and len(side_b) < 2:
            continue
        smaller = side_a if len(side_a) <= len(side_b) else side_b
        key = frozenset(smaller)
        if key in seen:
            continue
        seen.add(key)
        splits.append((bp.split_bitmask, sorted(smaller)))
    return tns, splits


def _unit_counters(unit_texts, names, ngram):
    """Per language, the list of per-unit (verse) n-gram Counters — each verse is
    cleaned + counted ONCE so the block bootstrap can resample by aggregating
    counters (fast) instead of re-cleaning whole texts every replicate.

    NOTE: verses are cleaned/counted independently here, whereas the headline tree
    cleans the *concatenated* text, so this bootstrap omits the handful of trigrams
    that span a verse boundary. The effect is negligible (boundary trigrams are a
    tiny fraction and mostly involve the word-gap), but it means the bootstrap
    feature set is a hair sparser than the estimate it brackets — documented, not a bug."""
    return {nm: [ngram_counter(clean(u), ngram) for u in unit_texts[nm]] for nm in names}


def _agg_counter(unit_counter_list, idx):
    from collections import Counter
    agg = Counter()
    for j in idx:
        agg.update(unit_counter_list[j])
    return agg


def _counters_to_D(counters):
    m = len(counters)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = js_div_counts(counters[i], counters[j])
    return D


def branch_support_block(unit_texts, names, n_boot=100, ngram=3, seed=0, method="upgma"):
    """Felsenstein (1985) SITE bootstrap branch support: resample the ALIGNED text
    units (verses) with replacement and re-infer, scoring each internal split of the
    reference tree. The statistically valid bootstrap (resamples DATA units) — unlike
    branch_support, which resamples a fitted multinomial. Fast: per-verse counters are
    precomputed once and replicates aggregate sampled counters. `unit_texts` maps
    name -> list of aligned raw verse strings; method in {'upgma','nj'}."""
    import dendropy
    uc = _unit_counters(unit_texts, names, ngram)           # clean + count ONCE
    n_units = len(uc[names[0]])
    full = [_agg_counter(uc[nm], range(n_units)) for nm in names]
    Dref = _counters_to_D(full)
    ref_newick = (nj_newick(Dref, names) if method == "nj"
                  else linkage_to_newick(upgma(Dref, names), names))
    tns, ref_splits = _reference_splits(ref_newick, names)
    rng = np.random.default_rng(seed)
    counts = [0] * len(ref_splits)
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        cs = [_agg_counter(uc[nm], idx) for nm in names]
        Db = _counters_to_D(cs)
        bnwk = (nj_newick(Db, names) if method == "nj"
                else linkage_to_newick(upgma(Db, names), names))
        bt = dendropy.Tree.get(data=bnwk, schema="newick", taxon_namespace=tns,
                               preserve_underscores=True)
        bt.is_rooted = False
        bt.encode_bipartitions()
        bsplits = {bp.split_bitmask for bp in bt.bipartition_encoding if not bp.is_trivial()}
        for i, (sb, _m) in enumerate(ref_splits):
            if sb in bsplits:
                counts[i] += 1
    clades = [{"members": m, "support": c / n_boot if n_boot else 0.0}
              for c, (sb, m) in zip(counts, ref_splits)]
    clades.sort(key=lambda c: (-len(c["members"]), c["members"]))
    return {"ref_newick": ref_newick, "clades": clades, "method": method, "n_boot": n_boot}


def bootstrap_ci_block(unit_texts, names, group, gold_newick, n_boot=100, ngram=3,
                       seed=0, method="upgma"):
    """Verse block-bootstrap 95% CIs for family-NN purity and normalized RF — the
    VALID replacement for the token bootstrap in bootstrap_ci (which resampled a
    fitted distribution). Resamples aligned verses with replacement; per-verse
    counters precomputed once for speed."""
    uc = _unit_counters(unit_texts, names, ngram)
    n_units = len(uc[names[0]])
    rng = np.random.default_rng(seed)
    purs, rfs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        cs = [_agg_counter(uc[nm], idx) for nm in names]
        D = _counters_to_D(cs)
        purs.append(nn_purity(D, names, group))
        nwk = nj_newick(D, names) if method == "nj" else linkage_to_newick(upgma(D, names), names)
        rfs.append(rf_corrected(nwk, gold_newick)[2])
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                     float(np.percentile(a, 97.5)))
    return {"purity": pct(purs), "rf": pct(rfs), "n_boot": n_boot}


# ---------- NCD with per-pair self-floor (no global clamp) ----------
def ncd_matrix_v2(texts, cap_bytes=None):
    """gzip NCD with a PER-PAIR self-compression floor instead of one global mean
    floor (the global clamp in ncd_matrix_fixed forces every genuinely close pair to
    exactly 0, destroying fine structure). Symmetric, rare-separator seam, equal byte
    budget. Still a crude Kolmogorov proxy — report comparatively."""
    raw = [t.encode("utf-8") for t in texts]
    if cap_bytes:
        raw = [b[:cap_bytes] for b in raw]
    SEP = b"\x01"
    C = [_gz(b) for b in raw]
    floor = [((_gz(b + SEP + b) - c) / c if c else 0.0) for b, c in zip(raw, C)]
    m = len(texts)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            cxy = 0.5 * (_gz(raw[i] + SEP + raw[j]) + _gz(raw[j] + SEP + raw[i]))
            d = (cxy - min(C[i], C[j])) / max(C[i], C[j])
            D[i, j] = D[j, i] = max(0.0, d - 0.5 * (floor[i] + floor[j]))
    return D


# ---------- symmetrized perplexity / cross-entropy distance (Gamallo 2017) ----------
def _lm(text, n, alpha):
    from collections import defaultdict, Counter
    ctx = defaultdict(Counter)
    for i in range(len(text) - n + 1):
        ctx[text[i:i + n - 1]][text[i + n - 1]] += 1
    return ctx


def _cross_entropy(text, model, n, alpha, V):
    tot = 0.0
    cnt = 0
    for i in range(len(text) - n + 1):
        c = text[i:i + n - 1]
        w = text[i + n - 1]
        counter = model.get(c)
        if counter is None:
            p = 1.0 / V
        else:
            N = sum(counter.values())
            p = (counter.get(w, 0) + alpha) / (N + alpha * V)
        tot += -math.log2(p)
        cnt += 1
    return tot / cnt if cnt else 0.0


def perplexity_distance_matrix(texts, n=3, alpha=0.1, cap=40000):
    """Symmetrized cross-entropy ('perplexity') distance between languages: train a
    smoothed order-n char model per language, measure the excess bits to encode A
    under B's model (and vice-versa) above each language's self-entropy. The
    field-standard char-LM alternative to raw JS (Gamallo et al. 2017). O(m^2*len),
    so texts are capped to `cap` chars."""
    T = [t[:cap] for t in texts]
    V = max(2, len(set("".join(T))))
    models = [_lm(t, n, alpha) for t in T]
    self_ce = [_cross_entropy(T[i], models[i], n, alpha, V) for i in range(len(T))]
    m = len(T)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            cij = _cross_entropy(T[i], models[j], n, alpha, V)
            cji = _cross_entropy(T[j], models[i], n, alpha, V)
            d = 0.5 * ((cij - self_ce[i]) + (cji - self_ce[j]))
            D[i, j] = D[j, i] = max(0.0, d)
    return D


# ---------- treelikeness diagnostic (delta score; Holland et al. 2002) ----------
def delta_score(D, max_full=70, n_sample=200000, seed=0):
    """Mean Holland (2002) delta score over quartets of a distance matrix: 0 =
    perfectly tree-like, higher = more reticulate/areal (borrowing) signal that a
    single tree cannot represent. Exhaustive for n<=max_full, else sampled."""
    import itertools
    import random as _r
    n = D.shape[0]
    if n <= max_full:
        quartets = itertools.combinations(range(n), 4)
    else:
        rng = _r.Random(seed)
        quartets = (tuple(sorted(rng.sample(range(n), 4))) for _ in range(n_sample))
    tot = 0.0
    cnt = 0
    for (i, j, k, l) in quartets:
        s = sorted([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]])
        denom = s[2] - s[0]
        tot += (s[2] - s[1]) / denom if denom > 1e-12 else 0.0
        cnt += 1
    return tot / cnt if cnt else 0.0

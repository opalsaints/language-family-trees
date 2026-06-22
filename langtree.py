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
    """Lowercase/casefold, keep only Unicode letters + single spaces."""
    text = text.casefold()
    if not keep_diacritics:
        # strip combining marks (e.g. é -> e) — used for ASCII-fold experiments
        text = "".join(
            c for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    out = "".join(ch if unicodedata.category(ch).startswith("L") else " " for ch in text)
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

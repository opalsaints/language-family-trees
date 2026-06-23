"""ARM: Bayesian phylogenetics (BEAST2) from ASJP cognate presence/absence.

Pipeline (parallels the distance-based arms, but Bayesian/character-based):
  1. Map our bible-corpus languages -> ISO 639-3 -> ASJP doculects (same path as
     asjp_tree.py: pick the doculect with the most concepts per ISO).
  2. AUTOMATED COGNATE CODING. ASJP ships no cognate judgments (the Cognacy column
     is empty), so we cluster the phonetic ASJPcode forms per concept into cognate
     sets by single-linkage on mean-normalized Levenshtein distance with a fixed
     threshold. NOTE: this is plain edit-distance cognate clustering, NOT lingpy
     LexStat (which trains a sound-correspondence scorer) — cognate_arm.py uses real
     LexStat; here the coding is kept simple/self-contained for the binary matrix.
     CAVEAT: the matrix keeps only variable cognate columns, so the model should add
     a Lewis Mkv ascertainment correction (conditioning on variability). The MCMC
     itself now RUNS (2M generations -> MCC tree, compared to gold below); only the
     ascertainment correction remains future work, so branch lengths/rates are not
     calibrated and we compare topology (GQD/RF) only.
     Each cognate set -> one BINARY character: 1 if the language has a form in that
     set, else 0 (absence). This is the Dollo-ish binary matrix Bayesian language
     phylogenetics (Gray & Atkinson 2003 etc.) runs on.
  3. Write a NEXUS (binary 'standard' datatype) AND a BEAST2 XML (binary CTMC
     substitution, strict clock, Yule tree prior, short chain 1e6 for feasibility)
     into beast/.
  4. Try to locate a BEAST2 'beast' executable / jar (shutil.which). If found: run
     BEAST2 + TreeAnnotator, read the MCC tree, compare to the Glottolog gold with
     BOTH rf_triple and gqd. If NOT found: graceful future-work exit (still report
     the cognate-matrix stats). Exits 0 either way.
"""
import csv
import os
import shutil
import subprocess
import sys

import numpy as np

import langtree as lt
import biblecorpus as bc

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(os.path.abspath(__file__))
ASJP = os.path.join(HERE, "corpus", "asjp", "cldf")
GLOTTOLOG = os.path.join(HERE, "corpus", "glottolog")
BEAST_DIR = os.path.join(HERE, "beast")
COG_THRESHOLD = 0.5     # normalized-Levenshtein single-linkage cognate cutoff
MIN_WORDS = 20          # need enough shared concepts for a meaningful language


# ---------- Levenshtein (same metric as asjp_tree.py) ----------
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


def single_linkage_clusters(forms, threshold):
    """Cluster a list of (lang_label, word) for ONE concept into cognate sets by
    single linkage on normalized Levenshtein distance. Returns list of sets of
    lang_labels (one set per cognate class)."""
    n = len(forms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if forms[i][1] == forms[j][1] or ndist(forms[i][1], forms[j][1]) < threshold:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(forms[i][0])
    return list(groups.values())


def safe(label):
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in label)


def main():
    os.makedirs(BEAST_DIR, exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    skipped = []

    # ---- 1. our languages -> ISO -> ASJP doculects ----
    d = bc.load(verse_cap=2000, char_cap=30000)
    names, rows, iso = d["names"], d["rows"], d["iso"]
    fam_of = {r[0]: r[1] for r in rows}

    iso_want = {iso[n] for n in names if iso[n]}
    iso_to_ids = {}
    with open(os.path.join(ASJP, "languages.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r["ISO639P3code"]
            if code in iso_want:
                iso_to_ids.setdefault(code, []).append(r["ID"])
    candidate_ids = {i for ids in iso_to_ids.values() for i in ids}
    print(f"[1] {len(names)} languages loaded; {len(iso_want)} distinct ISO codes; "
          f"{len(candidate_ids)} candidate ASJP doculects.")

    # forms[doculect][concept] = ASJPcode word
    forms_by_doc = {}
    with open(os.path.join(ASJP, "forms.csv"), encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        li, pi, fi = hdr.index("Language_ID"), hdr.index("Parameter_ID"), hdr.index("Form")
        for row in rd:
            if row[li] in candidate_ids:
                forms_by_doc.setdefault(row[li], {}).setdefault(row[pi], row[fi])

    # per ISO pick doculect with most concepts; map back to our labels
    keep, doc_forms, krows = [], {}, []
    for n in names:
        ids = [i for i in iso_to_ids.get(iso[n], []) if i in forms_by_doc]
        if not ids:
            skipped.append((n, "no ASJP doculect for ISO " + str(iso[n])))
            continue
        best = max(ids, key=lambda i: len(forms_by_doc[i]))
        if len(forms_by_doc[best]) < MIN_WORDS:
            skipped.append((n, f"only {len(forms_by_doc[best])} ASJP words (<{MIN_WORDS})"))
            continue
        keep.append(n)
        doc_forms[n] = forms_by_doc[best]
        krows.append(next(r for r in rows if r[0] == n))

    print(f"[1] ASJP usable for {len(keep)}/{len(names)} languages "
          f"(avg {np.mean([len(doc_forms[k]) for k in keep]):.0f} concepts each).")
    if skipped:
        print(f"[1] SKIPPED {len(skipped)} languages (logged):")
        for n, why in skipped:
            print(f"      - {n}: {why}")

    labels = [safe(n) for n in keep]
    label_of = dict(zip(keep, labels))

    # ---- 2. automated cognate coding -> binary character matrix ----
    all_concepts = set()
    for k in keep:
        all_concepts.update(doc_forms[k].keys())
    char_cols = []   # each: dict label->'1'/'0', plus meta (concept, set size)
    for concept in sorted(all_concepts, key=lambda c: (len(c), c)):
        present = [(k, doc_forms[k][concept]) for k in keep if concept in doc_forms[k]]
        if len(present) < 2:
            continue  # singleton concept carries no phylogenetic signal
        clusters = single_linkage_clusters(present, COG_THRESHOLD)
        for cset in clusters:
            if len(cset) < 2:
                continue  # autapomorphy (one language only) -> uninformative for tree
            col = {k: ("1" if k in cset else "0") for k in keep}
            char_cols.append((concept, len(cset), col))

    n_chars = len(char_cols)
    n_taxa = len(keep)
    # per-language presence count (1s)
    ones = {k: sum(1 for _, _, col in char_cols if col[k] == "1") for k in keep}
    total_ones = sum(ones.values())
    fill = total_ones / (n_chars * n_taxa) if n_chars and n_taxa else 0.0
    print(f"[2] cognate matrix: {n_taxa} languages x {n_chars} binary cognate "
          f"characters (threshold {COG_THRESHOLD}); fill {fill:.3f} "
          f"(mean {total_ones / n_taxa:.0f} present/lang).")

    # matrix string per taxon
    matrix = {k: "".join(col[k] for _, _, col in char_cols) for k in keep}

    # ---- 3a. write NEXUS ----
    nexus_path = os.path.join(BEAST_DIR, "cognates.nex")
    with open(nexus_path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\nBEGIN DATA;\n")
        f.write(f"  DIMENSIONS NTAX={n_taxa} NCHAR={n_chars};\n")
        f.write('  FORMAT DATATYPE=STANDARD SYMBOLS="01" MISSING=? GAP=-;\n')
        f.write("  MATRIX\n")
        w = max(len(l) for l in labels)
        for k in keep:
            f.write(f"    {label_of[k]:<{w}}  {matrix[k]}\n")
        f.write("  ;\nEND;\n")
    print(f"[3] wrote NEXUS -> {nexus_path}")

    # ---- 3b. write BEAST2 XML (binary CTMC, strict clock, Yule) ----
    xml_path = os.path.join(BEAST_DIR, "cognates.xml")
    write_beast_xml(xml_path, keep, label_of, matrix, n_chars, chain=2_000_000)
    print(f"[3] wrote BEAST2 XML -> {xml_path}")

    # gold tree (Glottolog, fall back to taxonomy rows) on the SAME taxa
    gold = None
    try:
        iso_set = {iso[k] for k in keep if iso[k]}
        lab_for = {iso[k]: label_of[k] for k in keep if iso[k]}
        gold, matched = lt.gold_newick_from_glottolog(iso_set, GLOTTOLOG, label_for=lab_for)
        if not gold or len(matched) < 4:
            raise RuntimeError("glottolog gold too small")
        print(f"[3] gold tree from Glottolog ({len(matched)} ISO matched).")
    except Exception as e:
        gold = lt.gold_newick_from_rows([(label_of[r[0]], r[1], r[2], r[3]) for r in krows])
        print(f"[3] gold tree from taxonomy rows (glottolog fallback: {e}).")

    # ---- 4. locate + run BEAST2 ----
    beast_exe = shutil.which("beast") or shutil.which("beast2")
    jar = None
    if not beast_exe:
        for cand in (os.environ.get("BEAST", ""),
                     "/Applications/BEAST 2.7.7/lib/launcher.jar",
                     os.path.expanduser("~/beast/lib/launcher.jar")):
            if cand and os.path.exists(cand):
                jar = cand
                break

    if not beast_exe and not jar:
        print("\n[4] BEAST2 executable/jar NOT found on this machine "
              "(shutil.which('beast')/('beast2') empty; no known jar path).")
        print("    -> FUTURE WORK: NEXUS + BEAST2 XML inputs are generated and ready.")
        print("       Run with:  beast -seed 42 beast/cognates.xml")
        print("       then:      treeannotator -burnin 10 beast/cognates.trees beast/mcc.tree")
        print("    Cognate-matrix stats (the deliverable that does not need BEAST):")
        print(f"       languages           = {n_taxa}")
        print(f"       cognate characters  = {n_chars}")
        print(f"       matrix fill (1s)    = {fill:.3f}")
        print(f"       gold taxa           = {n_taxa}")
        _report(arm_done=False, n_taxa=n_taxa, n_chars=n_chars, fill=fill,
                nexus=nexus_path, xml=xml_path, skipped=skipped,
                mcc=None, gold=gold, labels=labels)
        return

    # BEAST2 present: run it.
    print(f"\n[4] BEAST2 found: {beast_exe or jar}. Running short chain...")
    try:
        run_beast(beast_exe, jar, xml_path, BEAST_DIR)
        trees = os.path.join(BEAST_DIR, "cognates.trees")
        mcc = os.path.join(BEAST_DIR, "mcc.tree")
        run_treeannotator(beast_exe, jar, trees, mcc)
        mcc_newick = read_mcc_newick(mcc, labels)
        print("[4] MCC tree read; comparing to gold...")
        triple = lt.rf_triple(mcc_newick, gold, labels)
        g = lt.gqd(mcc_newick, gold, labels)
        print(f"    rf_triple: observed={triple['observed']:.3f} floor={triple['floor']:.3f} "
              f"null_p50={triple['null_p50']:.3f} rescaled={triple['rescaled']:.3f}")
        print(f"    gqd      : {g['gqd']:.3f} (resolved gold quartets {g['resolved_in_gold']})")
        # render the MCC tree (coloured by family) + dump a result artifact so the
        # notebook/README can cite the Bayesian arm without re-running BEAST2.
        fam_by_label = {label_of[k]: fam_of[k] for k in keep}
        fig_path = os.path.join(HERE, "figures", "beast_mcc_tree.png")
        try:
            save_mcc_figure(mcc, fam_by_label, fig_path)
            print(f"[4] saved {fig_path}")
        except Exception as fe:
            print(f"[4] figure skipped ({fe})")
        import json
        res = {"languages": n_taxa, "cognate_chars": n_chars, "matrix_fill": fill,
               "chain": 2_000_000, "rf_triple": triple, "gqd": g,
               "skipped": [n for n, _ in skipped]}
        with open(os.path.join(BEAST_DIR, "beast_result.json"), "w") as jf:
            json.dump(res, jf, indent=2)
        print(f"[4] wrote {os.path.join(BEAST_DIR, 'beast_result.json')}")
        _report(arm_done=True, n_taxa=n_taxa, n_chars=n_chars, fill=fill,
                nexus=nexus_path, xml=xml_path, skipped=skipped,
                mcc=mcc, gold=gold, labels=labels, triple=triple, gqd=g)
    except Exception as e:
        print(f"[4] BEAST2 run FAILED ({e}); inputs are still generated. Future-work exit.")
        _report(arm_done=False, n_taxa=n_taxa, n_chars=n_chars, fill=fill,
                nexus=nexus_path, xml=xml_path, skipped=skipped,
                mcc=None, gold=gold, labels=labels)


def write_beast_xml(path, keep, label_of, matrix, n_chars, chain=1_000_000):
    """Minimal BEAST2 v2.x XML: binary CTMC substitution, strict clock, Yule prior."""
    seqs = []
    for k in keep:
        lab = label_of[k]
        seqs.append(
            f'        <sequence id="seq_{lab}" spec="beast.base.evolution.alignment.Sequence" '
            f'taxon="{lab}" totalcount="2" value="{matrix[k]}"/>')
    seq_block = "\n".join(seqs)
    log_every = max(1000, chain // 1000)
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<beast beautitemplate='Standard' beautistatus='' namespace="beast.pkgmgmt:beast.base.core:beast.base.inference:beast.base.evolution.alignment:beast.base.evolution.tree:beast.base.evolution.tree.coalescent:beast.base.evolution.speciation:beast.base.evolution.operator:beast.base.inference.operator:beast.base.evolution.sitemodel:beast.base.evolution.substitutionmodel:beast.base.evolution.branchratemodel:beast.base.evolution.likelihood:beast.base.inference.util" required="" version="2.7">

    <data id="cognates" spec="beast.base.evolution.alignment.Alignment" dataType="binary">
{seq_block}
    </data>

    <map name="Uniform">beast.base.inference.distribution.Uniform</map>
    <map name="Exponential">beast.base.inference.distribution.Exponential</map>
    <map name="LogNormal">beast.base.inference.distribution.LogNormalDistributionModel</map>
    <map name="Normal">beast.base.inference.distribution.Normal</map>
    <map name="Beta">beast.base.inference.distribution.Beta</map>
    <map name="Gamma">beast.base.inference.distribution.Gamma</map>
    <map name="LaplaceDistribution">beast.base.inference.distribution.LaplaceDistribution</map>
    <map name="prior">beast.base.inference.distribution.Prior</map>
    <map name="InverseGamma">beast.base.inference.distribution.InverseGamma</map>
    <map name="OneOnX">beast.base.inference.distribution.OneOnX</map>

    <run id="mcmc" spec="beast.base.inference.MCMC" chainLength="{chain}">
        <state id="state" spec="beast.base.inference.State" storeEvery="5000">
            <tree id="Tree.t:cognates" spec="beast.base.evolution.tree.Tree" name="stateNode">
                <taxonset id="TaxonSet.cognates" spec="beast.base.evolution.alignment.TaxonSet">
                    <alignment idref="cognates"/>
                </taxonset>
            </tree>
            <parameter id="birthRate.t:cognates" spec="beast.base.inference.parameter.RealParameter" name="stateNode">1.0</parameter>
            <parameter id="clockRate.c:cognates" spec="beast.base.inference.parameter.RealParameter" name="stateNode">1.0</parameter>
            <parameter id="freqParameter.s:cognates" spec="beast.base.inference.parameter.RealParameter" dimension="2" lower="0.0" name="stateNode" upper="1.0">0.5</parameter>
        </state>

        <init id="RandomTree.t:cognates" spec="beast.base.evolution.tree.coalescent.RandomTree" estimate="false" initial="@Tree.t:cognates" taxa="@cognates">
            <populationModel id="ConstantPopulation0.t:cognates" spec="beast.base.evolution.tree.coalescent.ConstantPopulation">
                <parameter id="randomPopSize.t:cognates" spec="beast.base.inference.parameter.RealParameter" name="popSize">1.0</parameter>
            </populationModel>
        </init>

        <distribution id="posterior" spec="beast.base.inference.CompoundDistribution">
            <distribution id="prior" spec="beast.base.inference.CompoundDistribution">
                <distribution id="YuleModel.t:cognates" spec="beast.base.evolution.speciation.YuleModel" birthDiffRate="@birthRate.t:cognates" tree="@Tree.t:cognates"/>
                <prior id="YuleBirthRatePrior.t:cognates" name="distribution" x="@birthRate.t:cognates">
                    <Uniform id="Uniform.0" name="distr" upper="1000.0"/>
                </prior>
                <prior id="ClockPrior.c:cognates" name="distribution" x="@clockRate.c:cognates">
                    <Uniform id="Uniform.01" name="distr" upper="1000.0"/>
                </prior>
            </distribution>
            <distribution id="likelihood" spec="beast.base.inference.CompoundDistribution" useThreads="true">
                <distribution id="treeLikelihood.cognates" spec="beast.base.evolution.likelihood.TreeLikelihood" data="@cognates" tree="@Tree.t:cognates">
                    <siteModel id="SiteModel.s:cognates" spec="beast.base.evolution.sitemodel.SiteModel">
                        <parameter id="mutationRate.s:cognates" spec="beast.base.inference.parameter.RealParameter" estimate="false" name="mutationRate">1.0</parameter>
                        <parameter id="gammaShape.s:cognates" spec="beast.base.inference.parameter.RealParameter" estimate="false" name="shape">1.0</parameter>
                        <parameter id="proportionInvariant.s:cognates" spec="beast.base.inference.parameter.RealParameter" estimate="false" lower="0.0" name="proportionInvariant" upper="1.0">0.0</parameter>
                        <substModel id="GeneralSubstitutionModel.s:cognates" spec="beast.base.evolution.substitutionmodel.GeneralSubstitutionModel">
                            <parameter id="rates.s:cognates" spec="beast.base.inference.parameter.RealParameter" dimension="2" estimate="false" name="rates">1.0 1.0</parameter>
                            <frequencies id="estimatedFreqs.s:cognates" spec="beast.base.evolution.substitutionmodel.Frequencies" frequencies="@freqParameter.s:cognates"/>
                        </substModel>
                    </siteModel>
                    <branchRateModel id="StrictClock.c:cognates" spec="beast.base.evolution.branchratemodel.StrictClockModel" clock.rate="@clockRate.c:cognates"/>
                </distribution>
            </distribution>
        </distribution>

        <operator id="YuleBirthRateScaler.t:cognates" spec="beast.base.evolution.operator.kernel.BactrianScaleOperator" parameter="@birthRate.t:cognates" upper="10.0" weight="3.0"/>
        <operator id="CognatesTreeScaler.t:cognates" spec="beast.base.evolution.operator.kernel.BactrianScaleOperator" scaleFactor="0.5" tree="@Tree.t:cognates" upper="10.0" weight="3.0"/>
        <operator id="CognatesTreeRootScaler.t:cognates" spec="beast.base.evolution.operator.kernel.BactrianScaleOperator" rootOnly="true" scaleFactor="0.5" tree="@Tree.t:cognates" upper="10.0" weight="3.0"/>
        <operator id="CognatesUniformOperator.t:cognates" spec="beast.base.evolution.operator.kernel.BactrianNodeOperator" tree="@Tree.t:cognates" weight="30.0"/>
        <operator id="CognatesSubtreeSlide.t:cognates" spec="beast.base.evolution.operator.kernel.BactrianSubtreeSlide" tree="@Tree.t:cognates" weight="15.0"/>
        <operator id="CognatesNarrow.t:cognates" spec="beast.base.evolution.operator.Exchange" tree="@Tree.t:cognates" weight="15.0"/>
        <operator id="CognatesWide.t:cognates" spec="beast.base.evolution.operator.Exchange" isNarrow="false" tree="@Tree.t:cognates" weight="3.0"/>
        <operator id="CognatesWilsonBalding.t:cognates" spec="beast.base.evolution.operator.WilsonBalding" tree="@Tree.t:cognates" weight="3.0"/>
        <operator id="StrictClockRateScaler.c:cognates" spec="beast.base.evolution.operator.kernel.BactrianScaleOperator" parameter="@clockRate.c:cognates" upper="10.0" weight="3.0"/>
        <operator id="strictClockUpDownOperator.c:cognates" spec="beast.base.inference.operator.kernel.BactrianUpDownOperator" scaleFactor="0.75" weight="3.0">
            <up idref="clockRate.c:cognates"/>
            <down idref="Tree.t:cognates"/>
        </operator>
        <operator id="FrequenciesExchanger.s:cognates" spec="beast.base.inference.operator.kernel.BactrianDeltaExchangeOperator" delta="0.01" weight="0.1">
            <parameter idref="freqParameter.s:cognates"/>
        </operator>

        <logger id="tracelog" spec="beast.base.inference.Logger" fileName="cognates.log" logEvery="{log_every}" model="@posterior" sanitiseHeaders="true" sort="smart">
            <log idref="posterior"/>
            <log idref="likelihood"/>
            <log idref="prior"/>
            <log idref="treeLikelihood.cognates"/>
            <log id="TreeHeight.t:cognates" spec="beast.base.evolution.tree.TreeStatLogger" tree="@Tree.t:cognates"/>
            <log idref="YuleModel.t:cognates"/>
            <log idref="birthRate.t:cognates"/>
            <log idref="clockRate.c:cognates"/>
            <log idref="freqParameter.s:cognates"/>
        </logger>

        <logger id="screenlog" spec="beast.base.inference.Logger" logEvery="{log_every}">
            <log idref="posterior"/>
            <log idref="likelihood"/>
            <log idref="prior"/>
        </logger>

        <logger id="treelog.t:cognates" spec="beast.base.inference.Logger" fileName="cognates.trees" logEvery="{log_every}" mode="tree">
            <log id="TreeWithMetaDataLogger.t:cognates" spec="beast.base.evolution.TreeWithMetaDataLogger" tree="@Tree.t:cognates"/>
        </logger>

        <operatorschedule id="OperatorSchedule" spec="beast.base.inference.OperatorSchedule"/>
    </run>
</beast>
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def run_beast(exe, jar, xml_path, workdir):
    if exe:
        cmd = [exe, "-overwrite", "-seed", "42", xml_path]
    else:
        cmd = ["java", "-jar", jar, "-overwrite", "-seed", "42", xml_path]
    subprocess.run(cmd, cwd=workdir, check=True, timeout=3600)


def run_treeannotator(exe, jar, trees, mcc):
    ta = (shutil.which("treeannotator") if exe else None)
    # BEAST 2.7 renamed -heights -> -height (default CA = Common-Ancestor heights);
    # use the default by omitting it so we don't depend on a method-name spelling.
    if ta:
        cmd = [ta, "-burnin", "10", trees, mcc]
    else:
        cmd = ["java", "-cp", os.path.dirname(jar) + "/*",
               "beastfx.app.treeannotator.TreeAnnotator",
               "-burnin", "10", trees, mcc]
    subprocess.run(cmd, check=True, timeout=600)


def read_mcc_newick(mcc_path, labels):
    """Read a NEXUS MCC tree -> a plain Newick string over our `labels`."""
    import dendropy
    tns = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(path=mcc_path, schema="nexus", taxon_namespace=tns)
    return t.as_string(schema="newick").strip()


def save_mcc_figure(mcc_path, fam_by_label, out_path):
    """Draw the BEAST2 MCC phylogram (faithful branch lengths) coloured by family."""
    import dendropy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tns = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(path=mcc_path, schema="nexus", taxon_namespace=tns,
                          preserve_underscores=True)
    leaves = list(t.leaf_node_iter())
    yof = {lf: i for i, lf in enumerate(leaves)}
    xof = {}
    for node in t.preorder_node_iter():            # x = root->node distance
        xof[node] = 0.0 if node.parent_node is None \
            else xof[node.parent_node] + (node.edge.length or 0.0)
    for node in t.postorder_node_iter():           # internal y = mean of children
        if not node.is_leaf():
            ys = [yof[c] for c in node.child_nodes()]
            yof[node] = sum(ys) / len(ys)
    fams = sorted(set(fam_by_label.values()))
    cmap = plt.get_cmap("tab20")
    fcol = {f: cmap(i % 20) for i, f in enumerate(fams)}
    xmax = max(xof.values()) or 1.0

    fig, ax = plt.subplots(figsize=(9, max(6, len(leaves) * 0.22)))
    for node in t.preorder_node_iter():
        if node.parent_node is not None:
            ax.plot([xof[node.parent_node], xof[node]], [yof[node], yof[node]],
                    color="0.45", lw=0.8)
        kids = node.child_nodes()
        if kids:
            ys = [yof[c] for c in kids]
            ax.plot([xof[node], xof[node]], [min(ys), max(ys)], color="0.45", lw=0.8)
    for lf in leaves:
        lab = lf.taxon.label if lf.taxon else str(lf)
        fam = fam_by_label.get(lab, fam_by_label.get(lab.replace(" ", "_"), "?"))
        ax.text(xof[lf] + 0.01 * xmax, yof[lf], lab.replace("_", " "),
                va="center", fontsize=6, color=fcol.get(fam, "0.2"))
    ax.set_yticks([])
    ax.set_xlim(0, xmax * 1.25)
    ax.set_xlabel("substitutions / site (MCC, common-ancestor heights)")
    ax.set_title("BEAST2 Bayesian MCC tree — binary cognate matrix")
    handles = [Line2D([0], [0], color=fcol[f], lw=3) for f in fams]
    ax.legend(handles, fams, fontsize=6, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 0.98), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _report(arm_done, n_taxa, n_chars, fill, nexus, xml, skipped, mcc, gold,
            labels, triple=None, gqd=None):
    print("\n========== BEAST ARM SUMMARY ==========")
    print(f"  status            : {'RAN BEAST2 + compared to gold' if arm_done else 'inputs generated (BEAST2 not available) - future work'}")
    print(f"  languages (taxa)  : {n_taxa}")
    print(f"  cognate chars     : {n_chars}")
    print(f"  matrix fill       : {fill:.3f}")
    print(f"  NEXUS             : {nexus}")
    print(f"  BEAST2 XML        : {xml}")
    if skipped:
        print(f"  skipped languages : {len(skipped)} (see [1] log)")
    if triple is not None:
        print(f"  rf_triple         : {triple}")
        print(f"  gqd               : {gqd}")
    print("=======================================")


if __name__ == "__main__":
    main()

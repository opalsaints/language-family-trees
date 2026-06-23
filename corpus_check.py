"""GlotLID corpus-purity gate (data-quality gate requested by the review).

For every language in the curated 57-language Bible set AND its FLORES-200
counterpart, classify a sample of ~300 lines/verses with the GlotLID fastText
model and report the % predicted as the EXPECTED language. Languages scoring
below ~70% in-target are FLAGGED as possible contamination / mislabeling
(e.g. Kazakh-vs-Russian style swaps, or wrong-script romanization leakage).

Why this matters: js_matrix / NN-purity / tree-vs-gold all assume each text file
really *is* the language its label claims. GlotLID (2102 languages, fastText,
char n-grams) is an independent oracle that does not see our labels, so it
catches mislabeled or mixed files before they poison the distance matrices.

Notes / honest caveats baked in:
  * GlotLID predicts INDIVIDUAL languages; our Bible metadata uses several
    macrolanguage ISO codes (ara, nor, alb, nep, syr, tgl) and a couple of
    deprecated codes (pes->fas, lav->lvs). We therefore accept a *set* of
    GlotLID labels per language (ACCEPT), and ALSO report the single dominant
    predicted label so a human can eyeball what GlotLID actually thinks.
  * fasttext.predict() is broken under NumPy 2.x (np.array(copy=False)); we call
    the underlying C predictor m.f.predict(text,k,thr,'strict') which returns
    [(prob,label),...] and sidesteps the wrapper. Verified on eng -> eng_Latn.
  * GlotLID needs the NATIVE script. Romanized text would be classified as a
    Latin-script language and look "contaminated"; that is a romanization
    artifact, not corpus contamination. We classify RAW (native-script) text
    here and say so. (Romanized variants are handled elsewhere in the project.)
  * Graceful-skip: any language GlotLID has no acceptable label for, or any file
    we can't sample, is logged under SKIPPED rather than silently dropped.
"""
import glob
import os
import re
import sys

import biblecorpus as bc

HERE = os.path.dirname(os.path.abspath(__file__))
FLORES_DEV = os.path.join(HERE, "corpus", "flores200_dataset", "dev")
SAMPLE_N = 300            # lines/verses per language
FLAG_THRESHOLD = 70.0     # % in-target below which we flag (review asked ~70%)

# --- Bible-corpus script name -> ISO 15924 4-letter code (GlotLID convention) --
SCRIPT2CODE = {
    "Latin": "Latn", "Cyrillic": "Cyrl", "Devanagari": "Deva",
    "Ethiopic": "Ethi", "Greek": "Grek", "Hangul": "Hang", "Hebrew": "Hebr",
    "Kanjii": "Jpan", "Kannada": "Knda", "Malayalam": "Mlym", "Myanmar": "Mymr",
    "Syriac": "Syrc", "Telugu": "Telu", "Thai": "Thai", "Arabic": "Arab",
    "Chinese": "Hani",
}

# --- ISO-639-3 macro/legacy code -> set of acceptable GlotLID base codes -------
# (GlotLID emits individual-language codes; a macrolanguage corpus legitimately
#  matches any of its members. Verified each member exists in the model.)
ISO_ALIASES = {
    "nor": {"nob", "nno"},                       # Norwegian macro -> Bokmaal/Nynorsk
    "lav": {"lvs", "ltg"},                       # Latvian -> Standard/Latgalian
    "alb": {"als", "aln", "sqi"},                # Albanian macro -> Tosk/Gheg
    "pes": {"pes", "fas", "prs"},                # Persian (deprecated pes; model has fas)
    "nep": {"npi", "nep"},                       # Nepali macro -> npi
    "ara": {"arb", "ara", "acm", "arz", "apc", "ary", "ars", "acq", "aeb", "ajp"},  # Arabic macro -> any MSA/dialect
    "syr": {"syr", "aii", "syc", "cld"},         # Syriac -> Assyrian/Classical Neo-Aramaic
    "tgl": {"tgl", "fil"},                        # Tagalog <-> Filipino
}

# Bible label -> human-friendly expected description (for the printed table)
def base_of(label):
    """'__label__eng_Latn' or 'eng_Latn' -> 'eng'."""
    return label.replace("__label__", "").split("_")[0]


def script_of(label):
    return label.replace("__label__", "").split("_")[-1]


def load_glotlid():
    matches = glob.glob(os.path.join(
        HERE, "corpus", "glotlid_cache",
        "models--cis-lmu--glotlid", "snapshots", "*", "model.bin"))
    if not matches:
        sys.exit("FATAL: GlotLID model.bin not found under corpus/glotlid_cache/")
    import fasttext
    m = fasttext.load_model(matches[0])
    print(f"[loaded GlotLID] {os.path.relpath(matches[0], HERE)}  "
          f"({len(m.get_labels())} languages)")
    return m


def predict_one(model, text):
    """Top-1 (prob,label) via the C predictor (NumPy-2-safe). '' on failure."""
    t = text.replace("\n", " ").strip()
    if not t:
        return (0.0, "")
    try:
        res = model.f.predict(t, 1, 0.0, "strict")
    except Exception:
        try:
            res = model.f.predict(t, 1, 0.0, "ignore")
        except Exception:
            return (0.0, "")
    if not res:
        return (0.0, "")
    prob, lab = res[0]
    return (prob, lab.replace("__label__", ""))


def accept_set(iso, script_code):
    """Set of acceptable GlotLID base codes for an expected (iso, script)."""
    if iso in ISO_ALIASES:
        return set(ISO_ALIASES[iso])
    return {iso}


def classify_lines(model, lines, iso, script_code, extra_accept=None):
    """Return (pct_in_target, n_used, dominant_label, dominant_pct, script_pct).

    extra_accept: optional set of additional acceptable base codes (e.g. zho for
    Mandarin when scoring the FLORES zho_Hans file).
    """
    acc = accept_set(iso, script_code)
    if extra_accept:
        acc = acc | set(extra_accept)
    n_in = 0
    n_used = 0
    n_script = 0
    from collections import Counter
    counts = Counter()
    for ln in lines:
        prob, lab = predict_one(model, ln)
        if not lab:
            continue
        n_used += 1
        counts[lab] += 1
        if base_of(lab) in acc:
            n_in += 1
        if script_of(lab) == script_code:
            n_script += 1
    if n_used == 0:
        return (0.0, 0, "(none)", 0.0, 0.0)
    pct = 100.0 * n_in / n_used
    dom_lab, dom_n = counts.most_common(1)[0]
    return (pct, n_used, dom_lab, 100.0 * dom_n / n_used, 100.0 * n_script / n_used)


def bible_lines(units_for_lab, raw_for_lab):
    """Prefer per-verse units; fall back to splitting raw text into lines."""
    if units_for_lab:
        out = [v for v in units_for_lab if v and v.strip()]
        if out:
            return out[:SAMPLE_N]
    # fallback: split on sentence-ish boundaries / newlines
    txt = raw_for_lab or ""
    parts = re.split(r"[\n.!?。।]+", txt)
    return [p.strip() for p in parts if len(p.strip()) > 8][:SAMPLE_N]


# FLORES-200 uses different code/script tags than our Bible metadata for a few
# languages. These are deliberate, linguistically-justified bridges (NOT label
# laundering): Mandarin cmn==zho (FLORES splits by Simplified/Traditional Han);
# Serbian FLORES ships Cyrillic, our Bible Serbian is romanized Latin -> compare
# the Cyrillic FLORES file against the Cyrillic-accepting srp label.
FLORES_OVERRIDE = {
    "cmn_Hani": ["zho_Hans", "zho_Hant"],   # Mandarin -> Chinese (Simpl/Trad)
    "srp_Latn": ["srp_Cyrl"],               # Serbian: FLORES is Cyrillic only
}


def flores_path(iso, script_code):
    """Find a FLORES dev file matching iso_script, allowing macrolang aliases
    and a small set of justified FLORES code/script overrides."""
    direct = os.path.join(FLORES_DEV, f"{iso}_{script_code}.dev")
    if os.path.exists(direct):
        return direct, f"{iso}_{script_code}"
    for alias in sorted(accept_set(iso, script_code)):
        p = os.path.join(FLORES_DEV, f"{alias}_{script_code}.dev")
        if os.path.exists(p):
            return p, f"{alias}_{script_code}"
    for cand in FLORES_OVERRIDE.get(f"{iso}_{script_code}", []):
        p = os.path.join(FLORES_DEV, f"{cand}.dev")
        if os.path.exists(p):
            return p, cand
    return None, None


def main():
    model = load_glotlid()
    print("[loading Bible corpus] verse_cap to feed up to %d lines/lang" % SAMPLE_N)
    d = bc.load(verse_cap=SAMPLE_N, char_cap=400000, return_units=True)
    names = d["names"]
    iso = d["iso"]
    script = d["script"]
    units = d.get("units", {})
    raw = d["rawtext"]

    rows = []          # (lang, corpus, expected_label, pct, n, dom, dom_pct, script_pct)
    skipped = []       # (lang, corpus, reason)

    for lab in names:
        iso_c = iso[lab]
        sc_name = script[lab]
        sc_code = SCRIPT2CODE.get(sc_name)
        if sc_code is None:
            skipped.append((lab, "BIBLE", f"no ISO15924 code for script '{sc_name}'"))
            skipped.append((lab, "FLORES", f"no ISO15924 code for script '{sc_name}'"))
            continue
        expected = f"{iso_c}_{sc_code}"
        # also show what an aliased macrolang accepts
        acc = accept_set(iso_c, sc_code)
        exp_disp = expected if acc == {iso_c} else expected + " {" + "/".join(sorted(acc)) + "}"

        # ---- BIBLE (native script, raw) ----
        lines = bible_lines(units.get(lab), raw.get(lab))
        if not lines:
            skipped.append((lab, "BIBLE", "no usable text"))
        else:
            pct, n, dom, dompct, scpct = classify_lines(model, lines, iso_c, sc_code)
            rows.append((lab, "BIBLE", exp_disp, pct, n, dom, dompct, scpct))

        # ---- FLORES ----
        fp, fcode = flores_path(iso_c, sc_code)
        if fp is None:
            skipped.append((lab, "FLORES", f"no FLORES file for {expected} or aliases"))
        else:
            with open(fp, encoding="utf-8") as fh:
                flines = [l.strip() for l in fh if l.strip()][:SAMPLE_N]
            # if FLORES uses a different base code / script, score against it too
            f_iso, f_sc = base_of(fcode), script_of(fcode)
            extra = {f_iso} if f_iso != iso_c else None
            score_sc = f_sc            # use the FLORES file's actual script
            disp = exp_disp if fcode == expected else exp_disp + f"  [FLORES={fcode}]"
            pct, n, dom, dompct, scpct = classify_lines(
                model, flines, iso_c, score_sc, extra_accept=extra)
            rows.append((lab, "FLORES", disp, pct, n, dom, dompct, scpct))

    # ---------------- print table ----------------
    print()
    print("=" * 104)
    print("GlotLID CORPUS-PURITY GATE  (native script; sample up to %d lines; flag < %.0f%% in-target)"
          % (SAMPLE_N, FLAG_THRESHOLD))
    print("=" * 104)
    hdr = f"{'language':18s} {'corpus':6s} {'expected':28s} {'%in-target':>10s} {'n':>4s} {'dominant_pred':>16s} {'dom%':>5s} {'flag':>5s}"
    print(hdr)
    print("-" * 104)
    flagged = []
    rows.sort(key=lambda r: (r[1], r[0]))
    for (lab, corpus, exp, pct, n, dom, dompct, scpct) in rows:
        flag = "FLAG" if pct < FLAG_THRESHOLD else ""
        if flag:
            flagged.append((lab, corpus, exp, pct, dom, dompct))
        print(f"{lab:18s} {corpus:6s} {exp:28s} {pct:9.1f}% {n:4d} {dom:>16s} {dompct:4.0f}% {flag:>5s}")

    # ---------------- summary ----------------
    print("-" * 104)
    bible_rows = [r for r in rows if r[1] == "BIBLE"]
    flores_rows = [r for r in rows if r[1] == "FLORES"]

    def avg(rs):
        return sum(r[3] for r in rs) / len(rs) if rs else float("nan")

    print(f"BIBLE : {len(bible_rows)} langs, mean %in-target = {avg(bible_rows):.1f}%, "
          f"{sum(1 for r in bible_rows if r[3] < FLAG_THRESHOLD)} flagged")
    print(f"FLORES: {len(flores_rows)} langs, mean %in-target = {avg(flores_rows):.1f}%, "
          f"{sum(1 for r in flores_rows if r[3] < FLAG_THRESHOLD)} flagged")

    print()
    if flagged:
        print(f"*** {len(flagged)} FLAGGED (<{FLAG_THRESHOLD:.0f}% in-target) ***")
        for (lab, corpus, exp, pct, dom, dompct) in sorted(flagged, key=lambda x: x[3]):
            print(f"   FLAG  {lab:18s} [{corpus}]  {pct:5.1f}% in-target  "
                  f"(expected {exp}; GlotLID mostly says {dom} @ {dompct:.0f}%)")
    else:
        print("No languages flagged: every corpus/language is >= %.0f%% in-target." % FLAG_THRESHOLD)

    if skipped:
        print()
        print(f"--- SKIPPED ({len(skipped)}) ---")
        for (lab, corpus, reason) in skipped:
            print(f"   skip  {lab:18s} [{corpus}]  {reason}")

    return flagged


if __name__ == "__main__":
    main()

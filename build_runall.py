"""Generate RunAll.ipynb — the HEAVY-COMPUTE runner, designed to run on Colab.

LanguageTrees.ipynb is the lightweight *presentation* notebook (core results live,
heavy arms shown via committed figures). RunAll.ipynb is the opposite: it runs the
ENTIRE study's heavy compute ON COLAB — installs every dependency, downloads all
data (Bible corpus, ASJP, FLORES-200, Glottolog, GlotLID, BEAST2), runs every arm
script, regenerates all figures, and bundles them for download / Drive / GitHub —
so the laptop never has to. Open it on Colab (Pro recommended for speed) and
Runtime → Run all.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Run-All (heavy compute) — **run this on Colab**, not your laptop
### Language Family Trees from Information Theory — Complexity Lab (Jonathan Cowley & Nil Doğan)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opalsaints/language-family-trees/blob/main/RunAll.ipynb)

This notebook reproduces **every figure and number** of the study on Colab's machine: it installs all
dependencies, downloads all corpora/models, runs all 13 analysis scripts, and bundles the regenerated
figures. Runtime → **Run all**. On Colab **Pro** (more RAM / a faster CPU) it finishes comfortably; on
free Colab it still works but the romanization + bootstrap cells are slower. The clean *presentation*
notebook is `LanguageTrees.ipynb`; this one is the compute engine behind it.
""")

md("## 1. Clone the repo + install all dependencies")
code(r"""
import os, sys, subprocess, time
if not os.path.isdir("language-family-trees"):
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/opalsaints/language-family-trees"], check=True)
os.chdir("language-family-trees")
print("cwd:", os.getcwd())
# core + all arm dependencies
subprocess.run([sys.executable,"-m","pip","install","-q",
                "numpy","scipy","matplotlib","dendropy","uroman",
                "lingpy","epitran","panphon","pyglottolog","fasttext-wheel","splitspy",
                "huggingface_hub"], check=False)
print("deps installed.")
""")

md("## 2. Download all data (Bible · ASJP · FLORES-200 · Glottolog · GlotLID)")
code(r"""
import os, subprocess
os.makedirs("corpus", exist_ok=True)
def run(cmd): print("+", " ".join(cmd)); subprocess.run(cmd, check=False)

# Parallel Bible corpus (verse-aligned)
if not os.path.isdir("corpus/bible-corpus/bibles"):
    run(["git","clone","--depth","1","https://github.com/christos-c/bible-corpus","corpus/bible-corpus"])
# ASJP wordlists (CLDF)
if not os.path.exists("corpus/asjp/cldf/forms.csv"):
    run(["git","clone","--depth","1","https://github.com/lexibank/asjp","corpus/asjp"])
# FLORES-200 (modern register) — direct public tarball (no auth)
if not os.path.isdir("corpus/flores200_dataset/dev"):
    run(["bash","-lc","curl -sL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz "
                      "-o corpus/flores200.tgz && tar -xzf corpus/flores200.tgz -C corpus"])
# Glottolog (for the programmatic gold tree) — large, shallow
if not os.path.isdir("corpus/glottolog"):
    run(["git","clone","--depth","1","https://github.com/glottolog/glottolog","corpus/glottolog"])
# GlotLID fastText model (data-quality gate)
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download("cis-lmu/glotlid","model.bin",cache_dir="corpus/glotlid_cache")
except Exception as e:
    print("GlotLID download skipped:", e)
print("\ndata present:",
      os.path.isdir("corpus/bible-corpus/bibles"),
      os.path.exists("corpus/asjp/cldf/forms.csv"),
      os.path.isdir("corpus/flores200_dataset/dev"),
      os.path.isdir("corpus/glottolog"))
""")

md(r"""
## 3. (optional) BEAST2 — for the Bayesian arm

Colab is Linux, so the BEAST2 Linux build runs with its bundled JRE. This enables `beast_arm.py` to run
the MCMC (otherwise it just emits the NEXUS + XML scaffolding). Skip if you only want the distance arms.
""")
code(r"""
import os, glob, subprocess
if not glob.glob("beast/**/bin/beast", recursive=True) and not os.path.isdir("beast2"):
    subprocess.run(["bash","-lc",
        "curl -sL https://github.com/CompEvol/beast2/releases/download/v2.7.6/BEAST.v2.7.6.Linux.x86.tgz "
        "-o beast2.tgz && tar -xzf beast2.tgz && mv beast beast2 2>/dev/null; ls beast2/bin"], check=False)
beast_bin = next(iter(glob.glob("beast2/bin/beast")), None)
if beast_bin:
    os.environ["PATH"] = os.path.abspath("beast2/bin") + os.pathsep + os.environ["PATH"]
    print("BEAST2 on PATH:", beast_bin)
else:
    print("BEAST2 not set up — beast_arm.py will emit scaffolding only.")
""")

md(r"""
## 4. Run every analysis arm (this is the heavy compute)

Each script prints its results and writes its figure(s) to `figures/`. Times are wall-clock on Colab.
""")
code(r"""
import subprocess, time, sys
ARMS = [
    ("bible_poc.py",            "crux: trigram-JS vs baselines, honest scaling"),
    ("entropy_support_demo.py", "bias-corrected entropy + bootstrap branch support"),
    ("bible_romanize.py",       "cross-script romanization arm + D1 proof"),
    ("robustness.py",           "n-gram sweep + valid verse-block bootstrap"),
    ("controls.py",             "Latinate-orthography + vowelless-abjad controls"),
    ("flores_replicate.py",     "FLORES-200 modern-register replication"),
    ("asjp_tree.py",            "ASJP wordlist cross-check + Mantel"),
    ("cognate_arm.py",          "cognate (LexStat) descent phylogeny"),
    ("bible_ipa.py",            "IPA (epitran) arm — the negative result"),
    ("methods_compare.py",      "alternative distances + vocab-size bias"),
    ("reticulation.py",         "treelikeness (delta) + NeighborNet"),
    ("corpus_check.py",         "GlotLID data-quality gate"),
    ("scale_up.py",             "102-language breadth run"),
    ("beast_arm.py",            "BEAST2 cognate matrix + (optional) MCMC"),
]
results = {}
for script, desc in ARMS:
    print("\n" + "="*88 + f"\n>>> {script}  —  {desc}\n" + "="*88, flush=True)
    t = time.time()
    p = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.returncode != 0:
        print("STDERR (tail):\n", p.stderr[-2000:])
    results[script] = (p.returncode, round(time.time()-t, 1))
    print(f"[{script}] exit={p.returncode} in {results[script][1]}s", flush=True)
print("\n\nSUMMARY:")
for s,(rc,sec) in results.items():
    print(f"  {'OK ' if rc==0 else 'FAIL'} {s:26s} {sec:7.1f}s")
""")

md("## 5. Show the regenerated figures")
code(r"""
import os
from IPython.display import Image, display
for f in sorted(os.listdir("figures")):
    if f.endswith(".png"):
        print(f); display(Image(os.path.join("figures", f)))
""")

md(r"""
## 6. Save the results off Colab

Pick one. **(a)** download a zip of all figures; **(b)** copy to Google Drive; **(c)** commit back to
GitHub (needs a personal-access token — paste it into Colab's *Secrets* (🔑) as `GH_TOKEN`, never inline).
""")
code(r"""
# (a) zip + download
import shutil
from google.colab import files  # type: ignore
shutil.make_archive("figures_bundle", "zip", "figures")
files.download("figures_bundle.zip")
""")
code(r"""
# (b) copy to Google Drive  (uncomment)
# from google.colab import drive; drive.mount("/content/drive")
# import shutil, os
# dst = "/content/drive/MyDrive/ComplexityLab_figures"; os.makedirs(dst, exist_ok=True)
# for f in os.listdir("figures"):
#     if f.endswith(".png"): shutil.copy(os.path.join("figures", f), dst)
# print("copied figures to", dst)
""")
code(r"""
# (c) commit figures back to GitHub  (uncomment; requires a token in Colab Secrets as GH_TOKEN)
# from google.colab import userdata; import subprocess
# tok = userdata.get("GH_TOKEN")
# subprocess.run(["git","config","user.email","lipodipo3@gmail.com"]); subprocess.run(["git","config","user.name","opalsaints"])
# subprocess.run(["git","add","figures"]); subprocess.run(["git","commit","-m","Regenerate figures on Colab"])
# subprocess.run(["bash","-lc",f"git push https://opalsaints:{tok}@github.com/opalsaints/language-family-trees main"])
# print("pushed.")
""")

nb["cells"] = cells
nb["metadata"] = {"language_info": {"name": "python"}, "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
with open("RunAll.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote RunAll.ipynb with", len(cells), "cells")

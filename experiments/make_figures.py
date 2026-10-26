"""Draw figures 1-6 of the write-up from the E1/E2/E3/E5 result files.
Each figure is saved in the results/figures/ folder of its experiment.
(Figures 7-9 are drawn by 4_combined/e6_compare.py.)"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EXP

E1, E2, E3 = (EXP / d / "results" for d in ("1_cleanup_step", "2_new_counter", "3_fr_designs"))
for d in (E1, E2, E3):
    (d / "figures").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})


def series(rows, method, setting, key):
    pts = [(float(r["snr_db"]), float(r[key])) for r in rows
           if r["method"] == method and r["setting"] == setting]
    return zip(*sorted(pts))


e1 = list(csv.DictReader(open(E1 / "e1_baselines_and_refinement.csv")))
e2 = list(csv.DictReader(open(E2 / "e2_counting.csv")))

# 1. Oracle-count detection: FNR vs SNR
fig, ax = plt.subplots(figsize=(6, 3.6))
for m, st in [("Periodogram", "s--"), ("MUSIC", "^--"), ("ESPRIT", "v--"),
              ("DeepFreq (pretrained)", "o-"), ("DeepFreq (fine-tuned)", "d-")]:
    ax.plot(*series(e1, m, "oracle", "fnr_percent"), st, label=m, ms=4)
ax.set(xlabel="SNR (dB)", ylabel="False negative rate (%)", yscale="log",
       title="Frequency detection with the true count given")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(E1 / "figures" / "fig1_fnr_oracle.png")

# 2. Precision: median error vs SNR, showing NLS refinement
fig, ax = plt.subplots(figsize=(6, 3.6))
for m, st in [("MUSIC", "^--"), ("ESPRIT", "v--"), ("DeepFreq (fine-tuned)", "d-"),
              ("DeepFreq (fine-tuned) + interp", "x-"), ("DeepFreq (fine-tuned) + NLS refine", "o-")]:
    ax.plot(*series(e1, m, "oracle", "median_err_bins"), st, label=m, ms=4)
ax.set(xlabel="SNR (dB)", ylabel="Median error (DFT bins, 1/N)", yscale="log",
       title="Frequency precision with the true count given")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(E1 / "figures" / "fig2_precision_refinement.png")

# 3. Counting accuracy by counter
fig, ax = plt.subplots(figsize=(6, 3.6))
for c, st in [("MDL", "^--"), ("FC regression (fine-tuned)", "d-"), ("NLS-BIC", "s:"),
              ("Fusion MLP", "o-")]:
    pts = sorted((float(r["snr_db"]), float(r["count_accuracy_percent"])) for r in e2 if r["counter"] == c)
    ax.plot(*zip(*pts), st, label=c, ms=4)
ax.set(xlabel="SNR (dB)", ylabel="Exact count accuracy (%)", title="Counting the frequencies")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(E2 / "figures" / "fig3_counting.png")

# 4. Full pipeline Chamfer error
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(*series(e1, "MDL + ESPRIT", "full", "chamfer"), "v--", label="MDL + ESPRIT", ms=4)
ax.plot(*series(e1, "MDL + MUSIC", "full", "chamfer"), "^--", label="MDL + MUSIC", ms=4)
ax.plot(*series(e1, "DeepFreq (fine-tuned)", "full", "chamfer"), "d-", label="DeepFreq fine-tuned (current)", ms=4)
pts = sorted((float(r["snr_db"]), float(r["chamfer_peaks_nls"])) for r in e2 if r["counter"] == "Fusion MLP")
ax.plot(*zip(*pts), "o-", label="+ fusion counter + NLS refine (proposed)", ms=4)
ax.set(xlabel="SNR (dB)", ylabel="Chamfer error", yscale="log", title="Complete pipeline (count + locate)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(E1 / "figures" / "fig4_full_pipeline.png")

# 5. Architecture screening curves (equal 4k-step CPU budget)
fig, ax = plt.subplots(figsize=(6, 3.6))
for n in ["deepfreq20x64", "resfreq10x64", "attn2x64", "deepfreq8x32", "dilated4x32"]:
    log = json.load(open(EXP / "3_fr_designs" / "models" / f"{n}_log.json"))
    ax.plot([e["step"] for e in log], [e["fnr_by_snr"]["20"] for e in log], "o-", ms=3, label=n)
ax.axhline(1.25, color="k", ls=":", lw=1, label="pretrained DeepFreq (~156k steps)")
ax.set(xlabel="Training step", ylabel="FNR at 20 dB (%)", title="Early training speed, same budget")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(E3 / "figures" / "fig5_arch_screening.png")

long_path = E3 / "e5_long_runs.json"
if long_path.exists():
    long = json.load(open(long_path))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, r in long.items():
        snr = sorted(r["fnr_percent"], key=float)
        ax.plot([float(s) for s in snr], [r["fnr_percent"][s] for s in snr], "o-", ms=4, label=name)
    ax.set(xlabel="SNR (dB)", ylabel="False negative rate (%)", yscale="log",
           title="Architectures trained 60k steps (Mac GPU)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(E3 / "figures" / "fig6_arch_long.png")
print("figures in", E1, E2, E3)

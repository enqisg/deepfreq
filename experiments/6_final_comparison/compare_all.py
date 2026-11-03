"""Final comparison: original vs fine-tuned vs improved DeepFreq vs classical methods.

Every method runs on the same saved test sets (test_dataset/ for 0-50 dB,
test_dataset_low_snr/ for -5 dB; 1000 signals each, 50 samples, 1-10
frequencies).  None of these signals were used for training.

Methods
  DeepFreq (original)   2019 pretrained FR + FC models (pretrained_models/DeepFreq)
  DeepFreq (fine-tuned) your fine-tuned FR + FC models (checkpoint/finetune_experiment, epoch 310)
  DeepFreq (improved)   E6 system: FR trained further + new counter + clean-up (experiments/4_combined)
  Periodogram, MUSIC, ESPRIT   classical; own count from MDL
  NOMP                  classical (2016); own count from a CFAR stop that is told the true noise level
  CBLasso               the DeepFreq authors' saved results (test_dataset/cblasso_results, 0-50 dB only)
  CRB                   Cramer-Rao bound: the best possible frequency error for an unbiased method

Metrics per SNR
  fnr_true_count_percent   % of true frequencies missed when told the true count (lower is better)
  median_err_true_count    typical frequency error in DFT bins, true count (lower is better)
  count_accuracy_percent   % of signals where the method's own count is exactly right
  chamfer_own_count        DeepFreq paper's overall error with the method's own count (lower is better)
  ms_per_signal            CPU time, whole pipeline (own count), batch of 1000, 4 threads

  python experiments/6_final_comparison/compare_all.py          # run + plot
  python experiments/6_final_comparison/compare_all.py plot     # plot from saved csv
"""

import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "5_symmetry"))
from common import (ROOT, EXP, PRETRAINED, FINETUNED, XGRID, MAX_FREQ, load_test_set, to_complex,
                    load_deepfreq, fnr_percent, chamfer, matched_median, esprit, music_peaks,
                    periodogram_peaks, mdl_counts, interp_peaks, nls_refine)
from e2_counting import run_net
from e6_compare import load_counter, fusion_counts
from e6_improved_finetune import FR_PATH, COUNTER_PATH
from sym_baselines import nomp, crb_per_signal, noise_var
from data import fr
import util

OUT = EXP / "6_final_comparison" / "results"
FIG = OUT / "figures"
SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
METHODS = ["DeepFreq (original)", "DeepFreq (fine-tuned)", "DeepFreq (improved)",
           "Periodogram", "MUSIC", "ESPRIT", "NOMP", "CBLasso"]
STYLE = {"DeepFreq (original)": ("#1f77b4", "o", "-"), "DeepFreq (fine-tuned)": ("#ff7f0e", "s", "-"),
         "DeepFreq (improved)": ("#2ca02c", "D", "-"), "Periodogram": ("#7f7f7f", "v", "--"),
         "MUSIC": ("#9467bd", "^", "--"), "ESPRIT": ("#8c564b", "<", "--"), "NOMP": ("#d62728", ">", "--"),
         "CBLasso": ("#e377c2", "x", ":"), "CRB (best possible)": ("black", None, ":")}


def build_methods():
    cpu = torch.device("cpu")
    orig = load_deepfreq(PRETRAINED)
    tuned = load_deepfreq(FINETUNED)
    fr_new = util.load(str(FR_PATH), "fr", cpu)[0].eval()
    counter = load_counter(COUNTER_PATH)

    def deepfreq(models):
        def run(signals, snr, counts=None):
            rep, raw = run_net(*models, signals)
            cnt = np.clip(np.rint(raw), 1, MAX_FREQ).astype(int) if counts is None else counts
            return fr.find_freq(rep, cnt, XGRID, MAX_FREQ), cnt
        return run

    def improved(signals, snr, counts=None):
        rep, raw = run_net(fr_new, tuned[1], signals)
        cnt = fusion_counts(counter, signals, rep, raw) if counts is None else counts
        return nls_refine(to_complex(signals), interp_peaks(rep, cnt)), cnt

    def classical(locate):
        def run(signals, snr, counts=None):
            x = to_complex(signals)
            cnt = mdl_counts(x) if counts is None else counts
            return locate(x, cnt), cnt
        return run

    def nomp_run(signals, snr, counts=None):
        x = to_complex(signals)
        est = nomp(x, counts, None if counts is not None else snr)
        return est, (np.sum(est >= -0.5, 1) if counts is None else counts)

    return {"DeepFreq (original)": deepfreq(orig), "DeepFreq (fine-tuned)": deepfreq(tuned),
            "DeepFreq (improved)": improved, "Periodogram": classical(periodogram_peaks),
            "MUSIC": classical(music_peaks), "ESPRIT": classical(esprit), "NOMP": nomp_run}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    methods = build_methods()
    cbl = {k: json.load(open(ROOT / "test_dataset/cblasso_results" / k)) for k in ("fnr", "chamfer")}
    clean = to_complex(np.load(ROOT / "test_dataset/infdB.npy"))
    rows = []
    for snr in SNRS:
        signals, targets, counts = load_test_set(snr)
        for name, run in methods.items():
            oracle, _ = run(signals, snr, counts)
            t = time.perf_counter()
            est, cnt = run(signals, snr)
            ms = 1000 * (time.perf_counter() - t) / len(signals)
            rows.append({"snr_db": snr, "method": name,
                         "fnr_true_count_percent": fnr_percent(oracle, targets),
                         "median_err_true_count": matched_median(oracle, targets),
                         "count_accuracy_percent": 100 * np.mean(cnt == counts),
                         "chamfer_own_count": chamfer(est, targets), "ms_per_signal": ms})
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
        key = str(float(snr))
        if key in cbl["fnr"]:
            rows.append({"snr_db": snr, "method": "CBLasso", "fnr_true_count_percent": cbl["fnr"][key],
                         "median_err_true_count": "", "count_accuracy_percent": "",
                         "chamfer_own_count": cbl["chamfer"][key], "ms_per_signal": ""})
        if snr != -5:   # the -5 dB set has different clean signals, so no CRB there
            v = np.concatenate([crb_per_signal(targets[i], clean[i], noise_var(snr))[0]
                                for i in range(len(targets))])
            rows.append({"snr_db": snr, "method": "CRB (best possible)", "fnr_true_count_percent": "",
                         "median_err_true_count": float(0.6745 * np.median(np.sqrt(np.abs(v))) * 50),
                         "count_accuracy_percent": "", "chamfer_own_count": "", "ms_per_signal": ""})
    with open(OUT / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot()


def load_rows():
    with open(OUT / "comparison.csv") as fh:
        return list(csv.DictReader(fh))


def series(rows, method, key):
    r = [x for x in rows if x["method"] == method and x[key] != ""]
    return [int(x["snr_db"]) for x in r], [float(x[key]) for x in r]


def plot():
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    panels = [("fnr_true_count_percent", "Missed frequencies, true count (%)", True, "a"),
              ("median_err_true_count", "Median frequency error, true count (bins)", True, "b"),
              ("count_accuracy_percent", "Correct count (%)", False, "c"),
              ("chamfer_own_count", "Overall error, own count (Chamfer)", True, "d")]
    lines = METHODS + ["CRB (best possible)"]
    for key, label, logy, tag in panels:
        plt.figure(figsize=(6.4, 4.4))
        for m in lines:
            x, y = series(rows, m, key)
            if not x:
                continue
            c, mk, ls = STYLE[m]
            lw = 2.2 if m.startswith("DeepFreq") else 1.3
            plt.plot(x, y, color=c, marker=mk, linestyle=ls, lw=lw, ms=4, label=m)
        if logy:
            plt.yscale("log")
        plt.xlabel("SNR (dB)")
        plt.ylabel(label)
        plt.grid(alpha=0.3, which="both")
        plt.legend(fontsize=7.5, ncol=2)
        plt.tight_layout()
        plt.savefig(FIG / f"fig_{tag}_{key}.png", dpi=150)
        plt.close()

    # one 2x2 overview figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (key, label, logy, tag) in zip(axes.flat, panels):
        for m in lines:
            x, y = series(rows, m, key)
            if x:
                c, mk, ls = STYLE[m]
                ax.plot(x, y, color=c, marker=mk, linestyle=ls, lw=2.2 if m.startswith("DeepFreq") else 1.3,
                        ms=4, label=m)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"({tag}) {label}", fontsize=10)
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3, which="both")
    h, lab = axes[0, 0].get_legend_handles_labels()
    h2, lab2 = axes[0, 1].get_legend_handles_labels()
    for hh, ll in zip(h2, lab2):
        if ll not in lab:
            h.append(hh)
            lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_overview.png", dpi=150)
    plt.close(fig)

    # speed bar chart (20 dB)
    r = [x for x in rows if x["snr_db"] == "20" and x["ms_per_signal"] != ""]
    plt.figure(figsize=(6.4, 3.6))
    plt.barh([x["method"] for x in r], [float(x["ms_per_signal"]) for x in r],
             color=[STYLE[x["method"]][0] for x in r])
    plt.xscale("log")
    plt.xlabel("CPU time per signal (ms, batch of 1000, log scale)")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis="x", which="both")
    plt.tight_layout()
    plt.savefig(FIG / "fig_e_speed.png", dpi=150)
    plt.close()
    plot_fft(rows, panels)


def plot_fft(rows, panels):
    """The three DeepFreq versions against the FFT only.  The 'Periodogram' method
    is the FFT: |FFT|^2 of the signal zero-padded to the same 1000-point grid as
    the FR model, then the highest peaks are picked (count from MDL)."""
    lines = {"DeepFreq (original)": "DeepFreq (original)", "DeepFreq (fine-tuned)": "DeepFreq (fine-tuned)",
             "DeepFreq (improved)": "DeepFreq (improved)", "Periodogram": "FFT (zero-padded to 1000 points)"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (key, label, logy, tag) in zip(axes.flat, panels):
        for m, shown in lines.items():
            x, y = series(rows, m, key)
            c, mk, ls = STYLE[m]
            ax.plot(x, y, color=c, marker=mk, linestyle=ls, lw=2.2, ms=4, label=shown)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"({tag}) {label}", fontsize=10)
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3, which="both")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig_fft_vs_deepfreq.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot() if sys.argv[1:] == ["plot"] else main()

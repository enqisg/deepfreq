"""Experiment 6 (part 2): compare the improved DeepFreq with the original.

Systems (all on the saved test sets, -5 to 50 dB):
  (a) Original fine-tuned: FR + FC regression count + find_freq
  (b) Original FR + E2 fusion counter + peak interpolation + NLS clean-up
  (c) Improved: new FR (e6_improved_finetune.py) + its fusion counter + NLS
  (d) Small FR (8 layers, 32 filters, from E3/E5) + its fusion counter + NLS
Metrics per SNR: FNR with the true count, count accuracy, Chamfer error of the
full pipeline and median frequency error (bins) of the full pipeline.  Also
times (a), (c) and (d) end to end on CPU.

Run e6_improved_finetune.py (stages fr, counter, counter_small) first, then:
  python experiments/4_combined/e6_compare.py
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import (ROOT, EXP, FINETUNED, XGRID, MAX_FREQ, load_test_set, to_complex, chamfer, fnr_percent,
                    matched_median, interp_peaks, nls_refine)
from e2_counting import Counter, features, run_net
from e6_improved_finetune import FR_PATH, COUNTER_PATH, SMALL_COUNTER_PATH, load_small_fr
from data import fr
import util

OUT = EXP / "4_combined" / "results"
FIG = OUT / "figures"
SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 40, 50]
NAMES = {"a": "Original fine-tuned", "b": "Original + counter + clean-up", "c": "Improved",
         "d": "Small FR + counter + clean-up"}


def load_counter(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    net = Counter(len(d["mu"]))
    net.load_state_dict(d["state"])
    return net.eval(), d["mu"], d["sd"]


def fusion_counts(counter, signals, rep, raw):
    net, mu, sd = counter
    with torch.no_grad():
        return net(torch.from_numpy((features(signals, rep, raw) - mu) / sd)).argmax(1).numpy() + 1


def pipeline(system, models, signals, counts=None):
    """Return (estimates, counts used).  counts=None means the system counts itself."""
    fr_model, fc_model, counter = models[system]
    rep, raw = run_net(fr_model, fc_model, signals)
    if system == "a":
        cnt = np.clip(np.rint(raw), 1, MAX_FREQ).astype(int) if counts is None else counts
        return fr.find_freq(rep, cnt, XGRID, MAX_FREQ), cnt
    cnt = fusion_counts(counter, signals, rep, raw) if counts is None else counts
    return nls_refine(to_complex(signals), interp_peaks(rep, cnt)), cnt


def main():
    torch.set_num_threads(4)
    cpu = torch.device("cpu")
    fr_orig = util.load(str(FINETUNED[0]), "fr", cpu)[0].eval()
    fc_orig = util.load(str(FINETUNED[1]), "fc", cpu)[0].eval()
    fr_new = util.load(str(FR_PATH), "fr", cpu)[0].eval()
    models = {"a": (fr_orig, fc_orig, None),
              "b": (fr_orig, fc_orig, load_counter(EXP / "2_new_counter" / "models" / "fusion_counter.pt")),
              "c": (fr_new, fc_orig, load_counter(COUNTER_PATH)),
              "d": (load_small_fr(), fc_orig, load_counter(SMALL_COUNTER_PATH))}

    rows = []
    for snr in SNRS:
        signals, targets, counts = load_test_set(snr)
        for system in "abcd":
            oracle, _ = pipeline(system, models, signals, counts)
            est, cnt = pipeline(system, models, signals)
            rows.append({"snr_db": snr, "system": NAMES[system],
                         "fnr_true_count_percent": fnr_percent(oracle, targets),
                         "count_accuracy_percent": 100 * np.mean(cnt == counts),
                         "chamfer_full": chamfer(est, targets),
                         "median_err_bins_full": matched_median(est, targets)})
            print(rows[-1], flush=True)
    with open(OUT / "e6_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # end-to-end CPU time per signal at batch 1000 (20 dB), best of 3
    signals = load_test_set(20)[0]
    timing = {}
    for system in "acd":
        pipeline(system, models, signals[:100])
        best = np.inf
        for _ in range(3):
            t = time.perf_counter()
            pipeline(system, models, signals)
            best = min(best, time.perf_counter() - t)
        timing[NAMES[system]] = 1000 * best / len(signals)
    print("ms per signal", timing, flush=True)
    with open(OUT / "e6_timing.json", "w") as fh:
        json.dump({"ms_per_signal_cpu_batch1000": timing, "threads": 4}, fh, indent=1)

    plot(rows)


def plot(rows):
    FIG.mkdir(parents=True, exist_ok=True)
    figs = [("fig7_compare_fnr.png", "fnr_true_count_percent", "FNR with true count (%)", True),
            ("fig8_compare_count.png", "count_accuracy_percent", "Count accuracy (%)", False),
            ("fig9_compare_chamfer.png", "chamfer_full", "Chamfer error (full pipeline)", True)]
    for fname, key, label, logy in figs:
        plt.figure(figsize=(6, 4))
        for name in NAMES.values():
            r = [x for x in rows if x["system"] == name]
            plt.plot([x["snr_db"] for x in r], [x[key] for x in r], "o-", label=name)
        if logy:
            plt.yscale("log")
        plt.xlabel("SNR (dB)")
        plt.ylabel(label)
        plt.grid(alpha=0.3, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=150)
        plt.close()


if __name__ == "__main__":
    main()

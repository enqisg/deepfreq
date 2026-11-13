"""FFT on its own: count the FFT peaks instead of using MDL.

The FFT spectrum (zero-padded to 1000 points, as in compare_all.py) is
searched for peaks, and every peak at least `cut` times as tall as the
tallest one is counted as a frequency (at most 10, at least 1).  `cut` is one
number for all noise levels, picked on separate validation signals (new
signals, seed 777, same generator and noise as the test sets) so the test sets
are only used for the final numbers.

Writes results/fft_peak_count.csv and redraws the figures via compare_all.plot().

  python experiments/6_final_comparison/fft_peak_count.py
"""

import csv
import json

import numpy as np
import scipy.signal
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py
sys.path.insert(1, str(Path(__file__).resolve().parent))
from common import XGRID, MAX_FREQ, load_test_set, to_complex, fnr_percent, chamfer, matched_median
from data import fr, noise
from data.data import gen_signal
import compare_all

SNRS = compare_all.SNRS
CUTS = np.geomspace(1e-3, 0.5, 60)


def peak_heights(spec):
    """Sorted (tallest first) peak heights of each spectrum, relative to its tallest peak."""
    out = []
    for s in spec:
        h = scipy.signal.find_peaks(s, height=(None, None))[1]["peak_heights"]
        out.append(np.sort(h)[::-1] / h.max())
    return out


def counts_for(heights, cut):
    return np.array([np.clip(np.sum(h >= cut), 1, MAX_FREQ) for h in heights])


def validation_set():
    np.random.seed(777)
    torch.manual_seed(777)
    s, f, _ = gen_signal(1000, 50, MAX_FREQ, 1.0, "normal", "normal_floor", 0.1, variable_num_freq=True)
    counts = np.sum(f >= -0.5, axis=1)
    sets = []
    for snr in SNRS:
        x = noise.noise_torch(torch.tensor(s), 10 ** (snr / 10), "gaussian").numpy()
        sets.append(peak_heights(fr.periodogram(to_complex(x), XGRID)))
    return sets, counts


def main():
    val, val_counts = validation_set()
    acc = [np.mean([np.mean(counts_for(h, c) == val_counts) for h in val]) for c in CUTS]
    cut = float(CUTS[int(np.argmax(acc))])
    print(f"cut-off picked on validation: {cut:.4f} of the tallest peak "
          f"(mean count accuracy {100 * max(acc):.1f}%)", flush=True)

    rows = []
    for snr in SNRS:
        signals, targets, counts = load_test_set(snr)
        spec = fr.periodogram(to_complex(signals), XGRID)
        cnt = counts_for(peak_heights(spec), cut)
        est = fr.find_freq(spec, cnt, XGRID, MAX_FREQ)
        rows.append({"snr_db": snr, "method": "FFT only (peak count)", "fnr_true_count_percent": "",
                     "median_err_true_count": "",
                     "count_accuracy_percent": 100 * np.mean(cnt == counts),
                     "chamfer_own_count": chamfer(est, targets),
                     "mean_count_minus_true": float(np.mean(cnt - counts))})
        print(rows[-1], flush=True)
    with open(compare_all.OUT / "fft_peak_count.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(compare_all.OUT / "fft_peak_count_cut.json", "w") as fh:
        json.dump({"cut_fraction_of_tallest_peak": cut, "validation_mean_count_accuracy": float(max(acc))}, fh,
                  indent=1)
    compare_all.plot()


if __name__ == "__main__":
    main()

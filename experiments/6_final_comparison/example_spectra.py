"""Two example test signals (20 dB) seen in time and in frequency: the FFT
spectrum (zero-padded to 1000 points) against the original FR model's output,
with the true frequencies and each method's picked peaks (true count given).

Example 1: two frequencies close together (about 1 bin apart).
Example 2: weak frequencies, so a side-lobe ripple of a strong one is picked
instead of one of them.
Both are picked automatically as signals where the FFT misses one frequency
and the original FR model misses none.

  python experiments/6_final_comparison/example_spectra.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py
from common import ROOT, PRETRAINED, XGRID, MAX_FREQ, SIGNAL_DIM, load_test_set, to_complex, load_deepfreq
from data import fr

FIG = Path(__file__).resolve().parent / "results" / "figures"
SNR = 20


def wrap(d):
    return np.abs((d + 0.5) % 1 - 0.5)


def missed(est, tgt):
    return wrap(tgt[:, None] - est[None, :]).min(1) > 1 / (2 * SIGNAL_DIM)


def pick_examples(signals, targets, counts, fft_est, fr_est, clean):
    close, weak = None, None
    for i in np.argsort(-counts):   # prefer a few frequencies, not one
        k = counts[i]
        if not 3 <= k <= 5:
            continue
        t = targets[i][:k]
        m_fft, m_fr = missed(fft_est[i][fft_est[i] >= -0.5], t), missed(fr_est[i][fr_est[i] >= -0.5], t)
        if m_fft.sum() != 1 or m_fr.any():
            continue
        j = int(np.argmax(m_fft))
        gap = np.min(wrap(t[j] - np.delete(t, j))) * SIGNAL_DIM
        a = np.abs(np.linalg.lstsq(np.exp(2j * np.pi * np.outer(np.arange(SIGNAL_DIM), t)), clean[i], rcond=None)[0])
        rel = a[j] / a.max()
        if close is None and gap < 1.3 and rel > 0.5:
            close = (i, j)
        if weak is None and gap > 2 and rel < 0.3:   # weak, not close: hidden by ripples
            weak = (i, j)
        if close and weak:
            break
    return [close, weak]


def main():
    signals, targets, counts = load_test_set(SNR)
    clean = to_complex(np.load(ROOT / "test_dataset/infdB.npy"))
    x = to_complex(signals)
    fft_spec = fr.periodogram(x, XGRID)
    fr_model, _ = load_deepfreq(PRETRAINED)
    with torch.no_grad():
        fr_spec = fr_model(torch.from_numpy(signals).float()).numpy()
    fft_est = fr.find_freq(fft_spec, counts, XGRID, MAX_FREQ)
    fr_est = fr.find_freq(fr_spec, counts, XGRID, MAX_FREQ)
    examples = pick_examples(signals, targets, counts, fft_est, fr_est, clean)
    titles = ["Example 1: two frequencies about 1 bin apart", "Example 2: weak frequencies lose to side-lobe ripples"]

    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 5 * len(examples)),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    for row, ((i, j), title) in enumerate(zip(examples, titles)):
        k = counts[i]
        t = targets[i][:k]
        ax = axes[row, 0]
        n = np.arange(SIGNAL_DIM)
        ax.plot(n, clean[i].real, color="0.6", lw=1.2, label="clean signal (real part)")
        ax.plot(n, x[i].real, "o-", color="#1f77b4", ms=3, lw=1, label=f"noisy signal, {SNR} dB")
        ax.set_xlabel("Sample number (time)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title}\nTime domain: the 50 samples both methods receive", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        f_bins = XGRID * SIGNAL_DIM
        ax.plot(f_bins, fft_spec[i] / fft_spec[i].max(), color="0.45", lw=1.5, ls="--", label="FFT spectrum")
        ax.plot(f_bins, fr_spec[i] / fr_spec[i].max(), color="#1f77b4", lw=1.5, label="FR model output (original)")
        for q, tf in enumerate(t):
            ax.axvline(tf * SIGNAL_DIM, color="#2ca02c", lw=1.2, alpha=0.7,
                       label="true frequency" if q == 0 else None)
        for est, spec, c, m, lab in [(fft_est, fft_spec, "0.3", "v", "FFT picked peaks"),
                                     (fr_est, fr_spec, "#1f77b4", "o", "FR picked peaks")]:
            e = est[i][est[i] >= -0.5]
            idx = np.searchsorted(XGRID, e)
            idx = np.clip(idx, 0, len(XGRID) - 1)
            ax.plot(e * SIGNAL_DIM, spec[i][idx] / spec[i].max(), m, color=c, ms=8, mfc="none", mew=1.8, label=lab)
        ax.annotate("missed by the FFT", xy=(t[j] * SIGNAL_DIM, 1.02), xytext=(t[j] * SIGNAL_DIM, 1.17),
                    ha="center", fontsize=9, color="#d62728",
                    arrowprops={"arrowstyle": "->", "color": "#d62728"})
        lo = min(t) * SIGNAL_DIM - 3
        hi = max(t) * SIGNAL_DIM + 3
        ax.set_xlim(max(lo, -25), min(hi, 25))
        ax.set_ylim(-0.03, 1.28)
        ax.set_xlabel("Frequency (bins; 1 bin = 1/50 of the range)")
        ax.set_ylabel("Height (each scaled to its tallest peak)")
        ax.set_title(f"Frequency domain: FFT vs FR model, {k} true frequencies, true count given", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5)
        ax.grid(alpha=0.3)
        print(f"example {row + 1}: test signal {i}, frequencies (bins) {np.round(t * SIGNAL_DIM, 2)}, "
              f"FFT missed {t[j] * SIGNAL_DIM:.2f}", flush=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_example_spectra.png", dpi=150)


if __name__ == "__main__":
    main()

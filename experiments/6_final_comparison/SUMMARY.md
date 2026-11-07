# Results summary and possible reasons

Test data: the saved test sets (1000 signals per SNR, −5 to 50 dB, 50 samples,
1–10 frequencies), never used for training. Figures: `results/figures/`.
Numbers: `results/comparison.csv`.

The three DeepFreq versions:

- **Original**: 2019 pretrained FR model + FC model.
- **Fine-tuned**: the original trained 10 more epochs on 10,000 new signals (`finetune.ipynb`).
- **Improved**: fine-tuned FR model trained further (100k signals), a new counter
  instead of the FC model's count, and a clean-up step that fits sine waves to the raw signal.

## 1. Finding frequencies (true count given)

**Result.** All DeepFreq versions miss far fewer frequencies than the FFT and
the subspace methods (MUSIC, ESPRIT) at every SNR from 0 dB up. At 20 dB:
DeepFreq about 1–1.2 % missed, FFT 13.3 %, MUSIC 8.0 %, ESPRIT 5.3 %, CBLasso 2.3 %.
Above 25 dB, NOMP misses fewer (0.18 % at 30 dB vs 0.47 % for the improved version).

**Possible reasons.**
- The FFT of a 50-sample signal has peaks about one bin wide. Close
  frequencies merge into one peak and weak ones hide under the side lobes
  of strong ones, so the FFT keeps missing about 13 % even without noise.
  Weak frequencies are the bigger effect: at 50 dB the FFT misses 41 % of
  frequencies under a quarter of the strongest one's strength, 3 % of those at
  half strength or more, and 25 % of those within 1.25 bins of a neighbour
  (11–16 % for wider gaps).
- MUSIC and ESPRIT need an estimate of the signal's correlation matrix. With only
  50 samples that estimate is poor, so they need much less noise to work well.
- The FR model was trained on 200,000 example signals, many with close and weak
  frequencies, so it learned to draw sharp, separate peaks where the FFT cannot.
- NOMP fits sine waves directly to the signal, one at a time, and is told the
  true noise level. With little noise this fitting is very precise, so it
  overtakes DeepFreq at high SNR.

## 2. How exact the frequencies are

**Result.** The original and fine-tuned DeepFreq stop improving at about 0.02
bins, however little noise there is. The improved version keeps getting better
(0.0029 bins at 50 dB, about 7× better) and stays close to the Cramér–Rao bound
(the best possible) up to about 25 dB. The FFT stops at about 0.09 bins.

**Possible reasons.**
- The FR model draws blurred peaks (a Gaussian bump) on a fixed 1000-point grid,
  so reading off the top of a peak has a built-in limit of about 0.02 bins.
- The clean-up step in the improved version fits the frequencies to the raw
  signal instead, so it is not limited by the grid or the blur.
- The FFT's limit is higher because nearby frequencies pull each other's peaks off position.

## 3. Counting the frequencies

**Result (correct count at 20 / 50 dB).** Original 61.6 / 64.7 %, fine-tuned
75.6 / 84.4 %, improved 84.4 / 96.3 %, FFT + MDL 59.6 / 97.0 %, NOMP 86.4 / 76.5 %.

**Possible reasons.**
- The original FC model levels off at about 65 %. It only sees the FR spectrum,
  where two close or one weak frequency can look like one peak.
- Fine-tuning helped the FC model a lot, and the gain is in the FC model's own
  weights: the fine-tuned FC model counts 84.5 % correctly at 30 dB even on the
  original FR model's output (`extra_checks.py`). The spread of its raw count
  guesses fell from 1.09 to 0.41. A possible (untested) reason: the original FC
  training had stalled at a learning rate of about 1.5e-7, and fine-tuning
  restarted it at 1e-4.
- The new counter in the improved version also uses classical counting clues
  taken directly from the raw signal (eigenvalues), which show frequencies that the
  spectrum hides. So it is better at every SNR.
- MDL (used with the FFT) works very well when noise is low but poorly when it is
  high. NOMP's count gets worse at very high SNR, probably because its stopping rule then
  accepts tiny leftover errors as extra frequencies.

## 4. Overall error (count + location, Chamfer)

**Result.** The improved version has the lowest overall error of all methods
from 0 to 40 dB (tied with NOMP at 15–20 dB). Fine-tuned is second among the
DeepFreq versions, the original third. ESPRIT is best only at 45–50 dB. At
−5 dB every method fails (about half the frequencies missed).

**Possible reasons.** The overall error combines both parts, so the improved
version wins because it counts best *and* places frequencies most exactly.
At −5 dB the noise is stronger than many of the frequencies, so no method can
find them.

## 5. Fine-tuned vs original FR model: slightly less exact

**Result.** The fine-tuned FR model's frequency error is 5–10 % larger from
20 dB up (0.022 vs 0.020 bins at 50 dB); a bootstrap over the 1000 signals says
this gap is real. It also misses slightly more above 25 dB (0.74 % vs 0.61 % at
30 dB), but that gap is within chance (95 % range −0.03 to +0.29 points).
Equal from 0 to 20 dB. Checks: `extra_checks.py`.

**Possible reasons.**
- The learning rate stayed high. The original FR training ended at about 7e-8;
  fine-tuning restarted at 1e-4 and ended at 5e-5, about 700× higher, so the
  weights never settled.
- The run was short: about 790 update steps (10 epochs × 79 batches of 128)
  against about 156,000 originally. Training loss fell (0.0117 → 0.0114) while
  validation loss stayed at 0.0124, so it fitted its 10,000 signals slightly
  better without improving in general.
- The noise mix is **not** a reason: fine-tuning used the same noise setting
  (gaussian_blind, snr 1.0) as the original training.
- The loss shows at low noise because there the model's own small errors
  dominate; at high noise the noise error hides them.

In the improved version, the clean-up step takes misses at 30 dB from 0.74 % to
0.64 %, and the extra FR training takes them to 0.47 %.

## 6. Speed

**Result.** CPU time per signal: FFT 0.13 ms, ESPRIT 0.24 ms, MUSIC 0.6 ms,
DeepFreq about 5 ms (improved 5.6 ms), NOMP 0.8–7 ms.

**Possible reasons.** The networks themselves take about 1 ms per signal; most of
DeepFreq's time is the peak-picking step written in plain Python. The FFT is
one fast operation. So on this CPU DeepFreq is not faster than the
classical methods; a GPU, a faster peak picker or TorchScript
(about 2× faster, see `extra_speed_test/`) would close the gap.

## In one paragraph

DeepFreq finds frequencies much better than the FFT and the classical
subspace methods when there is moderate noise (0–25 dB), because the FR model
learned to separate close and weak frequencies that blur together in a
50-sample FFT. Fine-tuning mainly improved counting. The improved version adds a
better counter and a fitting step, so it is the most accurate method overall
from 0 to 40 dB. At very low noise, classical fitting (NOMP, ESPRIT) catches up
or wins on precision, and on this CPU DeepFreq is slower than the FFT.

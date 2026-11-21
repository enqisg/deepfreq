# 6. Final comparison: DeepFreq versions vs classical methods

One script, `compare_all.py`, runs every method on the same test signals and
draws the figures in `results/figures/`. Numbers are in `results/comparison.csv`.

```bash
python experiments/6_final_comparison/compare_all.py        # about 5 minutes on the iMac CPU
python experiments/6_final_comparison/compare_all.py plot   # redraw figures only
```

## Which data, and why

The saved **test sets** in the repository root:

- `test_dataset/`: 1000 signals at each SNR from 0 to 50 dB (steps of 5 dB),
- `test_dataset_low_snr/`: the −5 dB set (only this file is used from that folder).

Each signal has 50 samples and 1 to 10 frequencies. These sets were made once
with a fixed random seed and **never used for training**, so they are a fair
exam for every method, and the numbers are reproducible. They are also the
sets the 2019 DeepFreq paper used, so the CBLasso numbers saved by its authors
(`test_dataset/cblasso_results/`) can be put on the same plots.

## Methods

| Line | What it is |
|---|---|
| DeepFreq (original) | 2019 pretrained FR model + FC model (`pretrained_models/DeepFreq`) |
| DeepFreq (fine-tuned) | your fine-tuned FR + FC models (`checkpoint/finetune_experiment`, epoch 310) |
| DeepFreq (improved) | the best system from `4_combined/`: FR model trained further, new counter instead of the FC model's count, and the clean-up step |
| Periodogram | FFT spectrum, pick the highest peaks |
| MUSIC, ESPRIT | classical subspace methods |
| NOMP | greedy fit-and-subtract method (2016); it is told the true noise level to decide when to stop |
| CBLasso | the DeepFreq authors' saved results (0–50 dB only, no timing) |
| CRB | Cramér–Rao bound: the smallest error any unbiased method can reach |

Periodogram, MUSIC and ESPRIT count the frequencies with MDL, so they share the
same count accuracy.

## Measures

- **(a) Missed frequencies, true count (FNR, %)**: each method is told how many
  frequencies there are; how many true ones did it miss? Lower is better.
- **(b) Median frequency error, true count (bins)**: how close the found
  frequencies are, in units of 1/50. Lower is better; the CRB is the floor.
- **(c) Correct count (%)**: how often the method's own count is exactly right.
- **(d) Overall error, own count (Chamfer)**: the DeepFreq paper's overall
  measure for the full system (count + locations). Lower is better.
- **(e) Speed**: CPU time per signal, batch of 1000, 4 threads.

## Results

![overview](results/figures/fig_overview.png)

Selected numbers (−5 / 0 / 10 / 20 / 30 / 50 dB):

| Method | Missed, true count (%) | Correct count (%) | Overall error (Chamfer) |
|---|---|---|---|
| DeepFreq (original) | 45.8 / 23.1 / 6.16 / 1.25 / 0.61 / 0.51 | 11.8 / 22.3 / 42.6 / 61.6 / 64.1 / 64.7 | 1.05 / 0.203 / 0.066 / 0.033 / 0.026 / 0.026 |
| DeepFreq (fine-tuned) | 46.1 / 23.3 / 6.19 / 1.24 / 0.74 / 0.62 | 11.1 / 22.4 / 49.3 / 75.6 / 84.1 / 84.4 | 1.09 / 0.205 / 0.057 / 0.022 / 0.013 / 0.013 |
| DeepFreq (improved) | 45.0 / 23.4 / 6.18 / 1.00 / 0.47 / 0.47 | 10.9 / 29.7 / 57.0 / 84.4 / 92.6 / 96.3 | 1.08 / 0.182 / 0.052 / 0.015 / 0.0066 / 0.0051 |
| Periodogram | 45.5 / 29.7 / 14.6 / 13.3 / 13.4 / 13.3 | 9.7 / 11.0 / 28.1 / 59.6 / 86.2 / 97.0 | 0.65 / 0.559 / 0.253 / 0.074 / 0.071 / 0.071 |
| MUSIC | 52.4 / 38.4 / 18.0 / 7.97 / 4.23 / 1.17 | same as periodogram | 0.69 / 0.557 / 0.245 / 0.047 / 0.028 / 0.011 |
| ESPRIT | 56.2 / 40.7 / 16.2 / 5.30 / 2.14 / 0.53 | same as periodogram | 0.68 / 0.541 / 0.242 / 0.036 / 0.012 / 0.0034 |
| NOMP | 47.3 / 30.4 / 8.37 / 0.93 / 0.18 / 0.17 | 10.8 / 17.0 / 45.2 / 86.4 / 95.2 / 76.5 | 0.63 / 0.274 / 0.056 / 0.015 / 0.0070 / 0.011 |
| CBLasso | – / 36.8 / 9.89 / 2.31 / 1.14 / 1.05 | – | – / 0.995 / 0.153 / 0.069 / 0.048 / 0.041 |

CPU time per signal: periodogram 0.13 ms, ESPRIT 0.24 ms, MUSIC 0.6 ms,
DeepFreq about 5 ms (original and fine-tuned) and 5.6 ms (improved), NOMP
0.8–7 ms (slower when it finds more frequencies). See `fig_e_speed.png`.

## What it shows

- **Low and medium noise (0–15 dB): all three DeepFreq versions are the best
  at finding frequencies.** They miss far fewer than periodogram, MUSIC, ESPRIT
  and CBLasso, and slightly fewer than NOMP.
- **Fine-tuning mainly improved counting** (75.6 % vs 61.6 % correct at 20 dB),
  which roughly halves the overall error at 20 dB and above. Finding
  frequencies when the count is known is about the same as the original.
- **The improved version is the best DeepFreq at every SNR from 0 dB up**: it
  counts best (84 % at 20 dB, 96 % at 50 dB) and, thanks to the clean-up step,
  its frequencies are about 7× more exact than the original's at 50 dB. Its
  overall error is the lowest of all methods from 0 to 40 dB (tied with NOMP at
  15–20 dB); only ESPRIT is lower at 45–50 dB.
- **At very little noise (30 dB and above), the classical fitting methods catch
  up.** NOMP misses fewer frequencies and ESPRIT/NOMP get closer to the CRB.
  NOMP also needs the true noise level, which DeepFreq does not.
- **At −5 dB nothing works well**; all methods miss about half the frequencies.
  DeepFreq's overall error there is higher than the classical methods'.
- **Speed:** on this CPU the DeepFreq pipelines take about 5 ms per signal, so
  they are not faster than the simple classical methods. Most of that time is
  the peak-picking step in Python; the networks themselves take about 1 ms
  (see `extra_speed_test/`).

## DeepFreq vs the FFT only

![fft](results/figures/fig_fft_vs_deepfreq.png)

The "Periodogram" method above *is* the FFT: the signal is zero-padded to 1000
points (the same grid the FR model uses), the squared FFT magnitude is taken,
and the highest peaks are picked. `fig_fft_vs_deepfreq.png` shows only that
line against the three DeepFreq versions.

- The FFT misses about 13 % of frequencies even with almost no noise. With only
  50 samples its peaks are about 1 bin wide, so two frequencies close together
  merge into one peak, and the side lobes of a strong frequency can hide a weak one.
- Its error stops falling at about 0.09 bins for the same reason: nearby
  frequencies pull each other's peaks off position.
- All DeepFreq versions miss far fewer (about 1 % at 20 dB) because the FR model
  learned to draw sharp peaks even for close frequencies.
- The FFT itself has no way to count; here it uses MDL, which is worse than the
  DeepFreq counters below 25 dB and similar above.

## FFT on its own (no MDL)

`fft_peak_count.py` lets the FFT count its own peaks: every peak at least
0.068 times as tall as the tallest one counts as a frequency (one cut-off for
all noise levels, picked on 1000 separate validation signals, seed 777). It is
the cyan line in `fig_fft_vs_deepfreq.png`; in (a) and (b) it is the same as
the FFT line, since those use the true count.

- Correct count stays at about 34 % from 10 dB up (MDL: 97 % at 50 dB;
  improved DeepFreq: 96 %). Even the best cut-off chosen separately for each
  noise level gets at most 37 % on the validation signals.
- With little noise it counts about 1 too few on average: merged peaks and
  weak frequencies hidden by side lobes never show up as separate peaks.
- At 0 dB and below it counts 3–4 too many: noise bumps pass the cut-off.
- So counting FFT peaks runs into the same resolution and side-lobe limits
  as finding them; MDL avoids that by working on the raw signal.

## Change: the improved version's new counter now uses regression (2026-11-21)

The new counter in the improved version used to pick one of 10 counts
(classification). That switch was never needed, so it now outputs one number,
rounded and limited to 1–10, like the original FC model (`regression_counter.py`,
model in `models/fusion_counter_regression.pt`).

Only these things changed:
- the counter's last layer: 10 outputs → 1 output;
- its training loss: cross-entropy → mean squared error;
- the count: largest of 10 scores → the output rounded and limited to 1–10.

Kept the same: the 41 inputs, the 3 layers (41 → 128 → 128), the training data,
Adam 1e-3, batch 256, 60 epochs and seed. The fine-tuned version, the extra FR
training (improvement 3) and the clean-up step (improvement 1) are unchanged.

Necessary steps and what they affect:
- The counter had to be retrained, because its output layer changed. This is
  the only training done.
- `compare_all.py` was rerun to get the new improved numbers. Only the improved
  rows of `results/comparison.csv` were replaced; every other row was kept from
  the previous run (saved in `results/comparison_classification_counter.csv`).
  Reason: NOMP's code in `5_symmetry/` was changed by later work, so a full rerun
  would also have changed the NOMP numbers.
- Round-1 results (graphs a, b) of the improved version are identical, since
  they don't use the counter. Graphs c and d changed:

| SNR | Correct count, classification → regression (%) | Overall error (Chamfer) |
|---|---|---|
| −5 dB | 10.9 → 11.7 | 1.075 → 1.030 |
| 0 dB | 29.7 → 25.0 | 0.182 → 0.198 |
| 10 dB | 57.0 → 56.4 | 0.0522 → 0.0518 |
| 20 dB | 84.4 → 83.9 | 0.0149 → 0.0149 |
| 30 dB | 92.6 → 93.4 | 0.0066 → 0.0066 |
| 50 dB | 96.3 → 93.7 | 0.0051 → 0.0056 |

The regression counter is about as good overall: slightly worse at 0–5 dB and
45–50 dB, slightly better at 25–35 dB. It still counts better than the
fine-tuned FC model at every SNR from 0 dB up.

# 5. Symmetry-aware FR model and one-network detector (paper work)

Goal: a new idea beyond DeepFreq (2019) that is strong enough for a top
venue. The full plan, baselines and stopping criteria are in the
"DeepFreq paper strategy" document in the project.

## The idea

Frequency estimation has simple symmetries. Multiplying a signal by a pure
tone shifts every frequency by the same amount, so the answer should shift
too; changing the overall phase or loudness changes nothing. DeepFreq's FR
model starts with a general learned matrix and has to learn this from data.
Here it is built in:

- **MultitaperLift** (`sym_models.py`): learned windows (started as Slepian
  tapers) and one FFT, keeping only phase-free products between the windowed
  spectra. Exactly shift-equivariant and phase-invariant (checked: error 4e-7).
- **EquivariantDetector**: that front-end, circular residual convolutions
  along frequency, and a detection head (per-bin probability that a
  frequency is there + sub-bin offset). One network both locates and counts,
  so no separate FC model is needed.
- **Clean-up**: joint NLS refinement of the detected frequencies against the
  raw signal (`sym_baselines._joint_refine`).

## Files

| File | What it does |
|---|---|
| `sym_models.py` | FR models (`FR_MODELS`, DeepFreq's loss, incl. our cResFreq and SwinFreq reimplementations) and detectors (`DETECTORS`, detection loss), incl. the symmetry ablations `lindet` (no symmetry, DeepFreq's front-end), `cpxdet` (phase only, cResFreq's front-end) and `rawdet` (shift only) |
| `sym_train.py` | Trainer. Same training signals and noise as E3, plus a separate validation set (seed 777) so the saved test sets are only used for final numbers |
| `sym_eval_detector.py` | Test-set evaluation of a detector, with and without the clean-up; count threshold picked on validation data |
| `sym_baselines.py` | NOMP (Newtonized OMP, 2016) and the Cramér-Rao bound |
| `sym_eval_classical.py` | NOMP + CRB on every test set → `results/sym_nomp_crb.json` |
| `run_round2.sh` | Round 2 runs (20k steps) and evaluations |
| `hard_tests.py` | Robustness test sets (very close frequencies, 30 dB amplitude range, coloured noise, spikes) |
| `sym_eval_hard.py` | DeepFreq 2019, the improved DeepFreq (E6), NOMP and the detector on the robustness sets |
| `run_round3.sh` | Round 3 runs (60k steps, 3 seeds, count head, data efficiency) and evaluations |
| `sym_eval_fr.py` | FR-type learned baselines (DeepFreq 2019, cResFreq, SwinFreq, ...) with exactly the same count threshold, clean-up and likelihood check as the detector |
| `run_round4.sh` | Round 4 runs: symmetry ablation and the cResFreq / SwinFreq baselines (60k steps), started when round 3's GPU work ends |
| `models/`, `logs/`, `results/` | Checkpoints, training logs, result files |

## Results so far

Round 1, 4k training steps (about 13 min each on the iMac GPU), validation set,
FNR % with the true count / count accuracy %:

| Model | Params | 0 dB | 10 dB | 20 dB | 30 dB |
|---|---|---|---|---|---|
| eqdet8x64 (symmetry-aware) | 0.20M | 23.3 / 22.2 | 6.69 / 44.1 | 1.92 / 65.3 | 1.45 / 67.7 |
| eqdet8x64fixed (tapers not learned) | 0.20M | 23.4 / 21.8 | 6.99 / 44.2 | 2.08 / 64.5 | 1.66 / 66.7 |
| lindet8x64 (DeepFreq-style front-end) | 1.00M | 23.9 / 20.1 | 8.66 / 38.7 | 5.53 / 45.5 | 5.17 / 46.5 |
| eqfr8x64, DeepFreq loss | 0.21M | 46.5 | 44.3 | 44.8 | 44.8 |
| deepfreq20x64, DeepFreq loss | 1.05M | 90.7 | 89.7 | 90.0 | 90.0 |

- With everything else equal, the symmetry-aware front-end cuts misses at
  20–30 dB by about 3.5× and uses 5× fewer parameters.
- Models trained with DeepFreq's spectrum loss are still far from converged
  at 4k steps (DeepFreq's FR model needs about 60k, see `3_fr_designs`), so
  the detection loss is also a large part of the faster training.

NOMP (classical, given the true noise level) is a strong baseline that the
2019 paper did not include: with the true count it misses 0.93% of
frequencies at 20 dB and 0.18% at 30 dB, fewer than the 2019 DeepFreq
weights (1.25%, 0.61%), while DeepFreq is better at 0–10 dB. See
`results/sym_nomp_crb.json`.

## Round 2: 20k training steps

Test sets (1000 signals per SNR). Each cell: FNR % with the true count /
count accuracy % / Chamfer distance (bins, own count). Lower FNR and Chamfer
are better. "+ clean-up" is the joint NLS refinement. NOMP is told the true
noise level; the detector is not.

| Method | Params | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|---|
| eqdet16x64 + clean-up | 0.40M | **22.5** / 27.4 / **0.173** | **5.54** / 53.3 / **0.048** | **0.65** / 86.9 / **0.012** | 0.16 / 95.2 / **0.0046** | 0.16 / 95.5 / **0.0032** |
| eqdet8x64 + clean-up (seed 0) | 0.20M | 22.9 / 26.1 / 0.195 | 5.84 / 50.6 / 0.052 | 0.66 / 86.4 / 0.012 | 0.23 / 92.9 / 0.0054 | 0.19 / 93.0 / 0.0043 |
| lindet8x64 + clean-up (DeepFreq-style front-end) | 1.00M | 22.8 / 25.6 / 0.173 | 7.13 / 49.2 / 0.056 | 2.71 / 68.0 / 0.024 | 2.35 / 68.7 / 0.022 | 2.29 / 69.3 / 0.020 |
| DeepFreq 2019 (FR + FC model) | 1.05M FR + FC | 23.1 / 22.3 / 0.203 | 6.16 / 42.6 / 0.066 | 1.25 / 61.6 / 0.033 | 0.61 / 64.1 / 0.026 | 0.51 / 64.7 / 0.026 |
| Improved DeepFreq (E6) | | 23.4 / **29.7** / 0.182 | 6.18 / **57.0** / 0.052 | 1.00 / 84.4 / 0.015 | 0.47 / 92.6 / 0.0066 | 0.47 / **96.3** / 0.0051 |
| NOMP (true noise level) | – | 30.5 / 17.0 / 0.274 | 8.47 / 45.1 / 0.056 | 0.88 / **87.0** / 0.014 | **0.13** / **98.1** / 0.0054 | **0.10** / 93.6 / 0.0044 |

- The 0.4M-parameter detector has the lowest Chamfer distance of all
  methods at every SNR and the fewest misses from 0 to 20 dB.
- Same backbone and loss, only the front-end changed: the symmetry-aware
  one misses 4× fewer frequencies at 20 dB (0.66 vs 2.71 %) and 10× fewer
  at 30 dB (0.23 vs 2.35 %), with 5× fewer parameters. Two seeds of
  eqdet8x64 agree within about 0.1 points.
- Still behind: counting at 0–10 dB (E6 has a dedicated counter) and NOMP's
  misses and counts at 30 dB and above. Round 3 adds a count head, longer
  training and the likelihood check (`sym_eval_detector.hybrid`) aimed at these.

## Round 3 and 4 plan

- Round 3 (running): the detector with a count head at 60k steps, 3 seeds;
  the same model with DeepFreq's front-end (`lindet16x64c`); data efficiency
  (3k to 100k training signals, both front-ends); robustness sets.
- The likelihood check is now tuned on validation data at every 5 dB from
  0 to 50 dB with a finer SNR gate grid. Before, 15 dB was not in the tuning
  set and the check cost 5 points of count accuracy there (74.8 → 69.4 %);
  with the new tuning it no longer does, and 20–50 dB are unchanged.
- Round 4: a 2×2 symmetry ablation with the same backbone, head and size,
  changing only the front-end: neither symmetry (`lindet16x64c`), phase
  only (`cpxdet16x64c`), shift only (`rawdet16x64c`), both
  (`eqdet16x64c`); and the cResFreq and SwinFreq baselines, trained on the
  same data for 60k steps and scored with the same post-processing.

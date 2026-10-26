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
| `sym_models.py` | FR models (`FR_MODELS`, DeepFreq's loss) and detectors (`DETECTORS`, detection loss), incl. the `lindet` ablation with DeepFreq's learned-matrix front-end |
| `sym_train.py` | Trainer. Same training signals and noise as E3, plus a separate validation set (seed 777) so the saved test sets are only used for final numbers |
| `sym_eval_detector.py` | Test-set evaluation of a detector, with and without the clean-up; count threshold picked on validation data |
| `sym_baselines.py` | NOMP (Newtonized OMP, 2016) and the Cramér-Rao bound |
| `sym_eval_classical.py` | NOMP + CRB on every test set → `results/sym_nomp_crb.json` |
| `run_round2.sh` | Round 2 runs (20k steps) and evaluations |
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

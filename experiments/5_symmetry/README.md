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
| `sym_anm.py` | Atomic norm denoising (batched ADMM) on the test sets → `results/anm.json` |
| `sym_valse.py` | VALSE (variational Bayesian line spectra) on the test and robustness sets → `results/valse.json`, `results/hard_<set>_valse.json`; `start SPEC`: VALSE started from a learned system, and as a gated high-SNR clean-up → `results/valse_from_<tag>.json` |
| `sym_eval_classical.py` | NOMP + CRB on every test set → `results/sym_nomp_crb.json` |
| `run_round2.sh` | Round 2 runs (20k steps) and evaluations |
| `hard_tests.py` | Robustness test sets (very close frequencies, 30 dB amplitude range, coloured noise, spikes) |
| `sym_eval_hard.py` | DeepFreq 2019, the improved DeepFreq (E6), NOMP and the detector on the robustness sets |
| `run_round3.sh` | Round 3 runs (60k steps, 3 seeds, count head, data efficiency) and evaluations |
| `sym_eval_fr.py` | FR-type learned baselines (DeepFreq 2019, cResFreq, SwinFreq, ...) with exactly the same count threshold, clean-up and likelihood check as the detector |
| `sym_eval_hard_fr.py` | An FR baseline with our post-processing on the robustness sets |
| `sym_stats.py` | Per-signal scores on the test sets and paired bootstrap comparisons between two systems (95% intervals, one-sided p) |
| `sym_speed.py` | Run time per signal of every system on one CPU at 20 dB → `results/speed.json` |
| `sym_summary.py` | Tables over seeds (mean ± sd), data efficiency and the symmetry ablation from the `det_*.json` files |
| `run_round4.sh` | Round 4 runs: symmetry ablation and the cResFreq / SwinFreq baselines (60k steps), started when round 3's GPU work ends |
| `run_round5.sh` | Round 5 runs: shift augmentation without built-in symmetry, a colour-trained 60k-step detector and a second seed of the DeepFreq-front-end model, started when round 4's GPU work ends |
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

## Round 3, 4 and 5 plan

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
- Round 5: is data augmentation enough without built-in symmetry? DeepFreq's
  front-end (`lindet16x64c`) trained with a random frequency shift on every
  training signal (`--augment-shift`), on 3k, 10k and 100k signals, and ours
  with the same shifts on 3k, all 10k steps, to compare with round 3's
  data-efficiency runs. Then a 60k-step detector trained with random noise
  colours (`eqdet16x64c_colour60k`, scored on the robustness sets too) and a
  second seed of the 60k DeepFreq-front-end model for the ablation.

## Round 3 so far: 60k steps with a count head (seeds 0 and 1)

`eqdet16x64c`, 60k steps on the iMac GPU (about 10.5 h awake each), scored
in the cloud. Validation data (`validation_wide`, 0-50 dB, mean count
accuracy after the clean-up and check) picks the count head's count over
the threshold count on both seeds (77.2 vs 75.9 % and 76.6 vs 75.8 %), so
the final system takes K from the count head, then the clean-up and the
likelihood check (`sym_stats.py run dethead:...`).

Test sets, per-signal means from `sym_stats.py`: count accuracy % / Chamfer
(bins) / FNR % (true count given). "+ pp": the same count threshold tuning,
clean-up and likelihood check as ours.

| System | -5 dB | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|---|
| ours, seed 0 | 10.8 / 1.621 / 55.5 | 31.3 / 0.192 / 23.2 | 58.4 / 0.052 / 5.65 | 89.5 / 0.012 / 0.51 | 99.6 / 0.003 / 0.08 | 98.6 / 0.001 / 0.05 |
| ours, seed 1 | 10.8 / 1.687 / 54.7 | 33.6 / 0.182 / 23.2 | 59.0 / 0.050 / 5.63 | 88.9 / 0.012 / 0.51 | 99.8 / 0.003 / 0.14 | 98.8 / 0.001 / 0.11 |
| DeepFreq 2019 FR + pp | 10.8 / 1.056 / 47.4 | 27.4 / 0.182 / 24.5 | 54.4 / 0.053 / 6.45 | 87.8 / 0.013 / 1.13 | 97.8 / 0.004 / 0.54 | 96.4 / 0.002 / 0.42 |
| DeepFreq architecture, our 60k run, + pp | 10.7 / 1.292 / 47.7 | 26.4 / 0.183 / 24.5 | 50.7 / 0.057 / 7.29 | 85.6 / 0.014 / 1.61 | 96.3 / 0.004 / 0.95 | 93.0 / 0.004 / 1.02 |

Paired bootstrap (one-sided p < 0.05), both seeds:

- Against DeepFreq 2019 + pp: count better at every SNR from 0 to 50 dB
  (the 20k model without a count head was only level at 0-10 dB); FNR better
  from 0 to 50 dB; Chamfer better from 15 dB (seed 0) or 10 dB (seed 1),
  level at 5 dB, at 0 dB worse (seed 0) or level (seed 1).
- Against NOMP, ANM and VALSE: count and Chamfer better from 0 to 50 dB; FNR
  better from 0 to 50 dB against ANM and VALSE, and against NOMP (told the
  noise level) from 0 to 25 dB, level or better above. At -5 dB ours is
  worse than all three on FNR and Chamfer.
- The count head costs some Chamfer at 0-5 dB: when it says more
  frequencies than the threshold would, the extra ones are often false.
- The seeds agree within about 0.5 points except count at 0 dB (31.3 / 33.6).
- **-5 dB is a weak spot.** It is below the training range (noise up to the
  signal's power, SNR >= 0 dB, as for DeepFreq), and longer training made it
  worse: FNR 55 % against 47 % for DeepFreq (the 20k model: 47.6 %). Every
  system answers K = 1 there, so count accuracy (10.8 %) says nothing.

## Training noise down to -5 dB (fine-tune, both systems)

Both seeds of ours and the DeepFreq architecture (our 60k run, same data)
were fine-tuned in exactly the same way on the cloud CPU (about 43 min each):
3000 steps, learning rate 3e-4, noise std up to 1.78 times the signal's
(`--noise-snr 0.316`, SNR >= -5 dB), starting from the 60k checkpoints
(`--init`). Same table format:

| System | -5 dB | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|---|
| ours, seed 0, fine-tuned | 22.4 / 0.443 / 39.8 | 30.3 / 0.182 / 23.1 | 57.9 / 0.050 / 5.84 | 89.0 / 0.012 / 0.56 | 99.6 / 0.003 / 0.09 | 98.4 / 0.001 / 0.08 |
| ours, seed 1, fine-tuned | 21.3 / 0.440 / 40.0 | 32.0 / 0.184 / 23.1 | 58.2 / 0.051 / 5.67 | 88.7 / 0.012 / 0.52 | 99.5 / 0.003 / 0.09 | 98.6 / 0.001 / 0.09 |
| DeepFreq architecture, fine-tuned, + pp | 17.4 / 0.399 / 42.8 | 26.7 / 0.179 / 25.2 | 50.0 / 0.059 / 7.47 | 85.1 / 0.015 / 1.94 | 94.9 / 0.005 / 1.17 | 91.4 / 0.004 / 1.09 |

- It fixes -5 dB for ours (FNR 55 -> 40 %, count 10.8 -> 22 %, Chamfer
  1.6 -> 0.44) and leaves 0-50 dB about where it was (count within 1.6
  points; FNR at 40-50 dB 0.05 -> 0.08 % on seed 0).
- The DeepFreq architecture gains less at -5 dB and loses at 30-50 dB
  (FNR 0.95 -> 1.17 % at 30 dB; significantly worse count at 30-50 dB).
- Ours against the DeepFreq architecture fine-tuned the same way: count
  and FNR better at **every SNR from -5 to 50 dB** on both seeds; Chamfer
  better from 10 dB, level at 0-5 dB, worse at -5 dB (0.44 against 0.40).
- Against the released DeepFreq 2019 weights: FNR better at every SNR,
  count better at every SNR except 20 dB on seed 1 (level), Chamfer better
  at -5 dB and from 10 dB, level at 0-5 dB.

## Run time (one CPU, 20 dB)

`sym_speed.py` on the cloud machine (4 x86 cores, PyTorch 2.14, no GPU), ms per
signal. Networks with 4 threads, the rest single-threaded.

| Step | ms per signal |
|---|---|
| Our network with count head, batch 1000 (batch 1) | 2.4 (3.9) |
| DeepFreq FR + FC models, batch 1000 (batch 1) | 0.53 (2.7) |
| DeepFreq peak picking | 0.02 |
| Clean-up, joint NLS 100 steps | 5.7 |
| Likelihood check (add/drop, SNR-gated) | 19.0 |
| NOMP (CFAR stop) | 25.2 |
| ANM (ADMM, 1000 iterations, batched) | 416 |
| VALSE with its own count, plus clean-up | 469 |

- Our full system takes about 27 ms per signal, the same as DeepFreq with
  our post-processing (about 25 ms) and NOMP (25 ms), and about 15 to 17
  times less than ANM and VALSE.
- Our network has 2.5x fewer parameters than DeepFreq's FR model (0.42M
  against 1.05M) but needs more compute: 199 MFLOP per signal against 91
  for the FR and FC models together, and it is about 4.5x slower at batch
  1000. The 16 circular ResBlocks on the 250-point grid cost more than
  DeepFreq's upsampling path. Plain DeepFreq (no clean-up) is by far the
  fastest system at 0.55 ms, but it is also the least accurate.
- Post-processing dominates. Without the likelihood check our system takes
  about 8 ms per signal.

## Colour and wide-noise fine-tune (both systems, seed 0)

Same recipe as the wide fine-tune, plus `--augment-colour` (half the
training signals get AR(1) noise of random colour): 3000 steps, lr 3e-4,
`--noise-snr 0.316 --augment-colour --init`, for our 60k seed-0 detector
(`eqdet16x64c_60k_widecol3k_s0`) and DeepFreq's FR architecture trained 60k
steps (`deepfreq20x64_60k_widecol3k_s0`). `sym_train.py`'s coloured noise now
takes the same SNR range as the white noise (default unchanged). Count % /
Chamfer / FNR % from per-signal `sym_stats.py`:

| System | -5 dB | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|---|
| ours, colour + wide | 21.4 / 0.496 / 40.7 | 31.0 / 0.196 / 23.3 | 56.4 / 0.056 / 6.02 | 87.9 / 0.013 / 0.64 | 99.3 / 0.003 / 0.09 | 97.1 / 0.001 / 0.09 |
| DeepFreq architecture, colour + wide, + our post-processing | 15.6 / 0.446 / 44.4 | 23.8 / 0.192 / 25.6 | 47.7 / 0.064 / 7.97 | 84.8 / 0.015 / 1.71 | 94.9 / 0.006 / 1.04 | 92.8 / 0.004 / 1.05 |

Paired bootstrap, ours colour + wide against:
- DeepFreq's architecture with the same fine-tune: count and FNR better at
  every SNR from -5 to 50 dB (p < 0.001 each; +3 to +9 count points),
  Chamfer better from 10 dB, level at 0 to 5 dB, worse at -5 dB.
- NOMP, ANM and VALSE: count, Chamfer and FNR better at -5 dB too (count
  +10.6, +11.2 and +8.6 points), and count and FNR better at almost every
  SNR above (NOMP count level at 20 to 25 dB, FNR level from 25 dB).
- DeepFreq 2019 with our post-processing: FNR better everywhere, count
  better or level everywhere (level at 10, 20 and 50 dB), Chamfer worse at
  0 dB only.
- Our wide-only model: the colour training costs 0.3 to 4 count points at
  15 to 30 dB and at 50 dB, a little Chamfer below 25 dB and up to 0.9
  points of misses (at -5 dB); elsewhere level.

So one fine-tuned model now covers white and coloured noise from -5 dB up,
and against the same fine-tune of DeepFreq's architecture it wins on count
and misses everywhere, on the standard test sets and on all five robustness
sets. Seed 1 is being fine-tuned the same way.

## Robustness sets with the 60k-step models (seed 0)

`sym_eval_hard.py` and `sym_eval_hard_fr.py` on the five harder sets
(`hard_tests.py`), cloud CPU. Count accuracy % / Chamfer (bins), each the
mean over 0, 10, 20, 30, 40 and 50 dB; `python sym_summary.py hard` also
prints FNR. Bold rows: both systems after the same colour and wide-noise
fine-tune (next section); all other networks saw white noise only.

| System | 0.5 bins apart | 0.25 bins apart | 30 dB amplitude range | coloured (AR(1), 0.8) | spikes |
|---|---|---|---|---|---|
| ours, 60k | 59.5 / 0.054 | 48.2 / 0.056 | 74.1 / 0.057 | 58.9 / 0.304 | 38.2 / 0.213 |
| ours, 60k + wide fine-tune | 59.4 / 0.049 | 47.8 / 0.050 | 73.4 / 0.062 | 59.5 / 0.293 | 41.2 / 0.119 |
| **ours, 60k + colour and wide fine-tune** | 58.7 / 0.052 | 47.4 / 0.053 | 72.0 / 0.071 | 73.1 / 0.057 | 40.2 / 0.131 |
| DeepFreq 2019 FR model + our post-processing | 56.0 / 0.053 | 46.4 / 0.054 | 69.5 / 0.062 | 62.5 / 0.205 | 36.6 / 0.131 |
| DeepFreq architecture, 60k + our post-processing | 55.9 / 0.053 | 45.8 / 0.054 | 65.5 / 0.066 | 64.4 / 0.171 | 36.1 / 0.133 |
| DeepFreq architecture, 60k + wide fine-tune + our post-processing | 55.3 / 0.050 | 45.8 / 0.052 | 64.0 / 0.068 | 63.7 / 0.164 | 36.2 / 0.119 |
| **DeepFreq architecture, 60k + colour and wide fine-tune + our post-processing** | 55.2 / 0.052 | 45.1 / 0.053 | 63.5 / 0.072 | 66.7 / 0.114 | 35.1 / 0.131 |
| DeepFreq 2019 (FR + FC models, as released) | 40.5 / 0.072 | 34.4 / 0.070 | 48.8 / 0.081 | 44.5 / 0.235 | 30.7 / 0.144 |
| NOMP (true noise power) | 53.1 / 0.062 | 44.1 / 0.060 | 69.0 / 0.077 | 21.8 / 0.269 | 10.8 / 1.010 |
| VALSE + clean-up | 47.9 / 0.082 | 40.6 / 0.082 | 55.0 / 0.095 | 41.2 / 0.330 | 28.8 / 0.172 |
- Close frequencies and a wide amplitude range: ours is best on count
  everywhere (for example 74.1 against 69.5 % for the DeepFreq FR model with
  the same post-processing at a 30 dB amplitude range) and level or better
  on Chamfer and misses.
- Spikes: ours counts best from 10 to 50 dB (41-46 % against 35-41 %), but
  the 60k model was poor at 0 dB (11 against 22 %). The wide-noise fine-tune
  fixes that: mean count 41.2 % and Chamfer 0.119, level with the best.
- Coloured noise was the clear weak spot of the white-trained models. Ours
  was worse at 10 to 20 dB (count 18.5 and 50.3 % against 25.2 and 66.5 %
  for the DeepFreq FR model, and 37.2 and 72.2 % for DeepFreq's architecture
  trained 60k steps) and better from 40 dB; the wide fine-tune did not help.
- After the same colour and wide-noise fine-tune of both systems, ours is
  better than DeepFreq's architecture on count and misses on all five sets
  and level or better on Chamfer: coloured 73.1 against 66.7 % (Chamfer
  0.057 against 0.114), spikes 40.2 against 35.1 %, amplitude range 72.0
  against 63.5 %. Per SNR on the coloured set: 36.8 against 15.9 % at 0 dB,
  57.4 against 45.9 % at 10 dB, level at 20 to 30 dB, better at 40 to 50 dB.

## Likelihood check: local noise floor

The check adds or keeps a sinusoid only if it lowers a BIC-style score
enough. With the noise level taken from the whole residual, coloured noise
(strong at some frequencies, weak at others) made it add false sinusoids
where the noise is strong: count accuracy at 30 dB on the coloured set was
69.5 % (E6: 90.9 %). Now each add or keep is also judged against the
residual spectrum near its own frequency (median 1 to 3 bins away, divided
by ln 2, never below the global level). Chosen on a coloured validation set
(seed 777), not on the test sets.

| Count accuracy % | 20 dB | 30 dB | 40 dB | 50 dB |
|---|---|---|---|---|
| coloured, global floor | 21.6 | 69.5 | 84.7 | 84.2 |
| coloured, local floor | 53.7 | 87.3 | 92.7 | 90.7 |
| coloured, E6 | 65.3 | 90.9 | 92.9 | 95.2 |
| white test set, global → local | 89.0 → 88.8 | 99.2 → 99.2 | 99.2 → 99.4 | 97.6 → 97.9 |

On white noise it changes count accuracy by at most 0.3 points; on
spiky noise it changes nothing. `LOCAL = True` in `sym_eval_detector.py`.
Moves that relocate one sinusoid (`relocate=True`) help very close
frequencies (super-resolution set at 50 dB: 57 → 64 % count accuracy) but
cost about 5× the time, so they are off by default.

Metric note: `data/loss.fnr` pads missing estimates with −1, which on the
frequency circle equals 0, so a padded slot can "hit" a true frequency near 0.
`sym_stats.py` scores each signal without the padding.

## Fair comparison and significance (20k-step detector, 1 seed)

DeepFreq 2019's FR model, given exactly our post-processing (count by
threshold, clean-up, likelihood check with the local floor, all tuned on
validation data), is far better than the 2019 system with its FC model
(count accuracy at 30 dB 97.8 % vs 64.1 %). So much of the gain over the
2019 paper comes from the post-processing, and the paper says so. Against
that stronger baseline, paired bootstrap over the 1000 test signals per SNR
(10k resamples, `sym_stats.py`), difference ours − theirs with 95 % interval:

| | FNR, points (true count) | Count accuracy, points | Chamfer |
|---|---|---|---|
| vs DeepFreq 2019 FR + same post-processing | lower at 0–50 dB (0 dB −1.2 [−1.7, −0.6], 30 dB −0.41 [−0.58, −0.24]), all p < 0.001; −5 dB level | level at 0–10 dB; higher at 15–50 dB (15 dB +5.7 [+3.1, +8.4], 30 dB +1.4 [+0.4, +2.4]), p ≤ 0.03 | lower at 0–50 dB (p ≤ 0.035); **higher at −5 dB** |
| vs NOMP (true noise level) | lower at 0–20 dB (0 dB −7.6 [−8.4, −6.9]); level at 25–50 dB | higher at 0–50 dB (0–15 dB +8 to +11, 20–50 dB +1 to +4), p < 0.01 | lower at 0–50 dB; higher at −5 dB |

At −5 dB every system with a threshold count puts out close to 10
frequencies (over-counting 87–89 %); ours is no worse on misses or count
accuracy but has a higher Chamfer there. The count head in round 3 is meant
to fix this.

## Robustness with the fair baseline

`sym_eval_hard_fr.py` runs an FR baseline with our post-processing on the
robustness sets. Count accuracy % at 10 / 20 / 30 / 50 dB, ours (20k-step
detector) vs DeepFreq 2019's FR model with the same post-processing:

| Set | Ours | DeepFreq 2019 FR + same post-processing |
|---|---|---|
| frequencies 0.5 bins apart | 44.7 / 64.3 / 82.4 / 63.2 | 42.4 / 61.9 / 76.6 / 61.2 |
| 0.25 bins apart | 35.1 / 53.1 / 66.5 / 48.8 | 37.2 / 49.0 / 61.2 / 51.2 |
| 30 dB amplitude range | 39.7 / 74.6 / 98.8 / 97.9 | 40.2 / 70.3 / 93.3 / 93.8 |
| coloured noise (AR(1), 0.8) | **15.7 / 53.7 / 87.3** / 90.7 | **25.2 / 66.5 / 91.5** / 89.0 |
| spikes | 36.2 / 40.5 / 40.3 / 40.0 | 34.9 / 39.6 / 40.9 / 40.6 |

Ours is ahead on close frequencies (20–40 dB) and the wide amplitude range,
level on spikes (slightly fewer misses and lower Chamfer), and **behind on
coloured noise at 10–30 dB**. On pure noise scaled to signal power, our
detector fires 1.2 times per signal on white noise but 9.1 times on AR(1)
noise, while DeepFreq's FR model fires about 9.5 times on both: ours has
learned to tell white noise from sinusoids, and that cue does not carry over
to strongly coloured noise (AR(1) noise near 0 Hz looks like a narrow band of
tones over 50 samples). Options: train every learned method with random noise
colours, or whiten the input with a residual AR fit before the detector.

## Atomic norm denoising (ANM)

`sym_anm.py`: ANM (Bhaskar, Tang and Recht, 2013) with the true noise level
in its regularisation weight, as NOMP gets. Solved by a batched ADMM written
here (it matches a CVXPY solve to 1e-6 on test signals; 1000 steps up to
30 dB, 2000 above, where it converges more slowly). Frequencies are the dual
polynomial's peaks within delta of 1 (delta = 0.005, picked on validation),
then the same 100-step NLS clean-up as every other method. About 0.4–1.1 s
per signal on one CPU core.

Test sets, count accuracy % / FNR % with the true count / Chamfer:

| | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|
| ANM + clean-up | 14.3 / 29.8 / 0.395 | 40.1 / 8.53 / 0.065 | 85.8 / 1.26 / 0.0127 | 94.2 / 0.65 / 0.0050 | 84.1 / 1.34 / 0.0117 |
| ours (20k steps) | 27.4 / 23.0 / 0.173 | 53.3 / 5.71 / 0.049 | 88.8 / 0.64 / 0.0118 | 99.2 / 0.14 / 0.0031 | 97.9 / 0.12 / 0.0013 |

Paired bootstrap: ours is better on all three metrics at every SNR from 0 to
50 dB (p ≤ 0.015). ANM over-counts above 30 dB (15 % at 50 dB), a known
weakness of reading frequencies off the dual polynomial; an exact SDP solver
might do a little better there.

## VALSE (variational Bayesian line spectra)

`sym_valse.py`: VALSE (Badiu, Hansen and Fleury, IEEE TSP 2017), written
here from the paper (no code was released). Unlike NOMP and ANM it is given
nothing: it estimates the count, the noise level and the amplitude prior
itself, and has no setting to pick on validation data. Frequency posteriors
use the paper's Heuristic 2 (one von Mises each), with the mode found on a
16x grid plus Newton steps. Same output rules as the other methods (at most
10 frequencies, the strongest; the 100-step NLS clean-up; with the true
count, the strongest k, or residual peaks added NOMP-style). 0.06–0.9 s per
signal on one CPU core.

Check against the paper: 5 unit-amplitude frequencies at least 1 bin apart,
40 signals each, the count is right in 40/40 at 30 dB (the paper reports a
success rate close to one from 10 dB up to its highest SNR, 30 dB).

Test sets, count accuracy % / FNR % with the true count / Chamfer:

| | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|
| VALSE + clean-up | 19.7 / 30.5 / 0.357 | 49.9 / 8.24 / 0.067 | 82.4 / 1.65 / 0.0145 | 87.6 / 1.98 / 0.0060 | 50.7 / 4.10 / 0.0112 |
| ANM + clean-up | 14.3 / 29.8 / 0.395 | 40.1 / 8.53 / 0.065 | 85.8 / 1.26 / 0.0127 | 94.2 / 0.65 / 0.0050 | 84.1 / 1.34 / 0.0117 |
| ours (20k steps) | 27.4 / 23.0 / 0.173 | 53.3 / 5.71 / 0.049 | 88.8 / 0.64 / 0.0118 | 99.2 / 0.14 / 0.0031 | 97.9 / 0.12 / 0.0013 |

Paired bootstrap (`compare_eqdet16x64_20k_s0__valse.json`): ours is better
on all three metrics at every SNR from 0 to 50 dB (p ≤ 0.009). At −5 dB the
count and FNR are level and VALSE has the lower Chamfer (it outputs fewer
frequencies than our threshold count there).

VALSE is the best classical method at 0–15 dB without knowing the noise
level, but it over-counts more and more above 30 dB (49 % of signals at
50 dB). Its model is not the cause: started from the true frequencies, it
keeps the right count on 100/100 signals at 50 dB (37/100 from its usual
start). Its greedy start, one component at a time from the residual, splits
close frequencies, and at high SNR the iterations cannot merge them again.

Robustness sets (`hard_<set>_valse.json`), count % / FNR %:

| Set | SNR | VALSE | DeepFreq 2019 FR + same post-processing | ours (20k) |
|---|---|---|---|---|
| 0.5 bins apart | 20 dB | 66.5 / 10.5 | 61.9 / 12.0 | 64.3 / 11.3 |
| 0.5 bins apart | 50 dB | 40.9 / 14.4 | 61.2 / 10.3 | 63.2 / 9.47 |
| 0.25 bins apart | 20 dB | 54.4 / 23.8 | 49.0 / 27.9 | 53.1 / 27.5 |
| 0.25 bins apart | 30 dB | 57.3 / 21.5 | 61.2 / 25.3 | 66.5 / 25.2 |
| 30 dB amplitude range | 30 dB | 79.2 / 2.56 | 93.3 / 1.27 | 98.8 / 0.33 |
| coloured | 20 dB | 23.4 / 4.89 | 66.5 / 1.35 | 53.7 / 2.27 |
| spikes | 30 dB | 32.1 / 16.5 | 40.9 / 14.1 | 40.3 / 12.4 |

VALSE resolves frequencies 0.25 bins apart better than both learned models
when it is told the count (FNR 21–24 % against 25–28 % at 20–30 dB), and is
close to them at 0.5 bins up to 20 dB. It assumes white noise, and coloured noise
hurts it more than either learned model.

## A learned start for VALSE, and VALSE as a high-SNR clean-up

`sym_valse.py start SPEC`: VALSE started from a learned system's final
estimates instead of its greedy start (it still adds components from the
residual and iterates as usual). From our estimates, VALSE's count accuracy
at 50 dB goes from 50.7 % to 98.8 %. It is worse than our own output below
20 dB (0 dB: count 20.1 against 27.4 %; VALSE's own count selection is
weaker there), level at 20–25 dB and better from 30 dB up.

So, as an optional last step, a system's output is replaced by VALSE started
from it when the estimated SNR (from the system's own fit, no noise level
given) is above a gate picked on validation data (mean count accuracy over
0–50 dB): 25 dB for ours, 22.5 dB for DeepFreq 2019's FR model with the same
post-processing, which gets the same step for a fair comparison. It adds
about 0.1 s per signal on the standard test set and up to 0.8 s on the
closest frequencies (one CPU core). Count % / FNR % with the true count /
Chamfer (`valse_from_<tag>.json`):

| Set | SNR | ours | ours + VALSE | DeepFreq 2019 FR + same post-processing | the same + VALSE |
|---|---|---|---|---|---|
| test | 20 dB | 88.8 / 0.64 / 0.0118 | (below the gate) | 87.8 / 1.11 / 0.0128 | 87.8 / 1.12 / 0.0128 |
| test | 30 dB | 99.2 / 0.14 / 0.0031 | 99.7 / 0.02 / 0.0026 | 97.8 / 0.53 / 0.0039 | 99.2 / 0.16 / 0.0030 |
| test | 40 dB | 99.4 / 0.12 / 0.0015 | 99.7 / 0.06 / 0.0012 | 97.8 / 0.44 / 0.0025 | 99.0 / 0.24 / 0.0016 |
| test | 50 dB | 97.9 / 0.12 / 0.0013 | 98.8 / 0.06 / 0.0008 | 96.4 / 0.42 / 0.0023 | 98.2 / 0.17 / 0.0013 |
| 0.5 bins apart | 30 dB | 79.7 / 9.73 / 0.0184 | 80.8 / 6.83 / 0.0150 | 76.6 / 10.4 / 0.0197 | 79.7 / 7.20 / 0.0151 |
| 0.5 bins apart | 50 dB | 63.1 / 9.47 / 0.0259 | 70.7 / 7.13 / 0.0180 | 61.2 / 10.3 / 0.0263 | 68.8 / 7.74 / 0.0187 |
| 0.25 bins apart | 30 dB | 63.4 / 25.2 / 0.0228 | 63.1 / 18.1 / 0.0188 | 61.2 / 25.3 / 0.0246 | 61.9 / 18.8 / 0.0195 |
| 0.25 bins apart | 50 dB | 49.8 / 25.0 / 0.0350 | 61.1 / 18.8 / 0.0235 | 51.2 / 25.1 / 0.0356 | 60.0 / 19.2 / 0.0248 |

Paired bootstrap on the test sets: the step cuts our misses at every SNR from
30 to 50 dB (p ≤ 0.017) and raises our count at 30 and 45–50 dB (p ≤ 0.034); at
25 dB it changes nothing significant. With the step on both, ours still
misses fewer frequencies than DeepFreq 2019's FR model at every SNR from 0 to
50 dB (30 dB: 0.02 against 0.16 %, p ≤ 0.03), counts better from 15 to 45 dB
(p ≤ 0.04) and has lower Chamfer from 0 to 50 dB (p ≤ 0.035). With the step,
both learned systems also beat VALSE on its own at 0.25 bins with the true
count (FNR 18–19 % against 21.5 % at 30 dB), which they did not before.

## Screening: training with random noise colours

`sym_train.py --augment-colour`: half the training signals get complex AR(1)
noise, z[t] = a z[t-1] + sqrt(1 - |a|²) w[t], with |a| uniform in [0, 0.9]
and a random angle (so the noise is strongest at a random frequency), at the
same noise energy as the usual noise. Two 4k-step runs of `eqdet8x64` on the
cloud CPU, identical apart from this flag (`eqdet8x64_cpu_s0`,
`eqdet8x64_colour_s0`). Count accuracy % / FNR % with the true count:

| Set | SNR | white-trained | colour-trained |
|---|---|---|---|
| coloured | 0 dB | 11.9 / 33.0 | 17.5 / 26.3 |
| coloured | 10 dB | 32.1 / 10.7 | 47.8 / 5.86 |
| coloured | 20 dB | 64.2 / 2.40 | 67.2 / 1.85 |
| coloured | 30 dB | 88.7 / 0.92 | 87.7 / 0.74 |
| standard test | 10 dB | 51.9 / 6.41 | 48.8 / 6.40 |
| standard test | 30 dB | 97.1 / 0.82 | 97.5 / 0.88 |
| spikes | 30 dB | 39.0 / 13.5 | 36.8 / 13.8 |
| 30 dB amplitude range | 30 dB | 94.7 / 1.90 | 93.0 / 2.06 |
| 0.5 bins apart | 20 dB | 61.0 / 12.9 | 57.4 / 12.7 |
| 0.5 bins apart | 30 dB | 76.8 / 11.7 | 75.8 / 11.7 |

It fixes most of the coloured-noise loss at 0–20 dB for a small cost
elsewhere. The white-trained 4k-step model is also less hurt by colour (32 %
count accuracy at 10 dB) than the 20k-step `eqdet16x64` (16 %): longer
white-noise training makes the detector more white-specific. With this
flag, noise colour becomes a condition we train for, not an unseen one.

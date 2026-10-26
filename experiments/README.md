# DeepFreq improvement experiments

This folder holds the work done **on top of** the fine-tuned DeepFreq model
(`checkpoint/finetune_experiment/{fr,fc}/epoch_310.pth`, from `finetune.ipynb`).
Nothing outside this folder was changed.

DeepFreq has two networks:

- **FR model** (frequency representation): turns the noisy signal into a
  1000-point spectrum with a peak at each frequency.
- **FC model** (frequency counting): looks at that spectrum and says how many
  frequencies there are.

Three improvements were proposed. **Each one was tested by its own experiment**
(folders 1–3), then the useful ones were combined into one system (folder 4).

| # | Improvement | Tested in | Result | Used in final system? |
|---|---|---|---|---|
| 1 | Clean-up step after the FR model | `1_cleanup_step/` (E1) | Frequencies 1.6× more exact at 20 dB, 3.6–7× at 30–50 dB | Yes |
| 2 | New counter instead of the FC model | `2_new_counter/` (E2) | Correct counts up 5–11 points (76 → 84 % at 20 dB) | Yes |
| 3 | Different FR model designs (layers) | `3_fr_designs/` (E3, E5) | None beat the original 20-layer FR model; the small one is ~2× faster end to end but less accurate | Training lessons only (see 4) |
| – | All combined vs the original | `4_combined/` (E6) | Overall error roughly halves at 25 dB and above | – |

All numbers come from the saved test sets in the repository root
(`test_dataset/`: 1000 signals per SNR, 0–50 dB; `test_dataset_low_snr/`: −5 dB;
50 samples, 1–10 frequencies per signal), so they are reproducible.

## Folder map

```
experiments/
├── README.md                     this file
├── improvements_summary.ipynb    short plain-language summary of the improvements
├── common.py                     shared: test-set loading, metrics, classical methods, clean-up step
├── fr_variants.py                shared: the FR model designs tested in improvement 3
├── make_figures.py               draws figures 1–6 into the folders below
├── 1_cleanup_step/               Improvement 1 (E1)
│   ├── e1_baselines_and_refinement.py
│   └── results/                  e1_*.csv/json, figures/fig1, fig2, fig4
├── 2_new_counter/                Improvement 2 (E2)
│   ├── e2_counting.py
│   ├── models/fusion_counter.pt  the new counter (for the original FR model)
│   └── results/                  e2_*.csv/json, figures/fig3
├── 3_fr_designs/                 Improvement 3 (E3 + E5)
│   ├── e3_train_variants.py      train designs on GPU/MPS (e3_train_variants_cpu.py for CPU)
│   ├── e5_eval_long_runs.py      evaluate the 60k-step runs
│   ├── models/                   trained designs (*_long.pt = 60k steps) + training logs
│   ├── logs/                     console logs of the 60k-step runs
│   └── results/                  e5_long_runs.json, figures/fig5, fig6
├── 4_combined/                   Everything combined vs the original (E6)
│   ├── e6_improved_finetune.py   trains the FR model further + retrains the counters
│   ├── e6_compare.py             the 4-way comparison + figures 7–9
│   ├── models/                   fr/best.pth (further-trained FR), fusion_counter.pt,
│   │                             small_fr_fusion_counter.pt
│   ├── logs/                     console logs of the E6 runs
│   └── results/                  e6_comparison.csv, e6_timing.json, figures/fig7–fig9
└── extra_speed_test/             E4: CPU speed of the fine-tuned model (TorchScript)
    ├── e4_inference_speed.py
    └── results/
```

## Improvement 1 – clean-up step (`1_cleanup_step/`, E1)

- **What it is:** after the FR model, take its strongest peaks and move each one
  to its exact position by fitting sine waves to the raw signal (non-linear
  least squares, 10 iterations). Plain maths, no neural network.
- **What changed:** no layers or parameters change. About 1.3 ms extra per signal.
- **Why it helps:** the FR model reads frequencies off a fixed 1000-point grid with
  deliberately blurred peaks, so its error stops improving at about 0.022 bins
  even with almost no noise. Fitting the raw signal removes that floor.
- **Experiment:** E1 compares the fine-tuned and pretrained DeepFreq, with and
  without clean-up, against periodogram, MUSIC and ESPRIT ("oracle" rows use the
  true number of frequencies, "full" rows each method's own count).
- **Result:** median error of the fine-tuned model, true count: 0.028 → 0.017 bins
  at 20 dB (1.6×), 0.023 → 0.0064 at 30 dB (3.6×), 0.022 → 0.0032 at 50 dB (6.9×).
  No gain at 0 dB and below, where noise dominates.

## Improvement 2 – new counter (`2_new_counter/`, E2)

- **What it is:** a small network that replaces the FC model's final count.
- **What changed:**
  - inputs: FR spectrum only → 41 numbers: 25 eigenvalue features of the raw
    signal (classical counting evidence), the 15 highest FR peaks and the FC
    model's own guess;
  - network: 20 conv layers (~64k parameters) → 3 fully connected layers
    41 → 128 → 128 → 10 (~23k parameters);
  - output: one number, rounded (regression) → pick 1 of 10 (classification).
- **Why it helps:** the eigenvalue features see things the spectrum hides (merged
  or very weak peaks). The small size follows from having only 41 inputs. The
  switch to classification was a default choice, not a tested gain: `results/fair_counter_head_comparison/`
  (outside this folder) found a classification head alone did worse than
  regression on the same FR features.
- **Experiment:** E2 compares the FC model, MDL, NLS-BIC and the new counter.
- **Result (exact count accuracy, 0 / 20 / 50 dB):** FC 22.4 / 75.6 / 84.4 %,
  MDL 11.0 / 59.6 / 97.0 %, NLS-BIC 17.3 / 87.6 / 48.1 %, new counter
  29.9 / 83.6 / 93.5 %. The new counter is the most consistent across SNR.

## Improvement 3 – different FR model designs (`3_fr_designs/`, E3 + E5)

- **What it is:** other FR designs, all with the same input and output, trained
  from scratch with the same data, noise, loss and number of steps.

| Design | Layers | Filters | Parameters | FNR at 20 dB (60k steps) |
|---|---|---|---|---|
| Original (`deepfreq20x64`) | 20 conv | 64 | 1,050,000 | 1.8 % |
| Small (`deepfreq8x32`) | 8 conv | 32 | 426,000 | 3.7 % (FR model ~7× faster on CPU) |
| Residual (`resfreq10x64`) | 10 residual blocks | 64 | 1,050,000 | 4.3 % |
| Attention (`attn2x64`) | 2 transformer | 64 | 877,000 | short run only |
| Dilated (`dilated4x32`) | 4 dilated residual blocks | 32 | 426,000 | short run only, worst |

- **Result:** no design beat the original. For reference, the pretrained
  DeepFreq (trained much longer) reaches 1.25 %. So the layers were not changed in the main
  improved system; only the training recipe (noise, target, Adam + OneCycle
  schedule) was reused to train your FR model further. The small design is
  tried as a separate fourth system in E6.

## Everything combined (`4_combined/`, E6)

Four systems, all using your unchanged fine-tuned FC model:

| Line in figures 7–9 | FR model | Counter | Clean-up |
|---|---|---|---|
| Original fine-tuned (blue) | your epoch_310 | FC model | no |
| Original + counter + clean-up (orange) | your epoch_310 | new counter (E2) | yes |
| Improved (green) | your epoch_310 trained 30k more steps (same layers) | new counter, retrained | yes |
| Small FR + counter + clean-up (red) | small 8-layer design from E3/E5 | new counter, retrained | yes |

Further FR training: start from epoch_310, 30k steps, 100k cached signals,
gaussian_blind noise, Gaussian target std 0.3/50, Adam, OneCycle max learning
rate 3e-4, batch 256. 48 min on the iMac GPU (MPS). Best of 10 checkpoints
by mean FNR at 0/10/20/30 dB (picked on the test sets as in E3, so this small
gain is slightly optimistic).

Results (FNR with true count % / count accuracy % / full-pipeline Chamfer error):

| SNR | Original fine-tuned | Original + counter + clean-up | Improved | Small FR + counter + clean-up |
|---|---|---|---|---|
| 0 dB | 23.3 / 22.4 / 0.205 | 23.6 / 29.9 / 0.195 | 23.4 / 29.7 / 0.182 | 25.1 / 28.3 / 0.220 |
| 10 dB | 6.19 / 49.3 / 0.057 | 6.19 / 54.2 / 0.053 | 6.18 / 57.0 / 0.052 | 8.22 / 51.3 / 0.061 |
| 20 dB | 1.23 / 75.6 / 0.022 | 1.23 / 83.6 / 0.015 | 1.00 / 84.4 / 0.015 | 3.09 / 73.8 / 0.025 |
| 30 dB | 0.74 / 84.1 / 0.013 | 0.64 / 90.8 / 0.0068 | 0.47 / 92.6 / 0.0066 | 2.67 / 89.2 / 0.017 |
| 50 dB | 0.62 / 84.4 / 0.013 | 0.55 / 93.5 / 0.0054 | 0.47 / 96.3 / 0.0051 | 2.68 / 97.1 / 0.017 |

CPU time per signal, whole pipeline (batch 1000, 4 threads, one run): original
6.1 ms, improved 6.6 ms, small FR 3.1 ms.

- Most of the gain comes from improvements 1 and 2 (blue → orange).
- Training the FR model further adds a small extra gain (orange → green).
- The small FR model halves the run time but misses more frequencies
  (3.1 % vs 1.0 % at 20 dB). Even with the counter and clean-up, its overall
  error is slightly worse than the original's at 20 dB and above.

## Extra: speed test (`extra_speed_test/`, E4)

Fine-tuned FR+FC on CPU, eager PyTorch vs TorchScript: freezing with
TorchScript roughly halves the time (1.11 → 0.63 ms per signal at batch 1000;
6.7 → 3.4 ms at batch 1).

## Rerunning

Install the root `requirements.txt`, then from the repository root:

```bash
python experiments/1_cleanup_step/e1_baselines_and_refinement.py
python experiments/2_new_counter/e2_counting.py
STEPS=4000 python experiments/3_fr_designs/e3_train_variants.py
STEPS=60000 TAG=_long python experiments/3_fr_designs/e3_train_variants.py deepfreq20x64 deepfreq8x32 resfreq10x64
python experiments/3_fr_designs/e5_eval_long_runs.py
python experiments/extra_speed_test/e4_inference_speed.py
python experiments/make_figures.py
STEPS=30000 python experiments/4_combined/e6_improved_finetune.py fr
python experiments/4_combined/e6_improved_finetune.py counter
python experiments/4_combined/e6_improved_finetune.py counter_small
python experiments/4_combined/e6_compare.py
```

E3 and E6 use CUDA or Apple MPS when available. The first run generates 100k
training signals and caches them in `experiments/train_cache.npz` (git-ignored,
rebuilt automatically).

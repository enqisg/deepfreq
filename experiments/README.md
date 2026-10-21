# DeepFreq improvement experiments

Experiments that compare DeepFreq with classical frequency estimators and try
to improve its weakest parts (counting, precision, architecture, speed).
Everything is evaluated on the saved test sets in the repository root, so all
numbers are reproducible:

- `test_dataset/` – 1000 signals per SNR (0–50 dB and noiseless), 50 samples, 1–10 frequencies
- `test_dataset_low_snr/` – the same signals at −5 dB

The two DeepFreq models compared throughout are:

- **pretrained** – `pretrained_models/DeepFreq/` (the paper's weights)
- **fine-tuned** – `checkpoint/finetune_experiment/{fr,fc}/epoch_310.pth` (from `finetune.ipynb`)

## Layout

| Path | Contents |
|---|---|
| `common.py` | Shared helpers: test-set loading, DeepFreq loading, metrics (FNR, Chamfer, matched RMSE), classical methods (periodogram, MUSIC, ESPRIT, MDL), peak interpolation and NLS refinement |
| `fr_variants.py` | Candidate FR architectures with DeepFreq's interface (`[B,2,50]` → `[B,1000]`): `deepfreq20x64`, `deepfreq8x32`, `resfreq10x64`, `dilated4x32`, `attn2x64` |
| `e1_…` – `e5_…` | The five experiments (below) |
| `make_figures.py` | Draws `results/figures/fig1…fig6` from the CSV/JSON results |
| `checkpoints/` | Trained models and their training logs (`*_log.json`) |
| `logs/` | Console logs of the 60k-step training runs |
| `results/` | Result tables (`.csv`/`.json`) and `figures/` |

## Running

Install the root `requirements.txt`, then run
from the repository root:

```bash
python experiments/e1_baselines_and_refinement.py
python experiments/e2_counting.py
STEPS=4000 python experiments/e3_train_variants.py            # all variants, or name some
STEPS=60000 TAG=_long python experiments/e3_train_variants.py deepfreq20x64 deepfreq8x32 resfreq10x64
python experiments/e4_inference_speed.py
python experiments/e5_eval_long_runs.py
python experiments/make_figures.py
```

`e3_train_variants.py` uses CUDA or Apple MPS when available;
`e3_train_variants_cpu.py` is the same script for CPU-only machines. The first
E3 run generates 100k training signals and caches them in
`experiments/train_cache.npz` (git-ignored, rebuilt automatically).

## Experiments

**E1 – Classical baselines vs DeepFreq, and test-time refinement**
(`e1_baselines_and_refinement.py` → `results/e1_baselines_and_refinement.csv`, `e1_timing.json`; fig 1, 2, 4).
Periodogram, MUSIC, ESPRIT (with the true count, and with MDL counting) against
pretrained and fine-tuned DeepFreq, each optionally followed by peak
interpolation or a 10-iteration NLS refinement. "Oracle" rows give each method
the true number of frequencies; "full" rows use its own count.

**E2 – Better frequency counting** (`e2_counting.py` → `results/e2_counting.csv`,
`e2_counting_meta.json`, `checkpoints/fusion_counter.pt`; fig 3).
Compares DeepFreq's FC regression counter, MDL, NLS-BIC (fit K = 1…10 sinusoids
from DeepFreq's peaks, choose K by penalised likelihood) and a new **fusion
counter**: a small MLP on Hankel eigenvalues, the top FR peak heights and the
FC output.

**E3 – FR architecture variants trained from scratch on an equal budget**
(`e3_train_variants.py` → `checkpoints/<variant>.pt`, `<variant>_log.json`; fig 5).
Same training data, noise model, loss and step count for every variant.
The short screening runs are the checkpoints without `_long`; the 60k-step
runs (trained on the Mac GPU) are the `*_long.pt` checkpoints.

**E4 – Inference speed on CPU** (`e4_inference_speed.py` → `results/e4_inference_speed.json`).
Fine-tuned FR+FC in eager PyTorch vs frozen TorchScript, with 4 and 1 threads,
at batch 1000 and batch 1, alongside the classical methods and post-processing steps.

**E5 – Evaluation of the 60k-step models** (`e5_eval_long_runs.py` → `results/e5_long_runs.json`; fig 6).
Oracle-count FNR and median error per SNR, plus CPU and MPS timing, for the
`*_long.pt` models and the pretrained DeepFreq FR.

## Headline results

At 20 dB (from the result files):

| Method (full pipeline) | FNR % | Median error (bins) | Count accuracy % |
|---|---|---|---|
| MDL + MUSIC | 11.1 | 0.037 | 59.6 |
| MDL + ESPRIT | 9.8 | 0.031 | 59.6 |
| DeepFreq pretrained | 6.8 | 0.029 | 61.6 |
| DeepFreq fine-tuned | 2.3 | 0.028 | 75.6 |
| DeepFreq fine-tuned + NLS refine | 2.3 | 0.018 | 75.6 |

- **Location (E1):** with the true count, DeepFreq's FNR is ~1.2 % at 20 dB vs
  5.3 % for ESPRIT and 8.0 % for MUSIC. NLS refinement cuts DeepFreq's median
  error by about 1.6× at 20 dB (0.028 → 0.017 bins), rising to 3.6–7× at
  30–50 dB (0.023 → 0.0064 bins at 30 dB), for ~1.3 ms per signal.
- **Counting (E2):** counting is the main source of error. Exact count accuracy
  at 0 / 20 / 50 dB: FC 22.4 / 75.6 / 84.4 %, MDL 11.0 / 59.6 / 97.0 %,
  NLS-BIC 17.3 / 87.6 / 48.1 %, fusion MLP 29.9 / 83.6 / 93.5 %. The fusion
  counter is the most consistent across SNR.
- **Architectures (E5, 60k steps):** FNR at 20 dB is 1.8 % for `deepfreq20x64`,
  3.7 % for `deepfreq8x32` (2.5× fewer parameters, ~7× faster on CPU at batch
  1000) and 4.3 % for `resfreq10x64`; the pretrained DeepFreq (trained much
  longer) reaches 1.25 %.
- **Speed (E4):** freezing with TorchScript roughly halves FR+FC time on CPU
  (1.11 → 0.63 ms per signal at batch 1000; 6.7 → 3.4 ms at batch 1), making
  DeepFreq faster per signal than MUSIC (1.7 ms) and close to ESPRIT (0.44 ms).

## E6 – Improved DeepFreq built from the fine-tuned model

`e6_improved_finetune.py` and `e6_compare.py` combine the three improvements
and compare the result with the original fine-tuned model (which is kept
unchanged in `checkpoint/finetune_experiment/`).

1. **FR model, trained further:** starts from `checkpoint/finetune_experiment/fr/epoch_310.pth`
   and trains 30k more steps with the E3 recipe (100k cached signals,
   gaussian_blind noise, Gaussian target std 0.3/50, Adam, OneCycle with
   max learning rate 3e-4, batch 256). FNR with the true count is checked
   at 0/10/20/30 dB every 10 % of training and the best checkpoint is kept →
   `checkpoint/improved_experiment/fr/best.pth` (same format as the original,
   loads with `util.load`) and `fr/train_log.json`. 48 min on the iMac GPU (MPS).
2. **Counter:** the E2 fusion counter retrained on features from the new FR model
   (FC output still from the unchanged fine-tuned FC model) →
   `checkpoint/improved_experiment/fusion_counter.pt`.
3. **Clean-up:** peak interpolation + NLS refinement (`common.py`).

`e6_compare.py` evaluates three systems on both test sets:
(a) *Original fine-tuned* (FR + FC + `find_freq`), (b) *Original + counter + clean-up*
(original FR + E2 fusion counter + NLS) and (c) *Improved* (new FR + new counter + NLS)
→ `results/e6_comparison.csv`, `results/e6_timing.json`, figures
`fig7_compare_fnr.png`, `fig8_compare_count.png`, `fig9_compare_chamfer.png`.

Rerun from the repository root:

```bash
STEPS=30000 python experiments/e6_improved_finetune.py fr      # ~50 min on MPS
python experiments/e6_improved_finetune.py counter
python experiments/e6_compare.py
```

Results (FNR with true count % / count accuracy % / full-pipeline Chamfer):

| SNR | Original fine-tuned | Original + counter + clean-up | Improved |
|---|---|---|---|
| 0 dB | 23.3 / 22.4 / 0.205 | 23.6 / 29.9 / 0.195 | 23.4 / 29.7 / 0.182 |
| 10 dB | 6.19 / 49.3 / 0.057 | 6.19 / 54.2 / 0.053 | 6.18 / 57.0 / 0.052 |
| 20 dB | 1.23 / 75.6 / 0.022 | 1.23 / 83.6 / 0.015 | 1.00 / 84.4 / 0.015 |
| 30 dB | 0.74 / 84.1 / 0.013 | 0.64 / 90.8 / 0.0068 | 0.47 / 92.6 / 0.0066 |
| 50 dB | 0.62 / 84.4 / 0.013 | 0.55 / 93.5 / 0.0054 | 0.47 / 96.3 / 0.0051 |

Most of the gain over the original comes from the new counter and the NLS
clean-up (Chamfer error roughly halves at 25 dB and above). The extra FR
training adds a smaller gain (FNR 1.23 → 1.00 % at 20 dB, 0.74 → 0.47 % at
30 dB). Note the best FR checkpoint was picked on the 0/10/20/30 dB test sets
(as in E3), so that small FR gain is slightly optimistic. CPU time per signal
(batch 1000, 4 threads): 4.8 ms original vs 5.2 ms improved.

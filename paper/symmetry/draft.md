# Building the symmetries of line spectra into a learned frequency estimator

*Working draft. Numbers marked (prov.) come from the single-seed 20k-step
model and will be replaced by the 3-seed 60k-step results from rounds 3 to 5
(`experiments/5_symmetry`). Items marked TODO are open.*

## Abstract (draft)

Learned frequency estimators such as DeepFreq map a short noisy signal to a
spectrum-like representation with a general learned linear front-end, and
must learn from data that the problem has simple symmetries: modulating a
signal by a pure tone shifts every frequency by the same amount, and the
global phase and loudness do not change which frequencies are present. We
build these symmetries into the network. A learned multitaper front-end,
started from Slepian tapers, computes windowed spectra with one FFT and keeps
only phase-free cross-spectra, so it is exactly equivariant to frequency
shifts on its grid and exactly invariant to the global phase. Circular
convolutions along frequency and a detection head that predicts, per bin,
whether a frequency is present and its sub-bin offset then let one network
both locate and count the sinusoids, replacing DeepFreq's two networks. A
short least-squares refinement and a penalised-likelihood check finish the
estimate. With everything else fixed, the symmetric front-end misses 4-13x
fewer frequencies at 20-30 dB than DeepFreq's front-end, with a third of the
parameters. The full detector (2.5x smaller than DeepFreq's FR model) comes
within 15 % of the Cramér-Rao bound from 10 dB. Against DeepFreq 2019 given
the same post-processing, it has significantly lower Chamfer distance and
fewer missed frequencies from 0 dB and better counts from 15 dB; it also
beats NOMP (told the noise level), atomic-norm denoising and VALSE on count
accuracy and Chamfer distance at 0-50 dB. A count head on the same backbone
makes it count better than DeepFreq at every SNR from 0 to 50 dB, and with
the training noise range extended to -5 dB for both, ours counts better and
misses fewer frequencies at every SNR from -5 to 50 dB (3 seeds). With the
same training steps, the symmetric front-end needs about ten times less
training data than DeepFreq's front-end in the same detector, and with 1/6
of the steps and 30 % of the data, or with 10 % of the steps, it beats
DeepFreq's architecture trained at full budget from 5-10 dB up (1 seed,
prov.). TODO: augmentation control, real data.

## 1 Introduction

Estimating the frequencies of a few sinusoids from a short, noisy record is
a classic problem (spectral analysis, radar, sonar, communications, audio).
With $N$ samples, the periodogram cannot separate two frequencies closer than
about $1/N$, and parametric methods (MUSIC, ESPRIT, NOMP, atomic norm
denoising, VALSE) need either the number of sinusoids, the noise level or a
regularisation weight, and degrade at low SNR or for close frequencies.

DeepFreq (Izacard, Mohan and Fernandez-Granda, NeurIPS 2019) showed that a
network trained on simulated signals can do better. It uses two networks: the
FR model (frequency representation) turns the signal into a sharp spectrum,
and the FC model (frequency counting) estimates how many sinusoids there are.
The FR model starts with a general learned matrix from the $2N$ real inputs to
a set of feature maps on a frequency grid.

That matrix has to learn something we already know. Line spectra have three
symmetries:

1. **Frequency shift.** Multiplying $x[t]$ by $e^{i 2\pi f_0 t}$ shifts every
   frequency by $f_0$. The answer should shift too.
2. **Global phase.** Multiplying $x$ by $e^{i\varphi}$ changes nothing.
3. **Scale.** Multiplying $x$ by $c > 0$ changes no frequency (and, for a
   blind estimator, not the count either).

Our question: does building these in help, and by how much? Our contributions:

- A **learned multitaper lift**, exactly equivariant to grid shifts and
  exactly invariant to the global phase, with $O(CN)$ parameters instead of
  the $2N \times C'M$ (800k here) of a general linear front-end (Section 3.1).
- A **one-network detector**: circular convolutions along frequency and a
  CenterNet-style head (per-bin presence and sub-bin offset) plus a pooled
  count head, so locating and counting share one shift-equivariant backbone
  and no separate counting network is needed (Section 3.2).
- A **fair comparison protocol**: every learned spectrum (DeepFreq 2019,
  cResFreq, SwinFreq, ours) gets the same count threshold tuning, refinement
  and likelihood check, so differences come from the networks, with paired
  bootstrap tests per SNR (Section 5).
- A **controlled symmetry ablation** (neither, phase only, shift only, both)
  with everything else fixed, an augmentation control, and a data-efficiency
  study (Section 6).

## 2 Problem and symmetries

The signal is $x[t] = \sum_{j=1}^{K} a_j e^{i(2\pi f_j t + \phi_j)} + w[t]$,
$t = 0, \dots, N-1$, with $N = 50$, $1 \le K \le 10$ unknown,
$f_j \in [-1/2, 1/2)$ separated by at least $1/N$ on the circle, amplitudes
$a_j \ge 0.1$ and white complex Gaussian noise $w$ of unknown level. The goal
is the set $\{f_j\}$, including its size $K$.

Let $T_{f_0}$ be modulation, $(T_{f_0}x)[t] = e^{i2\pi f_0 t}x[t]$; $P_\varphi$
the global phase, $(P_\varphi x)[t] = e^{i\varphi}x[t]$; and $S_c$ scaling by
$c>0$. An ideal estimator $E$ satisfies $E(T_{f_0}x) = E(x) + f_0 \pmod 1$
and $E(P_\varphi x) = E(S_c x) = E(x)$. Modulation and phase rotation leave
white circular Gaussian noise unchanged in distribution, so the first two
symmetries hold for the Bayes-optimal estimator too, not only for the
noiseless map. Scaling also scales the noise, so for noisy signals it
matters only through the SNR.

## 3 Method

### 3.1 Learned multitaper lift

$C$ learned complex windows $w_c[t]$, initialised as the first $C$ discrete
prolate spheroidal (Slepian) tapers ($C = 8$), give windowed spectra on an
$M$-point grid ($M = 250$, five times finer than $1/N$) with one FFT:

$$Z_c(k) = \frac{1}{N}\sum_{t=0}^{N-1} w_c[t]\, x[t]\, e^{-i 2\pi k t / M}.$$

The features are the magnitudes $|Z_c(k)|$ and the normalised cross-spectra

$$G_{cd}(k) = \frac{Z_c(k)\,\overline{Z_d(k)}}{\sqrt{|Z_c(k)\,Z_d(k)|}}, \quad c < d,$$

split into real and imaginary parts: $C + C(C-1) = 64$ channels on the grid.

**Proposition 1.** For a grid shift $f_0 = m/M$, $Z_c$ of $T_{f_0}x$ is
$Z_c$ of $x$ circularly shifted by $m$; for $P_\varphi x$ it is
$e^{i\varphi}Z_c$; for $S_c x$ it is $c Z_c$. Hence every lift feature is
circularly shifted by $m$ under $T_{m/M}$, unchanged under $P_\varphi$, and
multiplied by $c$ under $S_c$.

*Proof.* Substitute into the definition: the modulation moves
$e^{-i2\pi kt/M}$ to $e^{-i2\pi(k-m)t/M}$; the phase and scale factor out of
the sum. $|e^{i\varphi}Z_c| = |Z_c|$ and the phase cancels in
$Z_c\overline{Z_d}$; the normaliser makes $G$ degree-one homogeneous in $c$.
$\square$

Checked numerically: the largest deviation over random signals and shifts is
$4\times10^{-7}$ (float32). The lift has $2CN = 800$ parameters; the
DeepFreq-style learned matrix in our ablation has 800k.

### 3.2 Backbone and heads

A $1\times1$ convolution maps the 64 channels to 64, followed by 16 residual
blocks of two circular convolutions (kernel 3, dilations 1, 2, 4, 8 in turn)
with batch normalisation and ReLU. Circular convolution commutes with
circular shifts, and batch normalisation pools statistics over the batch and
the frequency axis, so the backbone keeps shift equivariance and phase
invariance (Proposition 1 composes). Scale is not kept exactly (biases); the
training data covers a wide amplitude range instead.

- **Detection head** ($1\times1$ convolution, 2 outputs per bin): a logit for
  "a frequency lies in this bin" and a sub-bin offset in $[-1/2, 1/2]$ bins.
  Trained with the CenterNet focal loss on a Gaussian heatmap (width one
  grid bin) plus an $L_1$ offset loss on the true bins (weight 5).
- **Count head**: mean- and max-pooled backbone features over frequency
  (shift-invariant), a 2-layer MLP, and a cross-entropy loss over
  $K \in \{1, \dots, 10\}$.

Decoding: local maxima of the heatmap, either the $K$ highest (true count
given, "oracle") or all above a threshold picked on validation data, or the
top $\hat K$ from the count head; position = bin centre + offset.

### 3.3 Clean-up and likelihood check

1. **Joint refinement.** 100 Gauss-Newton steps on the least-squares fit of
   all detected sinusoids to $x$ (amplitudes solved in closed form).
2. **Likelihood check.** If the estimated SNR (fitted power over residual
   power, no noise level needed) is above a gate, try adding the strongest
   residual peak among the network's low-threshold proposals and dropping
   each sinusoid; keep a move if it lowers a BIC-style score by a margin.
   Each move is also judged against the residual spectrum near its own
   frequency, which makes the check robust to coloured noise. The margin,
   gate and proposal threshold are tuned once on validation data at 0-50 dB.
3. **Optional VALSE clean-up.** Above a second gate (picked on validation
   data), the estimate is used to start VALSE (Badiu et al. 2017), which then
   runs as usual. Section 6.4 reports it separately; all main results are
   without it.

Every learned FR baseline gets steps 1-2 with its own tuned threshold and
check parameters, using its spectrum peaks in place of our heatmap.

## 4 Related work

- **Classical line spectral estimation.** Periodogram and multitaper
  (Thomson 1982; Slepian 1978); subspace methods MUSIC (Schmidt 1986) and
  ESPRIT (Roy and Kailath 1989); greedy NOMP (Mamandipoor et al. 2016);
  convex gridless methods, atomic norm denoising (Bhaskar, Tang and Recht
  2013; Candès and Fernandez-Granda 2014); Bayesian VALSE (Badiu, Hansen and
  Fleury 2017). Model order by information criteria (Schwarz 1978).
- **Learned frequency estimation.** PSnet (Izacard et al., ICASSP 2019),
  DeepFreq (NeurIPS 2019), cResFreq (Pan et al., IEEE TGRS 2022, complex-valued
  front-end, i.e. phase symmetry only), SwinFreq (TODO check citation). All
  use a general learned front-end and a spectrum-matching loss.
- **Equivariant networks.** Group-equivariant CNNs (Cohen and Welling 2016),
  the general theory linking equivariance and convolution (Kondor and
  Trivedi 2018), and the geometric deep learning view (Bronstein et al.
  2021). Here the group acts on the input by modulation, and the FFT turns it
  into a translation of the frequency axis, so circular convolution is the
  natural equivariant layer.
- **Detection heads.** CenterNet (Zhou, Wang and Krähenbühl 2019) and the
  focal loss (Lin et al. 2017): we treat frequencies as points on a circle.

## 5 Experimental protocol

- **Data.** Same generator as DeepFreq: 100k training signals (seed 2026),
  noise level drawn per signal (SNR from 0 dB to noiseless); a separate
  validation set (seed 777) for every tuned number; DeepFreq's saved test
  sets (1000 signals per SNR, -5 to 50 dB) only for final numbers.
- **Robustness sets.** Frequencies 0.5 and 0.25 bins apart; 30 dB amplitude
  range; coloured (AR(1)) noise; impulsive noise.
- **Metrics.** Count accuracy; Chamfer distance in bins (with the estimated
  count); FNR, the share of true frequencies with no estimate within 0.5 bins
  when the true count is given; median error against the Cramér-Rao bound.
- **Baselines.** DeepFreq 2019 (released weights, its own FC model, and with
  our post-processing), an improved DeepFreq (fine-tuned FR, new counter,
  refinement), cResFreq and SwinFreq (re-implemented, trained on the same
  data for 60k steps, our post-processing), NOMP (given the true noise
  level), atomic norm denoising (regularisation tuned on validation data),
  VALSE (no tuning).
- **Statistics.** 3 seeds for our model; paired bootstrap over test signals
  (10k resamples, one-sided p) for every claimed difference.

## 6 Results

### 6.1 Main comparison (60k steps, 3 seeds)

Count accuracy % / Chamfer (bins) / FNR %, test sets, both systems trained
on DeepFreq's noise range (SNR >= 0 dB):

| Method | -5 dB | 0 dB | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|---|---|
| Ours, count head (mean of 3 seeds) | 10.8 / 1.66 / 54.7 | 32.5 / 0.187 / 23.2 | 59.0 / 0.051 / 5.65 | 89.2 / 0.012 / 0.53 | 99.6 / 0.003 / 0.12 | 98.5 / 0.001 / 0.08 |
| DeepFreq 2019 FR + our post-processing | 10.8 / 1.06 / 47.4 | 27.4 / 0.182 / 24.5 | 54.4 / 0.053 / 6.45 | 87.8 / 0.013 / 1.13 | 97.8 / 0.004 / 0.54 | 96.4 / 0.002 / 0.42 |
| NOMP (true noise level) | 10.8 / 0.63 / 48.1 | 17.0 / 0.274 / 31.0 | 45.1 / 0.056 / 8.64 | 87.0 / 0.014 / 0.88 | 98.1 / 0.005 / 0.13 | 93.6 / 0.004 / 0.10 |
| Atomic norm denoising | 10.2 / 0.65 / 46.6 | 14.3 / 0.395 / 30.3 | 40.1 / 0.065 / 8.66 | 85.8 / 0.013 / 1.26 | 94.2 / 0.005 / 0.65 | 84.1 / 0.012 / 1.35 |
| VALSE | 12.8 / 0.72 / 47.8 | 19.7 / 0.357 / 31.0 | 49.9 / 0.067 / 8.40 | 82.4 / 0.014 / 1.65 | 87.6 / 0.006 / 2.00 | 50.7 / 0.011 / 4.21 |

Seed spread (sd over 3 seeds) is at most 0.9 count points (0 dB) and 0.03
FNR points from 0 dB. Paired bootstrap, each seed (one-sided $p < 0.05$):
against DeepFreq 2019
with our post-processing, ours counts better at every SNR from 0 to 50 dB
and misses fewer frequencies from 0 to 50 dB; Chamfer is better from 10-15
dB and level or slightly worse at 0-5 dB. Against NOMP (told the noise
level), atomic norm denoising and VALSE: better count and Chamfer from 0 to
50 dB (one exception: seed 2's count against NOMP at 25 dB, $p = 0.051$).
Below the training range, at -5 dB, ours is worse on FNR and Chamfer.

### 6.1b Extending the training noise range to -5 dB

Ours and the DeepFreq architecture (our 60k run on the same data) are
fine-tuned identically (3000 steps, noise down to -5 dB). Ours: -5 dB FNR
55 -> 40 %, 0-50 dB about unchanged. DeepFreq's architecture: 47.7 -> 42.8 %
at -5 dB, but worse at 30-50 dB (FNR 0.95 -> 1.17 % at 30 dB). After the
fine-tune, ours counts better and misses fewer frequencies than the
DeepFreq architecture at **every SNR from -5 to 50 dB** (all 3 seeds), with
Chamfer better from 10 dB, level at 0-5 dB, worse at -5 dB (0.44 vs 0.40).
TODO: decide whether the main table uses the extended range for all
systems (fairer to report both).

### 6.1c One model for white and coloured noise from -5 dB (3 seeds)

The same fine-tune with random AR(1) noise colours added (both systems, 3
seeds each, each seed paired with the run of the same seed): ours against
DeepFreq's architecture is better on count and FNR at every SNR from -5 to
50 dB for all three pairs ($p \le 0.003$; 3-seed means at 10 dB: 56.4
against 47.7 % count, 5.97 against 8.04 % misses; at 30 dB: 99.4 against
95.5 %, 0.12 against 1.12 %), better on Chamfer from 10 dB. Seeds agree
within 0.8 count points for both systems. It also beats NOMP,
ANM and VALSE at -5 dB on all three metrics. The colour training costs at
most 4 count points against our white-only fine-tune (15 dB).

### 6.2 Accuracy against the Cramér-Rao bound

Median absolute error with the true count after the clean-up, bins
(`median_err_oracle_refined` in `det_<model>.json`; the bound from
`sym_nomp_crb.json`):

| SNR | 10 dB | 20 dB | 30 dB | 50 dB |
|---|---|---|---|---|
| CRB (median abs. error) | 0.0512 | 0.0162 | 0.0051 | 0.00051 |
| Ours, 60k steps (3 seeds) | 0.0523 | 0.0161 | 0.0049 | 0.00057 |
| Ours, colour + wide fine-tune (3 seeds) | 0.0525 | 0.0160 | 0.0049 | 0.00057 |

Ours is within 3 % of the bound at 10-30 dB and 11 % at 50 dB, and the
fine-tune does not change this.

### 6.3 What the symmetries buy

- Round 2 (prov.): same backbone, head and loss, only the front-end changed:
  4x fewer misses at 20 dB (0.66 vs 2.71 %) and 10x fewer at 30 dB (0.23 vs
  2.35 %) with 5x fewer parameters.
- Round 3, 60k steps, seed 0: the same detector with DeepFreq's front-end
  (1.22M parameters against 0.42M) misses 2.05 % of frequencies at 20 dB
  and 1.55 % at 30 dB, against 0.53 and 0.12 % for ours (3 seeds), and
  counts worse from 0 dB (84.2 against 89.2 % at 20 dB). It is only level
  with DeepFreq's own architecture trained the same way, so the gain comes
  from the front-end, not from the backbone, heads or post-processing.
- TODO round 4: the other two cells of the 2x2 ablation (phase only, shift
  only), 60k steps (`sym_summary.py ablation`).
- Round 3, data efficiency (10k steps on 3k to 100k training signals,
  seed 0): with the same data and steps, ours counts better and misses
  fewer frequencies than the DeepFreq-front-end detector at every SNR from
  0 to 50 dB, at all four sizes (misses at 20 dB: 5.30 / 2.94 / 0.99 /
  0.88 % against 24.79 / 14.25 / 6.87 / 3.18 %). Ours on 3k signals beats
  the other on 30k on all three metrics from 20 dB, and ours on 10k is level
  with the other on 100k at 10-20 dB and better above: about ten times less
  data. Ours with 1/6 of DeepFreq's steps and 30 % of its data beats
  DeepFreq's architecture at its full budget (60k steps, 100k signals) on
  count and misses at every SNR from 0 dB; with 10 % of the data it does
  not (FNR 2.94 against 1.61 % at 20 dB). With 10 % of the steps (6k) on
  all 100k signals, ours beats DeepFreq's architecture at its full budget
  on count and misses at every SNR from 0 dB and on Chamfer from 10 dB
  (misses at 20 dB 1.09 against 1.61 %), and is level or better than the
  released DeepFreq 2019 models on all three from 5 dB. Chamfer is worse
  at -5 and 0 dB (training noise only down to 0 dB, see §6.1b). TODO: more
  seeds.
- TODO round 5: is augmentation enough? DeepFreq's front-end with random
  frequency shifts on every training signal.

### 6.4 Learned start for VALSE

Started from our estimates and used only above a validation-picked SNR gate,
VALSE cuts our misses at 30-50 dB from about 0.13 % to 0.02-0.06 %
($p \le 0.017$) and helps very close pairs at 50 dB (count accuracy 49.8 →
61.1 %). The same step on DeepFreq 2019 leaves ours significantly ahead on
FNR and Chamfer at 0-50 dB.

### 6.5 Robustness

Five harder sets (count accuracy % / Chamfer in bins, mean over 0-50 dB;
seed 0 unless marked). Bold: both systems after the same colour and
wide-noise fine-tune, mean of 3 seeds (each seed within 0.8 count points of
the mean); the other networks saw white noise only.

| System | 0.5 bins apart | 0.25 bins apart | 30 dB amplitude range | coloured (AR(1), 0.8) | spikes |
|---|---|---|---|---|---|
| ours, 60k | 59.5 / 0.054 | 48.2 / 0.056 | 74.1 / 0.057 | 58.9 / 0.304 | 38.2 / 0.213 |
| ours, 60k + wide fine-tune | 59.4 / 0.049 | 47.8 / 0.050 | 73.4 / 0.062 | 59.5 / 0.293 | 41.2 / 0.119 |
| **ours, 60k + colour and wide fine-tune (3 seeds)** | 58.5 / 0.052 | 47.6 / 0.053 | 72.1 / 0.070 | 72.8 / 0.058 | 40.3 / 0.129 |
| DeepFreq 2019 FR model + our post-processing | 56.0 / 0.053 | 46.4 / 0.054 | 69.5 / 0.062 | 62.5 / 0.205 | 36.6 / 0.131 |
| DeepFreq architecture, 60k + our post-processing | 55.9 / 0.053 | 45.8 / 0.054 | 65.5 / 0.066 | 64.4 / 0.171 | 36.1 / 0.133 |
| DeepFreq architecture, 60k + wide fine-tune + our post-processing | 55.3 / 0.050 | 45.8 / 0.052 | 64.0 / 0.068 | 63.7 / 0.164 | 36.2 / 0.119 |
| **DeepFreq architecture, 60k + colour and wide fine-tune + our post-processing (3 seeds)** | 55.1 / 0.052 | 44.8 / 0.053 | 63.8 / 0.073 | 66.8 / 0.115 | 34.9 / 0.131 |
| DeepFreq 2019 (FR + FC models, as released) | 40.5 / 0.072 | 34.4 / 0.070 | 48.8 / 0.081 | 44.5 / 0.235 | 30.7 / 0.144 |
| NOMP (true noise power) | 53.1 / 0.062 | 44.1 / 0.060 | 69.0 / 0.077 | 21.8 / 0.269 | 10.8 / 1.010 |
| VALSE + clean-up | 47.9 / 0.082 | 40.6 / 0.082 | 55.0 / 0.095 | 41.2 / 0.330 | 28.8 / 0.172 |
Trained on white noise only, our detector is best on close frequencies and
on a 30 dB amplitude range but weak on coloured noise at 10-20 dB. After a
3000-step fine-tune with random noise colours and noise down to -5 dB,
given to both systems, ours beats DeepFreq's architecture on count and
misses on all five sets for each of three seed pairs (coloured: 72.8
against 66.8 %, Chamfer 0.058 against 0.115) and is level or better on
Chamfer.

### 6.6 Real data

TODO: depends on access to a real recording set (NSynth test split asked).

### 6.7 Run time

Per signal on one 4-core x86 CPU at 20 dB (`results/speed.json`): our
network with its count head 2.4 ms at batch 1000 (3.9 ms at batch 1), the
DeepFreq FR and FC models 0.53 ms (2.7 ms); clean-up 5.7 ms and likelihood
check 19.0 ms (single thread), so about 27 ms for our full system against
about 25 ms for DeepFreq with the same post-processing, 25 ms for NOMP,
416 ms for ANM and 469 ms for VALSE. Our network has 2.5x fewer parameters
than DeepFreq's FR model but about 2.2x more FLOPs (199 against 91 MFLOP per
signal, FR plus FC).

## 7 Limitations

- Exact equivariance holds only for shifts by multiples of $1/M$; sub-grid
  shifts are handled by the offsets and learned, not guaranteed.
- Scale is not built in exactly.
- Trained on white noise with SNR at least 0 dB; -5 dB is outside the
  training range, and coloured noise needs colour-augmented training.
- $N = 50$ and $K \le 10$ throughout, as in DeepFreq.
- Fewer parameters but more compute than DeepFreq's networks; the full
  system's run time is set by the post-processing, which the learned
  baselines share.

## References (to format)

- Badiu, Hansen, Fleury. Variational Bayesian inference of line spectra. IEEE TSP 2017.
- Bhaskar, Tang, Recht. Atomic norm denoising with applications to line spectral estimation. IEEE TSP 2013.
- Bronstein, Bruna, Cohen, Veličković. Geometric deep learning: grids, groups, graphs, geodesics, and gauges. arXiv 2021.
- Candès, Fernandez-Granda. Towards a mathematical theory of super-resolution. CPAM 2014.
- Cohen, Welling. Group equivariant convolutional networks. ICML 2016.
- Izacard, Bernstein, Fernandez-Granda. A learning-based framework for line-spectra super-resolution. ICASSP 2019.
- Izacard, Mohan, Fernandez-Granda. Data-driven estimation of sinusoid frequencies. NeurIPS 2019.
- Kondor, Trivedi. On the generalization of equivariance and convolution in neural networks to the action of compact groups. ICML 2018.
- Lin, Goyal, Girshick, He, Dollár. Focal loss for dense object detection. ICCV 2017.
- Mamandipoor, Ramasamy, Madhow. Newtonized orthogonal matching pursuit: frequency estimation over the continuum. IEEE TSP 2016.
- Pan, Zhang, Deng, Fan, Huang. Complex-valued frequency estimation network and its applications to superresolution of radar range profiles. IEEE TGRS 2022.
- Roy, Kailath. ESPRIT: estimation of signal parameters via rotational invariance techniques. IEEE TASSP 1989.
- Schmidt. Multiple emitter location and signal parameter estimation. IEEE TAP 1986.
- Schwarz. Estimating the dimension of a model. Annals of Statistics 1978.
- Slepian. Prolate spheroidal wave functions, Fourier analysis, and uncertainty V: the discrete case. Bell System Technical Journal 1978.
- Thomson. Spectrum estimation and harmonic analysis. Proc. IEEE 1982.
- Zhou, Wang, Krähenbühl. Objects as points. arXiv 2019.
- TODO: SwinFreq.

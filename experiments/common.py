"""Shared helpers for the DeepFreq improvement experiments.

All methods are evaluated on the saved test sets (1000 signals, 50 samples,
1-10 frequencies) so every number in the write-up is reproducible.
"""

from pathlib import Path
import sys
import time

import numpy as np
import scipy.signal
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
sys.path.insert(0, str(ROOT))
# one folder per improvement; added to the path so experiments can import each other
for _folder in ("1_cleanup_step", "2_new_counter", "3_fr_designs", "4_combined", "extra_speed_test"):
    sys.path.insert(1, str(EXP / _folder))

from data import fr, loss  # noqa: E402
from data.source_number import mdl_arr  # noqa: E402
import util  # noqa: E402

SIGNAL_DIM = 50
FR_SIZE = 1000
XGRID = np.linspace(-0.5, 0.5, FR_SIZE, endpoint=False)
MAX_FREQ = 10

PRETRAINED = (ROOT / "pretrained_models/DeepFreq/frequency_representation_module.pth",
              ROOT / "pretrained_models/DeepFreq/frequency_counting_module.pth")
FINETUNED = (ROOT / "checkpoint/finetune_experiment/fr/epoch_310.pth",
             ROOT / "checkpoint/finetune_experiment/fc/epoch_310.pth")


def load_test_set(snr_db):
    """Return (signals[N,2,50] float32, targets[N,10], true_counts[N])."""
    if snr_db == -5:
        folder = ROOT / "test_dataset_low_snr"
    else:
        folder = ROOT / "test_dataset"
    name = "infdB.npy" if snr_db == np.inf else f"{float(snr_db)}dB.npy"
    signals = np.load(folder / name)
    targets = np.load(folder / "f.npy")
    counts = np.sum(targets >= -0.5, axis=1)
    return signals, targets, counts


def to_complex(signals):
    return signals[:, 0].astype(np.float64) + 1j * signals[:, 1].astype(np.float64)


def load_deepfreq(paths):
    device = torch.device("cpu")
    fr_model = util.load(str(paths[0]), "fr", device)[0].eval()
    fc_model = util.load(str(paths[1]), "fc", device)[0].eval()
    return fr_model, fc_model


# ---------------------------------------------------------------- metrics

def fnr_percent(estimates, targets):
    return 100 * loss.fnr(estimates.copy(), targets.copy(), SIGNAL_DIM) / len(targets)


def chamfer(estimates, targets):
    return loss.chamfer(estimates.copy(), targets.copy()) / len(targets)


def matched_rmse(estimates, targets):
    """RMSE (in units of 1/N, i.e. DFT bins) between each true frequency and the
    closest estimate, counting only detections within half a bin.  This shows
    precision independently of missed detections."""
    errs = []
    for est, tgt in zip(estimates, targets):
        est = est[est >= -0.5]
        tgt = tgt[tgt >= -0.5]
        if len(est) == 0:
            continue
        d = np.abs(((tgt[:, None] - est[None, :]) + 0.5) % 1 - 0.5).min(axis=1)
        errs.extend(d[d < 1 / (2 * SIGNAL_DIM)])
    errs = np.asarray(errs) * SIGNAL_DIM
    return float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("nan")


def matched_median(estimates, targets):
    """Median distance (in DFT bins) from each true frequency to the closest
    estimate.  Robust to the occasional missed or merged peak, so it shows
    typical precision."""
    errs = []
    for est, tgt in zip(estimates, targets):
        est = est[est >= -0.5]
        tgt = tgt[tgt >= -0.5]
        if len(est):
            errs.extend(np.abs(((tgt[:, None] - est[None, :]) + 0.5) % 1 - 0.5).min(axis=1))
    return float(np.median(errs) * SIGNAL_DIM)


def pad(freq_lists, width=MAX_FREQ):
    out = -np.ones((len(freq_lists), width))
    for i, f in enumerate(freq_lists):
        f = np.sort(np.asarray(f)[:width])
        out[i, :len(f)] = f
    return out


# ------------------------------------------------------- classical methods

def hankel_batch(x, m):
    n = x.shape[1]
    idx = np.arange(m)[:, None] + np.arange(n - m + 1)[None, :]
    return x[:, idx]


def esprit(x, counts, m=25):
    """Standard (LS) ESPRIT on a Hankel matrix; gridless."""
    hank = hankel_batch(x, m)
    out = []
    for h, k in zip(hank, counts):
        k = max(int(k), 1)
        u, _, _ = np.linalg.svd(h, full_matrices=False)
        us = u[:, :k]
        phi = np.linalg.lstsq(us[:-1], us[1:], rcond=None)[0]
        out.append(np.angle(np.linalg.eigvals(phi)) / (2 * np.pi))
    return pad(out)


def music_peaks(x, counts, m=25):
    return fr.find_freq(fr.music(x, XGRID, counts, m), counts, XGRID, MAX_FREQ)


def periodogram_peaks(x, counts):
    return fr.find_freq(fr.periodogram(x, XGRID), counts, XGRID, MAX_FREQ)


def mdl_counts(x):
    return np.clip(mdl_arr(x, 25), 1, MAX_FREQ)


# --------------------------------------------- test-time refinement tricks

def interp_peaks(representation, counts):
    """find_freq plus 3-point parabolic interpolation around each peak, which
    removes the 1/1000 grid quantisation at zero cost."""
    step = XGRID[1] - XGRID[0]
    out = -np.ones((len(counts), MAX_FREQ))
    for n, (rep, k) in enumerate(zip(representation, counts)):
        k = max(int(k), 1)
        peaks, props = scipy.signal.find_peaks(rep, height=(None, None))
        if len(peaks) == 0:
            continue
        k = min(k, len(peaks))
        top = peaks[np.argsort(props["peak_heights"])[-k:]]
        left = rep[(top - 1) % FR_SIZE]
        mid = rep[top]
        right = rep[(top + 1) % FR_SIZE]
        denom = left - 2 * mid + right
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (left - right) / denom, 0.0)
        f = XGRID[top] + np.clip(delta, -0.5, 0.5) * step
        out[n, :k] = np.sort((f + 0.5) % 1 - 0.5)
    return out


def _steering(f, n=SIGNAL_DIM):
    return np.exp(2j * np.pi * np.outer(np.arange(n), f))


def nls_refine(x, estimates, iters=10):
    """Variable-projection Gauss-Newton refinement of the frequencies on the
    raw signal (amplitudes solved by least squares at every step).  This is a
    few small linear solves per signal."""
    n = np.arange(SIGNAL_DIM)
    out = estimates.copy()
    for i in range(len(x)):
        f = estimates[i][estimates[i] >= -0.5].astype(np.float64)
        if len(f) == 0:
            continue
        y = x[i]
        for _ in range(iters):
            a_mat = _steering(f)
            amp = np.linalg.lstsq(a_mat, y, rcond=None)[0]
            r = y - a_mat @ amp
            # Jacobian of the model w.r.t. each frequency (amplitudes fixed)
            jac = (2j * np.pi * n[:, None]) * a_mat * amp[None, :]
            jr = np.concatenate([jac.real, jac.imag])
            rr = np.concatenate([r.real, r.imag])
            step = np.linalg.lstsq(jr, rr, rcond=None)[0]
            step = np.clip(step, -0.25 / SIGNAL_DIM, 0.25 / SIGNAL_DIM)
            f = (f + step + 0.5) % 1 - 0.5
            if np.max(np.abs(step)) < 1e-7:
                break
        out[i, :len(f)] = np.sort(f)
    return out


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t

"""Harder test sets for the paper, generated deterministically on demand
(not stored in git) and cached in experiments/5_symmetry/test_sets/.

  superres05   frequencies at least 0.5 bins apart (training uses 1 bin)
  superres025  at least 0.25 bins apart
  widerange    amplitudes log-uniform over 30 dB (training: |N(0,1)| + 0.1)
  colored      AR(1) noise with coefficient 0.8 instead of white noise
  impulsive    white noise plus 5 large spikes per signal

Each set has 1000 signals of 50 samples, 1-10 frequencies, at SNR 0-50 dB.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py
from common import ROOT, SIGNAL_DIM, MAX_FREQ  # noqa: E402
from data import data  # noqa: E402
from data.noise import noise_torch  # noqa: E402

CACHE = ROOT / "experiments" / "5_symmetry" / "test_sets"
SNRS = [0, 10, 20, 30, 40, 50]
SETS = ["superres05", "superres025", "widerange", "colored", "impulsive"]


def _clean(name, n, rng_seed):
    np.random.seed(rng_seed)
    sep = {"superres05": 0.5, "superres025": 0.25}.get(name, 1.0)
    s, f, nf = data.gen_signal(n, SIGNAL_DIM, MAX_FREQ, sep, distance="normal", amplitude="normal_floor",
                               floor_amplitude=0.1, variable_num_freq=True)
    if name == "widerange":
        # rebuild the signals with log-uniform amplitudes over 30 dB
        t = np.arange(SIGNAL_DIM)
        for i in range(n):
            fi = f[i][f[i] > -1]
            amp = 10 ** (-1.5 * np.random.rand(len(fi)))
            ph = np.random.rand(len(fi)) * 2 * np.pi
            x = (amp * np.exp(1j * ph))[None, :] * np.exp(2j * np.pi * np.outer(t, fi))
            x = x.sum(1)
            x = x / np.sqrt(np.mean(np.abs(x) ** 2) / 2)
            s[i, 0], s[i, 1] = x.real, x.imag
    return s, f


def _noisy(name, s, snr_db, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    snr = 10 ** (snr_db / 10)
    st = torch.from_numpy(s)
    if name == "colored":
        w = torch.randn(st.shape)
        for t in range(1, SIGNAL_DIM):
            w[..., t] = 0.8 * w[..., t - 1] + np.sqrt(1 - 0.64) * w[..., t]
        w = w.reshape(len(s), -1)
        sig = st.reshape(len(s), -1)
        w = w * (torch.norm(sig, dim=1) / torch.norm(w, dim=1) / np.sqrt(snr))[:, None]
        return (sig + w).reshape(st.shape).numpy()
    out = noise_torch(st, snr, "gaussian")
    if name == "impulsive":
        idx = np.argsort(np.random.rand(len(s), SIGNAL_DIM), axis=1)[:, :5]
        spikes = np.zeros_like(s)
        for i in range(len(s)):
            spikes[i, :, idx[i]] = 2.0 * np.random.randn(5, 2)
        out = out + torch.from_numpy(spikes)
    return out.numpy().astype(np.float32)


def load(name, snr_db):
    """(signals[N,2,50] float32, targets[N,10], counts[N]) for a hard test set."""
    folder = CACHE / name
    if not (folder / f"{snr_db}dB.npy").exists():
        folder.mkdir(parents=True, exist_ok=True)
        s, f = _clean(name, 1000, 9000 + SETS.index(name))
        np.save(folder / "f.npy", f)
        np.save(folder / "clean.npy", s)
        for snr in SNRS:
            np.save(folder / f"{snr}dB.npy", _noisy(name, s, snr, 100 + snr))
    f = np.load(folder / "f.npy")
    return np.load(folder / f"{snr_db}dB.npy"), f, np.sum(f >= -0.5, axis=1)


if __name__ == "__main__":
    for name in SETS:
        sig, f, c = load(name, 20)
        print(name, sig.shape, c.mean().round(2), np.abs(sig).max().round(2))

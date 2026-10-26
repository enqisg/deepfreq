"""Extra classical baselines and the Cramer-Rao bound for the paper.

* NOMP (Newtonized orthogonal matching pursuit, Mamandipoor, Ramasamy and
  Madhow, IEEE TSP 2016): greedy detection on an oversampled grid, Newton
  refinement of each new frequency, cyclic refinement of all frequencies and
  a least-squares amplitude update.  It stops either after the true count
  (oracle) or when the best residual correlation falls below a CFAR
  threshold set from the noise level.
* crb_per_signal: the deterministic Cramer-Rao bound on the frequencies of a
  sum of complex sinusoids in white complex Gaussian noise.

On the saved test sets the signals are scaled so the noiseless signal has
mean |x[t]|^2 = 2, so the complex noise variance per sample is
sigma^2 = 2 * 10^(-SNR/10).
"""

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import SIGNAL_DIM, MAX_FREQ, pad

N_IDX = np.arange(SIGNAL_DIM)


def noise_var(snr_db):
    return 2.0 * 10 ** (-snr_db / 10)


def _atom(f):
    return np.exp(2j * np.pi * N_IDX * f)


def _newton(y_r, f, a, steps=1):
    """Newton steps on f for one sinusoid against residual y_r (y_r includes it)."""
    for _ in range(steps):
        x = _atom(f)
        dx = 2j * np.pi * N_IDX * x
        d2x = -(2 * np.pi * N_IDX) ** 2 * x
        r = y_r - a * x
        # S(f) = -||r||^2 ; derivatives wrt f
        der1 = 2 * np.real(np.conj(r) @ (a * dx))
        der2 = 2 * np.real(np.conj(r) @ (a * d2x)) - 2 * np.abs(a) ** 2 * np.real(np.vdot(dx, dx))
        if der2 < 0:
            f_new = f - der1 / der2
            if abs(f_new - f) < 1 / (4 * SIGNAL_DIM):
                f = (f_new + 0.5) % 1 - 0.5
        x = _atom(f)
        a = np.vdot(x, y_r) / SIGNAL_DIM
    return f, a


def _joint_refine(y, f, iters=5):
    """Variable-projection Gauss-Newton on all frequencies at once (as in
    common.nls_refine), so each NOMP step ends at a converged fit."""
    for _ in range(iters):
        a_mat = np.stack([_atom(v) for v in f], 1)
        amp = np.linalg.lstsq(a_mat, y, rcond=None)[0]
        r = y - a_mat @ amp
        jac = (2j * np.pi * N_IDX[:, None]) * a_mat * amp[None, :]
        step = np.linalg.lstsq(np.concatenate([jac.real, jac.imag]), np.concatenate([r.real, r.imag]),
                               rcond=None)[0]
        step = np.clip(step, -0.25 / SIGNAL_DIM, 0.25 / SIGNAL_DIM)
        f = (f + step + 0.5) % 1 - 0.5
        if np.max(np.abs(step)) < 1e-9:
            break
    return f


def nomp_one(y, k_max=MAX_FREQ, k_oracle=None, tau=None, oversample=4, r_s=1, r_c=3, joint_iters=20):
    m = oversample * SIGNAL_DIM
    grid = np.arange(m) / m - 0.5
    fs, amps = [], []
    r = y.copy()
    limit = k_oracle if k_oracle is not None else k_max
    while len(fs) < limit:
        spec = np.fft.fftshift(np.fft.fft(r, m))   # X(f) = sum r[t] e^{-2 pi i f t}
        # correlation with atom e^{+2 pi i f t} is conj-FFT at -f; use direct formula instead
        corr = np.exp(-2j * np.pi * np.outer(grid, N_IDX)) @ r
        j = int(np.argmax(np.abs(corr)))
        if k_oracle is None and np.abs(corr[j]) ** 2 / SIGNAL_DIM < tau:
            break
        f = grid[j]
        a = corr[j] / SIGNAL_DIM
        f, a = _newton(r, f, a, r_s)
        fs.append(f)
        amps.append(a)
        # cyclic refinement
        for _ in range(r_c):
            for i in range(len(fs)):
                others = sum(amps[q] * _atom(fs[q]) for q in range(len(fs)) if q != i) if len(fs) > 1 else 0
                fs[i], amps[i] = _newton(y - others, fs[i], amps[i], r_s)
        fs = list(_joint_refine(y, np.asarray(fs), iters=joint_iters))
        a_mat = np.stack([_atom(f) for f in fs], 1)
        amps = list(np.linalg.lstsq(a_mat, y, rcond=None)[0])
        r = y - a_mat @ np.asarray(amps)
        del spec
    return np.asarray(fs)


def nomp(x, counts=None, snr_db=None, p_fa=0.01, **kw):
    """Batch NOMP.  counts given -> oracle count; else CFAR stop using snr_db."""
    out = []
    for i, y in enumerate(x):
        if counts is not None:
            out.append(nomp_one(y, k_oracle=max(int(counts[i]), 1), **kw))
        else:
            s2 = noise_var(snr_db)
            n = SIGNAL_DIM
            tau = s2 * (np.log(n) - np.log(-np.log(1 - p_fa)))  # NOMP CFAR threshold
            f = nomp_one(y, tau=tau, **kw)
            out.append(f if len(f) else nomp_one(y, k_oracle=1, **kw))
    return pad(out)


def crb_per_signal(f_true, clean, sigma2):
    """CRB (variance, in cycles/sample squared) for each true frequency of one signal.
    Amplitudes are recovered from the noiseless signal by least squares."""
    f = f_true[f_true >= -0.5].astype(np.float64)
    a_mat = np.stack([_atom(v) for v in f], 1)
    amp = np.linalg.lstsq(a_mat, clean, rcond=None)[0]
    d = (2j * np.pi * N_IDX)[:, None] * a_mat
    proj = np.eye(SIGNAL_DIM) - a_mat @ np.linalg.pinv(a_mat)
    fim = (2 / sigma2) * np.real((d.conj().T @ proj @ d) * np.outer(amp.conj(), amp).T)
    return np.diag(np.linalg.pinv(fim)), f

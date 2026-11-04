"""Atomic norm denoising (ANM; Bhaskar, Tang and Recht, 2013), a gridless
convex baseline, solved by ADMM for a batch of signals at once.

    minimise  1/2 ||y - x||^2 + tau/2 (u_0 + t)
    s.t.      [[T(u), x], [x^H, t]] >= 0           (T(u) Hermitian Toeplitz)

tau = sigma (1 + 1/log N) sqrt(N log N + N log(4 pi log N)) uses the true
noise level, as NOMP does.  Frequencies are the peaks of the dual polynomial
|<a(f), (y - x)/tau>| that reach 1 - delta (delta picked on validation data);
with the true count, the k highest peaks.  "+ clean-up" refines them with the
same joint NLS as every other method.

Usage:
    python experiments/5_symmetry/sym_anm.py            # test sets -> results/anm.json
"""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # many small matrices: threads only slow it down
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SIGNAL_DIM, MAX_FREQ, load_test_set, to_complex, pad, fnr_percent, chamfer  # noqa: E402
from sym_baselines import noise_var, _joint_refine  # noqa: E402

N = SIGNAL_DIM
GRID = 4096
_F = np.arange(GRID) / GRID - 0.5
_A = np.exp(2j * np.pi * np.outer(np.arange(N), _F))           # N x GRID, unnormalised atoms
_I, _J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
_LAG = _J - _I                                                  # T[i, j] = u[j - i] (conj below the diagonal)
RES = ROOT / "experiments" / "5_symmetry" / "results"


def tau_for(sigma2):
    return np.sqrt(sigma2) * (1 + 1 / np.log(N)) * np.sqrt(N * np.log(N) + N * np.log(4 * np.pi * np.log(N)))


_LAG_T = torch.as_tensor(_LAG)
_ABS_T = _LAG_T.abs()
_P_UP = torch.stack([(_LAG_T == k).flatten() for k in range(N)], 1).to(torch.complex128)   # [N*N, N]
_P_LO = torch.stack([(_LAG_T == -k).flatten() for k in range(N)], 1).to(torch.complex128)
_DIAG_COUNT = 2.0 * (N - torch.arange(N, dtype=torch.float64))


def _toeplitz(u):
    """u [B, N] -> T(u) [B, N, N] with T[i, j] = u[j-i] for j >= i, conj(u[i-j]) otherwise."""
    up = u[:, _ABS_T]
    return torch.where(_LAG_T >= 0, up, up.conj())


def _toeplitz_fit(v0):
    """Least-squares Hermitian Toeplitz u for each [N, N] block: average of each
    diagonal (upper) and the conjugate of the matching lower diagonal."""
    flat = v0.reshape(len(v0), N * N)
    return (flat @ _P_UP + (flat @ _P_LO).conj()) / _DIAG_COUNT


def anm_admm(y, tau, iters=1000, rho0=None, adapt_every=10, mu=10.0):
    """Batched ADMM.  y [B, N] complex, tau [B].  Returns the denoised x [B, N].
    The penalty rho is balanced per signal every few steps (primal vs dual
    residual, Boyd et al. 2011, sec. 3.4.1); the dual Lambda is unscaled, so
    changing rho needs no rescaling."""
    b = len(y)
    y = torch.as_tensor(y, dtype=torch.complex128)
    tau = torch.as_tensor(np.broadcast_to(np.asarray(tau, float), (b,)).copy())
    rho = 1.0 / tau if rho0 is None else torch.as_tensor(np.broadcast_to(rho0, (b,)).copy())
    z = torch.zeros((b, N + 1, N + 1), dtype=torch.complex128)
    lam = torch.zeros_like(z)
    m = torch.zeros_like(z)
    for it in range(iters):
        r3 = rho[:, None, None]
        v = z + lam / r3
        t = v[:, N, N].real - tau / (2 * rho)
        x = (y + 2 * rho[:, None] * v[:, :N, N]) / (1 + 2 * rho[:, None])
        u = _toeplitz_fit(v[:, :N, :N])
        u[:, 0] = u[:, 0].real - tau / (2 * rho * N)
        m[:, :N, :N] = _toeplitz(u)
        m[:, :N, N] = x
        m[:, N, :N] = x.conj()
        m[:, N, N] = t
        w = m - lam / r3
        w = (w + w.conj().transpose(1, 2)) / 2
        ev, vec = map(torch.from_numpy, np.linalg.eigh(w.numpy()))   # numpy's batched eigh is much faster here
        z_old = z
        z = (vec * ev.clamp(min=0)[:, None, :]) @ vec.conj().transpose(1, 2)
        lam = lam + r3 * (z - m)
        if adapt_every and it % adapt_every == adapt_every - 1:
            r = torch.linalg.matrix_norm(z - m)
            s = rho * torch.linalg.matrix_norm(z - z_old)
            rho = torch.where(r > mu * s, 2 * rho, torch.where(s > mu * r, rho / 2, rho))
    return x.numpy()


def dual_poly(y, x, tau):
    return np.abs(((y - x) / tau[:, None]).conj() @ _A)       # [B, GRID]


def top_peaks(q, k=20):
    """The k highest local maxima of each row of q: frequencies and heights, highest first."""
    out_f = np.full((len(q), k), np.nan)
    out_q = np.zeros((len(q), k))
    for i, row in enumerate(q):
        idx = np.where((row >= np.roll(row, 1)) & (row > np.roll(row, -1)))[0]
        idx = idx[np.argsort(-row[idx])][:k]
        out_f[i, :len(idx)], out_q[i, :len(idx)] = _F[idx], row[idx]
    return out_f, out_q


def iters_for(snr):
    """ADMM converges more slowly at high SNR (checked on 100 test signals:
    count accuracy stops changing at about 1000 steps up to 30 dB, 2000 above)."""
    return 1000 if snr <= 30 else 2000


def solve(x, snr, chunk=250):
    tau = np.full(len(x), tau_for(noise_var(snr)))
    pf, pq = [], []
    for s in range(0, len(x), chunk):
        xh = anm_admm(x[s:s + chunk], tau[s:s + chunk], iters=iters_for(snr))
        f, q = top_peaks(dual_poly(x[s:s + chunk], xh, tau[s:s + chunk]))
        pf.append(f)
        pq.append(q)
    return np.concatenate(pf), np.concatenate(pq)


def choose(pf, pq, delta=None, counts=None):
    """Frequencies per signal: peaks reaching 1 - delta (at least one, at most
    MAX_FREQ), or the k highest with the true count k."""
    out = []
    for i in range(len(pf)):
        k = max(int(counts[i]), 1) if counts is not None else int(np.clip(np.sum(pq[i] >= 1 - delta), 1, MAX_FREQ))
        out.append(pf[i, :k][~np.isnan(pf[i, :k])])
    return out


DELTAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
VAL_SNRS = (0, 10, 20, 30, 40, 50)
TEST_SNRS = (-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
CACHE = RES / "anm_peaks"


def _job(args):
    kind, snr = args
    out = CACHE / f"{kind}_{snr}.npz"
    if out.exists():
        return str(out)
    if kind == "val":
        from sym_eval_detector import validation_wide
        val, _ = validation_wide(300)
        x = to_complex(val[snr])
    else:
        x = to_complex(load_test_set(snr)[0])
    t0 = time.time()
    pf, pq = solve(x, snr)
    np.savez_compressed(out, f=pf, q=pq, ms=1000 * (time.time() - t0) / len(x))
    print("solved", kind, snr, flush=True)
    return str(out)


def main(workers=3):
    from multiprocessing import Pool
    CACHE.mkdir(exist_ok=True)
    jobs = [("val", s) for s in VAL_SNRS] + [("test", s) for s in TEST_SNRS]
    with Pool(workers) as pool:
        list(pool.imap_unordered(_job, sorted(jobs, key=lambda j: -iters_for(j[1]))))
    from sym_eval_detector import validation_wide
    _, vf = validation_wide(300)
    vcnt = np.sum(vf >= -0.5, 1)
    acc = {}
    for d in DELTAS:
        acc[d] = [float(np.mean(np.array([len(e) for e in choose(*_load("val", s), delta=d)]) == vcnt)) for s in VAL_SNRS]
    delta = max(DELTAS, key=lambda d: (np.mean(acc[d]), d))
    print("validation count accuracy", {d: round(float(np.mean(a)), 3) for d, a in acc.items()}, "-> delta", delta)
    res = {"delta": delta, "val_count_acc": {str(d): a for d, a in acc.items()}, "snr": {}}
    for snr in TEST_SNRS:
        sig, tgt, cnt = load_test_set(snr)
        x = to_complex(sig)
        pf, pq = _load("test", snr)
        own_raw, orc_raw = choose(pf, pq, delta=delta), choose(pf, pq, counts=cnt)
        own = pad([_joint_refine(x[i], f, 100) for i, f in enumerate(own_raw)])
        orc = pad([_joint_refine(x[i], f, 100) for i, f in enumerate(orc_raw)])
        res["snr"][str(snr)] = {
            "count_acc": 100 * float(np.mean(np.sum(own >= -0.5, 1) == cnt)),
            "over_count": 100 * float(np.mean(np.sum(own >= -0.5, 1) > cnt)),
            "chamfer_raw": float(chamfer(pad(own_raw), tgt)), "chamfer": float(chamfer(own, tgt)),
            "fnr_oracle_raw": float(fnr_percent(pad(orc_raw), tgt)), "fnr_oracle": float(fnr_percent(orc, tgt)),
            "ms_solve": float(np.load(CACHE / f"test_{snr}.npz")["ms"])}
        print(snr, res["snr"][str(snr)], flush=True)
        np.savez_compressed(CACHE / f"est_{snr}.npz", own=own, oracle=orc)
    json.dump(res, open(RES / "anm.json", "w"), indent=1)


def _load(kind, snr):
    d = np.load(CACHE / f"{kind}_{snr}.npz")
    return d["f"], d["q"]


if __name__ == "__main__":
    main()

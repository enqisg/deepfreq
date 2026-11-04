"""VALSE (variational Bayesian line spectral estimation; Badiu, Hansen and
Fleury, IEEE TSP 2017), a gridless Bayesian baseline.  It estimates the
number of frequencies, the noise level and the amplitude prior itself, so it
needs neither the true noise level (which NOMP and ANM are given) nor any
setting picked on validation data.

Model: y = sum_l w_l a(theta_l) + noise with a_m(theta) = e^{i m theta},
L = N candidate components, w_l = 0 with probability 1 - rho and CN(0, tau)
otherwise, noise CN(0, nu).  Each iteration
  * picks the support and the weight posterior by greedy maximisation of the
    marginal likelihood over the binary support (the paper's Appendix A),
  * updates nu, tau and rho in closed form,
  * updates each active frequency posterior q(theta) ~ exp(Re(eta^H a(theta)))
    in turn, approximated by one von Mises (the paper's Heuristic 2: the mode
    by grid search plus Newton steps, the concentration by matching
    E[e^{i theta}] to a Laplace approximation at the mode).
The start follows the paper: components are added one at a time from the
autocorrelation of the residual, and nu starts at the mean of the smallest
quarter of the eigenvalues of the sample autocorrelation matrix.
It stops when the reconstructed signal changes by less than 1e-6 (relative),
or after 2000 iterations (almost all signals stop well before).

Like every other method here it reports at most MAX_FREQ = 10 frequencies
(the strongest) and at least one.  "+ clean-up" refines them with the same
joint NLS as every other method.  With the true count k (for the FNR
metric) it keeps the k strongest, or adds the largest residual periodogram
peak until it has k, as NOMP does.

Usage:
    python experiments/5_symmetry/sym_valse.py                 # test sets -> results/valse.json
    python experiments/5_symmetry/sym_valse.py hard [SET ...]  # robustness sets -> results/hard_<set>_valse.json
"""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # many tiny problems: one thread per process is fastest
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
from scipy.special import ive  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SIGNAL_DIM, MAX_FREQ, load_test_set, to_complex, pad, fnr_percent, chamfer  # noqa: E402
from sym_baselines import _joint_refine, _atom  # noqa: E402

N = SIGNAL_DIM
M_IDX = np.arange(N)
G = 16 * N                       # grid for the mode search
RES = ROOT / "experiments" / "5_symmetry" / "results"
CACHE = RES / "valse_est"


def _a1_inv(r):
    """Inverse of A1(k) = I1(k)/I0(k) (Fisher 1993, Best and Fisher's approximation)."""
    r = min(max(r, 0.0), 1 - 1e-15)
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)


def _vm_atom(mu, kappa):
    """E[a(theta)] for theta ~ von Mises(mu, kappa)."""
    if kappa <= 0:
        ratio = (M_IDX == 0).astype(float)
    elif kappa > 1e5:                 # scipy's ive gives NaN for huge kappa; this is accurate to 1e-7 here
        ratio = np.exp(-M_IDX ** 2 / (2 * kappa))
    else:
        ratio = ive(M_IDX, kappa) / ive(0, kappa)
    return np.exp(1j * M_IDX * mu) * ratio


def _heuristic2(eta, lags, newton=10):
    """von Mises approximation of q(theta) ~ exp(Re(sum_m conj(eta_m) e^{i m theta}))."""
    c = np.zeros(G, complex)
    c[lags] = np.conj(eta)
    vals = (np.fft.ifft(c) * G).real
    th = 2 * np.pi * int(np.argmax(vals)) / G
    m = lags.astype(float)
    d2 = 0.0
    for _ in range(newton):
        e = np.conj(eta) * np.exp(1j * m * th)
        d1 = -np.sum(m * e.imag)
        d2 = -np.sum(m ** 2 * e.real)
        if d2 >= 0:
            break
        step = float(np.clip(-d1 / d2, -np.pi / G, np.pi / G))
        th += step
        if abs(step) < 1e-12:
            break
    if d2 >= 0:
        return th, 0.0
    return th, _a1_inv(np.exp(1 / (2 * d2)))   # A1(kappa) = exp(-sigma^2 / 2), sigma^2 = -1 / f''


def _max_z(J, h, M, nu, rho, tau):
    """Greedy maximisation over the binary support (Badiu et al., Appendix A):
    returns the support, the weight posterior mean w and covariance C."""
    L = len(h)
    cnst = np.log(rho / (1 - rho) / tau)
    s = np.zeros(L, bool)
    K = 0
    w = np.zeros(L, complex)
    C = np.zeros((L, L), complex)
    for _ in range(4 * L):
        act, ina = np.where(s)[0], np.where(~s)[0]
        delta = np.full(L, -np.inf)
        if K < M - 1:
            j_ai = J[np.ix_(act, ina)]
            quad = np.real(np.sum(j_ai.conj() * (C[np.ix_(act, act)] @ j_ai), 0))
            v = nu / (M + nu / tau - quad / nu)
            u = v * (h[ina] - j_ai.conj().T @ w[act]) / nu
            delta[ina] = np.log(v) + np.abs(u) ** 2 / v + cnst
        if K > 0:
            dc = np.real(np.diag(C)[act])
            delta[act] = -np.log(dc) - np.abs(w[act]) ** 2 / dc - cnst
        k = int(np.argmax(delta))
        if not delta[k] > 0:
            break
        if not s[k]:
            j = int(np.searchsorted(ina, k))
            ctemp = C[np.ix_(act, act)] @ J[act, k] / nu
            w[act] -= ctemp * u[j]
            w[k] = u[j]
            C[np.ix_(act, act)] += v[j] * np.outer(ctemp, ctemp.conj())
            C[act, k] = -v[j] * ctemp
            C[k, act] = np.conj(C[act, k])
            C[k, k] = v[j]
            s[k] = True
            K += 1
        else:
            s[k] = False
            K -= 1
            rest = np.where(s)[0]
            w[rest] -= C[rest, k] * w[k] / C[k, k]
            C[np.ix_(rest, rest)] -= np.outer(C[rest, k], C[k, rest]) / C[k, k]
            w[k] = 0
            C[k, :] = 0
            C[:, k] = 0
        C = (C + C.conj().T) / 2
    return K, s, w, C


def valse_one(y, max_iter=2000, tol=1e-6, start=None):
    """Returns (frequencies in cycles/sample, complex amplitudes, noise variance, iterations).
    start: optional frequencies (cycles/sample) used for the first components of
    the start in place of the residual estimates (a diagnostic only)."""
    M = L = N
    A = np.zeros((M, L), complex)
    J = np.zeros((L, L), complex)
    h = np.zeros(L, complex)
    w = np.zeros(L, complex)
    C = np.zeros((L, L), complex)
    theta = np.zeros(L)
    lags = np.arange(1, N)
    yy = float(np.real(np.vdot(y, y)))

    r_all = np.array([np.vdot(y[:N - k], y[k:]) for k in range(N)])
    ev = np.sort(np.linalg.eigvalsh(scipy.linalg.toeplitz(r_all / M)))
    nu = max(float(np.mean(ev[:M // 4])), 1e-12)
    k0 = L // 2
    rho = k0 / L
    tau = max((yy / M - nu) / (rho * L), 1e-12)

    res = y.copy()
    xro = None
    for l in range(L):
        s_r = np.array([np.vdot(res[:N - k], res[k:]) for k in lags])
        theta[l], kappa = _heuristic2(2 * s_r / (M + nu / tau) / nu, lags)
        if start is not None and l < len(start):
            theta[l] = 2 * np.pi * start[l]
        A[:, l] = _vm_atom(theta[l], kappa)
        jl = A[:, :l].conj().T @ A[:, l]
        J[:l, l] = jl
        J[l, :l] = jl.conj()
        J[l, l] = M
        h[l] = np.vdot(A[:, l], y)
        ct, wt = C[:l, :l], w[:l].copy()
        v = nu / (M + nu / tau - np.real(jl.conj() @ ct @ jl) / nu)
        u = v * (h[l] - jl.conj() @ wt) / nu
        ctemp = ct @ jl / nu
        w[:l] = wt - ctemp * u
        w[l] = u
        C[:l, :l] = ct + v * np.outer(ctemp, ctemp.conj())
        C[:l, l] = -v * ctemp
        C[l, :l] = np.conj(C[:l, l])
        C[l, l] = v
        res = y - A[:, :l + 1] @ w[:l + 1]
        if l == k0 - 1:
            xro = A[:, :l + 1] @ w[:l + 1]

    s = np.zeros(L, bool)
    for it in range(max_iter):
        K, s, w, C = _max_z(J, h, M, nu, rho, tau)
        if K > 0:
            ss = np.ix_(s, s)
            nu = float(np.real(yy - 2 * np.real(np.vdot(h[s], w[s])) + w[s].conj() @ J[ss] @ w[s]
                               + np.trace(J[ss] @ C[ss])) / M)
            nu = max(nu, 1e-12)
            tau = float(np.real(np.vdot(w[s], w[s]) + np.trace(C[ss])) / K)
            rho = K / L if K < L else (L - 1) / L
        else:
            rho = 1 / L
        inz = np.where(s)[0]
        for i in inz:
            oth = inz[inz != i]
            r = y - A[:, oth] @ w[oth]
            eta = 2 / nu * (r * np.conj(w[i]) - A[:, oth] @ C[oth, i])
            theta[i], kappa = _heuristic2(eta, M_IDX)
            A[:, i] = _vm_atom(theta[i], kappa)
        if K > 0:
            J[:, s] = A.conj().T @ A[:, s]
            J[s, :] = J[:, s].conj().T
        J[np.arange(L), np.arange(L)] = M
        h[s] = A[:, s].conj().T @ y
        xr = A[:, s] @ w[s]
        nx = np.linalg.norm(xro)
        if (nx > 0 and np.linalg.norm(xr - xro) / nx < tol) or (nx == 0 and np.linalg.norm(xr) == 0):
            break
        xro = xr
    f = (theta[s] / (2 * np.pi) + 0.5) % 1 - 0.5
    return f, w[s], nu, it + 1


def _greedy_add(y, f, k, oversample=4):
    """Add the largest residual periodogram peak (with a short joint NLS) until there are k."""
    grid = np.arange(oversample * N) / (oversample * N) - 0.5
    steer = np.exp(-2j * np.pi * np.outer(grid, M_IDX))
    f = np.asarray(f, float)
    while len(f) < k:
        if len(f):
            a_mat = np.stack([_atom(v) for v in f], 1)
            r = y - a_mat @ np.linalg.lstsq(a_mat, y, rcond=None)[0]
        else:
            r = y
        f = _joint_refine(y, np.append(f, grid[int(np.argmax(np.abs(steer @ r)))]), 5)
    return f


def estimate(x, counts):
    """VALSE on each signal: own estimate (raw and cleaned up), true-count
    estimate (cleaned up), the raw VALSE count and run time."""
    own_raw, own, orc, k_raw, iters = [], [], [], [], []
    t0 = time.time()
    for y, k in zip(x, counts):
        f, a, _, n_it = valse_one(y)
        f = f[np.argsort(-np.abs(a))]            # strongest first
        k_raw.append(len(f))
        iters.append(n_it)
        mine = f[:MAX_FREQ] if len(f) else _greedy_add(y, f, 1)
        own_raw.append(mine)
        own.append(_joint_refine(y, mine, 100))
        orc.append(_joint_refine(y, _greedy_add(y, f[:k], k), 100))
    ms = 1000 * (time.time() - t0) / len(x)
    return pad(own_raw), pad(own), pad(orc), np.array(k_raw), np.array(iters), ms


TEST_SNRS = (-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


def _job(args):
    kind, set_name, snr = args
    out = CACHE / (f"est_{snr}.npz" if kind == "test" else f"hard_{set_name}_{snr}.npz")
    if out.exists():
        return str(out)
    if kind == "test":
        sig, tgt, cnt = load_test_set(snr)
    else:
        from hard_tests import load
        sig, tgt, cnt = load(set_name, snr)
    own_raw, own, orc, k_raw, iters, ms = estimate(to_complex(sig), cnt)
    np.savez_compressed(out, own_raw=own_raw, own=own, oracle=orc, k_raw=k_raw, iters=iters, ms=ms)
    print("done", kind, set_name, snr, round(ms, 1), "ms/signal", flush=True)
    return str(out)


def _summary(d, tgt, cnt, fnr_fn):
    own = d["own"]
    k = np.sum(own >= -0.5, 1)
    return {"count_acc": 100 * float(np.mean(k == cnt)), "over_count": 100 * float(np.mean(k > cnt)),
            "raw_count_over_10": 100 * float(np.mean(d["k_raw"] > MAX_FREQ)),
            "chamfer_raw": float(chamfer(d["own_raw"], tgt)), "chamfer": float(chamfer(own, tgt)),
            "fnr_oracle": float(fnr_fn(d["oracle"], tgt)), "median_iters": float(np.median(d["iters"])), "hit_iter_cap": 100 * float(np.mean(d["iters"] >= 2000)),
            "ms_per_signal": float(d["ms"])}


def main(workers=4):
    from multiprocessing import Pool
    CACHE.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "hard":
        from hard_tests import SETS, SNRS, load
        from sym_eval_hard import fnr_at
        sets = sys.argv[2:] or SETS
        with Pool(workers) as pool:
            list(pool.imap_unordered(_job, [("hard", s, snr) for s in sets for snr in SNRS]))
        for set_name in sets:
            thr = {"superres05": 0.25, "superres025": 0.125}.get(set_name, 0.5)
            res = {"fnr_threshold_bins": thr, "snr": {}}
            for snr in SNRS:
                _, tgt, cnt = load(set_name, snr)
                d = np.load(CACHE / f"hard_{set_name}_{snr}.npz")
                sm = _summary(d, tgt, cnt, lambda e, t: fnr_at(e, t, thr))
                res["snr"][snr] = {"VALSE + clean-up": {"fnr_oracle": sm["fnr_oracle"], "count_acc": sm["count_acc"],
                                                        "chamfer": sm["chamfer"]}}
                print(set_name, snr, {a: round(b, 4) for a, b in sm.items()}, flush=True)
            json.dump(res, open(RES / f"hard_{set_name}_valse.json", "w"), indent=1)
        return
    with Pool(workers) as pool:
        list(pool.imap_unordered(_job, [("test", None, s) for s in TEST_SNRS]))
    res = {"snr": {}}
    for snr in TEST_SNRS:
        _, tgt, cnt = load_test_set(snr)
        res["snr"][str(snr)] = _summary(np.load(CACHE / f"est_{snr}.npz"), tgt, cnt, fnr_percent)
        print(snr, {a: round(b, 4) for a, b in res["snr"][str(snr)].items()}, flush=True)
    json.dump(res, open(RES / "valse.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Experiment 2: better frequency counting (model-order selection).

The E1 benchmark showed counting is DeepFreq's weakest link: full-pipeline
Chamfer error is roughly twice the oracle-count error.  Compared here:
  * FC regression counter (pretrained / fine-tuned DeepFreq)
  * MDL (classical, Hankel eigenvalues)
  * NLS-BIC: fit K = 1..10 sinusoids initialised from DeepFreq's peaks, refine
    by NLS, pick K by a penalised likelihood (penalty tuned on validation data)
  * Fusion counter: a small MLP on cheap features that combine the learned
    and the classical evidence: sorted log-eigenvalues of the Hankel matrix,
    the top FR peak heights and the FC output.
"""

import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import (ROOT, EXP, FINETUNED, XGRID, MAX_FREQ, SIGNAL_DIM, load_test_set, to_complex,
                    load_deepfreq, chamfer, fnr_percent, hankel_batch, interp_peaks, nls_refine,
                    mdl_counts, _steering)
from data import data
from data.noise import noise_torch
import scipy.signal

OUT = EXP / "2_new_counter" / "results"
SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 40, 50]
N_FEAT_PEAKS = 15


def features(signals, rep, fc_raw):
    x = to_complex(signals)
    hank = hankel_batch(x, 25)
    ev = np.linalg.svd(hank, compute_uv=False) ** 2           # descending
    ev = np.log(ev / ev.sum(axis=1, keepdims=True) + 1e-12)
    peaks = np.zeros((len(rep), N_FEAT_PEAKS))
    for i, r in enumerate(rep):
        p, props = scipy.signal.find_peaks(r, height=(None, None))
        h = np.sort(props["peak_heights"])[::-1][:N_FEAT_PEAKS]
        peaks[i, :len(h)] = h
    return np.concatenate([ev, peaks, fc_raw[:, None]], axis=1).astype(np.float32)


def nls_rss_profile(signals, rep, kmax=MAX_FREQ):
    """Residual energy after an NLS fit with K = 1..kmax sinusoids."""
    x = to_complex(signals)
    rss = np.zeros((len(x), kmax))
    for k in range(1, kmax + 1):
        est = nls_refine(x, interp_peaks(rep, np.full(len(x), k)), iters=5)
        for i in range(len(x)):
            f = est[i][est[i] >= -0.5]
            a = _steering(f)
            amp = np.linalg.lstsq(a, x[i], rcond=None)[0]
            rss[i, k - 1] = np.sum(np.abs(x[i] - a @ amp) ** 2)
    return rss


def bic_counts(rss, mult):
    k = np.arange(1, rss.shape[1] + 1)
    crit = 2 * SIGNAL_DIM * np.log(rss / SIGNAL_DIM + 1e-12) + mult * 3 * k * np.log(2 * SIGNAL_DIM)
    return np.argmin(crit, axis=1) + 1


class Counter(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, MAX_FREQ))

    def forward(self, x):
        return self.net(x)


def make_split(n, seed, fr_model, fc_model):
    np.random.seed(seed)
    torch.manual_seed(seed)
    s, f, nf = data.gen_signal(n, SIGNAL_DIM, MAX_FREQ, 1.0, distance="normal", amplitude="normal_floor",
                               floor_amplitude=0.1, variable_num_freq=True)
    noisy = noise_torch(torch.from_numpy(s), 1.0, "gaussian_blind").numpy()
    rep, fc_raw = run_net(fr_model, fc_model, noisy)
    return noisy, rep, fc_raw, nf


def run_net(fr_model, fc_model, signals):
    reps, raws = [], []
    with torch.no_grad():
        for i in range(0, len(signals), 1000):
            r = fr_model(torch.from_numpy(signals[i:i + 1000]).float())
            reps.append(r.numpy())
            raws.append(fc_model(r).view(-1).numpy())
    return np.concatenate(reps), np.concatenate(raws)


def main():
    torch.set_num_threads(2)
    fr_model, fc_model = load_deepfreq(FINETUNED)

    t = time.time()
    tr_s, tr_rep, tr_raw, tr_n = make_split(60_000, 11, fr_model, fc_model)
    va_s, va_rep, va_raw, va_n = make_split(3_000, 12, fr_model, fc_model)
    print("data", time.time() - t, flush=True)
    xtr, xva = features(tr_s, tr_rep, tr_raw), features(va_s, va_rep, va_raw)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6

    # ---- fusion MLP counter
    torch.manual_seed(0)
    net = Counter(xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt = torch.from_numpy((xtr - mu) / sd)
    yt = torch.from_numpy(tr_n - 1).long()
    xv = torch.from_numpy((xva - mu) / sd)
    best, best_state = 0, None
    for epoch in range(60):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = nn.functional.cross_entropy(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (net(xv).argmax(1).numpy() + 1 == va_n).mean()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0:
            print("mlp epoch", epoch, "val acc", acc, flush=True)
    net.load_state_dict(best_state)
    torch.save({"state": best_state, "mu": mu, "sd": sd}, EXP / "2_new_counter" / "models" / "fusion_counter.pt")
    print("fusion val acc", best, "| fc val acc", (np.clip(np.rint(va_raw), 1, 10) == va_n).mean(), flush=True)

    # ---- tune NLS-BIC penalty on validation
    va_rss = nls_rss_profile(va_s[:1000], va_rep[:1000])
    grid = np.linspace(0.2, 3.0, 29)
    accs = [(bic_counts(va_rss, m) == va_n[:1000]).mean() for m in grid]
    mult = float(grid[int(np.argmax(accs))])
    print("bic mult", mult, "val acc", max(accs), flush=True)

    rows = []
    for snr in SNRS:
        signals, targets, counts = load_test_set(snr)
        x = to_complex(signals)
        rep, raw = run_net(fr_model, fc_model, signals)
        t0 = time.perf_counter()
        feats = features(signals, rep, raw)
        with torch.no_grad():
            fus = net(torch.from_numpy((feats - mu) / sd)).argmax(1).numpy() + 1
        t_fus = time.perf_counter() - t0
        t0 = time.perf_counter()
        rss = nls_rss_profile(signals, rep)
        bic = bic_counts(rss, mult)
        t_bic = time.perf_counter() - t0
        methods = {"FC regression (fine-tuned)": (np.clip(np.rint(raw), 1, 10).astype(int), 0.0),
                   "MDL": (mdl_counts(x), None),
                   "NLS-BIC": (bic, t_bic),
                   "Fusion MLP": (fus, t_fus)}
        for name, (cnt, secs) in methods.items():
            est = interp_peaks(rep, cnt)
            est_n = nls_refine(x, est)
            row = {"snr_db": snr, "counter": name,
                   "count_accuracy_percent": 100 * np.mean(cnt == counts),
                   "within_one_percent": 100 * np.mean(np.abs(cnt - counts) <= 1),
                   "chamfer_peaks": chamfer(est, targets),
                   "chamfer_peaks_nls": chamfer(est_n, targets),
                   "counter_ms_per_signal": secs if secs is None else secs}
            rows.append(row)
            print(row, flush=True)
    with open(OUT / "e2_counting.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "e2_counting_meta.json", "w") as fh:
        json.dump({"fusion_val_acc": float(best), "bic_mult": mult, "bic_val_acc": float(max(accs))}, fh)


if __name__ == "__main__":
    main()

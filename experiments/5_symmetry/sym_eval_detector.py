"""Evaluate a trained detector (fr_variants.DETECTORS) on the saved test sets.

The count threshold is picked on the validation set (never on test).  For
each SNR it reports, with the true count and with the detector's own count:
FNR, count accuracy, Chamfer and median error, both raw (heatmap peak +
offset) and after the joint NLS clean-up (sym_baselines._joint_refine), and
the "hybrid" system: clean-up plus a likelihood check that adds or drops
sinusoids (sym_baselines.add_drop_one) when the estimated SNR is high,
adding only where the detector itself sees a weaker peak.  The penalty, SNR
gate and proposal threshold are also picked on validation data.

Usage: python experiments/5_symmetry/sym_eval_detector.py CHECKPOINT [MODEL]
       -> 5_symmetry/results/det_<checkpoint name>.json
"""

import json
import time

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import ROOT, load_test_set, to_complex, fnr_percent, chamfer, matched_median
from sym_models import DETECTORS
from sym_baselines import _joint_refine, _rss, add_drop_one
from common import pad
from sym_train import decode, validation_set, VAL_SNRS

SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


@torch.no_grad()
def head_counts(model, signals, batch=500):
    """Count head's answer (1..10), or None for models without one."""
    if getattr(model, "count_head", None) is None:
        return None
    out = [model.forward_all(torch.from_numpy(signals[i:i + batch]).float())[2]
           for i in range(0, len(signals), batch)]
    return torch.cat(out).argmax(1).numpy() + 1


@torch.no_grad()
def raw_outputs(model, signals, batch=500):
    lo, of = [], []
    for i in range(0, len(signals), batch):
        a, b = model(torch.from_numpy(signals[i:i + batch]).float())
        lo.append(a)
        of.append(b)
    return torch.cat(lo), torch.cat(of)


def refine(signals, est, iters=100):
    x = to_complex(signals)
    out = est.copy()
    for i in range(len(x)):
        f = est[i][est[i] >= -0.5]
        if len(f):
            out[i, :len(f)] = np.sort(_joint_refine(x[i], f.astype(np.float64), iters))
    return out


def pick_threshold(model):
    val_s, val_f = validation_set()
    counts = np.sum(val_f >= -0.5, axis=1)
    outs = {snr: raw_outputs(model, val_s[snr]) for snr in VAL_SNRS}
    best = None
    for th in np.arange(0.2, 0.81, 0.05):
        acc = np.mean([np.mean(np.sum(decode(*outs[s], None, th) >= -0.5, 1) == counts) for s in VAL_SNRS])
        if best is None or acc > best[1]:
            best = (float(th), float(acc))
    return best[0]


def snr_estimate(y, f):
    """Signal-to-noise ratio in dB estimated from the fit residual (no noise
    level is given): fitted power over residual power."""
    rss = _rss(y, f)
    return 10 * np.log10(max(np.sum(np.abs(y) ** 2) - rss, 1e-9) / max(rss, 1e-12))


def hybrid(signals, est_refined, mult, gate_db, proposals=None):
    """Likelihood check (add/drop by penalised likelihood) on top of the
    refined detector output, used only when the signal's estimated SNR (from
    the fit residual, no noise level given) is above gate_db.  Each sinusoid
    is also judged against the noise near its own frequency (LOCAL).  proposals:
    optional [N, 10] candidate frequencies; adds are then limited to them."""
    x = to_complex(signals)
    out = []
    for i, (y, e) in enumerate(zip(x, est_refined)):
        f = e[e >= -0.5]
        allowed = None if proposals is None else proposals[i][proposals[i] >= -0.5]
        out.append(add_drop_one(y, f, mult, allowed=allowed, local=LOCAL) if snr_estimate(y, f) > gate_db else f)
    return pad(out)


TUNE_SNRS = (0, 5, 10, 15, 20, 25, 30, 40, 50)
PLOWS = (0.05, 0.1, 0.2, 0.3)
# judge each sinusoid against the noise near its own frequency, not only the
# average noise power (sym_baselines.add_drop_one): robust to coloured noise,
# costs about 1 point of count accuracy at 20 dB on white noise
LOCAL = True
MULTS = (2.5, 2.0, 1.5)
GATES = (25, 20, 17.5, 15, 12.5, 10)


def validation_wide(n=300):
    """The first n validation signals (same seed as sym_train.validation_set)
    at SNR 0-50 dB, for tuning the likelihood check across the whole range."""
    from data import data
    from data.noise import noise_torch
    from common import SIGNAL_DIM, MAX_FREQ
    np.random.seed(777)
    s, f, _ = data.gen_signal(1000, SIGNAL_DIM, MAX_FREQ, 1.0, distance="normal", amplitude="normal_floor",
                              floor_amplitude=0.1, variable_num_freq=True)
    out = {}
    for snr in TUNE_SNRS:
        torch.manual_seed(5000 + snr)
        out[snr] = noise_torch(torch.from_numpy(s[:n]), 10 ** (snr / 10), "gaussian").numpy()
    return out, f[:n]


def pick_hybrid(model, th, n=300, decoder=None):
    """Penalty multiplier, SNR gate and proposal threshold for the likelihood
    check, on validation data at 0-50 dB (ties go to the more conservative
    setting: larger penalty, higher gate, higher proposal threshold).  The
    add/drop result does not depend on the gate, so it is computed once per
    (proposal threshold, penalty) and the gates are applied afterwards."""
    dec = decoder or (lambda lo, of, c, t: decode(lo, of, c, t))
    val_s, val_f = validation_wide(n)
    counts = np.sum(val_f >= -0.5, axis=1)
    k_ref, snr_hat, k_ad = {}, {}, {}
    for snr in TUNE_SNRS:
        lo, of = raw_outputs(model, val_s[snr])
        ref = refine(val_s[snr], dec(lo, of, None, th))
        x = to_complex(val_s[snr])
        fs = [e[e >= -0.5] for e in ref]
        k_ref[snr] = np.array([len(f) for f in fs])
        snr_hat[snr] = np.array([snr_estimate(y, f) for y, f in zip(x, fs)])
        for plow in PLOWS:
            props = dec(lo, of, None, plow)
            for mult in MULTS:
                k = k_ref[snr].copy()
                for i in np.where(snr_hat[snr] > min(GATES))[0]:
                    k[i] = len(add_drop_one(x[i], fs[i], mult, allowed=props[i][props[i] >= -0.5], local=LOCAL))
                k_ad[snr, plow, mult] = k
    best = None
    for plow in sorted(PLOWS, reverse=True):
        for mult in MULTS:
            for gate in GATES:
                acc = np.mean([np.mean(np.where(snr_hat[s] > gate, k_ad[s, plow, mult], k_ref[s]) == counts)
                               for s in TUNE_SNRS])
                if best is None or acc > best[3] + 1e-9:
                    best = (mult, gate, plow, float(acc))
    return best[:3]


def main():
    ckpt = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else ckpt.stem.rsplit("_s", 1)[0].replace("_smoke", "")
    model = DETECTORS[name]()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    torch.set_num_threads(4)
    th = pick_threshold(model)
    mult, gate, plow = pick_hybrid(model, th)
    res = {"checkpoint": str(ckpt), "threshold": th, "hybrid_mult": mult, "hybrid_gate_db": gate,
           "hybrid_proposal_threshold": plow, "snr": {}}
    for snr in SNRS:
        sig, tgt, cnt = load_test_set(snr)
        t0 = time.perf_counter()
        lo, of = raw_outputs(model, sig)
        t_net = time.perf_counter() - t0
        eo = decode(lo, of, cnt)
        ef = decode(lo, of, None, th)
        t0 = time.perf_counter()
        efr = refine(sig, ef)
        t_ref = time.perf_counter() - t0
        eor = refine(sig, eo)
        t0 = time.perf_counter()
        efh = hybrid(sig, efr, mult, gate, decode(lo, of, None, plow))
        t_hyb = time.perf_counter() - t0
        k = np.sum(ef >= -0.5, 1)
        kh = np.sum(efh >= -0.5, 1)
        k_head = head_counts(model, sig)
        if k_head is not None:
            efc = refine(sig, decode(lo, of, k_head))
            efch = hybrid(sig, efc, mult, gate, decode(lo, of, None, plow))
            kch = np.sum(efch >= -0.5, 1)
        r = {"fnr_oracle_raw": fnr_percent(eo, tgt), "fnr_oracle_refined": fnr_percent(eor, tgt),
             "median_err_oracle_refined": matched_median(eor, tgt),
             "count_acc": float(100 * np.mean(k == cnt)), "over_count": float(100 * np.mean(k > cnt)),
             "chamfer_raw": chamfer(ef, tgt), "chamfer_refined": chamfer(efr, tgt),
             "median_err_raw": matched_median(ef, tgt), "median_err_refined": matched_median(efr, tgt),
             "count_acc_hybrid": float(100 * np.mean(kh == cnt)), "chamfer_hybrid": chamfer(efh, tgt),
             "median_err_hybrid": matched_median(efh, tgt),
             "ms_net": t_net / len(sig) * 1e3, "ms_refine": t_ref / len(sig) * 1e3,
             "ms_hybrid": t_hyb / len(sig) * 1e3}
        if k_head is not None:
            r.update({"count_acc_head": float(100 * np.mean(k_head == cnt)), "chamfer_head": chamfer(efc, tgt),
                      "count_acc_head_hybrid": float(100 * np.mean(kch == cnt)),
                      "chamfer_head_hybrid": chamfer(efch, tgt)})
        res["snr"][snr] = r
        print(snr, {a: round(b, 4) for a, b in r.items()}, flush=True)
    out = ROOT / "experiments" / "5_symmetry" / "results" / f"det_{ckpt.stem}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""Evaluate a trained detector (fr_variants.DETECTORS) on the saved test sets.

The count threshold is picked on the validation set (never on test).  For
each SNR it reports, with the true count and with the detector's own count:
FNR, count accuracy, Chamfer and median error, both raw (heatmap peak +
offset) and after the joint NLS clean-up (sym_baselines._joint_refine).

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
from sym_baselines import _joint_refine
from sym_train import decode, validation_set, VAL_SNRS

SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


@torch.no_grad()
def raw_outputs(model, signals, batch=500):
    lo, of = [], []
    for i in range(0, len(signals), batch):
        a, b = model(torch.from_numpy(signals[i:i + batch]).float())
        lo.append(a)
        of.append(b)
    return torch.cat(lo), torch.cat(of)


def refine(signals, est, iters=20):
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


def main():
    ckpt = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else ckpt.stem.rsplit("_s", 1)[0].replace("_smoke", "")
    model = DETECTORS[name]()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    torch.set_num_threads(4)
    th = pick_threshold(model)
    res = {"checkpoint": str(ckpt), "threshold": th, "snr": {}}
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
        k = np.sum(ef >= -0.5, 1)
        r = {"fnr_oracle_raw": fnr_percent(eo, tgt), "fnr_oracle_refined": fnr_percent(eor, tgt),
             "median_err_oracle_refined": matched_median(eor, tgt),
             "count_acc": float(100 * np.mean(k == cnt)), "over_count": float(100 * np.mean(k > cnt)),
             "chamfer_raw": chamfer(ef, tgt), "chamfer_refined": chamfer(efr, tgt),
             "median_err_raw": matched_median(ef, tgt), "median_err_refined": matched_median(efr, tgt),
             "ms_net": t_net / len(sig) * 1e3, "ms_refine": t_ref / len(sig) * 1e3}
        res["snr"][snr] = r
        print(snr, {a: round(b, 4) for a, b in r.items()}, flush=True)
    out = ROOT / "experiments" / "5_symmetry" / "results" / f"det_{ckpt.stem}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

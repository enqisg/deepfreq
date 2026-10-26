"""NOMP (oracle count and CFAR count) and the Cramer-Rao bound on every saved
test set -> results/sym_nomp_crb.json."""

import json
import time

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import ROOT, load_test_set, to_complex, fnr_percent, chamfer, matched_median
from sym_baselines import nomp, crb_per_signal, noise_var

SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

if __name__ == "__main__":
    clean = to_complex(np.load(ROOT / "test_dataset/infdB.npy"))
    res = {}
    for snr in SNRS:
        sig, tgt, cnt = load_test_set(snr)
        x = to_complex(sig)
        t0 = time.time()
        eo = nomp(x, cnt)
        t1 = time.time()
        ef = nomp(x, None, snr)
        k = np.sum(ef >= -0.5, 1)
        r = {"nomp_oracle_fnr": fnr_percent(eo, tgt), "nomp_oracle_median_err_bins": matched_median(eo, tgt),
             "nomp_cfar_count_acc": float(100 * np.mean(k == cnt)), "nomp_cfar_chamfer": chamfer(ef, tgt),
             "nomp_cfar_median_err_bins": matched_median(ef, tgt), "nomp_ms_per_signal": (t1 - t0) / len(x) * 1e3}
        if snr != -5:   # the -5 dB set uses different clean signals
            v = np.concatenate([crb_per_signal(tgt[i], clean[i], noise_var(snr))[0] for i in range(len(x))])
            sd = np.sqrt(np.abs(v)) * 50
            r.update({"crb_median_std_bins": float(np.median(sd)),
                      "crb_median_abs_err_bins": float(0.6745 * np.median(sd))})
        res[snr] = r
        print(snr, {a: round(b, 4) for a, b in r.items()}, flush=True)
    with open(ROOT / "experiments/5_symmetry/results/sym_nomp_crb.json", "w") as fh:
        json.dump(res, fh, indent=1)

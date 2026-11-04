"""Evaluate FR-type learned baselines (a spectrum on the 1000-point grid) with
exactly the same post-processing as the detector, so that only the network
differs:

* count: spectrum peaks above a threshold picked on the validation set,
* the joint NLS clean-up and the likelihood check (penalty, SNR gate and
  proposal threshold also picked on validation data).

The spectrum s is turned into detector-style outputs: p = s / (1 + s) as the
"probability" at each bin (so the threshold grid 0.2-0.8 covers spectrum
heights 0.25-4; DeepFreq's target has height 1 per frequency) and a
3-point parabolic sub-bin offset.

Usage: python experiments/5_symmetry/sym_eval_fr.py NAME
  NAME: deepfreq2019 (the 2019 pretrained FR model), any fr_variants.VARIANTS
  key with a 3_fr_designs/models/<key>_long.pt checkpoint (60k steps), or
  CHECKPOINT MODEL for a sym_models.FR_MODELS model trained by sym_train.py
  -> 5_symmetry/results/fr_<name>.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, PRETRAINED, load_test_set, load_deepfreq, fnr_percent, chamfer, matched_median  # noqa: E402
from sym_models import FR_MODELS  # noqa: E402
import sym_eval_detector as E  # noqa: E402
from sym_train import decode  # noqa: E402


class SpectrumAsDetector(nn.Module):
    """Wraps an FR model so that sym_train.decode and the detector
    evaluation code can be reused unchanged."""

    def __init__(self, fr):
        super().__init__()
        self.fr = fr
        self.m = 1000
        self.count_head = None

    def forward(self, inp):
        s = self.fr(inp)
        logits = torch.log(s.clamp(min=1e-6))          # sigmoid(logits) = s / (1 + s) for s > 0
        left, right = torch.roll(s, 1, -1), torch.roll(s, -1, -1)
        den = left - 2 * s + right
        off = torch.where(den.abs() > 1e-9, 0.5 * (left - right) / den, torch.zeros_like(s))
        return logits, off.clamp(-0.5, 0.5)


def load(argv):
    if argv[0] == "deepfreq2019":
        return "deepfreq2019", load_deepfreq(PRETRAINED)[0]
    if len(argv) == 1:
        from fr_variants import VARIANTS
        m = VARIANTS[argv[0]]()
        m.load_state_dict(torch.load(ROOT / "experiments" / "3_fr_designs" / "models" / f"{argv[0]}_long.pt",
                                     map_location="cpu"))
        return f"{argv[0]}_60k", m
    ckpt = Path(argv[0])
    m = FR_MODELS[argv[1]]()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return ckpt.stem, m


def main():
    name, fr = load(sys.argv[1:])
    model = SpectrumAsDetector(fr.eval()).eval()
    torch.set_num_threads(4)
    th = E.pick_threshold(model)
    mult, gate, plow = E.pick_hybrid(model, th)
    res = {"model": name, "threshold": th, "hybrid_mult": mult, "hybrid_gate_db": gate,
           "hybrid_proposal_threshold": plow, "params": sum(p.numel() for p in fr.parameters()), "snr": {}}
    for snr in E.SNRS:
        sig, tgt, cnt = load_test_set(snr)
        t0 = time.perf_counter()
        lo, of = E.raw_outputs(model, sig)
        t_net = time.perf_counter() - t0
        eo, ef = decode(lo, of, cnt), decode(lo, of, None, th)
        t0 = time.perf_counter()
        efr = E.refine(sig, ef)
        t_ref = time.perf_counter() - t0
        eor = E.refine(sig, eo)
        t0 = time.perf_counter()
        efh = E.hybrid(sig, efr, mult, gate, decode(lo, of, None, plow))
        t_hyb = time.perf_counter() - t0
        k, kh = np.sum(ef >= -0.5, 1), np.sum(efh >= -0.5, 1)
        r = {"fnr_oracle_raw": fnr_percent(eo, tgt), "fnr_oracle_refined": fnr_percent(eor, tgt),
             "median_err_oracle_refined": matched_median(eor, tgt),
             "count_acc": float(100 * np.mean(k == cnt)), "over_count": float(100 * np.mean(k > cnt)),
             "chamfer_raw": chamfer(ef, tgt), "chamfer_refined": chamfer(efr, tgt),
             "median_err_refined": matched_median(efr, tgt),
             "count_acc_hybrid": float(100 * np.mean(kh == cnt)), "chamfer_hybrid": chamfer(efh, tgt),
             "median_err_hybrid": matched_median(efh, tgt),
             "ms_net": t_net / len(sig) * 1e3, "ms_refine": t_ref / len(sig) * 1e3,
             "ms_hybrid": t_hyb / len(sig) * 1e3}
        res["snr"][snr] = r
        print(snr, {a: round(b, 4) for a, b in r.items()}, flush=True)
    out = ROOT / "experiments" / "5_symmetry" / "results" / f"fr_{name}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

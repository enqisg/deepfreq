"""Validation pick of the count method for one detector.

Count accuracy on validation_wide (300 signals per SNR, 0-50 dB) with K from
the count head and with K from the count threshold, both followed by the same
refinement and likelihood check, using the settings in det_<stem>.json.  The
method with the higher mean over SNRs is the one reported for that model
(sym_summary.py reads results/countpick_<stem>.json).

    python sym_count_pick.py models/<stem>.pt <model name> [results/countpick_<stem>.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "5_symmetry"))
sys.path.insert(0, str(ROOT / "experiments"))
import sym_eval_detector as E  # noqa: E402
from sym_train import decode  # noqa: E402
from sym_models import DETECTORS  # noqa: E402

torch.set_num_threads(2)
ckpt = Path(sys.argv[1])
cfg = json.load(open(ROOT / "experiments/5_symmetry/results" / f"det_{ckpt.stem}.json"))
m = DETECTORS[sys.argv[2]]()
m.load_state_dict(torch.load(ckpt, map_location="cpu"))
m.eval()
val, f = E.validation_wide(300)
cnt = np.sum(f >= -0.5, 1)
out = {}
for snr, sig in val.items():
    sig = sig.astype(np.float32)
    lo, of = E.raw_outputs(m, sig)
    prop = decode(lo, of, None, cfg["hybrid_proposal_threshold"])
    kh = E.head_counts(m, sig)
    r = {}
    for name, k in (("threshold", None), ("head", kh)):
        est = E.hybrid(sig, E.refine(sig, decode(lo, of, k, cfg["threshold"])), cfg["hybrid_mult"],
                       cfg["hybrid_gate_db"], prop)
        r[name] = float(100 * np.mean(np.sum(est >= -0.5, 1) == cnt))
    r["head_raw"] = float(100 * np.mean(kh == cnt))
    out[snr] = r
    print(snr, r, flush=True)
print("mean", {k: round(np.mean([v[k] for v in out.values()]), 2) for k in ("threshold", "head", "head_raw")})
dst = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "experiments/5_symmetry/results" / f"countpick_{ckpt.stem}.json"
json.dump(out, open(dst, "w"), indent=1)

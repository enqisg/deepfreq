"""Robustness sets for an FR-type baseline given exactly our post-processing
(count by threshold, NLS clean-up, likelihood check), with the settings
picked on validation data in results/fr_<name>.json by sym_eval_fr.py.
This is the fair counterpart of "Symmetry-aware detector + checks" in
sym_eval_hard.py.

Usage: python experiments/5_symmetry/sym_eval_hard_fr.py NAME [SET ...]
       NAME as in sym_stats.py: deepfreq2019, deepfreq20x64_60k, cresfreq10x64_60k_s0, ...
       -> 5_symmetry/results/hard_<set>_fr_<name>.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import to_complex, chamfer, matched_median  # noqa: E402
from hard_tests import load, SETS, SNRS  # noqa: E402
from sym_eval_detector import raw_outputs, refine, hybrid  # noqa: E402
from sym_eval_hard import fnr_at, OUT  # noqa: E402
from sym_stats import run_fr  # noqa: E402
from sym_train import decode  # noqa: E402

KEY = "{} FR model + same post-processing"


def main():
    name = sys.argv[1]
    sets = sys.argv[2:] or SETS
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    _, model, cfg = run_fr(name)
    for set_name in sets:
        thr = {"superres05": 0.25, "superres025": 0.125}.get(set_name, 0.5)
        res = {"fnr_threshold_bins": thr, "model": name, "snr": {}}
        for snr in SNRS:
            sig, tgt, cnt = load(set_name, snr)
            lo, of = raw_outputs(model, sig)
            eo = refine(sig, decode(lo, of, cnt))
            ef = hybrid(sig, refine(sig, decode(lo, of, None, cfg["threshold"])), cfg["hybrid_mult"],
                        cfg["hybrid_gate_db"], decode(lo, of, None, cfg["hybrid_proposal_threshold"]))
            k = np.sum(ef >= -0.5, 1)
            res["snr"][snr] = {KEY.format(name): {
                "fnr_oracle": fnr_at(eo, tgt, thr), "count_acc": float(100 * np.mean(k == cnt)),
                "chamfer": chamfer(ef, tgt), "median_err": matched_median(ef, tgt)}}
            print(set_name, snr, {a: round(b, 4) for a, b in res["snr"][snr][KEY.format(name)].items()}, flush=True)
        with open(OUT / f"hard_{set_name}_fr_{name}.json", "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""Robustness: every method on the harder test sets (hard_tests.py).

Methods: DeepFreq (2019 weights: FR model + FC model), the E6 system
(4_combined, improved FR model + fusion counter + NLS), NOMP (given the true
noise power; true count and CFAR count) and a symmetry-aware detector with
the NLS clean-up and likelihood check (settings from its det_*.json, which
were picked on validation data).

Usage: python experiments/5_symmetry/sym_eval_hard.py DET_CHECKPOINT DET_MODEL [SET ...]
       -> 5_symmetry/results/hard_<set>_<checkpoint name>.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import (ROOT, EXP, PRETRAINED, FINETUNED, XGRID, MAX_FREQ, SIGNAL_DIM, to_complex, chamfer,
                    matched_median)
from data import fr
import util
from hard_tests import load, SETS, SNRS
from sym_baselines import nomp
from sym_models import DETECTORS
from sym_eval_detector import raw_outputs, refine, hybrid, head_counts
from sym_train import decode

OUT = EXP / "5_symmetry" / "results"


def fnr_at(est, tgt, thr_bins):
    """FNR (%) where a true frequency counts as found if an estimate is within thr_bins bins."""
    miss, tot = 0.0, 0
    for e, t in zip(est, tgt):
        e, t = e[e >= -0.5], t[t >= -0.5]
        d = np.abs(((t[:, None] - e[None, :]) + 0.5) % 1 - 0.5).min(1) * SIGNAL_DIM if len(e) else np.full(len(t), 9.)
        miss += np.mean(d > thr_bins)
        tot += 1
    return 100 * miss / tot


def deepfreq_2019(fr_model, fc_model, signals, counts=None):
    with torch.no_grad():
        rep = fr_model(torch.from_numpy(signals).float())
        raw = fc_model(rep).numpy().reshape(-1)
    rep = rep.numpy()
    cnt = np.clip(np.rint(raw), 1, MAX_FREQ).astype(int) if counts is None else counts
    return fr.find_freq(rep, cnt, XGRID, MAX_FREQ), cnt


def main():
    ckpt, det_name = Path(sys.argv[1]), sys.argv[2]
    sets = sys.argv[3:] or SETS
    torch.set_num_threads(4)
    cfg = json.load(open(OUT / f"det_{ckpt.stem}.json"))
    det = DETECTORS[det_name]()
    det.load_state_dict(torch.load(ckpt, map_location="cpu"))
    det.eval()
    cpu = torch.device("cpu")
    df_fr = util.load(str(PRETRAINED[0]), "fr", cpu)[0].eval()
    df_fc = util.load(str(PRETRAINED[1]), "fc", cpu)[0].eval()
    from e6_compare import pipeline, load_counter
    from e6_improved_finetune import FR_PATH, COUNTER_PATH
    e6_models = {"c": (util.load(str(FR_PATH), "fr", cpu)[0].eval(),
                       util.load(str(FINETUNED[1]), "fc", cpu)[0].eval(), load_counter(COUNTER_PATH))}
    for name in sets:
        thr = {"superres05": 0.25, "superres025": 0.125}.get(name, 0.5)
        res = {"fnr_threshold_bins": thr, "detector": str(ckpt), "snr": {}}
        for snr in SNRS:
            sig, tgt, cnt = load(name, snr)
            x = to_complex(sig)
            out = {}

            def score(key, oracle_est, full_est, k_full):
                out[key] = {"fnr_oracle": fnr_at(oracle_est, tgt, thr),
                            "count_acc": float(100 * np.mean(k_full == cnt)),
                            "chamfer": chamfer(full_est, tgt), "median_err": matched_median(full_est, tgt)}

            eo, _ = deepfreq_2019(df_fr, df_fc, sig, cnt)
            ef, k = deepfreq_2019(df_fr, df_fc, sig)
            score("DeepFreq 2019", eo, ef, k)
            eo, _ = pipeline("c", e6_models, sig, cnt)
            ef, k = pipeline("c", e6_models, sig)
            score("E6 system", eo, ef, k)
            eo = nomp(x, cnt)
            ef = nomp(x, None, snr)
            score("NOMP (true noise power)", eo, ef, np.sum(ef >= -0.5, 1))
            lo, of = raw_outputs(det, sig)
            eo = refine(sig, decode(lo, of, cnt))
            ef = hybrid(sig, refine(sig, decode(lo, of, None, cfg["threshold"])), cfg["hybrid_mult"],
                        cfg["hybrid_gate_db"], decode(lo, of, None, cfg["hybrid_proposal_threshold"]))
            score("Symmetry-aware detector + checks", eo, ef, np.sum(ef >= -0.5, 1))
            k_head = head_counts(det, sig)
            if k_head is not None:
                ef = hybrid(sig, refine(sig, decode(lo, of, k_head)), cfg["hybrid_mult"], cfg["hybrid_gate_db"],
                            decode(lo, of, None, cfg["hybrid_proposal_threshold"]))
                score("Symmetry-aware detector, count head + checks", eo, ef, np.sum(ef >= -0.5, 1))
            res["snr"][snr] = out
            print(name, snr, {m: {a: round(b, 4) for a, b in v.items()} for m, v in out.items()}, flush=True)
        with open(OUT / f"hard_{name}_{ckpt.stem}.json", "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""Per-signal test scores and paired bootstrap comparisons.

`run` scores one system on every test signal (same 1000 signals per SNR for
all systems) and saves, per SNR, whether its count is right, its Chamfer
distance, and the fraction of true frequencies it misses with the true count:
    results/persig_<name>.npz
`compare` gives, per SNR, the mean difference A - B with a 95% paired
bootstrap interval (resampling test signals) and the share of resamples in
which A is not better (a one-sided p-value).

Systems (each uses the settings picked on validation data, stored in its
results json):
    det:<checkpoint>:<model>   a detector, final system (clean-up + likelihood check)
    fr:<name>                  an FR baseline as in sym_eval_fr.py (results/fr_<name>.json)
    nomp                       NOMP, CFAR stop with the true noise level, 100-step refinement

Usage:
    python experiments/5_symmetry/sym_stats.py run det:experiments/5_symmetry/models/eqdet16x64_20k_s0.pt:eqdet16x64
    python experiments/5_symmetry/sym_stats.py run fr:deepfreq2019
    python experiments/5_symmetry/sym_stats.py compare eqdet16x64_20k_s0 fr_deepfreq2019
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SIGNAL_DIM, load_test_set, to_complex  # noqa: E402
import sym_eval_detector as E  # noqa: E402
from sym_train import decode  # noqa: E402

RES = ROOT / "experiments" / "5_symmetry" / "results"
METRICS = {"count_acc": "higher", "chamfer": "lower", "fnr": "lower"}


def _circ(a, b):
    """|a - b| on the unit circle of frequencies, broadcast."""
    return np.abs((a - b + 0.5) % 1 - 0.5)


def per_signal_fnr(est, tgt):
    """Fraction of each signal's true frequencies with no estimate within half
    a bin (data/loss.fnr, per signal)."""
    out = np.zeros(len(tgt))
    for i, (e, t) in enumerate(zip(est, tgt)):
        t, e = t[t >= -0.5], e[e >= -0.5]
        if len(e) == 0:
            out[i] = 1.0
            continue
        d = _circ(t[:, None], e[None, :]).min(1)
        out[i] = np.mean(d > 1 / (2 * SIGNAL_DIM))
    return out


def per_signal_chamfer(est, tgt):
    """Chamfer distance per signal (data/loss.chamfer, per signal)."""
    out = np.zeros(len(tgt))
    for i, (e, t) in enumerate(zip(est, tgt)):
        t, e = t[t >= -0.5], e[e >= -0.5]
        if len(e) == 0:
            out[i] = np.nan
            continue
        d = _circ(t[:, None], e[None, :])
        out[i] = d.min(1).sum() + d.min(0).sum()
    return out


def _scores(est_own, est_oracle, tgt, cnt):
    return {"count_acc": (np.sum(est_own >= -0.5, 1) == cnt).astype(float),
            "chamfer": per_signal_chamfer(est_own, tgt), "fnr": per_signal_fnr(est_oracle, tgt)}


def run_detector(ckpt, model_name):
    from sym_models import DETECTORS
    cfg = json.load(open(RES / f"det_{Path(ckpt).stem}.json"))
    model = DETECTORS[model_name]()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return Path(ckpt).stem, model.eval(), cfg


def run_fr(name):
    import sym_eval_fr as F
    cfg = json.load(open(RES / f"fr_{name}.json"))
    if name == "deepfreq2019":
        argv = ["deepfreq2019"]
    elif name.endswith("_60k") and (ROOT / "experiments" / "3_fr_designs" / "models" /
                                    f"{name[:-4]}_long.pt").exists():
        argv = [name[:-4]]
    else:
        argv = [str(ROOT / "experiments" / "5_symmetry" / "models" / f"{name}.pt"), name.rsplit("_60k", 1)[0]]
    _, fr = F.load(argv)
    return f"fr_{name}", F.SpectrumAsDetector(fr.eval()).eval(), cfg


def run(spec):
    out = {}
    if spec == "nomp":
        from sym_baselines import nomp
        name = "nomp"
        for snr in E.SNRS:
            sig, tgt, cnt = load_test_set(snr)
            x = to_complex(sig)
            out[snr] = _scores(nomp(x, snr_db=snr, joint_iters=100), nomp(x, counts=cnt, joint_iters=100), tgt, cnt)
            print(name, snr, {k: round(float(np.nanmean(v)), 4) for k, v in out[snr].items()}, flush=True)
    else:
        kind, rest = spec.split(":", 1)
        name, model, cfg = run_detector(*rest.rsplit(":", 1)) if kind == "det" else run_fr(rest)
        for snr in E.SNRS:
            sig, tgt, cnt = load_test_set(snr)
            lo, of = E.raw_outputs(model, sig)
            efr = E.refine(sig, decode(lo, of, None, cfg["threshold"]))
            efh = E.hybrid(sig, efr, cfg["hybrid_mult"], cfg["hybrid_gate_db"],
                           decode(lo, of, None, cfg["hybrid_proposal_threshold"]))
            out[snr] = _scores(efh, E.refine(sig, decode(lo, of, cnt)), tgt, cnt)
            print(name, snr, {k: round(float(np.nanmean(v)), 4) for k, v in out[snr].items()}, flush=True)
    np.savez_compressed(RES / f"persig_{name}.npz",
                        **{f"{m}_{snr}": v[m] for snr, v in out.items() for m in METRICS})


def compare(a, b, n_boot=10000, seed=0):
    da, db = np.load(RES / f"persig_{a}.npz"), np.load(RES / f"persig_{b}.npz")
    rng = np.random.default_rng(seed)
    res = {}
    for snr in E.SNRS:
        res[snr] = {}
        for m, better in METRICS.items():
            x, y = da[f"{m}_{snr}"], db[f"{m}_{snr}"]
            ok = ~(np.isnan(x) | np.isnan(y))
            d = (x - y)[ok] * (100 if m != "chamfer" else 1)
            idx = rng.integers(0, len(d), (n_boot, len(d)))
            boot = d[idx].mean(1)
            worse = np.mean(boot <= 0) if better == "higher" else np.mean(boot >= 0)
            res[snr][m] = {"diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                            float(np.percentile(boot, 97.5))],
                           "p_not_better": float(worse)}
        print(snr, "  ".join(f"{m} {r['diff']:+.4g} [{r['ci95'][0]:+.3g}, {r['ci95'][1]:+.3g}] p={r['p_not_better']:.3f}"
                             for m, r in res[snr].items()))
    with open(RES / f"compare_{a}__{b}.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    if sys.argv[1] == "run":
        for s in sys.argv[2:]:
            run(s)
    else:
        compare(sys.argv[2], sys.argv[3])

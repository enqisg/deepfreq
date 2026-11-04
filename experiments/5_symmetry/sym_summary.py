"""Summary tables from the detector result files (results/det_*.json).

  python experiments/5_symmetry/sym_summary.py seeds PREFIX [PREFIX...]
      mean ± sd over seeds of det_<PREFIX>_s<k>.json, e.g. eqdet16x64c_60k
  python experiments/5_symmetry/sym_summary.py data
      data efficiency: det_<model>_n<N>[aug]_s0.json for both front-ends
  python experiments/5_symmetry/sym_summary.py ablation
      the 2x2 symmetry ablation at 60k steps (seed 0, or mean over seeds)

Each cell is FNR % with the true count / count accuracy % / Chamfer (bins),
after the clean-up and likelihood check.  For models with a count head the
count head's count is used; --both adds rows with the threshold count.
Missing files are shown as "-".
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
SNRS = ["-5", "0", "10", "20", "30", "50"]
NTRAIN = [3000, 10000, 30000, 100000]
ABLATION = [("neither (DeepFreq's front-end)", "lindet16x64c_60k"), ("phase only", "cpxdet16x64c_60k"),
            ("shift only", "rawdet16x64c_60k"), ("both (ours)", "eqdet16x64c_60k")]


def load(stem):
    p = RES / f"det_{stem}.json"
    return json.load(open(p)) if p.exists() else None


def metrics(d, snr, head=True):
    r = d["snr"].get(snr)
    if r is None:
        return None
    if head and "count_acc_head_hybrid" in r:
        return r["fnr_oracle_refined"], r["count_acc_head_hybrid"], r["chamfer_head_hybrid"]
    if "count_acc_hybrid" in r:
        return r["fnr_oracle_refined"], r["count_acc_hybrid"], r["chamfer_hybrid"]
    return r["fnr_oracle_refined"], r["count_acc"], r["chamfer_refined"]  # older files, no likelihood check


def seeds(prefix):
    return [d for d in (load(f"{prefix}_s{k}") for k in range(10)) if d is not None]


def cell(runs, snr, head=True):
    vals = [m for m in (metrics(d, snr, head) for d in runs) if m is not None]
    if not vals:
        return "-"
    a = np.array(vals)
    mu, sd = a.mean(0), a.std(0, ddof=1) if len(a) > 1 else None
    if sd is None:
        return f"{mu[0]:.2f} / {mu[1]:.1f} / {mu[2]:.4f}"
    return f"{mu[0]:.2f}±{sd[0]:.2f} / {mu[1]:.1f}±{sd[1]:.1f} / {mu[2]:.4f}±{sd[2]:.4f}"


def table(rows):
    print("| System | runs | " + " | ".join(f"{s} dB" for s in SNRS) + " |")
    print("|---" * (len(SNRS) + 2) + "|")
    for row in rows:
        label, runs, head = row if len(row) == 3 else (*row, True)
        print(f"| {label} | {len(runs)} | " + " | ".join(cell(runs, s, head) for s in SNRS) + " |")


def main():
    mode, args = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    both = "--both" in sys.argv
    if mode == "seeds":
        rows = [(p, seeds(p)) for p in args]
        if both:
            rows += [(p + " (threshold count)", seeds(p), False) for p in args]
        table(rows)
    elif mode == "data":
        rows = []
        for model, label in [("eqdet16x64c", "ours"), ("lindet16x64c", "DeepFreq front-end")]:
            for aug in ["", "aug"]:
                for n in NTRAIN:
                    d = load(f"{model}_n{n}{aug}_s0")
                    if d is not None:
                        rows.append((f"{label}{' + shift aug.' if aug else ''}, {n} signals", [d]))
        table(rows)
    elif mode == "ablation":
        table([(label, seeds(p)) for label, p in ABLATION])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

"""Experiment 1+2: classical baselines vs DeepFreq, and test-time refinement.

Oracle-count metrics (FNR, matched RMSE) isolate how well each method locates
frequencies when told how many there are.  Full-pipeline metrics (count
accuracy, Chamfer) include each method's own model-order estimate.
"""

import csv
import json

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import (ROOT, EXP, PRETRAINED, FINETUNED, XGRID, MAX_FREQ, Timer, load_test_set,
                    to_complex, load_deepfreq, fnr_percent, chamfer, matched_rmse, matched_median,
                    esprit, music_peaks, periodogram_peaks, mdl_counts, interp_peaks,
                    nls_refine)
from data import fr

OUT = EXP / "1_cleanup_step" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SNRS = [-5, 0, 5, 10, 15, 20, 25, 30, 40, 50]


def deepfreq_outputs(models, signals):
    fr_model, fc_model = models
    with torch.no_grad():
        rep_t = fr_model(torch.from_numpy(signals).float())
        raw = fc_model(rep_t).view(-1).numpy()
    counts = np.clip(np.rint(raw), 1, MAX_FREQ).astype(int)
    return rep_t.numpy(), counts


def main():
    torch.set_num_threads(4)
    models = {"DeepFreq (pretrained)": load_deepfreq(PRETRAINED),
              "DeepFreq (fine-tuned)": load_deepfreq(FINETUNED)}
    rows, timing = [], {}
    for snr in SNRS:
        signals, targets, counts = load_test_set(snr)
        x = to_complex(signals)

        def add(method, est, kind, count_est=None, seconds=None):
            row = {"snr_db": snr, "method": method, "setting": kind,
                   "fnr_percent": fnr_percent(est, targets),
                   "matched_rmse_bins": matched_rmse(est, targets),
                   "median_err_bins": matched_median(est, targets),
                   "chamfer": chamfer(est, targets),
                   "count_accuracy_percent": (100 * np.mean(count_est == counts)
                                              if count_est is not None else ""),
                   "seconds_per_1000": seconds if seconds is not None else ""}
            rows.append(row)
            print(f"{snr:>4} dB | {kind:6s} | {method:38s} | FNR {row['fnr_percent']:6.2f}% | "
                  f"RMSE {row['matched_rmse_bins']:.4f} bins | Chamfer {row['chamfer']:.5f}"
                  + (f" | count acc {row['count_accuracy_percent']:.1f}%" if count_est is not None else ""),
                  flush=True)

        # ---- oracle count: how well frequencies are located
        with Timer() as t:
            est = periodogram_peaks(x, counts)
        add("Periodogram", est, "oracle", seconds=t.elapsed)
        with Timer() as t:
            est = music_peaks(x, counts)
        add("MUSIC", est, "oracle", seconds=t.elapsed)
        with Timer() as t:
            est = esprit(x, counts)
        add("ESPRIT", est, "oracle", seconds=t.elapsed)

        for name, mdl in models.items():
            with Timer() as t_net:
                rep, fc_counts = deepfreq_outputs(mdl, signals)
            with Timer() as t_peak:
                est = fr.find_freq(rep, counts, XGRID, MAX_FREQ)
            add(name, est, "oracle", seconds=t_net.elapsed + t_peak.elapsed)
            with Timer() as t_i:
                est_i = interp_peaks(rep, counts)
            add(name + " + interp", est_i, "oracle", seconds=t_net.elapsed + t_i.elapsed)
            with Timer() as t_n:
                est_n = nls_refine(x, est_i)
            add(name + " + NLS refine", est_n, "oracle",
                seconds=t_net.elapsed + t_i.elapsed + t_n.elapsed)

            # ---- full pipeline with the network's own count
            est_fc = fr.find_freq(rep, fc_counts, XGRID, MAX_FREQ)
            add(name, est_fc, "full", fc_counts)
            est_fc_n = nls_refine(x, interp_peaks(rep, fc_counts))
            add(name + " + NLS refine", est_fc_n, "full", fc_counts)

        with Timer() as t_mdl:
            mdl_c = mdl_counts(x)
        with Timer() as t:
            est = music_peaks(x, mdl_c)
        add("MDL + MUSIC", est, "full", mdl_c, t.elapsed + t_mdl.elapsed)
        with Timer() as t:
            est = esprit(x, mdl_c)
        add("MDL + ESPRIT", est, "full", mdl_c, t.elapsed + t_mdl.elapsed)

    with open(OUT / "e1_baselines_and_refinement.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ---- latency for a single signal (batch of 1), the deployment case
    signals, targets, counts = load_test_set(10)
    x = to_complex(signals)
    fr_model, fc_model = models["DeepFreq (fine-tuned)"]
    one = torch.from_numpy(signals[:1]).float()
    with torch.no_grad():
        for _ in range(5):
            fr_model(one)
        with Timer() as t:
            for i in range(200):
                rep = fr_model(torch.from_numpy(signals[i:i + 1]).float())
                fc_model(rep)
    timing["DeepFreq FR+FC, batch=1 (ms)"] = 1000 * t.elapsed / 200
    with torch.no_grad():
        with Timer() as t:
            rep = fr_model(torch.from_numpy(signals).float())
            fc_model(rep)
    timing["DeepFreq FR+FC, batch=1000 (ms per signal)"] = t.elapsed
    with Timer() as t:
        esprit(x[:200], counts[:200])
    timing["ESPRIT (ms per signal)"] = 1000 * t.elapsed / 200
    with Timer() as t:
        music_peaks(x[:200], counts[:200])
    timing["MUSIC 1000-pt grid (ms per signal)"] = 1000 * t.elapsed / 200
    with Timer() as t:
        mdl_counts(x[:200])
    timing["MDL count (ms per signal)"] = 1000 * t.elapsed / 200
    est = interp_peaks(rep.numpy()[:200], counts[:200])
    with Timer() as t:
        nls_refine(x[:200], est)
    timing["NLS refine, 10 iters (ms per signal)"] = 1000 * t.elapsed / 200
    print(json.dumps(timing, indent=2))
    with open(OUT / "e1_timing.json", "w") as fh:
        json.dump(timing, fh, indent=2)


if __name__ == "__main__":
    main()

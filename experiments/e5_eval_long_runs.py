"""Experiment 5: evaluate the FR models trained for 60k steps on the Mac GPU.

For every checkpoint: oracle-count FNR and median frequency error at each test
SNR, plus inference time per signal (batch of 1000 and batch of 1) on CPU and,
when available, on the Apple GPU (MPS).  The pretrained DeepFreq FR is timed
the same way for reference.
Usage: python e5_eval_long_runs.py   (writes results/e5_long_runs.json)
"""

import json
import time

import numpy as np
import torch

from common import ROOT, XGRID, MAX_FREQ, fnr_percent
from fr_variants import VARIANTS
from data import fr

SNRS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
CKPT = ROOT / "experiments" / "checkpoints"
OUT = ROOT / "experiments" / "results"


def median_err_bins(est, targets):
    errs = []
    for e, t in zip(est, targets):
        e, t = e[e >= -0.5], t[t >= -0.5]
        if len(e):
            errs.extend(np.abs(((t[:, None] - e[None, :]) + 0.5) % 1 - 0.5).min(axis=1) * 50)
    return float(np.median(errs))


def timing(model, device, sig):
    model = model.to(device).eval()
    x = torch.from_numpy(sig).float().to(device)
    sync = torch.mps.synchronize if device.type == "mps" else (lambda: None)
    with torch.no_grad():
        model(x[:16]); sync()
        t = time.perf_counter(); model(x); sync()
        batch = 1000 * (time.perf_counter() - t) / len(x)
        t = time.perf_counter()
        for i in range(200):
            model(x[i:i + 1])
        sync()
        single = 1000 * (time.perf_counter() - t) / 200
    return {"ms_per_signal_batch1000": batch, "ms_latency_batch1": single}


def main():
    targets = np.load(ROOT / "test_dataset" / "f.npy")
    counts = np.sum(targets >= -0.5, axis=1)
    sig10 = np.load(ROOT / "test_dataset" / "10.0dB.npy")
    devices = [torch.device("cpu")] + ([torch.device("mps")] if torch.backends.mps.is_available() else [])
    models = {}
    for name in VARIANTS:
        path = CKPT / f"{name}_long.pt"
        if path.exists():
            m = VARIANTS[name]()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            models[name + " (60k steps)"] = m
    pre = ROOT / "pretrained_models" / "DeepFreq" / "frequency_representation_module.pth"
    if pre.exists():
        m = VARIANTS["deepfreq20x64"]()
        m.load_state_dict(torch.load(pre, map_location="cpu", weights_only=False)["model"])
        models["DeepFreq pretrained (paper)"] = m

    res = {}
    for name, m in models.items():
        m.eval()
        r = {"params": sum(p.numel() for p in m.parameters()), "fnr_percent": {}, "median_err_bins": {}}
        for snr in SNRS:
            sig = np.load(ROOT / "test_dataset" / f"{float(snr)}dB.npy")
            with torch.no_grad():
                rep = m(torch.from_numpy(sig).float()).numpy()
            est = fr.find_freq(rep, counts, XGRID, MAX_FREQ)
            r["fnr_percent"][snr] = fnr_percent(est, targets)
            r["median_err_bins"][snr] = median_err_bins(est, targets)
        for d in devices:
            r[f"timing_{d.type}"] = timing(m, d, sig10)
        m.to("cpu")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "e5_long_runs.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

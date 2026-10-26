"""Experiment 3: train FR architecture variants from scratch on an equal budget.

Every variant sees the same pre-generated training signals, the same noise
model as DeepFreq (gaussian_blind, SNR >= 0 dB), the same loss (squared error
to the Gaussian-kernel target) and the same number of optimisation steps.
Usage: python experiments/3_fr_designs/e3_train_variants_cpu.py [variant ...]
"""

import json
import os
import sys
import time

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import ROOT, EXP, XGRID, MAX_FREQ, load_test_set, fnr_percent
from fr_variants import VARIANTS
from data import data, fr
from data.noise import noise_torch

STEPS = int(os.environ.get("STEPS", 4000))
TAG = os.environ.get("TAG", "")
BATCH = 256
N_TRAIN = 100_000
LR = 1e-3
OUT = EXP / "3_fr_designs" / "models"
CACHE = ROOT / "experiments" / "train_cache.npz"


def training_set():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["s"], d["fr"]
    np.random.seed(2026)
    s, f, _ = data.gen_signal(N_TRAIN, 50, MAX_FREQ, 1.0, distance="normal", amplitude="normal_floor",
                              floor_amplitude=0.1, variable_num_freq=True)
    target = fr.freq2fr(f, XGRID, "gaussian", 0.3 / 50).astype(np.float32)
    np.savez(CACHE, s=s, fr=target)
    return s, target


def quick_val(model):
    model.eval()
    res = {}
    for snr in (0, 10, 20, 30):
        sig, tgt, cnt = load_test_set(snr)
        with torch.no_grad():
            rep = model(torch.from_numpy(sig).float()).numpy()
        res[snr] = fnr_percent(fr.find_freq(rep, cnt, XGRID, MAX_FREQ), tgt)
    model.train()
    return res


def train(name):
    torch.manual_seed(0)
    s, target = training_set()
    s, target = torch.from_numpy(s), torch.from_numpy(target)
    model = VARIANTS[name]()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    gen = torch.Generator().manual_seed(1)
    log = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=gen)
        noisy = noise_torch(s[idx], 1.0, "gaussian_blind")
        loss = ((model(noisy) - target[idx]) ** 2).sum() / BATCH
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(500, STEPS // 10) == 0 or step == STEPS:
            val = quick_val(model)
            log.append({"step": step, "loss": loss.item(), "minutes": (time.time() - t0) / 60,
                        "fnr_by_snr": val})
            print(name, log[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / f"{name}{TAG}.pt")
    with open(OUT / f"{name}{TAG}_log.json", "w") as fh:
        json.dump(log, fh, indent=1)


if __name__ == "__main__":
    torch.set_num_threads(4)
    for n in sys.argv[1:] or list(VARIANTS):
        train(n)

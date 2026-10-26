"""Experiment 6 (part 1): build an improved DeepFreq from the fine-tuned model.

Stage "fr" (default): continue training the fine-tuned FR model
(checkpoint/finetune_experiment/fr/epoch_310.pth, same 20x64 architecture) with
the recipe that worked in E3: 100k cached training signals, gaussian_blind
noise, Gaussian target (std 0.3/50), Adam with a OneCycle schedule (lower peak
learning rate because we start from trained weights), batch 256.  The
checkpoint with the best mean oracle-count FNR at 0/10/20/30 dB is kept.

Stage "counter": retrain the E2 fusion counter (same features, MLP, data and
schedule) using the new FR model; the FC output still comes from the unchanged
fine-tuned FC model.

Stage "counter_small": the same counter retraining for the small 8-layer,
32-filter FR design from E3/E5 (experiments/3_fr_designs/models/deepfreq8x32_long.pt),
i.e. the version of improvement 3 that changes the FR layers.

Usage (from the repository root):
  STEPS=30000 python experiments/4_combined/e6_improved_finetune.py fr
  python experiments/4_combined/e6_improved_finetune.py counter
  python experiments/4_combined/e6_improved_finetune.py counter_small
  python experiments/4_combined/e6_improved_finetune.py probe     # time 200 steps only
Uses CUDA or Apple MPS automatically when available.  Nothing in
checkpoint/finetune_experiment/ or the other experiment folders is modified.
"""

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import ROOT, EXP, FINETUNED
from e3_train_variants import DEVICE, N_TRAIN, BATCH, training_set, quick_val
from e2_counting import Counter, features, make_split
from fr_variants import VARIANTS
from data.noise import noise_torch
import util

STEPS = int(os.environ.get("STEPS", 30000))
MAX_LR = 3e-4
OUT = EXP / "4_combined" / "models"
FR_PATH = OUT / "fr" / "best.pth"
COUNTER_PATH = OUT / "fusion_counter.pt"
SMALL_FR_PATH = EXP / "3_fr_designs" / "models" / "deepfreq8x32_long.pt"
SMALL_COUNTER_PATH = OUT / "small_fr_fusion_counter.pt"


def load_small_fr():
    model = VARIANTS["deepfreq8x32"]()
    model.load_state_dict(torch.load(SMALL_FR_PATH, map_location="cpu"))
    return model.eval()


def train_fr(steps, save=True):
    torch.manual_seed(0)
    original = torch.load(FINETUNED[0], map_location="cpu", weights_only=False)
    model = util.load(str(FINETUNED[0]), "fr", torch.device("cpu"))[0].to(DEVICE).train()
    s, target = training_set()
    s, target = torch.from_numpy(s).to(DEVICE), torch.from_numpy(target).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(1)
    every = max(1, steps // 10)
    log = []
    if save:
        val = quick_val(model)
        log.append({"step": 0, "minutes": 0.0, "fnr_by_snr": val, "mean_fnr": float(np.mean(list(val.values())))})
        print(log[-1], flush=True)
    best, best_state, best_step = np.inf, None, None
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=gen).to(DEVICE)
        noisy = noise_torch(s[idx], 1.0, "gaussian_blind")
        loss = ((model(noisy) - target[idx]) ** 2).sum() / BATCH
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if save and (step % every == 0 or step == steps):
            val = quick_val(model)
            mean = float(np.mean(list(val.values())))
            log.append({"step": step, "loss": loss.item(), "minutes": (time.time() - t0) / 60,
                        "fnr_by_snr": val, "mean_fnr": mean})
            print(log[-1], flush=True)
            if mean < best:
                best, best_step = mean, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    minutes = (time.time() - t0) / 60
    if not save:
        return minutes
    ckpt = copy.copy(original)
    ckpt["model"] = best_state
    ckpt["epoch"] = f"{original['epoch']}+{best_step}steps"
    ckpt["args"] = copy.copy(original["args"])
    ckpt["args"].output_dir = str(OUT.relative_to(ROOT))
    (OUT / "fr").mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, FR_PATH)
    with open(OUT / "fr" / "train_log.json", "w") as fh:
        json.dump({"steps": steps, "max_lr": MAX_LR, "device": DEVICE.type, "train_minutes": minutes,
                   "best_step": best_step, "best_mean_fnr": best, "log": log}, fh, indent=1, default=float)
    print("saved", FR_PATH, "best step", best_step, "mean FNR", best, "minutes", minutes, flush=True)


def train_counter(fr_model, out_path):
    """Same as the fusion-counter part of e2_counting.main, with another FR."""
    torch.set_num_threads(2)
    fc_model = util.load(str(FINETUNED[1]), "fc", torch.device("cpu"))[0].eval()
    tr_s, tr_rep, tr_raw, tr_n = make_split(60_000, 11, fr_model, fc_model)
    va_s, va_rep, va_raw, va_n = make_split(3_000, 12, fr_model, fc_model)
    xtr, xva = features(tr_s, tr_rep, tr_raw), features(va_s, va_rep, va_raw)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6

    torch.manual_seed(0)
    net = Counter(xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt = torch.from_numpy((xtr - mu) / sd)
    yt = torch.from_numpy(tr_n - 1).long()
    xv = torch.from_numpy((xva - mu) / sd)
    best, best_state = 0, None
    for epoch in range(60):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = nn.functional.cross_entropy(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (net(xv).argmax(1).numpy() + 1 == va_n).mean()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0:
            print("mlp epoch", epoch, "val acc", acc, flush=True)
    torch.save({"state": best_state, "mu": mu, "sd": sd}, out_path)
    print("fusion val acc", best, "| fc val acc", (np.clip(np.rint(va_raw), 1, 10) == va_n).mean(), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    stage = sys.argv[1] if len(sys.argv) > 1 else "fr"
    if stage == "probe":
        training_set()
        m = train_fr(200, save=False)
        print(f"200 steps on {DEVICE.type}: {m:.2f} min -> {STEPS} steps ~ {m * STEPS / 200:.0f} min", flush=True)
    elif stage == "fr":
        train_fr(STEPS)
    elif stage == "counter":
        train_counter(util.load(str(FR_PATH), "fr", torch.device("cpu"))[0].eval(), COUNTER_PATH)
    elif stage == "counter_small":
        train_counter(load_small_fr(), SMALL_COUNTER_PATH)
    else:
        raise SystemExit("stage must be fr, counter, counter_small or probe")

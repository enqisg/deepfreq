"""Trainer for the symmetry-aware FR models (paper experiments).

Same training signals, noise model and optimiser as E3, with two additions:
  * a separate validation set (its own seed) for logging and checkpoint
    choice, so the saved test sets are only used for the final numbers;
  * a detection loss for the EquivariantDetector models (focal loss on a
    per-bin heatmap plus an L1 loss on the sub-bin offset).

Usage (from the repository root):
  python experiments/5_symmetry/sym_train.py MODEL [--steps 4000] [--seed 0] [--ntrain 100000]
                                  [--augment-shift] [--tag _x] [--device auto]
MODEL is any key of sym_models.FR_MODELS (trained with DeepFreq's loss) or
sym_models.DETECTORS (trained with the detection loss).
"""

import argparse
import json
import time

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
from common import ROOT, XGRID, MAX_FREQ, SIGNAL_DIM, fnr_percent, chamfer
from sym_models import FR_MODELS, DETECTORS
from data import data, fr
from data.noise import noise_torch

OUT = ROOT / "experiments" / "5_symmetry" / "models"
CACHE = ROOT / "experiments" / "train_cache_f.npz"
VAL_CACHE = ROOT / "experiments" / "val_cache.npz"
VAL_SNRS = (0, 10, 20, 30)


def training_set():
    """The E3 training signals (seed 2026) plus their true frequencies."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d["s"], d["f"]
    np.random.seed(2026)
    s, f, _ = data.gen_signal(100_000, SIGNAL_DIM, MAX_FREQ, 1.0, distance="normal",
                              amplitude="normal_floor", floor_amplitude=0.1, variable_num_freq=True)
    np.savez(CACHE, s=s, f=f)
    return s, f


def validation_set():
    """1000 clean signals per SNR with a fixed seed, noise added once."""
    if VAL_CACHE.exists():
        d = np.load(VAL_CACHE)
        return {snr: d[f"s{snr}"] for snr in VAL_SNRS}, d["f"]
    np.random.seed(777)
    torch.manual_seed(777)
    s, f, _ = data.gen_signal(1000, SIGNAL_DIM, MAX_FREQ, 1.0, distance="normal",
                              amplitude="normal_floor", floor_amplitude=0.1, variable_num_freq=True)
    noisy = {snr: noise_torch(torch.from_numpy(s), 10 ** (snr / 10), "gaussian").numpy() for snr in VAL_SNRS}
    np.savez(VAL_CACHE, f=f, **{f"s{snr}": v for snr, v in noisy.items()})
    return noisy, f


def gaussian_target(f, sigma=0.3 / SIGNAL_DIM):
    """DeepFreq's Gaussian-kernel target, in torch, on the 1000-point grid."""
    grid = torch.as_tensor(XGRID, dtype=torch.float32, device=f.device)
    d = (grid[None, None, :] - f[:, :, None] + 0.5) % 1 - 0.5
    t = torch.exp(-d ** 2 / sigma ** 2) * (f[:, :, None] > -1).float()
    return t.sum(1)


def detection_targets(f, m, sigma_bins=1.0):
    """Heatmap (Gaussian bumps, 1 at the nearest bin) and offsets on an m-point grid."""
    valid = f > -1
    pos = (f + 0.5) * m                      # continuous bin coordinate
    centre = torch.round(pos).long() % m
    offset = pos - torch.round(pos)          # in [-0.5, 0.5]
    grid = torch.arange(m, device=f.device).float()
    d = (grid[None, None, :] - centre[:, :, None].float() + m / 2) % m - m / 2
    heat = torch.exp(-d ** 2 / (2 * sigma_bins ** 2)) * valid[:, :, None].float()
    heat = heat.max(1).values
    off = torch.zeros(f.shape[0], m, device=f.device)
    mask = torch.zeros(f.shape[0], m, device=f.device)
    b = torch.arange(f.shape[0], device=f.device)[:, None].expand_as(centre)
    off[b[valid], centre[valid]] = offset[valid]
    mask[b[valid], centre[valid]] = 1.0
    return heat, off, mask


def focal_loss(logits, heat, alpha=2.0, beta=4.0):
    """CenterNet's penalty-reduced focal loss, normalised by the number of frequencies."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos = heat.ge(0.999).float()
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - heat) ** beta * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def decode(logits, offsets, counts=None, threshold=0.5, max_freq=MAX_FREQ):
    """Peaks of the heatmap (3-bin local maxima) -> frequencies.

    counts given: take the top-k peaks (oracle count); otherwise every peak
    whose probability is above the threshold."""
    m = logits.shape[-1]
    p = torch.sigmoid(logits)
    is_peak = (p >= torch.roll(p, 1, -1)) & (p > torch.roll(p, -1, -1))
    score = torch.where(is_peak, p, torch.zeros_like(p))
    top, idx = score.topk(max_freq, dim=-1)
    freq = (idx.float() + torch.gather(offsets, -1, idx)) / m - 0.5
    freq = (freq + 0.5) % 1 - 0.5
    if counts is None:
        keep = top > threshold
        keep[:, 0] = True                      # at least one frequency
    else:
        keep = torch.arange(max_freq, device=logits.device)[None, :] < torch.as_tensor(
            counts, device=logits.device)[:, None].clamp(min=1)
    out = torch.where(keep, freq, torch.full_like(freq, -1.0)).cpu().numpy()
    res = -np.ones_like(out)
    for i, row in enumerate(out):
        v = row[row >= -0.5]
        res[i, :len(v)] = np.sort(v)
    return res


@torch.no_grad()
def evaluate(model, signals, f, device, detector, threshold=0.5, batch=500):
    """FNR with the true count; for detectors also count accuracy and Chamfer."""
    model.eval()
    counts = np.sum(f >= -0.5, axis=1)
    outs = []
    for i in range(0, len(signals), batch):
        outs.append(model(torch.from_numpy(signals[i:i + batch]).float().to(device)))
    model.train()
    if not detector:
        rep = torch.cat(outs).cpu().numpy()
        return {"fnr": fnr_percent(fr.find_freq(rep, counts, XGRID, MAX_FREQ), f)}
    logits = torch.cat([o[0] for o in outs])
    offs = torch.cat([o[1] for o in outs])
    est_oracle = decode(logits, offs, counts)
    est_full = decode(logits, offs, None, threshold)
    k_hat = np.sum(est_full >= -0.5, axis=1)
    return {"fnr": fnr_percent(est_oracle, f), "count_acc": float(100 * np.mean(k_hat == counts)),
            "chamfer": chamfer(est_full, f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment-shift", action="store_true",
                    help="multiply each training signal by a random tone (frequency-shift augmentation)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    detector = args.model in DETECTORS
    torch.manual_seed(args.seed)
    model = (DETECTORS if detector else FR_MODELS)[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    s, f = training_set()
    s, f = torch.from_numpy(s[:args.ntrain]).to(device), torch.from_numpy(f[:args.ntrain]).to(device)
    val_s, val_f = validation_set()
    tvec = torch.arange(SIGNAL_DIM, device=device).float()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(args.seed + 1)
    name = f"{args.model}{args.tag}_s{args.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "params": n_params, "device": str(device), "evals": []}
    best = None
    t0 = time.time()
    every = max(250, args.steps // 20)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(s), (args.batch,), generator=gen).to(device)
        sig, fb = s[idx], f[idx]
        if args.augment_shift:
            f0 = torch.rand(args.batch, 1, device=device) - 0.5
            rot = torch.exp(2j * np.pi * f0 * tvec[None])
            z = torch.complex(sig[:, 0], sig[:, 1]) * rot
            sig = torch.stack([z.real, z.imag], 1)
            fb = torch.where(fb > -1, (fb + f0 + 0.5) % 1 - 0.5, fb)
        noisy = noise_torch(sig, 1.0, "gaussian_blind")
        if detector:
            logits, offs = model(noisy)
            heat, off_t, mask = detection_targets(fb, model.m)
            loss = focal_loss(logits, heat) + 5.0 * ((offs - off_t).abs() * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = ((model(noisy) - gaussian_target(fb)) ** 2).sum() / args.batch
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % every == 0 or step == args.steps:
            res = {snr: evaluate(model, val_s[snr], val_f, device, detector) for snr in VAL_SNRS}
            score = np.mean([r["fnr"] for r in res.values()])
            log["evals"].append({"step": step, "loss": loss.item(), "minutes": (time.time() - t0) / 60,
                                 "val": res, "mean_val_fnr": score})
            print(name, step, round(loss.item(), 3), round(score, 2),
                  {k: {a: round(b, 3) for a, b in v.items()} for k, v in res.items()}, flush=True)
            if best is None or score < best:
                best = score
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / f"{name}.pt")
    log["minutes"] = (time.time() - t0) / 60
    log["best_mean_val_fnr"] = best
    with open(OUT / f"{name}_log.json", "w") as fh:
        json.dump(log, fh, indent=1)


if __name__ == "__main__":
    main()

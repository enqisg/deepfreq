"""Improved DeepFreq's new counter with a regression output instead of classification.

Same as e6_improved_finetune.train_counter (same 41 inputs: 25 log eigenvalues of
the raw signal, the 15 highest FR peaks and the FC model's raw guess; same
3-layer MLP 41 -> 128 -> 128; same data, Adam 1e-3, batch 256, 60 epochs), but
the last layer outputs one number trained with mean squared error on the true
count, and the count is that number rounded and limited to 1-10, like the
original FC model.  The checkpoint with the best validation counting accuracy
is kept.

  python experiments/6_final_comparison/regression_counter.py
"""

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py
from common import FINETUNED, MAX_FREQ
from e2_counting import features, make_split
from e6_improved_finetune import FR_PATH
import util

OUT_PATH = Path(__file__).resolve().parent / "models" / "fusion_counter_regression.pt"


class RegressionCounter(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def to_count(raw):
    return np.clip(np.rint(raw), 1, MAX_FREQ).astype(int)


def load_regression_counter(path=OUT_PATH):
    d = torch.load(path, map_location="cpu", weights_only=False)
    net = RegressionCounter(len(d["mu"]))
    net.load_state_dict(d["state"])
    return net.eval(), d["mu"], d["sd"]


def regression_counts(counter, signals, rep, raw):
    net, mu, sd = counter
    with torch.no_grad():
        return to_count(net(torch.from_numpy((features(signals, rep, raw) - mu) / sd)).numpy())


def main():
    torch.set_num_threads(4)
    cpu = torch.device("cpu")
    fr_model = util.load(str(FR_PATH), "fr", cpu)[0].eval()
    fc_model = util.load(str(FINETUNED[1]), "fc", cpu)[0].eval()
    tr_s, tr_rep, tr_raw, tr_n = make_split(60_000, 11, fr_model, fc_model)
    va_s, va_rep, va_raw, va_n = make_split(3_000, 12, fr_model, fc_model)
    xtr, xva = features(tr_s, tr_rep, tr_raw), features(va_s, va_rep, va_raw)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6

    torch.manual_seed(0)
    net = RegressionCounter(xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt = torch.from_numpy((xtr - mu) / sd)
    yt = torch.from_numpy(tr_n.astype(np.float32))
    xv = torch.from_numpy((xva - mu) / sd)
    best, best_state = 0, None
    for epoch in range(60):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = nn.functional.mse_loss(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (to_count(net(xv).numpy()) == va_n).mean()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0:
            print("mlp epoch", epoch, "val acc", acc, flush=True)
    OUT_PATH.parent.mkdir(exist_ok=True)
    torch.save({"state": best_state, "mu": mu, "sd": sd}, OUT_PATH)
    print("regression counter val acc", best, "| fc val acc", (to_count(va_raw) == va_n).mean(), flush=True)


if __name__ == "__main__":
    main()

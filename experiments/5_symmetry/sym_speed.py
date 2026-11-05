"""Run time per signal of each system on the same CPU, at 20 dB.

Networks: the FR model and FC model of DeepFreq (2019) and our detector with
its count head, in plain PyTorch, batch 1000 (throughput) and batch 1
(latency), with 4 threads and 1 thread.  Post-processing: DeepFreq's own peak
picking, and the clean-up (100-step joint NLS) plus likelihood check that both
learned systems get in the comparisons.  Classical methods: NOMP (CFAR stop),
ANM (ADMM, the iteration count used in the tests) and VALSE, single thread
and unbatched apart from ANM, which is batched over signals as in sym_anm.py.

Usage: python experiments/5_symmetry/sym_speed.py [CHECKPOINT MODEL]
       (default: models/eqdet16x64c_60k_s0.pt eqdet16x64c)
       -> 5_symmetry/results/speed.json
"""

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # experiments/: common.py
sys.path.insert(0, str(HERE))
from common import PRETRAINED, XGRID, MAX_FREQ, Timer, load_test_set, to_complex, load_deepfreq  # noqa: E402
from data import fr  # noqa: E402
import sym_eval_detector as E  # noqa: E402
from sym_models import DETECTORS  # noqa: E402
from sym_train import decode  # noqa: E402
from sym_baselines import nomp, _joint_refine  # noqa: E402
import sym_anm  # noqa: E402
import sym_valse  # noqa: E402

SNR = 20


def bench(fn, inputs, repeats=3):
    fn(inputs[:8])
    best = np.inf
    for _ in range(repeats):
        with Timer() as t:
            fn(inputs)
        best = min(best, t.elapsed)
    return 1000 * best / len(inputs)


def latency(fn, inputs, n=200):
    fn(inputs[:1])
    with Timer() as t:
        for i in range(n):
            fn(inputs[i:i + 1])
    return 1000 * t.elapsed / n


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "models" / "eqdet16x64c_60k_s0.pt"
    name = sys.argv[2] if len(sys.argv) > 2 else "eqdet16x64c"
    cfg = json.load(open(HERE / "results" / f"det_{ckpt.stem}.json"))
    det = DETECTORS[name]()
    det.load_state_dict(torch.load(ckpt, map_location="cpu"))
    det.eval()
    fr_model, fc_model = load_deepfreq(PRETRAINED)

    sig, tgt, cnt = load_test_set(SNR)
    sig = sig.astype(np.float32)
    sig_t = torch.from_numpy(sig)
    res = {"machine": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
           "torch": torch.__version__, "snr_db": SNR, "checkpoint": ckpt.name,
           "params": {"ours": sum(p.numel() for p in det.parameters()),
                      "deepfreq_fr": sum(p.numel() for p in fr_model.parameters()),
                      "deepfreq_fc": sum(p.numel() for p in fc_model.parameters())},
           "ms": {}}
    ms = res["ms"]

    @torch.no_grad()
    def ours_net(s):
        return det.forward_all(s)

    @torch.no_grad()
    def df_net(s):
        rep = fr_model(s)
        return rep, fc_model(rep)

    @torch.no_grad()
    def df_fr_only(s):
        return fr_model(s)

    for threads in (4, 1):
        torch.set_num_threads(threads)
        ms[f"ours network + count head, batch 1000, {threads} thr"] = bench(ours_net, sig_t)
        ms[f"DeepFreq FR+FC, batch 1000, {threads} thr"] = bench(df_net, sig_t)
        ms[f"DeepFreq FR only, batch 1000, {threads} thr"] = bench(df_fr_only, sig_t)
        ms[f"ours network + count head, batch 1, {threads} thr"] = latency(ours_net, sig_t)
        ms[f"DeepFreq FR+FC, batch 1, {threads} thr"] = latency(df_net, sig_t)

    torch.set_num_threads(1)
    with torch.no_grad():
        lo, of, head = det.forward_all(sig_t)
        rep, raw = df_net(sig_t)
    k_head = head.argmax(1).numpy() + 1
    k_df = np.clip(np.rint(raw.numpy().reshape(-1)), 1, MAX_FREQ).astype(int)
    rep = rep.numpy()
    n = 300
    ms["DeepFreq peak picking (its own count)"] = bench(
        lambda r: fr.find_freq(r, k_df[:len(r)], XGRID, MAX_FREQ), rep[:n])
    ms["ours decode (count head)"] = bench(lambda a: decode(a, of[:len(a)], k_head[:len(a)], cfg["threshold"]), lo[:n])
    est = decode(lo, of, k_head, cfg["threshold"])
    prop = decode(lo, of, None, cfg["hybrid_proposal_threshold"])
    ms["clean-up: joint NLS, 100 steps"] = bench(lambda s: E.refine(s, est[:len(s)]), sig[:n], 1)
    est_r = E.refine(sig, est)
    ms["likelihood check (add/drop, SNR-gated)"] = bench(
        lambda s: E.hybrid(s, est_r[:len(s)], cfg["hybrid_mult"], cfg["hybrid_gate_db"], prop[:len(s)]), sig[:n], 1)

    x = to_complex(sig)
    ms["NOMP (CFAR stop)"] = bench(lambda xx: nomp(xx, None, SNR), x[:100], 1)
    m = 50
    ms[f"ANM (ADMM, {sym_anm.iters_for(SNR)} it, batch {m})"] = bench(lambda xx: sym_anm.solve(xx, SNR, chunk=m), x[:m], 1)

    def valse_own(xx):
        for y in xx:
            f, a, _, _ = sym_valse.valse_one(y)
            f = f[np.argsort(-np.abs(a))][:MAX_FREQ]
            _joint_refine(y, f if len(f) else sym_valse._greedy_add(y, f, 1), 100)
    ms["VALSE (own count) + clean-up"] = bench(valse_own, x[:20], 1)

    for k, v in ms.items():
        print(f"{k:58s} {v:9.3f}", flush=True)
    with open(HERE / "results" / "speed.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

"""Experiment 4: how fast can the existing DeepFreq pipeline run on a CPU?

Times the fine-tuned FR+FC networks in plain PyTorch, then after freezing with
TorchScript (which folds BatchNorm into the convolutions), and with half the
CPU threads.  Also times the classical methods and the post-processing steps
on the same machine so the speed comparison is like for like.
"""

import json

import numpy as np
import torch

from common import (ROOT, FINETUNED, XGRID, MAX_FREQ, Timer, load_test_set, to_complex,
                    load_deepfreq, esprit, music_peaks, mdl_counts, interp_peaks, nls_refine)
from data import fr

OUT = ROOT / "experiments" / "results"


def bench(fn, inputs, repeats=3):
    fn(inputs[:8])
    best = np.inf
    for _ in range(repeats):
        with Timer() as t:
            fn(inputs)
        best = min(best, t.elapsed)
    return 1000 * best / len(inputs)


def latency(fn, inputs, n=300):
    fn(inputs[:1])
    with Timer() as t:
        for i in range(n):
            fn(inputs[i:i + 1])
    return 1000 * t.elapsed / n


def main():
    res = {}
    signals, targets, counts = load_test_set(10)
    x = to_complex(signals)
    sig_t = torch.from_numpy(signals).float()
    fr_model, fc_model = load_deepfreq(FINETUNED)

    def eager(s):
        with torch.no_grad():
            return fc_model(fr_model(s))

    example = sig_t[:4]
    with torch.no_grad():
        fr_frozen = torch.jit.freeze(torch.jit.trace(fr_model, example))
        fc_frozen = torch.jit.freeze(torch.jit.trace(fc_model, fr_model(example)))

    def frozen(s):
        with torch.no_grad():
            return fc_frozen(fr_frozen(s))

    with torch.no_grad():
        ref = eager(sig_t[:100])
        assert torch.allclose(ref, frozen(sig_t[:100]), atol=1e-3)

    for threads in (4, 1):
        torch.set_num_threads(threads)
        res[f"DeepFreq FR+FC eager, batch 1000, {threads} thr (ms/signal)"] = bench(eager, sig_t)
        res[f"DeepFreq FR+FC frozen, batch 1000, {threads} thr (ms/signal)"] = bench(frozen, sig_t)
        res[f"DeepFreq FR+FC eager, batch 1, {threads} thr (ms)"] = latency(eager, sig_t)
        res[f"DeepFreq FR+FC frozen, batch 1, {threads} thr (ms)"] = latency(frozen, sig_t)

    torch.set_num_threads(4)
    with torch.no_grad():
        rep = fr_model(sig_t).numpy()
    res["Peak picking on 1000-pt FR (ms/signal)"] = bench(lambda r: fr.find_freq(r, counts[:len(r)], XGRID, MAX_FREQ), rep)
    res["Peak picking + interpolation (ms/signal)"] = bench(lambda r: interp_peaks(r, counts[:len(r)]), rep)
    est = interp_peaks(rep, counts)
    res["NLS refine 10 it (ms/signal)"] = bench(lambda xx: nls_refine(xx, est[:len(xx)]), x[:300], 1)
    res["ESPRIT, known K (ms/signal)"] = bench(lambda xx: esprit(xx, counts[:len(xx)]), x[:300], 1)
    res["MUSIC 1000-pt grid, known K (ms/signal)"] = bench(lambda xx: music_peaks(xx, counts[:len(xx)]), x[:300], 1)
    res["MDL count (ms/signal)"] = bench(mdl_counts, x[:300], 1)

    for k, v in res.items():
        print(f"{k:60s} {v:8.3f}")
    with open(OUT / "e4_inference_speed.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()

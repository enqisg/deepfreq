"""Symmetry-aware FR models for the paper (improvement 5).

* MultitaperLift: learned windows + one FFT, exactly equivariant to
  frequency shifts and invariant to the global phase.
* EquivariantFR: that lift + circular residual convolutions, trained with
  DeepFreq's loss (drop-in FR model, [B,2,50] -> [B,1000]).
* EquivariantDetector: the same backbone with a detection head (per-bin
  probability + sub-bin offset), so one network locates and counts.
* LinearLift: DeepFreq's learned-matrix front-end, for ablations.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, fr_variants.py
import common  # noqa: E402,F401  (puts the repository root on the path)
from fr_variants import ResBlock, deepfreq_fr  # noqa: E402

class MultitaperLift(nn.Module):
    """Learned multitaper front-end.

    C learned complex windows w_c (initialised as DPSS tapers) give the
    windowed spectra Z_c(f) = sum_t w_c[t] x[t] exp(-2i pi f t) on an M-point
    grid (one FFT).  Modulating the signal by exp(2i pi f0 t) shifts every
    Z_c by f0, so the lift is exactly equivariant to frequency shifts.  The
    output features use only Z_c conj(Z_d), which removes the global phase,
    and are rescaled by sqrt(|Z_c Z_d|) so they grow linearly with amplitude.
    """

    def __init__(self, n=50, tapers=8, m=250):
        super().__init__()
        import scipy.signal.windows
        w = scipy.signal.windows.dpss(n, max(tapers / 2, 1.0), tapers) * (n ** 0.5)
        self.wr = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.wi = nn.Parameter(torch.zeros(tapers, n))
        self.m = m
        iu = torch.triu_indices(tapers, tapers, 1)
        self.register_buffer("iu", iu)
        self.out_ch = tapers + 2 * iu.shape[1]

    def forward(self, inp):
        x = torch.complex(inp[:, 0], inp[:, 1])
        w = torch.complex(self.wr, self.wi)
        z = torch.fft.fftshift(torch.fft.fft(x[:, None, :] * w[None], n=self.m), dim=-1)
        z = z / x.shape[-1]
        mag = z.abs()
        cross = z[:, self.iu[0]] * z[:, self.iu[1]].conj()
        cross = cross / torch.sqrt(cross.abs() + 1e-6)
        return torch.cat([mag, cross.real, cross.imag], dim=1)


class EquivariantFR(nn.Module):
    """FR model built only from frequency-shift-equivariant parts: multitaper
    lift on an M-point grid, circular residual convolutions along frequency,
    and a transposed convolution up to the 1000-point output grid."""

    def __init__(self, tapers=8, ch=64, n_blocks=8, dilations=(1, 2, 4, 8), m=250, out_size=1000,
                 kernel_out=25):
        super().__init__()
        self.lift = MultitaperLift(50, tapers, m)
        up = out_size // m
        self.inp = nn.Sequential(nn.Conv1d(self.lift.out_ch, ch, 1), nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch, dilations[i % len(dilations)]) for i in range(n_blocks)])
        self.out_layer = nn.ConvTranspose1d(ch, 1, kernel_out, stride=up, bias=False)
        self.out_size = out_size
        self.shift = (kernel_out - 1) // 2

    def forward(self, inp):
        y = self.out_layer(self.blocks(self.inp(self.lift(inp)))).squeeze(1)
        # lift bin j sits at output up*j + shift; move it to up*j and wrap the
        # tails around so the output stays periodic in frequency
        n = self.out_size
        y = torch.nn.functional.pad(y, (0, 2 * n - y.shape[-1]))
        y = torch.roll(y, -self.shift, dims=-1)
        return y.view(y.shape[0], 2, n).sum(1)


FR_MODELS = {
    "deepfreq20x64": lambda: deepfreq_fr(20, 64),
    "deepfreq8x32": lambda: deepfreq_fr(8, 32),
    "eqfr8x64": lambda: EquivariantFR(8, 64, 8),
    "eqfr16x64": lambda: EquivariantFR(8, 64, 16),
}


class LinearLift(nn.Module):
    """DeepFreq-style front-end for ablations: a general learned matrix from
    the 100 input numbers to ch x M features (no built-in symmetry)."""

    def __init__(self, n=50, ch=32, m=250):
        super().__init__()
        self.lin = nn.Linear(2 * n, ch * m, bias=False)
        self.out_ch, self.m = ch, m

    def forward(self, inp):
        return self.lin(inp.reshape(inp.shape[0], -1)).view(inp.shape[0], self.out_ch, self.m)


class EquivariantDetector(nn.Module):
    """Symmetry-aware FR model with a detection head instead of a spectrum.

    For every bin of the M-point grid it predicts the logit that a frequency
    lies in that bin and the sub-bin offset of that frequency, so locating and
    counting come from one network (count = number of confident peaks).
    """

    def __init__(self, tapers=8, ch=64, n_blocks=8, dilations=(1, 2, 4, 8), m=250, lift="multitaper",
                 learn_tapers=True):
        super().__init__()
        self.lift = MultitaperLift(50, tapers, m) if lift == "multitaper" else LinearLift(50, 32, m)
        if not learn_tapers:
            self.lift.wr.requires_grad_(False)
            self.lift.wi.requires_grad_(False)
        self.m = m
        self.inp = nn.Sequential(nn.Conv1d(self.lift.out_ch, ch, 1), nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch, dilations[i % len(dilations)]) for i in range(n_blocks)])
        self.head = nn.Conv1d(ch, 2, 1)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[0] = -4.0  # start with "no frequency here" (focal-loss prior)

    def forward(self, inp):
        out = self.head(self.blocks(self.inp(self.lift(inp))))
        return out[:, 0], torch.tanh(out[:, 1]) * 0.5   # logits [B,M], offsets in bins [B,M]


DETECTORS = {
    "eqdet8x64": lambda: EquivariantDetector(8, 64, 8),
    "eqdet16x64": lambda: EquivariantDetector(8, 64, 16),
    "eqdet16x64m500": lambda: EquivariantDetector(8, 64, 16, m=500),
    # ablations
    "lindet8x64": lambda: EquivariantDetector(8, 64, 8, lift="linear"),
    "eqdet8x64fixed": lambda: EquivariantDetector(8, 64, 8, learn_tapers=False),
}

"""Candidate frequency-representation (FR) architectures.

All share DeepFreq's interface: input [B, 2, 50] (real/imag), output [B, 1000]
pseudo-spectrum on the same grid, so they plug into the existing pipeline.
"""

import torch
import torch.nn as nn

import modules


def deepfreq_fr(n_layers=20, n_filters=64):
    """The original DeepFreq FR module (paper configuration by default)."""
    return modules.FrequencyRepresentationModule(signal_dim=50, n_filters=n_filters, inner_dim=125,
                                                 n_layers=n_layers, upsampling=8, kernel_size=3,
                                                 kernel_out=25)


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False, padding_mode="circular"),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False, padding_mode="circular"),
            nn.BatchNorm1d(ch))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.body(x))


class ResFreqFR(nn.Module):
    """DeepFreq with residual blocks (the idea behind cResFreq), optionally
    dilated so that fewer layers still see the whole spectrum."""

    def __init__(self, n_blocks=10, ch=64, dilations=(1,), inner_dim=125, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.ch = ch
        self.in_layer = nn.Linear(100, inner_dim * ch, bias=False)
        self.blocks = nn.Sequential(*[ResBlock(ch, dilations[i % len(dilations)]) for i in range(n_blocks)])
        self.out_layer = nn.ConvTranspose1d(ch, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1,
                                            bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        x = self.in_layer(inp.reshape(bsz, -1)).view(bsz, self.ch, -1)
        x = self.blocks(x)
        return self.out_layer(x).view(bsz, -1)


class AttnFR(nn.Module):
    """Light transformer variant (the idea behind SwinFreq): a learned linear
    'matched filter' to 125 frequency tokens, then a few self-attention layers
    so every frequency bin can see every other one in one step."""

    def __init__(self, n_layers=2, ch=64, heads=4, inner_dim=125, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.ch = ch
        self.in_layer = nn.Linear(100, inner_dim * ch, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, inner_dim, ch))
        layer = nn.TransformerEncoderLayer(ch, heads, dim_feedforward=2 * ch, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out_layer = nn.ConvTranspose1d(ch, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1,
                                            bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        x = self.in_layer(inp.reshape(bsz, -1)).view(bsz, self.ch, -1).transpose(1, 2)
        x = self.encoder(x + self.pos).transpose(1, 2)
        return self.out_layer(x).view(bsz, -1)


VARIANTS = {
    "deepfreq20x64": lambda: deepfreq_fr(20, 64),
    "deepfreq8x32": lambda: deepfreq_fr(8, 32),
    "resfreq10x64": lambda: ResFreqFR(10, 64),
    "dilated4x32": lambda: ResFreqFR(4, 32, dilations=(1, 2, 4, 8)),
    "attn2x64": lambda: AttnFR(2, 64),
}

"""RMVPE vocal pitch estimator (Wei et al., InterSpeech 2023).

Network definition adapted from Retrieval-based-Voice-Conversion-WebUI
(infer/rmvpe.py, MIT licence) so the official rmvpe.pt weights load as-is.
Inference is rewritten to run the song as a batch of overlapping chunks and to
decode salience -> Hz with vectorised numpy instead of a per-frame loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as mel_filters

SR = 16000
HOP = 160  # 10 ms frames -> 100 fps
N_FFT = 1024
LOG_FLOOR = float(np.log(1e-5))


class ConvBlockRes(nn.Module):
    def __init__(self, in_ch, out_ch, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, (1, 1))

    def forward(self, x):
        skip = self.shortcut(x) if hasattr(self, "shortcut") else x
        return self.conv(x) + skip


class ResEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_ch, out_ch, momentum)])
        self.conv.extend(ConvBlockRes(out_ch, out_ch, momentum) for _ in range(n_blocks - 1))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, in_size, n_encoders, kernel_size, n_blocks, out_ch=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(ResEncoderBlock(in_ch, out_ch, kernel_size, n_blocks, momentum))
            in_ch, out_ch = out_ch, out_ch * 2
        self.out_channel = out_ch

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            skips.append(t)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_ch, out_ch, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_ch, out_ch, None, n_blocks, momentum)])
        self.layers.extend(
            ResEncoderBlock(out_ch, out_ch, None, n_blocks, momentum) for _ in range(n_inters - 1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride, 1, output_padding=out_padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_ch * 2, out_ch, momentum)])
        self.conv2.extend(ConvBlockRes(out_ch, out_ch, momentum) for _ in range(n_blocks - 1))

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            self.layers.append(ResDecoderBlock(in_ch, in_ch // 2, stride, n_blocks, momentum))
            in_ch //= 2

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_ch=1, en_out_ch=16):
        super().__init__()
        self.encoder = Encoder(in_ch, 128, en_de_layers, kernel_size, n_blocks, en_out_ch)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2), en_out_ch=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_out_ch=en_out_ch)
        self.cnn = nn.Conv2d(en_out_ch, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid()
        )

    def forward(self, mel):  # mel: (batch, 128, frames)
        x = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(x)).transpose(1, 2).flatten(-2)
        return self.fc(x)  # (batch, frames, 360)


class RMVPE:
    CENTS = 20 * np.arange(360) + 1997.3794084376191

    def __init__(self, weights_path, device="cpu"):
        self.device = device
        self.model = E2E()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval().to(device)
        basis = mel_filters(sr=SR, n_fft=N_FFT, n_mels=128, fmin=30, fmax=8000, htk=True)
        self.mel_basis = torch.from_numpy(basis).float()
        self.window = torch.hann_window(N_FFT)

    def mel(self, audio: np.ndarray) -> torch.Tensor:
        spec = torch.stft(torch.from_numpy(audio).float(), N_FFT, HOP, N_FFT,
                          window=self.window, center=True, return_complex=True).abs()
        return torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))  # (128, frames)

    @torch.inference_mode()
    def salience(self, audio: np.ndarray, chunk=2048, context=128, batch=4) -> np.ndarray:
        """Run the network over overlapping chunks, batched, and stitch the centres."""
        mel = self.mel(audio)
        n = mel.shape[-1]
        win = chunk + 2 * context  # multiple of 32, as the U-Net requires
        padded = F.pad(mel, (context, chunk + context), value=LOG_FLOOR)
        windows = padded.unfold(-1, win, chunk).permute(1, 0, 2)  # (n_chunks, 128, win)
        out = []
        for i in range(0, windows.shape[0], batch):
            hidden = self.model(windows[i:i + batch].to(self.device))
            out.append(hidden[:, context:context + chunk].float().cpu())
        return torch.cat(out).reshape(-1, 360)[:n].numpy()

    def decode(self, sal: np.ndarray):
        """Salience (frames, 360) -> (f0 Hz, confidence). Local weighted average around the peak."""
        center = sal.argmax(1)
        idx = np.clip(center[:, None] + np.arange(-4, 5), 0, 359)
        w = np.take_along_axis(sal, idx, 1)
        cents = (w * self.CENTS[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
        return 10 * 2 ** (cents / 1200), sal.max(1)

    def __call__(self, audio: np.ndarray):
        return self.decode(self.salience(audio))

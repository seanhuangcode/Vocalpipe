"""YAMNet in PyTorch: Google's AudioSet sound classifier (521 classes, Apache 2.0).

Among its classes are music genres (Pop, Hip hop, Rock, Jazz, Classical, Electronic,
Country, Reggae, ...) and kinds of voice (Singing, Rapping, Speech, Choir, ...), which is
what genre.py uses. The network is a MobileNet v1 over 0.96 s log-mel patches, hopped every
0.48 s. This is a line-for-line port of the reference (tensorflow/models,
research/audioset/yamnet) that loads Google's released weights, yamnet.h5 (15 MB), which is
downloaded on first use. Features match tf.signal: periodic Hann window, unpadded STFT,
HTK mel bands without the DC bin, and TensorFlow's asymmetric "same" padding.
"""

import hashlib
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
WEIGHTS_SHA256 = "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yamnet.h5"

SR = 16000
WINDOW, HOP, N_FFT = 400, 160, 512  # 25 ms frames every 10 ms
MEL_BANDS, MEL_LO, MEL_HI, LOG_OFFSET = 64, 125.0, 7500.0, 0.001
PATCH, PATCH_HOP = 96, 48  # 0.96 s patches every 0.48 s
PATCH_SECONDS = PATCH_HOP * HOP / SR  # time between patches
# (kernel, stride, filters) of the 14 layers; the first is a plain conv, the rest depthwise-separable
LAYERS = [(3, 2, 32), (3, 1, 64), (3, 2, 128), (3, 1, 128), (3, 2, 256), (3, 1, 256), (3, 2, 512),
          (3, 1, 512), (3, 1, 512), (3, 1, 512), (3, 1, 512), (3, 1, 512), (3, 2, 1024), (3, 1, 1024)]


def mel_matrix() -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix(64, 257, 16000, 125, 7500): (257, 64)."""
    mel = lambda f: 1127.0 * np.log1p(np.asarray(f, float) / 700.0)  # noqa: E731
    hz = np.linspace(0.0, SR / 2, N_FFT // 2 + 1)[1:]
    edges = np.linspace(mel(MEL_LO), mel(MEL_HI), MEL_BANDS + 2)
    m = mel(hz)[:, None]
    lower, centre, upper = edges[:-2], edges[1:-1], edges[2:]
    w = np.maximum(0.0, np.minimum((m - lower) / (centre - lower), (upper - m) / (upper - centre)))
    return np.pad(w, ((1, 0), (0, 0)))


def log_mel_patches(audio: np.ndarray) -> np.ndarray:
    """16 kHz mono audio -> (n_patches, 96, 64) log-mel patches, padded like the reference."""
    audio = np.asarray(audio, np.float32)
    first = int((PATCH * HOP + WINDOW - HOP))  # samples for one patch: 0.975 s
    hop = PATCH_HOP * HOP
    n = max(len(audio), first)
    n = first + int(np.ceil((n - first) / hop)) * hop
    audio = np.pad(audio, (0, n - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, WINDOW)[::HOP]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / WINDOW)  # periodic Hann
    spec = np.abs(np.fft.rfft(frames * window, n=N_FFT))
    log_mel = np.log(spec @ mel_matrix() + LOG_OFFSET).astype(np.float32)
    return np.lib.stride_tricks.sliding_window_view(log_mel, PATCH, axis=0)[::PATCH_HOP].transpose(0, 2, 1).copy()


def _same_pad(x, stride, kernel=3):
    """TensorFlow 'same' padding: any odd padding goes after, not before."""
    pads = []
    for size in (x.shape[3], x.shape[2]):
        total = max((int(np.ceil(size / stride)) - 1) * stride + kernel - size, 0)
        pads += [total // 2, total - total // 2]
    return F.pad(x, pads)


class _BatchNorm(nn.Module):
    """Inference batch norm with a shift but no scale (YAMNet's batchnorm_scale=False)."""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("beta", torch.zeros(channels))
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("var", torch.ones(channels))

    def forward(self, x):
        return (x - self.mean[:, None, None]) / torch.sqrt(self.var[:, None, None] + 1e-4) + self.beta[:, None, None]


class _Layer(nn.Module):
    def __init__(self, cin, cout, stride, separable):
        super().__init__()
        self.stride, self.separable = stride, separable
        if separable:
            self.depthwise = nn.Conv2d(cin, cin, 3, stride, groups=cin, bias=False)
            self.bn_dw = _BatchNorm(cin)
            self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn_pw = _BatchNorm(cout)
        else:
            self.conv = nn.Conv2d(cin, cout, 3, stride, bias=False)
            self.bn = _BatchNorm(cout)

    def forward(self, x):
        x = _same_pad(x, self.stride)
        if not self.separable:
            return F.relu(self.bn(self.conv(x)))
        x = F.relu(self.bn_dw(self.depthwise(x)))
        return F.relu(self.bn_pw(self.pointwise(x)))


class YAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, cin = [], 1
        for i, (_, stride, cout) in enumerate(LAYERS):
            layers.append(_Layer(cin, cout, stride, separable=i > 0))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.logits = nn.Linear(1024, 521)

    def forward(self, patches):  # (n, 96, 64) log-mel -> (n, 521) scores, (n, 1024) embeddings
        x = self.layers(patches[:, None])
        emb = x.mean((2, 3))
        return torch.sigmoid(self.logits(emb)), emb

    def load_h5(self, path):
        import h5py

        with h5py.File(path, "r") as f:
            def tensor(group, leaf):  # Keras nests each layer's weights as group/group/leaf
                return torch.from_numpy(np.array(f[f"{group}/{group}/{leaf}"]))

            def bn(module, group):
                module.beta.copy_(tensor(group, "beta:0"))
                module.mean.copy_(tensor(group, "moving_mean:0"))
                module.var.copy_(tensor(group, "moving_variance:0"))

            for i, layer in enumerate(self.layers, 1):
                if not layer.separable:
                    layer.conv.weight.data.copy_(tensor(f"layer{i}/conv", "kernel:0").permute(3, 2, 0, 1))
                    bn(layer.bn, f"layer{i}/conv/bn")
                else:
                    layer.depthwise.weight.data.copy_(tensor(f"layer{i}/depthwise_conv", "depthwise_kernel:0").permute(2, 3, 0, 1))
                    bn(layer.bn_dw, f"layer{i}/depthwise_conv/bn")
                    layer.pointwise.weight.data.copy_(tensor(f"layer{i}/pointwise_conv", "kernel:0").permute(3, 2, 0, 1))
                    bn(layer.bn_pw, f"layer{i}/pointwise_conv/bn")
            self.logits.weight.data.copy_(tensor("logits", "kernel:0").T)
            self.logits.bias.data.copy_(tensor("logits", "bias:0"))
        return self


def download_weights(path=DEFAULT_WEIGHTS):
    """Fetch Google's yamnet.h5 (15 MB) once, checking it arrived intact."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(WEIGHTS_URL, timeout=60, context=ctx) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("yamnet.h5 download was corrupted")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


@lru_cache(maxsize=1)
def get_model(path=DEFAULT_WEIGHTS) -> YAMNet:
    return YAMNet().load_h5(download_weights(path)).eval()


@torch.inference_mode()
def classify(audio16k: np.ndarray, model: YAMNet | None = None, batch=64):
    """Per-patch class scores (n_patches, 521) for 16 kHz mono audio; patch k covers
    k * 0.48 s .. k * 0.48 + 0.96 s."""
    model = model or get_model()
    patches = torch.from_numpy(log_mel_patches(audio16k))
    out = [model(patches[i:i + batch])[0] for i in range(0, len(patches), batch)]
    return torch.cat(out).numpy() if out else np.zeros((0, 521), np.float32)

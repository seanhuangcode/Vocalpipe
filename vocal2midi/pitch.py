import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import rmvpe
from .audio import load_audio
from .onsets import audio_features

SR = rmvpe.SR
HOP = rmvpe.HOP
FPS = SR / HOP  # 100 frames per second

DEFAULT_RMVPE = Path(__file__).resolve().parent.parent / "models" / "rmvpe.pt"


def frame_rms_db(audio: np.ndarray, n_frames: int) -> np.ndarray:
    """Loudness per 10 ms frame, aligned with the (centred) pitch frames."""
    padded = np.pad(audio, rmvpe.N_FFT // 2, mode="reflect")
    frames = sliding_window_view(padded, rmvpe.N_FFT)[::HOP][:n_frames]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    out = np.full(n_frames, -120.0)
    out[: len(rms)] = 20 * np.log10(rms + 1e-9)
    return out


@lru_cache(maxsize=2)
def _rmvpe(weights: str, device: str):
    return rmvpe.RMVPE(weights, device)


class FCPE:
    """FCPE (Fast Context-based Pitch Estimation, `pip install torchfcpe`): a lighter singing-voice
    pitch model whose weights ship with the package, so nothing to download. It predicts the same
    kind of 360-bin pitch salience as RMVPE and is decoded the same way."""

    def __init__(self, device="cpu"):
        from torchfcpe import spawn_bundled_infer_model

        self.net = spawn_bundled_infer_model(device=device)
        self.cents = self.net.model.cent_table.cpu().numpy()
        self.device = device

    def __call__(self, audio: np.ndarray):
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(audio).float()[None, :, None].to(self.device)
            sal = self.net.model(self.net.wav2mel(x, SR))[0].float().cpu().numpy()
        idx = np.clip(sal.argmax(1)[:, None] + np.arange(-4, 5), 0, sal.shape[1] - 1)
        w = np.take_along_axis(sal, idx, 1)
        cents = (w * self.cents[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
        return 10 * 2 ** (cents / 1200), sal.max(1)


@lru_cache(maxsize=2)
def _fcpe(device: str):
    return FCPE(device)


def track_pitch(vocals_path, backend="rmvpe", device="cpu", weights=DEFAULT_RMVPE):
    """Returns dict with per-frame arrays: time, f0 (Hz), conf (0-1), rms_db."""
    t0 = time.time()
    audio = load_audio(vocals_path, SR)

    if backend == "fcpe":
        f0, conf = _fcpe(device)(audio)
    elif backend == "rmvpe":
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"RMVPE weights not found at {weights}. Download with:\n"
                "  curl -L -o models/rmvpe.pt "
                "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt\n"
                "or run with --pitch fcpe (no download) or --pitch crepe"
            )
        f0, conf = _rmvpe(str(weights), device)(audio)
    elif backend in ("crepe", "crepe-full"):
        import torch
        import torchcrepe

        f0, conf = torchcrepe.predict(
            torch.from_numpy(audio)[None], SR, HOP, 50, 1100,
            "full" if backend == "crepe-full" else "tiny",
            batch_size=2048, device=device, return_periodicity=True,
            decoder=torchcrepe.decode.weighted_argmax,
        )
        f0, conf = f0[0].cpu().numpy(), conf[0].cpu().numpy()
    else:
        raise ValueError(f"unknown pitch backend {backend!r}")

    n = len(f0)
    flux, hf = audio_features(audio, n)
    print(f"  pitch ({backend}) in {time.time() - t0:.1f}s on {device}: {n} frames")
    return {
        "time": np.arange(n) / FPS,
        "f0": f0.astype(np.float32),
        "conf": conf.astype(np.float32),
        "rms_db": frame_rms_db(audio, n).astype(np.float32),
        "flux": flux,
        "hf": hf,
    }


def add_onset_features(track: dict, vocals_path) -> bool:
    """Add the note-onset model's audio cues to a track cached before they existed."""
    if "flux" in track and "hf" in track:
        return False
    track["flux"], track["hf"] = audio_features(load_audio(vocals_path, SR), len(track["f0"]))
    return True

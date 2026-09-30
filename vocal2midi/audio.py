import subprocess

import numpy as np
import soundfile as sf
import torch


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(path, sr: int, channels: int = 1, max_seconds: float | None = None) -> np.ndarray:
    """Decode any format ffmpeg understands, resampled to `sr` (optionally just the start).

    Returns float32 array shaped (samples,) for mono or (channels, samples).
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    if channels == 1:
        return audio.copy()
    return audio.reshape(-1, channels).T.copy()


def write_wav(path, audio: np.ndarray, sr: int):
    """Write (samples,) or (channels, samples) float audio as 16-bit PCM."""
    data = audio.T if audio.ndim == 2 else audio
    peak = np.abs(data).max() if data.size else 0
    if peak > 1:
        data = data / peak
    sf.write(str(path), data, sr, subtype="PCM_16")

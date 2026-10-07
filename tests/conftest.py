"""Shared helpers: synthetic voices and pitch tracks with known notes, no models needed."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import vocal2midi  # noqa: E402,F401  (sets MPS fallback before torch loads)

FPS = 100
SR = 16000


def voice(midi, level_db, sr=SR, fps=FPS):
    """A harmonic 'voice' following per-frame MIDI pitch (nan = silent) and loudness (dB)."""
    midi, level_db = np.asarray(midi, float), np.asarray(level_db, float)
    n = len(midi) * sr // fps
    t_fr, t = np.arange(len(midi)) / fps, np.arange(n) / sr
    f0 = 440 * 2 ** ((np.interp(t, t_fr, np.nan_to_num(midi, nan=60.0)) - 69) / 12)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    amp = np.interp(t, t_fr, np.where(np.isfinite(midi), 10 ** (level_db / 20), 0.0))
    y = amp * sum(np.sin(k * phase) / k for k in range(1, 9)) * 0.2
    return (y + 1e-4 * np.random.default_rng(0).standard_normal(n)).astype(np.float32)


def track(midi, level_db, fps=FPS):
    """The pitch track a good tracker would produce for voice(midi, level_db)."""
    from vocal2midi.onsets import audio_features
    from vocal2midi.pitch import frame_rms_db

    midi = np.asarray(midi, float)
    audio = voice(midi, level_db)
    n = len(midi)
    sung = np.isfinite(midi)
    f0 = np.where(sung, 440 * 2 ** ((np.nan_to_num(midi, nan=69.0) - 69) / 12), 0.0)
    conf = np.convolve(np.where(sung, 0.92, 0.02), np.ones(3) / 3, mode="same")
    flux, hf = audio_features(audio, n)
    return {"time": np.arange(n) / fps, "f0": f0.astype(np.float32), "conf": conf.astype(np.float32),
            "rms_db": frame_rms_db(audio, n).astype(np.float32), "flux": flux, "hf": hf}


def notes_curve(spec, fps=FPS, glide=0.0):
    """Per-frame MIDI curve from [(seconds, midi or None), ...] (None = silence)."""
    out = []
    for dur, m in spec:
        out += [np.nan if m is None else float(m)] * int(round(dur * fps))
    x = np.array(out)
    if glide:  # legato transitions between consecutive pitches
        k = max(1, int(glide * fps))
        sm = np.convolve(np.pad(np.nan_to_num(x, nan=0.0), k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]
        x = np.where(np.isfinite(x), sm, np.nan)
    return x

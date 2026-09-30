"""Learned note-onset detector for singing.

A gradient-boosted classifier looks at every 10 ms frame and predicts "a new note
starts here" from pitch movement (how far the pitch moves across the frame at
several time scales, and how steady it is either side), loudness and voicing
changes, and consonant cues (spectral flux, high-frequency energy). Notes are the
spans between predicted onsets; each gets the pitch the singer holds in its middle.

Trained on Vocadito (Bittner et al., 2021, CC BY 4.0): 40 solo-singing recordings
with expert note annotations. See tools/train_onsets.py and tools/benchmark_notes.py.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, median_filter

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "note_onsets.joblib"
LAGS = (1, 2, 3, 5, 8, 12)
FEATURE_VERSION = 1
VOICED_CONF = 0.4


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _shift(a, k):
    """a[t + k], zero-filled at the edges."""
    out = np.zeros_like(a)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def audio_features(audio16k: np.ndarray, n_frames: int):
    """Consonant cues per 10 ms frame: onset strength (spectral flux) and >4 kHz energy share."""
    import librosa

    flux = librosa.onset.onset_strength(y=audio16k, sr=16000, hop_length=160, n_mels=64, center=True)
    power = np.abs(librosa.stft(audio16k, n_fft=1024, hop_length=160, center=True)) ** 2
    hf = power[256:].sum(0) / (power.sum(0) + 1e-10)

    def fit(a):
        return np.pad(a, (0, max(0, n_frames - len(a))))[:n_frames].astype(np.float32)

    flux = fit(flux)
    return flux / (np.percentile(flux, 95) + 1e-9), fit(hf)


def pitch_curve(track):
    """(glitch-free MIDI pitch, voiced mask) as used for features and note pitches."""
    f0, conf = track["f0"], track["conf"]
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440), np.nan)
    voiced = (conf > VOICED_CONF) & np.isfinite(midi)
    clean = midi.copy()
    for s, e in _runs(voiced):
        clean[s:e] = median_filter(midi[s:e], size=min(5, (e - s) | 1), mode="nearest")
    return clean, voiced


def frame_features(track) -> np.ndarray:
    conf, rms, flux, hf = track["conf"], track["rms_db"], track["flux"], track["hf"]
    n = len(conf)
    clean, voiced = pitch_curve(track)
    # carry the last voiced pitch through gaps so pitch differences stay defined
    idx = np.where(voiced, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    held = clean[idx]
    held = np.where(np.isnan(held), np.nanmedian(clean) if voiced.any() else 60.0, held)
    level = rms - np.percentile(rms, 95)
    cols = [voiced.astype(float), conf, level, flux, hf]
    for k in LAGS:
        step = _shift(held, k) - _shift(held, -k)
        cols += [step, np.abs(step), _shift(level, k) - _shift(level, -k), _shift(conf, k) - _shift(conf, -k)]
    for k in (2, 4, 8):
        cols += [flux - _shift(flux, -k), hf - _shift(hf, -k), _shift(voiced.astype(float), -k)]
    before = sliding_window_view(np.pad(held, (8, 0), mode="edge"), 8)[:n]
    after = sliding_window_view(np.pad(held, (0, 8), mode="edge"), 8)[:n]
    after_med = np.median(after, 1)
    cols += [before.std(1), after.std(1), after_med - np.median(before, 1), np.abs(after_med - np.round(after_med))]
    peak = maximum_filter1d(flux, 9)
    cols += [flux - peak, (flux >= peak).astype(float)]
    return np.stack(cols, 1).astype(np.float32)


@lru_cache(maxsize=2)
def model_info(path=MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"note onset model not found at {path} — run tools/train_onsets.py")
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("note onset model was trained on different features — retrain it")
    return meta


@lru_cache(maxsize=2)
def get_model(path=MODEL_PATH) -> "OnsetModel":
    return OnsetModel(path)


class OnsetModel:
    def __init__(self, path=MODEL_PATH):
        import joblib

        self.threshold = model_info(path)["threshold"]
        self.model = joblib.load(path)

    def onset_probability(self, track) -> np.ndarray:
        return self.model.predict_proba(frame_features(track))[:, 1]


def segment(prob, voiced, threshold, min_frames, gap_frames=6):
    """Note spans (start, end) in frames: onsets are peaks of the onset probability, and every
    voiced stretch starts a note; a note ends at the next onset or when voicing stops."""
    smooth = np.convolve(prob, np.ones(3) / 3, mode="same")
    peaks = np.flatnonzero((smooth >= threshold) & (smooth >= maximum_filter1d(smooth, 7)) & voiced)
    starts = {s for s, e in _runs(voiced) if e - s >= 4 and not np.any(np.abs(peaks - s) <= 4)}
    onsets = []
    for c in sorted(set(peaks.tolist()) | starts):
        if not onsets or c - onsets[-1] >= min_frames:
            onsets.append(c)
    spans = []
    for k, s in enumerate(onsets):
        e = onsets[k + 1] if k + 1 < len(onsets) else len(prob)
        gaps = [(a, b) for a, b in _runs(~voiced[s:e]) if b - a > gap_frames]
        if gaps:
            e = s + gaps[0][0]
        if e - s >= min_frames:
            spans.append((s, e))
    return spans

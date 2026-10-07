"""Tuning, key (and key changes) and tempo of a song, read from its backing track.

Instruments hold steady pitches, so the backing track gives a far more reliable
reading of the song's reference pitch (A4 = 440 Hz or not) and its key than the
vocal alone, which slides and bends around notes.
"""

import numpy as np

from .audio import load_audio
from .notes import MAJOR_PROFILE, MINOR_PROFILE, NOTE_NAMES

HOP = 512


def _corr_with_keys(chroma: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Correlation of each chroma column (12, n) with the profile in all 12 transpositions: (n, 12)."""
    keys = np.stack([np.roll(profile, r) for r in range(12)])
    keys = (keys - keys.mean(1, keepdims=True)) / keys.std(1, keepdims=True)
    c = (chroma - chroma.mean(0)) / (chroma.std(0) + 1e-9)
    return c.T @ keys.T / 12


def key_changes(chroma: np.ndarray, frame_rate: float, root: int, mode: str, window=6.0, hop=2.0,
                switch_cost=20.0, gain=10.0) -> list:
    """Where the song moves to another key, e.g. up a step for the last chorus.

    The backing's chroma is averaged over `window`-second stretches every `hop` seconds and
    compared with the song's key in all 12 transpositions; a Viterbi path through those
    picks the key of every stretch, paying `switch_cost` for each change (so a change has to
    hold for ~20 s to win). The mode stays the song's: a minor song moving up a tone is in
    another minor key. Returns [[start_seconds, "E major"], ...], one entry if it never changes."""
    profile = MAJOR_PROFILE if mode == "major" else MINOR_PROFILE
    n = chroma.shape[1]
    w, h = max(1, int(window * frame_rate)), max(1, int(hop * frame_rate))
    starts = list(range(0, max(n - w, 0) + 1, h))
    if len(starts) < 3:
        return [[0.0, f"{NOTE_NAMES[root]} {mode}"]]
    windows = np.stack([chroma[:, s:s + w].mean(1) for s in starts], 1)
    emit = gain * _corr_with_keys(windows, profile)  # (windows, 12)
    score = emit[0].copy()
    back = np.zeros((len(starts), 12), int)
    for t in range(1, len(starts)):
        stay, best = score, score.max() - switch_cost
        back[t] = np.where(stay >= best, np.arange(12), score.argmax())
        score = np.maximum(stay, best) + emit[t]
    path = [int(score.argmax())]
    for t in range(len(starts) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path = path[::-1]
    out = []
    for t, r in enumerate(path):
        if not out or r != path[t - 1]:
            at = 0.0 if not out else (starts[t] + w / 2 - h / 2) / frame_rate
            out.append([round(at, 2), f"{NOTE_NAMES[r]} {mode}"])
    return out


def analyze_backing(path, sr=22050) -> dict:
    import librosa

    y = load_audio(path, sr)
    tuning = float(librosa.estimate_tuning(y=y, sr=sr))  # semitones off A440, -0.5..0.5
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, hop_length=HOP)
    score, root, mode = max((np.corrcoef(chroma.mean(1), np.roll(profile, r))[0, 1], r, mode)
                            for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE))
                            for r in range(12))
    keys = key_changes(chroma, sr / HOP, root, mode)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr))[0])
    return {"tuning": round(tuning, 4), "key": f"{NOTE_NAMES[root]} {mode}", "key_confidence": round(float(score), 3),
            "keys": keys, "tempo": round(tempo, 1)}

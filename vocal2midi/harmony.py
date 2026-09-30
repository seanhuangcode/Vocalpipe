"""Tuning and key of a song, read from its backing track.

Instruments hold steady pitches, so the backing track gives a far more reliable
reading of the song's reference pitch (A4 = 440 Hz or not) and its key than the
vocal alone, which slides and bends around notes.
"""

import numpy as np

from .audio import load_audio
from .notes import MAJOR_PROFILE, MINOR_PROFILE, NOTE_NAMES


def analyze_backing(path, sr=22050) -> dict:
    import librosa

    y = load_audio(path, sr)
    tuning = float(librosa.estimate_tuning(y=y, sr=sr))  # semitones off A440, -0.5..0.5
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning).mean(1)
    score, root, mode = max((np.corrcoef(chroma, np.roll(profile, r))[0, 1], r, mode)
                            for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE))
                            for r in range(12))
    return {"tuning": round(tuning, 4), "key": f"{NOTE_NAMES[root]} {mode}", "key_confidence": round(float(score), 3)}

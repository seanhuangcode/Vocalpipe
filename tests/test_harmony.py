import numpy as np

from vocal2midi.harmony import key_changes
from vocal2midi.notes import MAJOR_PROFILE, MINOR_PROFILE

RATE = 10.0  # chroma frames per second


def chroma(*segments, noise=0.15, seed=0):
    """Chroma frames (12, n) for [(seconds, root, profile), ...]."""
    rng = np.random.default_rng(seed)
    cols = [np.roll(profile, root)[:, None] * np.ones((1, int(sec * RATE))) for sec, root, profile in segments]
    c = np.concatenate(cols, 1)
    return np.clip(c / c.max() + noise * rng.standard_normal(c.shape), 0, None)


def test_last_chorus_up_a_step():
    c = chroma((120, 2, MAJOR_PROFILE), (60, 4, MAJOR_PROFILE))
    keys = key_changes(c, RATE, root=2, mode="major")
    assert [k for _, k in keys] == ["D major", "E major"]
    assert abs(keys[1][0] - 120) <= 6


def test_no_change():
    assert key_changes(chroma((180, 9, MINOR_PROFILE)), RATE, root=9, mode="minor") == [[0.0, "A minor"]]


def test_short_excursion_is_not_a_key_change():
    c = chroma((60, 0, MAJOR_PROFILE), (6, 7, MAJOR_PROFILE), (60, 0, MAJOR_PROFILE))
    assert key_changes(c, RATE, root=0, mode="major") == [[0.0, "C major"]]

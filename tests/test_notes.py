"""Note extraction on pitch tracks with known notes (see conftest.py)."""

import numpy as np
import pytest
from conftest import FPS, notes_curve, track

from vocal2midi.notes import PREVIOUS, Settings, extract_notes, speech_score


def levels(curve, db=-6.0):
    return np.where(np.isfinite(curve), db, -80.0)


def summary(notes):
    return [(round(float(n.start), 2), round(float(n.end), 2), n.pitch, n.spoken) for n in notes]


@pytest.fixture(scope="module")
def vibrato_note():
    curve = notes_curve([(0.5, None), (3.0, 64), (0.5, None)])
    curve[50:350] += 0.9 * np.sin(2 * np.pi * 6 * np.arange(300) / FPS)  # ±90 cents at 6 Hz, operatic
    return track(curve, levels(curve))


def test_vibrato_is_one_held_note(vibrato_note):
    notes, info = extract_notes(vibrato_note, FPS, Settings())
    assert summary(notes) == [(0.5, 3.5, 64, False)]
    assert info["vibrato"]["share"] > 0.9 and 70 <= info["vibrato"]["extent_cents"] <= 100


def test_previous_version_chopped_vibrato(vibrato_note):
    notes, _ = extract_notes(vibrato_note, FPS, Settings(**PREVIOUS))
    assert len(notes) > 5 and all(n.pitch == 64 for n in notes)


def test_legato_change_of_note():
    curve = notes_curve([(0.5, None), (1.0, 60), (1.0, 62), (0.5, None)], glide=0.06)
    notes, _ = extract_notes(track(curve, levels(curve)), FPS, Settings())
    assert [n.pitch for n in notes] == [60, 62]
    assert abs(notes[1].start - 1.5) <= 0.03


def test_quiet_ghost_note_is_dropped():
    curve = notes_curve([(0.5, None), (0.5, 60), (0.5, 62), (0.5, 64), (1.0, None), (0.5, 67), (0.5, None)])
    db = levels(curve)
    db[300:350] = -40  # 34 dB under the singing: bleed, a reverb tail, a backing vocal
    tr = track(curve, db)
    assert [n.pitch for n in extract_notes(tr, FPS, Settings())[0]] == [60, 62, 64]
    assert [n.pitch for n in extract_notes(tr, FPS, Settings(ghost_db=0, ghost_local_db=0))[0]] == [60, 62, 64, 67]


def test_note_ends_where_the_singing_does():
    curve = notes_curve([(0.5, None), (1.0, 62), (1.2, 62), (0.5, None)])
    db = levels(curve)
    db[150:270] = np.linspace(-20, -45, 120)  # reverb tail after the voice stops at 1.5 s
    tr = track(curve, db)
    (note,) = extract_notes(tr, FPS, Settings())[0]
    assert abs(note.end - 1.5) <= 0.06
    (old,) = extract_notes(tr, FPS, Settings(release_db=0))[0]
    assert old.end > 2.5


def rap_curve(seed=3, syllables=16):
    rng = np.random.default_rng(seed)
    spec = [(0.5, None)]
    for _ in range(syllables):
        spec += [(0.12, 50), (0.05, None)]
    curve = notes_curve(spec + [(0.5, None)])
    idx = np.flatnonzero(np.isfinite(curve))
    for a in range(0, len(idx), 12):  # every syllable glides, like speech
        seg = idx[a:a + 12]
        curve[seg] += np.linspace(rng.uniform(-1, 3), rng.uniform(-3, 1), len(seg))
    return curve


def test_rapped_phrase_is_rhythm_only():
    curve = rap_curve()
    notes, info = extract_notes(track(curve, levels(curve)), FPS, Settings())
    assert len(notes) >= 12 and all(n.spoken for n in notes)
    assert len({n.pitch for n in notes}) == 1  # one line at the speaking pitch
    assert info["spoken_share"] == 1.0
    assert not any(n.spoken for n in extract_notes(track(curve, levels(curve)), FPS, Settings(spoken=0))[0])


def test_sung_phrase_is_not_spoken():
    curve = notes_curve([(0.5, None)] + [(0.4, p) for p in (60, 62, 64, 65, 67, 65, 64, 62)] + [(0.5, None)], glide=0.04)
    notes, info = extract_notes(track(curve, levels(curve)), FPS, Settings())
    assert [n.pitch for n in notes] == [60, 62, 64, 65, 67, 65, 64, 62]
    assert not any(n.spoken for n in notes) and info["spoken_share"] == 0.0


def test_speech_evidence_sways_a_borderline_phrase():
    curve = rap_curve()
    tr = track(curve, levels(curve))
    n = len(curve)
    sung_says = (np.zeros(n), np.ones(n))  # a classifier sure it hears singing
    rapped = extract_notes(tr, FPS, Settings(spoken=0.75))[0]
    vetoed = extract_notes(tr, FPS, Settings(spoken=0.75), speech=sung_says)[0]
    assert sum(n.spoken for n in vetoed) <= sum(n.spoken for n in rapped)


def test_speech_score_range():
    curve = rap_curve()
    tr = track(curve, levels(curve))
    notes, info = extract_notes(tr, FPS, Settings(spoken=0))
    score = speech_score(notes, curve, np.isfinite(curve), FPS)
    assert 0.6 <= score <= 1.0


def test_key_changes_follow_the_song():
    curve = notes_curve([(0.5, None)] + [(0.4, p) for p in (60, 62, 64, 65, 67)] + [(0.5, None)]
                        + [(0.4, p) for p in (62, 64, 66, 67, 69)] + [(0.5, None)])
    keys = [[0.0, "C major"], [3.0, "D major"]]
    notes, info = extract_notes(track(curve, levels(curve)), FPS, Settings(key="C major", key_changes=keys))
    assert info["keys"] == keys
    assert [n.pitch for n in notes] == [60, 62, 64, 65, 67, 62, 64, 66, 67, 69]


def test_note_dicts_say_if_spoken():
    curve = rap_curve()
    notes, _ = extract_notes(track(curve, levels(curve)), FPS, Settings())
    d = notes[0].to_dict()
    assert d["spoken"] is True and d["cents"] == 0.0

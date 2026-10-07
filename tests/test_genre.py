import numpy as np
import pytest

from vocal2midi import genre
from vocal2midi.notes import Settings


def scores(**classes):
    """Fake YAMNet output: 20 patches of music with the given class scores."""
    s = np.zeros((20, 521), np.float32)
    s[:, genre.CLASSES["Music"]] = 0.8
    for name, value in classes.items():
        s[:, genre.CLASSES[name.replace("_", " ")]] = value
    return s


def test_families_pool_their_classes():
    fam, music = genre.genre_scores(scores(Hip_hop_music=0.4, Pop_music=0.1))
    assert max(fam, key=fam.get) == "Hip-hop" and music == pytest.approx(0.8)
    fam, _ = genre.genre_scores(scores(House_music=0.2, Techno=0.2, Pop_music=0.3))
    assert fam["Electronic"] == pytest.approx(1 - 0.8 * 0.8)


def test_describe_and_style_nudges():
    fam, music = genre.genre_scores(scores(Pop_music=0.30, Hip_hop_music=0.22))
    assert genre.describe(fam, music)["genre"] == "Pop"
    style = {"rapped": 0.5, "autotune": False, "runs": 0.0, "range": ["C3", "C4"],
             "vibrato": {"share": 0.0, "rate": None, "extent_cents": None}}
    g = genre.describe(fam, music, style, tempo=92.0)
    assert g["genre"] == "Hip-hop" and "rapped verses" in g["tags"] and g["tempo"] == 92.0
    assert abs(sum(p for _, p in g["genres"]) - 1) < 0.2


def test_no_genre_without_music():
    fam, music = genre.genre_scores(np.zeros((10, 521), np.float32))
    assert genre.describe(fam, music)["genre"] is None


def test_profiles_leave_changed_settings_alone():
    assert genre.profile(Settings(), "Hip-hop").spoken == 0.5
    assert genre.profile(Settings(spoken=0.7), "Hip-hop").spoken == 0.7
    assert genre.profile(Settings(), "R&B / Soul").run_share == 0.1
    assert genre.profile(Settings(), "Rock") == Settings()
    assert genre.profile(Settings(), None) == Settings()


def test_speech_curve():
    s = np.zeros((10, 521), np.float32)
    s[:5, genre.CLASSES["Speech"]] = 0.6  # first 2.4 s spoken, then sung
    s[5:, genre.CLASSES["Singing"]] = 0.6
    share, weight = genre.speech_curve(s, 600)
    assert share[50] > 0.9 and share[400] < 0.1 and weight[50] == 1.0
    share, weight = genre.speech_curve(np.zeros((0, 521), np.float32), 100)
    assert weight.sum() == 0

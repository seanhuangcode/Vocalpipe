import numpy as np

from vocal2midi.vibrato import cycle_mean, find_vibrato, turning_points

FPS = 100
t = np.arange(300) / FPS


def test_wide_vibrato_flattens_to_its_note():
    v = find_vibrato(60 + 1.0 * np.sin(2 * np.pi * 6 * t), np.ones(300, bool))
    assert v.mask.mean() > 0.95
    assert np.abs(v.centre[30:-30] - 60).max() < 0.1
    s = v.summary(np.ones(300, bool))
    assert 5.5 <= s["rate"] <= 6.5 and 90 <= s["extent_cents"] <= 110


def test_pop_vibrato():
    v = find_vibrato(67 + 0.3 * np.sin(2 * np.pi * 5.5 * t), np.ones(300, bool))
    assert v.mask.mean() > 0.9 and np.abs(v.centre[30:-30] - 67).max() < 0.05


def test_note_change_inside_vibrato_stays_where_it_happens():
    truth = np.where(t < 1.5, 60.0, 62.0)
    v = find_vibrato(truth + 0.6 * np.sin(2 * np.pi * 5.8 * t), np.ones(300, bool))
    crossing = int(np.argmax(v.centre > 61.0))
    assert abs(crossing - 150) <= 3
    assert np.abs(v.centre[:120] - 60).max() < 0.15 and np.abs(v.centre[180:270] - 62).max() < 0.15


def test_runs_and_noise_are_not_vibrato():
    steps = np.repeat([67, 65, 64, 62, 60, 59, 57, 55], 12).astype(float)  # a run at 8 notes/s
    assert not find_vibrato(steps, np.ones(len(steps), bool)).mask.any()
    noise = 60 + 0.05 * np.random.default_rng(0).standard_normal(300)
    assert not find_vibrato(noise, np.ones(300, bool)).mask.any()
    slow = np.repeat([62, 64] * 4, 18).astype(float)  # neighbour notes, ~3 per second
    assert not find_vibrato(slow, np.ones(len(slow), bool)).mask.any()


def test_only_voiced_frames():
    x = 60 + 1.0 * np.sin(2 * np.pi * 6 * t)
    voiced = np.ones(300, bool)
    voiced[100:200] = False
    v = find_vibrato(x, voiced)
    assert not v.mask[100:200].any() and np.array_equal(v.centre[100:200], x[100:200])


def test_turning_points_and_cycle_mean():
    x = np.sin(2 * np.pi * np.arange(100) / 20)
    pts = turning_points(x, 0.5)
    assert np.allclose(np.diff(pts), 10, atol=1)
    assert np.abs(cycle_mean(x, np.full(100, 20.0))[20:80]).max() < 1e-6

"""Vibrato: find it, measure it, and see through it to the note being held.

Vibrato is a regular 4-8 Hz pitch wobble around a held note: up to a semitone either way in
opera, 20-50 cents in pop, R&B and country. Listeners and transcribers hear its centre as
the note. Left in the pitch curve, every swing looks like a note change: the onset model
fires on the swings and chops held notes into fragments, and each fragment's pitch depends
on which part of a swing it caught.

find_vibrato() follows the pitch's turning points (where it turns back by a clear margin)
in every voiced stretch. Vibrato shows up as a run of alternating swings of similar size,
each lasting half a cycle of a 4-8.5 Hz wobble. A run of notes doesn't: it moves mostly one
way, and it doesn't keep turning back at a steady rate. Inside vibrato the curve is replaced
by its average over exactly one cycle, which cancels the wobble but keeps real note changes,
as short ramps centred where they happen.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Vibrato:
    centre: np.ndarray  # the pitch curve with vibrato averaged out (MIDI)
    mask: np.ndarray  # frames inside vibrato
    rate: np.ndarray  # Hz per frame (0 outside vibrato)
    extent: np.ndarray  # semitones either side of the centre per frame (0 outside)

    def summary(self, voiced) -> dict:
        """Share of voiced time with vibrato, and its typical rate and extent."""
        v = self.mask & voiced
        share = float(v.sum() / max(voiced.sum(), 1))
        return {"share": round(share, 3),
                "rate": round(float(np.median(self.rate[v])), 2) if v.any() else None,
                "extent_cents": round(float(np.median(self.extent[v]) * 100), 1) if v.any() else None}


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def turning_points(x: np.ndarray, threshold: float):
    """Indices where the curve peaks or dips, counting only turns back by >= threshold.
    They alternate peak, dip, peak, ... (or dip, peak, ...)."""
    pts = []
    hi = lo = 0
    trend = 0
    for i in range(1, len(x)):
        if x[i] > x[hi]:
            hi = i
        if x[i] < x[lo]:
            lo = i
        if trend >= 0 and x[hi] - x[i] >= threshold and (trend > 0 or hi > lo):
            pts.append(hi)
            trend, lo = -1, i
        elif trend <= 0 and x[i] - x[lo] >= threshold and (trend < 0 or lo > hi):
            pts.append(lo)
            trend, hi = 1, i
    return pts


def cycle_mean(x: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Average of `x` over one (fractional) period centred on each frame. Near the ends the
    window slides inward rather than shrinking, so it always spans a whole cycle."""
    n = len(x)
    c = np.concatenate(([0.0], np.cumsum(x)))
    p = np.minimum(period, n)
    a = np.clip(np.arange(n) + 0.5 - p / 2, 0, n - p)
    grid = np.arange(n + 1)
    return (np.interp(a + p, grid, c) - np.interp(a, grid, c)) / np.maximum(p, 1e-9)


def find_vibrato(midi: np.ndarray, voiced: np.ndarray, fps: float = 100.0, min_extent: float = 0.15,
                 max_extent: float = 1.4, rate: tuple = (4.0, 8.5), min_swings: int = 3) -> Vibrato:
    """Vibrato in a (glitch-free) MIDI pitch curve; see the module docstring.

    min_extent / max_extent: semitones either side of the centre. Bigger swings are real note
    changes; smaller ones are drift. min_swings: half-cycles in a row it takes to call it vibrato.
    """
    n = len(midi)
    centre = np.array(midi, dtype=float)
    mask = np.zeros(n, bool)
    rates, extents = np.zeros(n), np.zeros(n)
    half_lo, half_hi = fps / (2 * rate[1]), fps / (2 * rate[0])
    for s, e in _runs(voiced & np.isfinite(centre)):
        if e - s < 2 * half_lo * min_swings:
            continue
        x = centre[s:e].copy()
        pts = turning_points(x, 1.5 * min_extent)
        if len(pts) < min_swings + 1:
            continue
        dur = np.diff(pts)
        swing = np.abs(np.diff(x[pts]))
        ok = (dur >= half_lo * 0.8) & (dur <= half_hi * 1.15) & (swing >= 2 * min_extent) & (swing <= 2 * max_extent)
        period = np.zeros(e - s)
        for a, b in _runs(ok):  # swings a..b-1 qualify, spanning turning points a..b
            if b - a < min_swings:
                continue
            d = dur[a:b]
            if d.std() > 0.35 * d.mean():  # not steady enough for vibrato
                continue
            p = 2 * float(np.median(d))
            lo = max(0, pts[a] - int(p / 2))
            hi = min(e - s, pts[b] + int(p / 2) + 1)
            period[lo:hi] = p
            extents[s + lo:s + hi] = float(np.median(swing[a:b])) / 2
            rates[s + lo:s + hi] = fps / p
        inside = period > 0
        if inside.any():
            smooth = cycle_mean(x, np.where(inside, period, 1.0))
            centre[s:e] = np.where(inside, smooth, x)
            mask[s:e] = inside
    return Vibrato(centre, mask, np.where(mask, rates, 0.0), np.where(mask, extents, 0.0))

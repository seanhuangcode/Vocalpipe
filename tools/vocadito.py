"""Vocadito helpers for training and benchmarking note extraction.

Vocadito (Bittner et al., 2021, CC BY 4.0): 40 solo-singing recordings with expert
note annotations. Download vocadito.zip (58 MB) from https://zenodo.org/records/5578807
and unzip it; pass the folder with --data. Only the audio and note annotations are used.
"""

import contextlib
import io
import sys
from pathlib import Path

import mir_eval
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vocal2midi.notes import midi_to_hz  # noqa: E402
from vocal2midi.pitch import FPS, track_pitch  # noqa: E402

TRACKS = list(range(1, 41))


class Vocadito:
    def __init__(self, root, device="mps"):
        self.root = Path(root)
        if not (self.root / "Audio").is_dir():
            sys.exit(f"{self.root} doesn't look like the unzipped vocadito folder (no Audio/)")
        self.cache = self.root / "_pitch_cache"
        self.cache.mkdir(exist_ok=True)
        self.device = device

    def reference(self, i, annotator="A1"):
        a = np.loadtxt(self.root / f"Annotations/Notes/vocadito_{i}_notes{annotator}.csv", delimiter=",", ndmin=2)
        return np.c_[a[:, 0], a[:, 0] + a[:, 2]], a[:, 1]

    def track(self, i):
        path = self.cache / f"{i}.npz"
        if not path.exists():
            with contextlib.redirect_stdout(io.StringIO()):
                np.savez(path, **track_pitch(self.root / f"Audio/vocadito_{i}.wav", "rmvpe", self.device))
        return dict(np.load(path))

    def onset_labels(self, i, n_frames, tolerance=2):
        """1 within ±tolerance frames (±20 ms) of an annotated note onset."""
        y = np.zeros(n_frames, np.int8)
        for s in self.reference(i)[0][:, 0]:
            c = int(round(s * FPS))
            y[max(0, c - tolerance):c + tolerance + 1] = 1
        return y


def score(ref_intervals, ref_hz, notes):
    """Note F1 (onset ±50 ms + pitch ±50 cents), F1 incl. offsets, and time-weighted pitch accuracy."""
    est_int = np.array([[n.start, n.end] for n in notes], float).reshape(-1, 2)
    est_hz = midi_to_hz(np.array([n.pitch for n in notes], float)).reshape(-1)
    if not len(est_int):
        return {"f1": 0.0, "f1_offsets": 0.0, "on_pitch": 0.0, "wrong_pitch": 0.0, "missing": 1.0}
    f1 = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_int, est_hz, onset_tolerance=0.05, pitch_tolerance=50, offset_ratio=None)[2]
    f1_off = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_int, est_hz, onset_tolerance=0.05, pitch_tolerance=50, offset_ratio=0.2)[2]
    # During each reference note: is the transcription on the right pitch, a wrong one, or silent?
    grid = np.arange(0, max(ref_intervals[:, 1].max(), est_int[:, 1].max()) + 0.01, 0.01)

    def raster(intervals, hz):
        out = np.full(len(grid), np.nan)
        for (s, e), h in zip(intervals, hz):
            out[(grid >= s) & (grid < e)] = 69 + 12 * np.log2(h / 440)
        return out

    ref, est = raster(ref_intervals, ref_hz), raster(est_int, est_hz)
    active = np.isfinite(ref)
    right = active & np.isfinite(est) & (np.abs(ref - est) <= 0.5)
    wrong = active & np.isfinite(est) & (np.abs(ref - est) > 0.5)
    return {"f1": f1, "f1_offsets": f1_off, "on_pitch": right.sum() / active.sum(),
            "wrong_pitch": wrong.sum() / active.sum(), "missing": 1 - (right.sum() + wrong.sum()) / active.sum()}


def evaluate(data, make_notes, tracks=TRACKS, annotator="A1"):
    rows = [score(*data.reference(i, annotator), make_notes(i)) for i in tracks]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def objective(m):
    return m["f1"] + m["on_pitch"] - m["wrong_pitch"]


def fmt(label, m):
    return (f"{label:<36} F1 {m['f1']:.3f} | F1 incl. offsets {m['f1_offsets']:.3f} | "
            f"right pitch {m['on_pitch']:.1%} | WRONG pitch {m['wrong_pitch']:.1%} | missing {m['missing']:.1%}")

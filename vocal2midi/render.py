import csv
import json
import shutil
import subprocess

import numpy as np
import pretty_midi

from .audio import write_wav
from .notes import midi_to_hz

FIELDS = ["start", "end", "duration", "pitch", "name", "hz", "note_hz", "cents", "velocity", "confidence"]


def write_midi(notes, path, program=0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="Vocal melody")
    inst.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end) for n in notes]
    pm.instruments.append(inst)
    pm.write(str(path))


def write_tables(notes, csv_path, json_path, meta):
    rows = [n.to_dict() for n in notes]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)
    with open(json_path, "w") as f:
        json.dump({**meta, "notes": rows}, f, indent=1)


def synth_piano(notes, duration, sr=44100):
    """Quick additive piano-like synth: decaying, slightly inharmonic partials."""
    out = np.zeros(int((duration + 2) * sr), dtype=np.float32)
    partials = np.array([1.0, 0.45, 0.3, 0.16, 0.1, 0.06, 0.04, 0.025])
    k = np.arange(1, len(partials) + 1)
    release = 0.15
    for n in notes:
        f = float(midi_to_hz(n.pitch))
        freqs = k * f * np.sqrt(1 + 0.0004 * k**2)
        keep = freqs < sr / 2
        dur = n.end - n.start
        t = np.arange(int((dur + release) * sr)) / sr
        decay = (0.8 + 0.6 * k[keep]) * (f / 261.6) ** 0.5
        tone = (partials[keep, None] * np.sin(2 * np.pi * freqs[keep, None] * t)
                * np.exp(-decay[:, None] * t)).sum(0)
        env = np.minimum(t / 0.004, 1.0)
        tail = t > dur
        env[tail] *= np.exp(-(t[tail] - dur) / (release / 4))
        s = int(n.start * sr)
        seg = (tone * env * (n.velocity / 127) * 0.3).astype(np.float32)
        out[s:s + len(seg)] += seg[: len(out) - s]
    peak = np.abs(out).max()
    return out / peak * 0.9 if peak > 0 else out


def render_piano(notes, midi_path, wav_path, duration, soundfont=None, sr=44100):
    if soundfont and shutil.which("fluidsynth"):
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", str(sr), "-F",
                        str(wav_path), str(soundfont), str(midi_path)], check=True)
        return
    write_wav(wav_path, synth_piano(notes, duration, sr), sr)


def print_notes(notes, info, limit=40):
    if not notes:
        print("  no notes found")
        return
    lo = min(notes, key=lambda n: n.pitch)
    hi = max(notes, key=lambda n: n.pitch)
    print(f"  {len(notes)} notes | range {lo.name} ({lo.note_hz:.1f} Hz) – {hi.name} ({hi.note_hz:.1f} Hz)"
          f" | singer tuning offset {info['tuning_cents']:+.0f} cents")
    mode = "autotune detected → notes snapped to key" if info["autotune"] else "natural singing"
    print(f"  key {info['key']} | {mode} ({info['on_note_ratio']:.0%} of frames within 10 cents of a note)")
    print(f"  {'#':>4} {'start':>7} {'dur':>5}  {'note':<4} {'sung Hz':>8} {'note Hz':>8} {'cents':>6} {'vel':>4}")
    for i, n in enumerate(notes[:limit]):
        print(f"  {i + 1:>4} {n.start:7.2f} {n.duration:5.2f}  {n.name:<4} {n.hz:8.1f} "
              f"{n.note_hz:8.1f} {n.cents:+6.0f} {n.velocity:4d}")
    if len(notes) > limit:
        print(f"  ... {len(notes) - limit} more (see notes.csv / viewer.html)")

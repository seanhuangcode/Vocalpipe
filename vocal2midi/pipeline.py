"""Song -> separated stems -> pitch track -> notes, shared by the CLI (main.py) and the app
server (serve.py). Models stay loaded between songs, so only the first song pays for loading
them.

`preview_seconds` runs just the start of the song into out/preview/ (a lighter separation
pass), so the app can start singing within seconds while the full song is processed.
"""

import json
from pathlib import Path

import numpy as np

from .harmony import analyze_backing
from .notes import Settings, extract_notes
from .pitch import DEFAULT_RMVPE, FPS, add_onset_features, track_pitch
from .render import render_piano, write_midi, write_tables
from .separate import separate_vocals
from .viewer import write_viewer

STAGES = ("Separating vocals", "Tracking pitch", "Finding notes", "Rendering")


def is_fresh(target: Path, *sources: Path) -> bool:
    return target.exists() and all(target.stat().st_mtime >= s.stat().st_mtime for s in sources)


def run(song: Path, out: Path, settings: Settings | None = None, *, model="htdemucs_ft", pitch="rmvpe",
        rmvpe_weights=DEFAULT_RMVPE, device="cpu", force=False, no_separate=False, soundfont=None,
        preview_seconds=None, progress=None, log=print):
    """Process `song` into `out`. Returns (notes, info, meta).

    progress(stage, fraction) reports where it is; fraction is 0..1 within the stage.
    """
    song, out = Path(song), Path(out)
    preview = bool(preview_seconds)
    if preview:
        out = out / "preview"
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac: None)

    # 1. Separate vocals
    if no_separate:
        vocals, backing = song, None
    else:
        vocals, backing = out / "vocals.wav", out / "instrumental.wav"
        if force or not is_fresh(vocals, song):
            log(f"[1/4] separating vocals ({model}{', preview' if preview else ''})")
            say(STAGES[0], 0.0)
            separate_vocals(song, vocals, backing, model, device, overlap=0.1 if preview else 0.25,
                            max_seconds=preview_seconds, progress=(lambda f: say(STAGES[0], f)) if progress else None)
        else:
            log("[1/4] vocals cached")

    # 2. Pitch track
    say(STAGES[1], 0.0)
    cache = out / "pitch.npz"
    track = None
    if is_fresh(cache, vocals) and not force:
        track = dict(np.load(cache))
        if str(track.pop("backend", "")) != pitch:
            track = None
    if track is None:
        log(f"[2/4] tracking pitch ({pitch})")
        track = track_pitch(vocals, pitch, device, rmvpe_weights)
        np.savez_compressed(cache, backend=pitch, **track)
    else:
        log("[2/4] pitch cached")
    if add_onset_features(track, vocals):  # pitch cached before the onset model existed
        np.savez_compressed(cache, backend=pitch, **track)

    # Song tuning and key, read from the backing track (steadier than the voice)
    harmony = {}
    if backing is not None:
        harmony_cache = out / "backing.json"
        if is_fresh(harmony_cache, backing) and not force:
            harmony = json.loads(harmony_cache.read_text())
        else:
            harmony = analyze_backing(backing)
            harmony_cache.write_text(json.dumps(harmony))

    # 3. Notes
    say(STAGES[2], 0.0)
    log(f"[3/4] extracting notes ({ {'hybrid': 'onset model + HMM', 'ml': 'onset model', 'hmm': 'HMM'}[(settings or Settings()).method] })")
    cfg = settings or Settings()
    cfg = Settings(**{**cfg.__dict__, "reference_tuning": harmony.get("tuning"), "key": cfg.key or harmony.get("key")})
    notes, info = extract_notes(track, FPS, cfg)
    duration = len(track["f0"]) / FPS
    meta = {"source": song.name, "pitch_backend": pitch, "tuning_cents": info["tuning_cents"],
            "key": info["key"], "autotune": info["autotune"], "duration": round(duration, 3),
            "preview": preview}
    write_tables(notes, out / "notes.csv", out / "notes.json", meta)
    if preview:
        return notes, info, meta

    # 4. MIDI, piano render + piano-roll viewer
    say(STAGES[3], 0.0)
    log("[4/4] rendering piano + viewer")
    write_midi(notes, out / "melody.mid")
    render_piano(notes, out / "melody.mid", out / "piano.wav", duration, soundfont)
    rel = lambda p: None if p is None else Path(p).resolve().relative_to(out.resolve(), walk_up=True).as_posix()
    write_viewer(out / "viewer.html", song.stem, notes, info, FPS, duration, pitch,
                 {"vocals": rel(vocals), "backing": rel(backing)})
    return notes, info, meta

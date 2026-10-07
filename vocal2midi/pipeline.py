"""Song -> separated stems -> pitch track -> notes, shared by the CLI (main.py) and the app
server (serve.py). Models stay loaded between songs, so only the first song pays for loading
them.

`preview_seconds` runs just the start of the song into out/preview/ (a lighter separation
pass), so the app can start singing within seconds while the full song is processed.
"""

import json
from pathlib import Path

import numpy as np

from . import genre as genres
from .audio import load_audio
from .harmony import analyze_backing
from .notes import Settings, extract_notes
from .pitch import DEFAULT_RMVPE, FPS, SR, add_onset_features, track_pitch
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

    # Song tuning, key (and key changes) and tempo, read from the backing track (steadier than the voice)
    harmony = {}
    if backing is not None:
        harmony_cache = out / "backing.json"
        if is_fresh(harmony_cache, backing) and not force:
            harmony = json.loads(harmony_cache.read_text())
        if "keys" not in harmony:  # analysed before key changes and tempo were read
            harmony = analyze_backing(backing)
            harmony_cache.write_text(json.dumps(harmony))

    # Genre: what YAMNet hears in the mix, and whether the vocal sounds sung or spoken
    cfg = settings or Settings()
    sounds = listen(song, vocals, out, cfg.genre, preview_seconds, force, log)
    families, music = genres.genre_scores(sounds["mix"]) if sounds else ({}, 0.0)
    guess = cfg.genre if cfg.genre not in ("auto", "off") else (genres.describe(families, music)["genre"] if sounds else None)

    # 3. Notes
    say(STAGES[2], 0.0)
    log(f"[3/4] extracting notes ({ {'hybrid': 'onset model + HMM', 'ml': 'onset model', 'hmm': 'HMM'}[cfg.method] }"
        f"{', tuned for ' + guess if guess in genres.PROFILES else ''})")
    cfg = genres.profile(Settings(**{**cfg.__dict__, "reference_tuning": harmony.get("tuning"),
                                     "key": cfg.key or harmony.get("key"),
                                     "key_changes": None if cfg.key else harmony.get("keys")}), guess)
    speech = genres.speech_curve(sounds["vocals"], len(track["f0"]), FPS) if sounds else None
    notes, info = extract_notes(track, FPS, cfg, speech=speech)
    duration = len(track["f0"]) / FPS
    style = genres.vocal_style(notes, info)
    if cfg.genre == "off":
        song_genre = None
    elif sounds:
        song_genre = genres.describe(families, music, style, harmony.get("tempo"))
        if cfg.genre != "auto":  # told what it is
            song_genre.update(genre=cfg.genre, confidence=1.0)
    else:
        song_genre = {"genre": None if cfg.genre == "auto" else cfg.genre, "confidence": 0.0, "genres": [],
                      "tempo": harmony.get("tempo"), "tags": [], "style": style}
    meta = {"source": song.name, "pitch_backend": pitch, "tuning_cents": info["tuning_cents"],
            "key": info["key"], "keys": info["keys"], "autotune": info["autotune"], "duration": round(duration, 3),
            "preview": preview, "genre": song_genre}
    write_tables(notes, out / "notes.csv", out / "notes.json", meta)
    if song_genre:
        (out / "genre.json").write_text(json.dumps(song_genre))
        log(f"      genre: {genres.text(song_genre)}")
    if preview:
        return notes, info, meta

    # 4. MIDI, piano render + piano-roll viewer
    say(STAGES[3], 0.0)
    log("[4/4] rendering piano + viewer")
    write_midi(notes, out / "melody.mid")
    render_piano(notes, out / "melody.mid", out / "piano.wav", duration, soundfont)
    rel = lambda p: None if p is None else Path(p).resolve().relative_to(out.resolve(), walk_up=True).as_posix()
    write_viewer(out / "viewer.html", song.stem, notes, info, FPS, duration, pitch,
                 {"vocals": rel(vocals), "backing": rel(backing)}, song_genre)
    return notes, info, meta


def listen(song: Path, vocals: Path, out: Path, mode="auto", preview_seconds=None, force=False, log=print):
    """YAMNet scores for the mix and the vocal, cached in out/yamnet.npz; None if genre
    recognition is off or YAMNet can't be loaded (e.g. offline the first time)."""
    if mode == "off":
        return None
    cache = out / "yamnet.npz"
    if not force and is_fresh(cache, song, vocals):
        return dict(np.load(cache))
    try:
        from . import yamnet

        model = yamnet.get_model()
        sounds = {"mix": yamnet.classify(load_audio(song, SR, max_seconds=preview_seconds), model),
                  "vocals": yamnet.classify(load_audio(vocals, SR), model)}
    except Exception as e:  # noqa: BLE001 -- the notes don't depend on it
        log(f"      genre recognition unavailable ({type(e).__name__}: {e})")
        return None
    np.savez_compressed(cache, **{k: v.astype(np.float16) for k, v in sounds.items()})
    return sounds


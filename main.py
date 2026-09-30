"""Song -> isolated vocals -> pitch track -> MIDI notes -> piano render + interactive viewer.

    venv/bin/python main.py golden.mp3
    venv/bin/python main.py song1.mp3 song2.wav --open
    venv/bin/python main.py vocals.wav --no-separate     # input is already a vocal stem

Each song gets a folder in output/ containing:
    vocals.wav, instrumental.wav    separated stems (cached)
    pitch.npz                       frame-level f0 / confidence / loudness (cached)
    melody.mid                      the vocal melody as MIDI notes
    notes.csv, notes.json           every note with start, duration, name, Hz, cents off
    piano.wav                       the MIDI rendered as piano, to compare against the vocals
    viewer.html                     interactive piano roll: play along, click keys/notes to test
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import vocal2midi  # noqa: F401  (sets MPS fallback before torch loads)

from vocal2midi import pipeline
from vocal2midi.audio import pick_device
from vocal2midi.notes import Settings
from vocal2midi.pitch import DEFAULT_RMVPE
from vocal2midi.render import print_notes


def process(song: Path, args, device: str):
    out = Path(args.output) / song.stem
    print(f"\n== {song.name} -> {out}/")
    t0 = time.time()
    cfg = Settings(method=args.notes, onset_threshold=args.sensitivity, smooth=not args.detailed, min_note=args.min_note,
                   conf_threshold=args.threshold, change_cost=args.steadiness, legato=args.legato,
                   auto_tuning=not args.no_tuning, autotune=args.autotune, key=args.key)
    notes, info, _ = pipeline.run(song, out, cfg, model=args.model, pitch=args.pitch, rmvpe_weights=args.rmvpe,
                                  device=device, force=args.force, no_separate=args.no_separate,
                                  soundfont=args.soundfont)
    print_notes(notes, info, limit=args.show)
    print(f"done in {time.time() - t0:.1f}s → open {out / 'viewer.html'}")
    if args.open:
        subprocess.run(["open", str(out / "viewer.html")])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("songs", nargs="+", type=Path)
    p.add_argument("-o", "--output", default="output")
    p.add_argument("--model", default="htdemucs_ft",
                   help="demucs model (htdemucs_ft = best; htdemucs = slightly faster)")
    p.add_argument("--pitch", default="rmvpe", choices=["rmvpe", "crepe", "crepe-full"],
                   help="pitch tracker (rmvpe is fastest + most accurate for singing)")
    p.add_argument("--rmvpe", default=DEFAULT_RMVPE, type=Path, help="path to rmvpe.pt")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--no-separate", action="store_true", help="input is already an isolated vocal")
    p.add_argument("--notes", default="hybrid", choices=["hybrid", "ml", "hmm"],
                   help="note detection: hybrid = learned onsets decide where notes start, the HMM decides "
                        "which note (default, most accurate); ml = onsets only; hmm = pitch-only")
    p.add_argument("--sensitivity", type=float, default=None,
                   help="onset probability that starts a new note (default from the model, ~0.25); "
                        "higher = fewer, longer, smoother notes; lower = catches quicker runs")
    p.add_argument("--detailed", action="store_true",
                   help="keep every split-second note of fast runs instead of folding slides into neighbours")
    p.add_argument("--min-note", type=float, default=0.06, help="shortest note in seconds")
    p.add_argument("--steadiness", type=float, default=6.0,
                   help="hmm: how much evidence a note change needs: higher = fewer, longer notes")
    p.add_argument("--legato", type=float, default=0.25,
                   help="hold notes through gaps shorter than this many seconds (0 = detached notes)")
    p.add_argument("--threshold", type=float, default=0.4, help="voicing confidence threshold")
    p.add_argument("--no-tuning", action="store_true", help="don't compensate for singer tuning offset")
    p.add_argument("--autotune", default="auto", choices=["auto", "on", "off"],
                   help="autotuned vocals: snap notes to the key and drop glitches (auto-detected by default)")
    p.add_argument("--key", help='force the key, e.g. "G major", "F#m" (default: read from the backing track)')
    p.add_argument("--soundfont", type=Path, help=".sf2 piano to render with fluidsynth")
    p.add_argument("--show", type=int, default=30, help="how many notes to print")
    p.add_argument("--force", action="store_true", help="ignore cached stems / pitch")
    p.add_argument("--open", action="store_true", help="open the viewer when done")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    for song in args.songs:
        if not song.exists():
            sys.exit(f"not found: {song}")
        process(song, args, device)


if __name__ == "__main__":
    main()

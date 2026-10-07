"""Score note extraction on synthetic singing in the styles of different genres.

    venv/bin/python tools/style_benchmark.py                       # every style, FCPE pitch
    venv/bin/python tools/style_benchmark.py --pitch rmvpe         # the app's pitch model
    venv/bin/python tools/style_benchmark.py --styles rap opera --clips 8
    venv/bin/python tools/style_benchmark.py --variants default previous   # before/after

The performances (tools/singsynth.py) come with their exact notes, so this measures what
Vocadito can't: opera vibrato, R&B runs, rapped verses, hard autotune, rock grit, jazz blue
notes, key changes, big reverb and harmonies. Each clip also gets a backing track, so tuning
and key come from the same backing-track analysis the app runs.

Columns: note F1 (onset within 50 ms and pitch within 50 cents), F1 counting offsets too,
share of sung time on the right / a wrong pitch, notes invented where nobody sang (per
minute), share of rapped time transcribed as melody (it should be marked spoken), and share
of sung time wrongly marked spoken. Clips, backing analysis and pitch tracks are cached in
output/.benchmark/.
"""

import argparse
import contextlib
import io
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import singsynth  # noqa: E402
from vocadito import score  # noqa: E402

from vocal2midi.harmony import analyze_backing  # noqa: E402
from vocal2midi.notes import PREVIOUS, Settings, extract_notes  # noqa: E402
from vocal2midi.pitch import FPS, track_pitch  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "output" / ".benchmark"
VARIANTS = {"default": {}, "previous": PREVIOUS}


def clip_seed(style, k):
    return zlib.crc32(style.encode()) % 100_000 * 100 + k


def load_clip(style, k, backend, device, seconds=24.0):
    """(reference, pitch track) for clip k of a style, rendering and tracking it on first use."""
    stem = CACHE / f"{style}_{k}"
    ref_path = stem.with_suffix(".json")
    if not ref_path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        audio, ref, _ = singsynth.render(style, clip_seed(style, k), seconds)
        sf.write(stem.with_suffix(".wav"), audio, singsynth.SR)
        sf.write(f"{stem}_backing.wav", singsynth.backing(ref, seconds, clip_seed(style, k)), singsynth.SR)
        ref["backing"] = analyze_backing(f"{stem}_backing.wav")
        ref_path.write_text(json.dumps(ref))
    ref = json.loads(ref_path.read_text())
    track_path = Path(f"{stem}.{backend}.npz")
    if not track_path.exists():
        with contextlib.redirect_stdout(io.StringIO()):
            track = track_pitch(stem.with_suffix(".wav"), backend, device)
        np.savez_compressed(track_path, **track)
    return ref, dict(np.load(track_path))


def _raster(notes, grid, pad=0.0):
    out = np.zeros(len(grid), bool)
    for s, e in notes:
        out[(grid >= s - pad) & (grid < e + pad)] = True
    return out


def evaluate(ref, notes, duration, tuning_cents=0.0):
    """Metrics for one clip (see the module docstring). Pitches are compared in the song's own
    tuning, as the extractor estimated it: a song tuned 30 cents flat still has an A4 in it."""
    shift = 2 ** (-tuning_cents / 1200)
    ref = {**ref, "notes": [{**n, "hz": n["hz"] * shift} for n in ref["notes"]]}
    sung = [n for n in ref["notes"] if not n["spoken"]]
    rapped = [n for n in ref["notes"] if n["spoken"]]
    melodic = [n for n in notes if not getattr(n, "spoken", False)]
    spoken = [n for n in notes if getattr(n, "spoken", False)]
    grid = np.arange(0, duration, 0.01)
    out = {}
    if sung:
        out.update(score(np.array([[n["start"], n["end"]] for n in sung]), np.array([n["hz"] for n in sung]), melodic))
    anyone = _raster([(n["start"], n["end"]) for n in ref["notes"]], grid, pad=0.1)
    invented = sum(1 for n in melodic if not anyone[(grid >= n.start) & (grid < n.end)].any())
    out["extra_per_min"] = invented / (duration / 60)
    est_mel = _raster([(n.start, n.end) for n in melodic], grid)
    if rapped:
        r = _raster([(n["start"], n["end"]) for n in rapped], grid)
        out["rap_as_melody"] = (r & est_mel).sum() / max(r.sum(), 1)
    if sung:
        s = _raster([(n["start"], n["end"]) for n in sung], grid)
        out["sung_as_spoken"] = (s & _raster([(n.start, n.end) for n in spoken], grid)).sum() / max(s.sum(), 1)
    return out


COLUMNS = [("f1", "F1", "{:.3f}"), ("f1_offsets", "F1+off", "{:.3f}"), ("on_pitch", "right", "{:.1%}"),
           ("wrong_pitch", "WRONG", "{:.1%}"), ("extra_per_min", "extra/min", "{:.1f}"),
           ("rap_as_melody", "rap→melody", "{:.0%}"), ("sung_as_spoken", "sung→spoken", "{:.1%}")]


def table(rows):
    head = f"{'':<22}" + "".join(f"{label:>12}" for _, label, _ in COLUMNS)
    lines = [head]
    for name, m in rows:
        lines.append(f"{name:<22}" + "".join(f"{(fmt.format(m[k]) if k in m and np.isfinite(m[k]) else '–'):>12}"
                                               for k, _, fmt in COLUMNS))
    return "\n".join(lines)


def mean_metrics(per_clip):
    keys = {k for m in per_clip for k in m}
    return {k: float(np.mean([m[k] for m in per_clip if k in m])) for k in keys}


def parse_set(items):
    """--set name=value overrides for Settings, typed like the field's default."""
    out = {}
    for item in items or []:
        name, value = item.split("=", 1)
        default = Settings.__dataclass_fields__[name].default
        out[name] = (value.lower() in ("1", "true", "yes", "on")) if isinstance(default, bool) else \
            None if value.lower() == "none" else type(default)(value) if default is not None else float(value)
    return out


def speech_evidence(style, k, n_frames):
    """YAMNet's speech-vs-singing reading of a clip's vocal, as the app computes it (cached)."""
    from vocal2midi import yamnet
    from vocal2midi.genre import speech_curve

    path = CACHE / f"{style}_{k}.yamnet.npy"
    if not path.exists():
        audio, _ = sf.read(CACHE / f"{style}_{k}.wav", dtype="float32")
        np.save(path, yamnet.classify(audio).astype(np.float16))
    return speech_curve(np.load(path).astype(np.float32), n_frames, FPS)


def run(styles, clips, backend, device, variants, a_cappella=False, overrides=None, listen=False):
    results = {v: {} for v in variants}
    for style in styles:
        data = [load_clip(style, k, backend, device) for k in range(clips)]
        speech = [speech_evidence(style, k, len(t["f0"])) if listen else None for k, (_, t) in enumerate(data)]
        for v in variants:
            per_clip = []
            for (ref, track), heard in zip(data, speech):
                bk = ref["backing"]
                backing = {} if a_cappella else {"reference_tuning": bk["tuning"], "key": bk["key"], "key_changes": bk.get("keys")}
                cfg = Settings(**{**backing, **VARIANTS[v], **(overrides or {})})
                notes, info = extract_notes(track, FPS, cfg, speech=heard)
                per_clip.append(evaluate(ref, notes, len(track["f0"]) / FPS, info["tuning_cents"]))
            results[v][style] = mean_metrics(per_clip)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styles", nargs="+", default=list(singsynth.STYLES), choices=list(singsynth.STYLES))
    ap.add_argument("--clips", type=int, default=6, help="clips per style")
    ap.add_argument("--pitch", default="fcpe", choices=["fcpe", "rmvpe", "crepe", "crepe-full"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", nargs="+", default=["default"], choices=list(VARIANTS))
    ap.add_argument("--a-cappella", action="store_true", help="no backing track: no reference tuning or key")
    ap.add_argument("--set", nargs="+", metavar="NAME=VALUE", help="override note Settings, e.g. --set vibrato=false")
    ap.add_argument("--listen", action="store_true", help="let YAMNet's speech/singing reading of the vocal sway rap calls, as the app does")
    ap.add_argument("--json", type=Path, help="also write the results here")
    args = ap.parse_args()
    t0 = time.time()
    results = run(args.styles, args.clips, args.pitch, args.device, args.variants, args.a_cappella, parse_set(args.set),
                  args.listen)
    for v, per_style in results.items():
        print(f"\n== {v} ({args.pitch} pitch, {args.clips} clips per style{', a cappella' if args.a_cappella else ''})")
        rows = [(f"{s} ({singsynth.STYLES[s].genre.split(' (')[0]})"[:22], m) for s, m in per_style.items()]
        sung = [m for s, m in per_style.items() if "f1" in m]
        if sung:
            rows.append(("MEAN (sung styles)", mean_metrics(sung)))
        print(table(rows))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

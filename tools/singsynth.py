"""Synthetic singing in the styles of different genres, with exact note ground truth.

    venv/bin/python tools/singsynth.py rnb out.wav            # render one example
    venv/bin/python tools/singsynth.py --list                 # the styles

tools/style_benchmark.py uses these to check note extraction on the ways different genres
are sung, which the Vocadito recordings (amateur, a cappella) don't cover: opera vibrato,
R&B runs, rapped verses, hard autotune, rock scoops and falls, country slides, jazz blue
notes and swing, key changes, big reverb, harmonies and instruments bleeding into the
separated vocal.

The voice is a source-filter model: phase-coherent harmonics that follow the f0 contour,
shaped by vowel formants and spectral tilt, plus breath, consonant noise and optional grit.
The f0 contour follows the usual model of sung pitch (Saitou et al., 2005): second-order
transitions between notes with overshoot, preparation (scoops), vibrato with a delayed
onset, drift and jitter. Rapped syllables follow speech intonation instead: a declining
phrase line with pitch accents and no held pitches.
"""

import argparse
import json
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.signal import bilinear, butter, fftconvolve, lfilter, sosfilt

SR = 16000
CTRL = 1000  # control rate (Hz) of the pitch and loudness curves
FPS = 100  # frame rate of the ground-truth pitch track

SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10),
          "mixolydian": (0, 2, 4, 5, 7, 9, 10)}
MODE_OF = {"major": "major", "mixolydian": "major", "minor": "minor", "dorian": "minor"}
# vowel formants (F1, F2, F3) for a male voice; female formants sit ~15% higher
VOWELS = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (300, 2200, 2950), "o": (570, 840, 2410),
          "u": (330, 870, 2240), "ae": (660, 1720, 2410), "uh": (520, 1190, 2390)}
NASAL = (260, 1000, 2300)
BANDWIDTHS = (80, 110, 170, 260)


@dataclass(frozen=True)
class Style:
    name: str
    genre: str
    tempo: tuple = (88, 124)
    voices: tuple = ("female", "male")
    rhythm: tuple = ((0.5, 4), (1, 4), (1.5, 1.5), (2, 1.5), (3, 0.5), (0.25, 0.5))  # (beats, weight)
    phrase: tuple = (6, 12)  # beats per sung phrase
    rest: tuple = (1, 3)  # beats between phrases
    scale: str = "major"
    steps: tuple = ((0, 0.18), (1, 0.42), (2, 0.2), (3, 0.09), (4, 0.06), (5, 0.03), (7, 0.02))
    span: int = 14  # semitones the melody covers
    slur: float = 0.15  # chance a note changes pitch within the syllable (no consonant)
    runs: float = 0.0  # chance a held syllable becomes a melisma run
    run_len: tuple = (4, 8)
    run_rate: tuple = (9.0, 13.0)  # run notes per second
    vibrato: tuple = (20, 45)  # cents (peak)
    vib_rate: tuple = (5.2, 6.4)
    vib_delay: tuple = (0.2, 0.35)
    vib_prob: float = 0.7
    scoop: float = 0.25  # chance a detached note starts from below
    scoop_depth: tuple = (60, 180)  # cents
    fall: float = 0.1  # chance a phrase's last note falls off
    glide: tuple = (0.04, 0.09)  # seconds, legato transitions
    overshoot: float = 0.12
    intonation: float = 15  # cents (sd) a note is off its pitch
    tilt: tuple = (-9.0, -6.0)  # source spectral slope, dB/octave
    breath: float = 0.04
    grit: float = 0.0
    swing: float = 0.0
    blue: float = 0.0  # chance a minor third / seventh is bent up (blue note)
    autotune: tuple | None = None  # retune time range (s)
    reverb: tuple | None = (-16, 1.2)  # (wet dB, RT60 s)
    harmony: float = 0.0  # chance a phrase has a backing harmony a third above
    bleed: float | None = None  # dB of instrument residue left in the separated vocal
    rap: float = 0.0  # share of phrases that are rapped
    modulate: bool = False  # key change half way through
    singer_formant: float = 0.0  # dB boost near 3 kHz (trained classical voices)
    twang: float = 0.0  # dB boost near 3.3 kHz


BALLAD_RHYTHM = ((0.5, 2), (1, 4), (1.5, 2), (2, 3), (3, 1.5), (4, 1))
STYLES = {s.name: s for s in [
    Style("pop", "Pop", runs=0.04, bleed=-40),
    Style("ballad", "Pop ballad", tempo=(60, 78), rhythm=BALLAD_RHYTHM, vibrato=(30, 60), vib_prob=0.9, slur=0.25,
          runs=0.08, reverb=(-11, 2.2)),
    Style("opera", "Classical / opera", tempo=(60, 92), rhythm=BALLAD_RHYTHM, span=19, vibrato=(55, 110),
          vib_rate=(5.3, 6.6), vib_delay=(0.04, 0.1), vib_prob=1.0, glide=(0.07, 0.15), slur=0.35, scoop=0.05,
          intonation=12, tilt=(-7.0, -4.0), singer_formant=14, reverb=(-9, 2.4), breath=0.02),
    Style("rnb", "R&B / soul", tempo=(64, 96), rhythm=((0.5, 3), (1, 3), (1.5, 1.5), (2, 2), (0.25, 1)),
          scale="minor", runs=0.3, vibrato=(25, 50), slur=0.25, scoop=0.3, blue=0.12, glide=(0.05, 0.12),
          reverb=(-12, 1.6), harmony=0.2),
    Style("rap", "Hip-hop (rapped)", tempo=(84, 100), rap=1.0, reverb=(-22, 0.6), tilt=(-6.0, -3.0), bleed=-38),
    Style("hiphop", "Hip-hop (rapped verse, sung hook)", tempo=(84, 100), rap=0.5, scale="minor", reverb=(-18, 0.9)),
    Style("autotune", "Pop / EDM (hard autotune)", tempo=(96, 140), autotune=(0.0, 0.012), intonation=30,
          scoop=0.3, vibrato=(10, 30), vib_prob=0.5, runs=0.05, reverb=(-16, 1.0), bleed=-38),
    Style("rock", "Rock", tempo=(100, 150), rhythm=((0.5, 5), (1, 4), (1.5, 1), (2, 1.5), (0.25, 0.5)),
          scale="mixolydian", scoop=0.45, scoop_depth=(100, 260), fall=0.3, grit=0.5, vibrato=(15, 35),
          vib_prob=0.4, intonation=22, tilt=(-5.0, -2.0), reverb=(-15, 1.1), bleed=-36),
    Style("jazz", "Jazz / blues", tempo=(100, 170), swing=0.33, rhythm=((0.5, 5), (1, 3), (1.5, 1.5), (2, 1), (3, 0.5)),
          scale="dorian", blue=0.2, scoop=0.4, fall=0.25, vib_delay=(0.3, 0.45), intonation=18, reverb=(-14, 1.4)),
    Style("country", "Country", tempo=(80, 120), glide=(0.08, 0.18), slur=0.3, scoop=0.3, scoop_depth=(60, 150),
          vibrato=(15, 35), vib_rate=(5.8, 7.0), twang=8, reverb=(-16, 1.0)),
    Style("folk", "Folk (plain, untrained)", vibrato=(0, 15), vib_prob=0.3, intonation=25, scoop=0.12,
          reverb=(-20, 0.8), tilt=(-10.0, -7.0)),
    Style("modulation", "Autotuned pop with a key change", modulate=True, autotune=(0.0, 0.01), intonation=25,
          bleed=-40),
    Style("reverb", "Ballad drenched in reverb", tempo=(60, 78), rhythm=BALLAD_RHYTHM, vibrato=(30, 60),
          vib_prob=0.9, slur=0.25, reverb=(-4, 3.0)),
    Style("harmony", "Pop with backing harmonies", harmony=0.7, bleed=-32),
]}


@dataclass
class Ev:
    """One sung note (or rapped syllable)."""
    start: float  # vowel onset (s): where a transcriber would put the note
    dur: float  # time until the next note in the rhythm
    pitch: int  # intended MIDI note
    sung: float  # MIDI pitch aimed at: tuning + intonation error (autotune: what's heard)
    kind: str = "note"  # "note" | "run" | "rap"
    phrase: int = 0
    link: str = "rest"  # how it follows the previous note: rest | cons | voiced | glottal | slur
    cons: str = ""  # consonant before the vowel
    cons_dur: float = 0.0
    vowel: str = "a"
    level: float = 0.0  # dB
    end: float = 0.0  # where its voicing stops (or the next legato note takes over)
    blue: bool = False
    stressed: bool = False
    rap_base: float = 0.0  # rapped syllables: speech pitch line at the syllable


def _midi_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _lowpass_noise(rng, n, cutoff, fs):
    sos = butter(2, cutoff, fs=fs, output="sos")
    x = sosfilt(sos, rng.standard_normal(n))
    return x / (x.std() + 1e-12)


# ---------------------------------------------------------------- score

def compose(style: Style, seed: int, seconds: float = 24.0):
    """The notes: rhythm, melody, syllables. Returns (events, info)."""
    rng = np.random.default_rng(seed)
    voice = str(rng.choice(style.voices))
    tempo = float(rng.uniform(*style.tempo))
    beat = 60.0 / tempo
    tuning = float(rng.choice([0.0, rng.uniform(-0.35, 0.35)], p=[0.6, 0.4]) + rng.normal(0, 0.02))
    tonic = int(rng.integers(0, 12))
    center = (67 if voice == "female" else 55) + int(rng.integers(-3, 4))
    keys = [(0.0, tonic)]
    evs: list[Ev] = []
    t, phrase, idx = float(rng.uniform(0.4, 1.0)), 0, None
    while t < seconds - 2.0:
        if style.modulate and len(keys) == 1 and t > seconds * 0.45:
            tonic = (tonic + int(rng.choice([1, 2]))) % 12
            keys.append((t, tonic))
            idx = None
        pcs = [(tonic + s) % 12 for s in SCALES[style.scale]]
        lo, hi = center - style.span // 2, center + style.span - style.span // 2
        pitches = [p for p in range(lo - 2, hi + 3) if p % 12 in pcs]
        if idx is None:
            idx = int(np.argmin([abs(p - center) for p in pitches]))
        if rng.random() < style.rap:
            t = _rap_phrase(rng, style, evs, t, beat, voice, phrase)
        else:
            t, idx = _sung_phrase(rng, style, evs, t, beat, pitches, idx, phrase, tonic)
        phrase += 1
        t += float(rng.uniform(*style.rest)) * beat
    evs = [ev for ev in evs if ev.start + 0.2 < seconds - 1.0]
    _finish(rng, evs, tuning, style)
    for ev in evs:
        ev.end = min(ev.end, seconds - 0.8)
    info = {"style": style.name, "genre": style.genre, "seed": seed, "voice": voice, "tempo": round(tempo, 2),
            "tuning": round(tuning, 4), "scale": style.scale,
            "keys": [(round(k, 3), f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][r]} {MODE_OF[style.scale]}")
                     for k, r in keys]}
    return evs, info


def _walk(rng, style, pitches, idx):
    steps, w = zip(*style.steps)
    step = int(rng.choice(steps, p=np.array(w) / sum(w)))
    nxt = idx + step * (1 if rng.random() < 0.5 else -1)
    if not 0 <= nxt < len(pitches):
        nxt = idx - (nxt - idx)
    return int(np.clip(nxt, 0, len(pitches) - 1))


def _sung_phrase(rng, style, evs, t0, beat, pitches, idx, phrase, tonic):
    beats = rng.uniform(*style.phrase)
    durs, w = zip(*style.rhythm)
    w = np.array(w) / sum(w)
    pos, first, level = 0.0, True, rng.uniform(-6, 0)
    vowels = list(VOWELS)
    while pos < beats:
        d = float(rng.choice(durs, p=w))
        if not first and rng.random() < 0.07:
            pos += 0.5  # a short breath inside the phrase
        start = t0 + pos * beat
        if style.swing and abs(pos % 1 - 0.5) < 1e-6:
            start += style.swing * beat / 2
        start += rng.normal(0, 0.01)
        if not first:
            idx = _walk(rng, style, pitches, idx)
        pitch = pitches[idx]
        vowel = str(rng.choice(vowels))
        lvl = level + rng.normal(0, 1.5) + (1.0 if abs(pos % 1) < 1e-6 else 0.0)
        blue = (style.blue > 0 and (pitch - tonic) % 12 in (3, 10) and rng.random() < style.blue * 3)
        if style.runs and d * beat >= 0.4 and rng.random() < style.runs:
            # melisma: a quick run of notes on one syllable, landing on a held note
            n = int(rng.integers(style.run_len[0], style.run_len[1] + 1))
            step_t = 1.0 / rng.uniform(*style.run_rate)
            n = max(2, min(n, int((d * beat - 0.15) / step_t)))
            direction = -1 if rng.random() < 0.7 else 1
            j, ts = idx, start
            for k in range(n):
                evs.append(Ev(ts, step_t, pitches[j], 0, "run", phrase, "slur" if k else "", "", 0, vowel, lvl))
                nxt = j + direction * (1 if rng.random() < 0.8 else 2)
                if not 0 <= nxt < len(pitches):  # turn around at the edge of the range
                    direction = -direction
                    nxt = j + direction
                j = nxt
                ts += step_t
            idx = j
            evs.append(Ev(ts, start + d * beat - ts, pitches[idx], 0, "note", phrase, "slur", "", 0, vowel, lvl))
        else:
            evs.append(Ev(start, d * beat, pitch, 0, "note", phrase, "", "", 0, vowel, lvl, blue=bool(blue)))
        pos += d
        first = False
    return t0 + pos * beat, idx


def _rap_phrase(rng, style, evs, t0, beat, voice, phrase):
    bars = int(rng.choice([1, 2]))
    six = beat / 4
    speak = (57 if voice == "female" else 48) + rng.uniform(-2, 3)
    hi, lo = speak + rng.uniform(1.5, 4), speak - rng.uniform(1, 3)
    pos, total = 0, 16 * bars
    level = rng.uniform(-4, 0)
    while pos < total:
        r = rng.random()
        if r < 0.07:
            pos += 1
            continue
        units = 1 if r < 0.62 else 2 if r < 0.9 else 3
        start = t0 + pos * six + rng.normal(0, 0.008)
        base = hi + (lo - hi) * pos / total
        stressed = pos % 4 == 0 or rng.random() < 0.3
        evs.append(Ev(start, units * six, int(round(base)), 0, "rap", phrase, "", "", 0, str(rng.choice(list(VOWELS))),
                      level + rng.normal(0, 2) + (2 if stressed else 0), stressed=bool(stressed), rap_base=base))
        pos += units
    return t0 + pos * six


def _finish(rng, evs, tuning, style):
    """Consonants, links and note ends."""
    for k, ev in enumerate(evs):
        prev = evs[k - 1] if k else None
        ev.sung = ev.pitch + tuning + float(np.clip(rng.normal(0, style.intonation / 100), -0.42, 0.42))
        if ev.kind == "rap":
            ev.sung = ev.rap_base + tuning
        if prev is None or prev.phrase != ev.phrase or ev.start - (prev.start + prev.dur) > 0.05:
            ev.link = "rest"
        elif ev.link == "slur":
            pass  # run notes
        elif ev.kind != "rap" and ev.pitch != prev.pitch and rng.random() < style.slur:
            ev.link = "slur"
            ev.vowel = prev.vowel  # same syllable
        else:
            r = rng.random()
            ev.cons, ev.link, ev.cons_dur = (("s", "cons", rng.uniform(0.06, 0.12)) if r < 0.3 else
                                             ("t", "cons", rng.uniform(0.04, 0.08)) if r < 0.65 else
                                             ("m", "voiced", rng.uniform(0.04, 0.09)) if r < 0.9 else
                                             ("'", "glottal", rng.uniform(0.02, 0.04)))
            if ev.kind == "rap" and ev.link == "glottal":
                ev.cons, ev.link, ev.cons_dur = "t", "cons", rng.uniform(0.03, 0.06)
            ev.cons_dur = float(min(ev.cons_dur, 0.45 * prev.dur))
        if ev.link == "rest" and prev is not None and prev.phrase == ev.phrase:
            ev.cons, ev.cons_dur = ("s", float(rng.uniform(0.05, 0.1))) if rng.random() < 0.5 else ("", 0.0)
    for k, ev in enumerate(evs):
        nxt = evs[k + 1] if k + 1 < len(evs) else None
        if nxt is not None and nxt.link != "rest":
            ev.end = nxt.start - (nxt.cons_dur if nxt.link in ("cons", "voiced", "glottal") else 0.0)
        else:
            ev.end = ev.start + ev.dur * (rng.uniform(0.55, 0.8) if ev.kind == "rap" else rng.uniform(0.8, 0.97))
            if nxt is not None and nxt.cons_dur:
                ev.end = min(ev.end, nxt.start - nxt.cons_dur - 0.02)
        if ev.kind == "rap":
            ev.end = min(ev.end, ev.start + ev.dur * rng.uniform(0.6, 0.85))
        ev.end = max(ev.end, ev.start + 0.03)


# ---------------------------------------------------------------- performance

@lru_cache(maxsize=4096)
def _step_response(glide, overshoot):
    """How pitch moves to a new note: a 2nd-order system's step response (Saitou et al., 2005)
    with a 10-90% rise time of `glide` seconds, and the delay to its half-way point."""
    zeta = 1.0 if overshoot <= 1e-3 else -np.log(overshoot) / np.sqrt(np.pi ** 2 + np.log(overshoot) ** 2)
    w = (1 - 0.4167 * zeta + 2.917 * zeta ** 2) / glide
    b, a = bilinear([w * w], [1, 2 * zeta * w, w * w], fs=CTRL)
    step = lfilter(b, a, np.ones(int(CTRL * max(1.0, 8 * glide))))
    return step, int(np.argmax(step >= 0.5))


def perform(evs, info, style: Style, rng, seconds):
    """Control-rate curves: pitch (MIDI, nan when silent), loudness, formants, noise events."""
    n = int(seconds * CTRL)
    t = np.arange(n) / CTRL
    pitch = np.full(n, np.nan)
    amp = np.zeros(n)
    formants = np.zeros((n, 3))
    formants[:] = VOWELS["a"]
    noises = []  # (start, dur, kind, level)
    scale_f = 1.15 if info["voice"] == "female" else 1.0

    chains, cur = [], []
    for k, ev in enumerate(evs):
        if ev.link in ("rest", "cons") and cur:
            chains.append(cur)
            cur = []
        cur.append(k)
    if cur:
        chains.append(cur)

    for chain in chains:
        ce = [evs[k] for k in chain]
        final = chain[-1] + 1 >= len(evs) or evs[chain[-1] + 1].link == "rest"
        rel = rng.uniform(0.08, 0.2) if final else rng.uniform(0.03, 0.06)
        att = rng.uniform(0.01, 0.025) if ce[0].kind == "rap" else rng.uniform(0.015, 0.04)
        s = int(ce[0].start * CTRL)
        e = min(n, int((ce[-1].end + rel / 2) * CTRL))
        if e - s < 5:
            continue
        seg = np.arange(s, e)
        ts = t[seg]
        # ---- pitch
        if ce[0].kind == "rap":
            u = np.empty(len(seg))
            for k, ev in enumerate(ce):
                a0 = int(ev.start * CTRL) - s
                a1 = min(len(seg), int(ce[k + 1].start * CTRL) - s) if k + 1 < len(ce) else len(seg)
                x = (np.arange(a1 - a0) / CTRL) / max(ev.end - ev.start, 0.03)
                accent = (rng.uniform(1.0, 4.0) if ev.stressed else rng.uniform(-0.3, 0.8)) * np.sin(np.pi * np.clip(x / 1.2, 0, 1)) ** 1.5
                slope = rng.uniform(-1.5, 1.0) * x
                u[max(a0, 0):a1] = (ev.sung + accent + slope)[: a1 - max(a0, 0)]
            y = u + 0.2 * _lowpass_noise(rng, len(seg), 12, CTRL) * 0.3
        else:
            # every note change is a step response; quick runs move faster than legato slides
            glide, over = rng.uniform(*style.glide), style.overshoot * rng.uniform(0.5, 1.5)
            y = np.full(len(seg), ce[0].sung)
            for k, ev in enumerate(ce[1:], 1):
                g = min(glide, 0.45 * (ev.start - ce[k - 1].start)) if "run" in (ev.kind, ce[k - 1].kind) else glide
                resp, d50 = _step_response(round(max(g, 0.02), 3), round(over, 2))
                mid = ev.start - (ev.cons_dur / 2 if ev.link == "voiced" else 0.0)
                j = int(mid * CTRL) - d50 - s
                part = resp[: max(0, len(seg) - max(j, 0))] if j >= 0 else resp[-j:][: len(seg)]
                y[max(j, 0):max(j, 0) + len(part)] += (ev.sung - ce[k - 1].sung) * part
                y[max(j, 0) + len(part):] += ev.sung - ce[k - 1].sung
            x0 = ts - ce[0].start
            if ce[0].link in ("rest", "cons") and rng.random() < style.scoop:
                dur = rng.uniform(0.06, 0.16)
                y -= rng.uniform(*style.scoop_depth) / 100 * (1 - _smoothstep(x0 / dur))
            else:
                y += rng.normal(0, 0.12) * np.clip(1 - x0 / 0.05, 0, 1)
            for k, ev in enumerate(ce):
                a0, a1 = int(ev.start * CTRL) - s, min(len(seg), int(ev.end * CTRL) - s)
                if a1 <= a0:
                    continue
                if ev.blue:  # bend up from the minor third
                    xb = (ts[a0:a1] - ev.start) / max(0.4 * (ev.end - ev.start), 0.05)
                    y[a0:a1] += rng.uniform(0.3, 0.55) * _smoothstep(xb)
                delay = rng.uniform(*style.vib_delay)
                if ev.end - ev.start > delay + 0.15 and rng.random() < style.vib_prob and style.vibrato[1] > 0:
                    ext = rng.uniform(*style.vibrato) / 100
                    rate = rng.uniform(*style.vib_rate)
                    stop = len(seg) if k == len(ce) - 1 else a1
                    xs = ts[a0:stop] - ev.start
                    env = np.clip((xs - delay) / 0.2, 0, 1)
                    if k < len(ce) - 1:
                        env *= np.clip((ev.end - ts[a0:stop]) / 0.05, 0, 1)
                    inst = rate * (1 + 0.05 * np.sin(2 * np.pi * 0.8 * xs + rng.uniform(0, 6.3)))
                    y[a0:stop] += ext * env * np.sin(2 * np.pi * np.cumsum(inst) / CTRL + rng.uniform(0, 6.3))
            if final and rng.random() < style.fall:
                fd = rng.uniform(0.12, 0.28)
                xf = (ts - (ce[-1].end - fd)) / fd
                y -= rng.uniform(1.0, 4.0) * np.clip(xf, 0, None) ** 2
        drift = sum(rng.uniform(0.01, 0.03) * np.sin(2 * np.pi * rng.uniform(0.1, 1.2) * ts + rng.uniform(0, 6.3))
                    for _ in range(3))
        y = y + drift + 0.025 * _lowpass_noise(rng, len(seg), 30, CTRL)
        pitch[seg] = y

        # ---- loudness
        lv = np.zeros(len(seg))
        for k, ev in enumerate(ce):
            a0 = max(0, int(ev.start * CTRL) - s)
            a1 = len(seg) if k + 1 == len(ce) else min(len(seg), int(ce[k + 1].start * CTRL) - s)
            if a1 <= a0:
                continue
            ramp = np.linspace(0, rng.uniform(-3, 3), a1 - a0)
            lv[a0:a1] = (ev.level + ramp)[: a1 - a0]
            if k and ev.link in ("voiced", "glottal"):
                c0 = int((ev.start - ev.cons_dur) * CTRL) - s
                depth = 10.0 if ev.link == "voiced" else 18.0
                dip = np.zeros(len(seg))
                dip[max(c0, 0):a0] = depth
                lv -= np.convolve(dip, np.ones(10) / 10, mode="same")
                if ev.link == "voiced":
                    formants[s + max(c0, 0):s + a0] = np.array(NASAL)
        lv = np.convolve(np.pad(lv, 8, mode="edge"), np.ones(15) / 15, mode="same")[8:-8]
        env = 10 ** (lv / 20)
        env *= _smoothstep((ts - ce[0].start) / att)
        env *= 1 - _smoothstep((ts - (ce[-1].end - rel / 2)) / rel)
        if style.vibrato[1] > 0:
            env *= 1 + 0.04 * _lowpass_noise(rng, len(seg), 8, CTRL)
        amp[seg] = np.maximum(amp[seg], env)
        # ---- vowels
        for k, ev in enumerate(ce):
            a0 = max(0, int(ev.start * CTRL) - s)
            a1 = len(seg) if k + 1 == len(ce) else min(len(seg), int((ce[k + 1].start - ce[k + 1].cons_dur) * CTRL) - s)
            formants[s + a0:s + a1] = np.array(VOWELS[ev.vowel]) * scale_f
        # ---- consonants and breaths
        first = ce[0]
        if first.cons_dur and first.link in ("cons", "rest"):
            noises.append((first.start - first.cons_dur, first.cons_dur, first.cons or "s", first.level))
        if first.link == "rest" and (chain[0] == 0 or evs[chain[0] - 1].phrase != first.phrase):
            if rng.random() < 0.6:
                noises.append((first.start - rng.uniform(0.25, 0.45), rng.uniform(0.15, 0.3), "breath", first.level))
    voiced = np.isfinite(pitch) & (amp > 0.03 * amp.max())
    # smooth formant jumps (coarticulation)
    formants = np.stack([np.convolve(np.pad(formants[:, i], 15, mode="edge"), np.ones(31) / 31, mode="same")[15:-15]
                         for i in range(3)], 1)
    return pitch, amp, formants, noises, voiced


def _autotune(pitch, voiced, evs, info, style, rng):
    """Hard pitch correction: snap to the nearest note of the scale, `retune` seconds smoothing."""
    retune = rng.uniform(*style.autotune)
    tuning = info["tuning"]
    keys = [(k, ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(name.split()[0]))
            for k, name in info["keys"]]
    out = pitch.copy()
    cur, a = None, np.exp(-1 / (retune * CTRL)) if retune > 1e-4 else 0.0
    y = None
    for i in range(len(pitch)):
        m = pitch[i]
        if not np.isfinite(m):
            cur = y = None
            continue
        if not voiced[i] and cur is None:
            continue
        tonic = [r for k, r in keys if k <= i / CTRL][-1]
        pcs = [(tonic + s) % 12 for s in SCALES[style.scale]]
        if cur is None or abs(m - cur) > 0.62:
            base = m - tuning
            cands = [p for p in range(int(base) - 2, int(base) + 3) if p % 12 in pcs]
            cur = min(cands, key=lambda p: abs(p - base)) + tuning
        y = cur if y is None else a * y + (1 - a) * cur
        out[i] = y + 0.04 * (m - cur)
    for ev in evs:
        if ev.kind == "rap":
            continue
        a0, a1 = int(ev.start * CTRL), int(ev.end * CTRL)
        mid = out[a0 + (a1 - a0) // 5: a1 - (a1 - a0) // 5]
        mid = mid[np.isfinite(mid)]
        if len(mid):
            ev.sung = float(np.median(mid))
            ev.pitch = int(round(ev.sung - tuning))
    return out


# ---------------------------------------------------------------- sound

def _voice(pitch, amp, formants, voice, style, rng, n_audio):
    """Harmonics of the f0 contour, shaped by the vowel formants and spectral tilt."""
    t_ctrl = np.arange(len(pitch)) / CTRL
    filled = pitch.copy()
    ok = np.isfinite(filled)
    if not ok.any():
        return np.zeros(n_audio)
    filled[~ok] = np.interp(t_ctrl[~ok], t_ctrl[ok], filled[ok])
    t_audio = np.arange(n_audio) / SR
    f0 = _midi_hz(np.interp(t_audio, t_ctrl, filled))
    f0 *= 1 + 0.0025 * _lowpass_noise(rng, n_audio, 120, SR)  # jitter
    phase = 2 * np.pi * np.cumsum(f0) / SR
    hop = SR // 200
    fr_t = np.arange(n_audio // hop + 2) * hop / SR
    f0_fr = np.interp(fr_t, t_audio, f0)
    F = np.stack([np.interp(fr_t, t_ctrl, formants[:, i]) for i in range(3)], 1)
    K = int(min(120, 7600 / max(f0.min(), 50)))
    k = np.arange(1, K + 1)
    freqs = f0_fr[:, None] * k[None, :]
    H = np.ones_like(freqs)
    f4 = 3400 * (1.15 if voice == "female" else 1.0)
    for i, B in enumerate(BANDWIDTHS):
        Fi = F[:, i:i + 1] if i < 3 else np.full((len(fr_t), 1), f4)
        H *= Fi ** 2 / np.sqrt((Fi ** 2 - freqs ** 2) ** 2 + (B * freqs) ** 2)
    boost = style.singer_formant * np.exp(-0.5 * ((freqs - 2900) / 350) ** 2) + \
        style.twang * np.exp(-0.5 * ((freqs - 3300) / 300) ** 2)
    tilt = rng.uniform(*style.tilt)
    A = H * k[None, :] ** (tilt / 6.02) * 10 ** (boost / 20)
    A *= np.clip((7600 - freqs) / 400, 0, 1)
    A /= np.sqrt((A ** 2).sum(1, keepdims=True) / 2) + 1e-12
    y = np.zeros(n_audio)
    for c0 in range(0, n_audio, 4000):
        idx = np.arange(c0, min(n_audio, c0 + 4000))
        fr = idx / hop
        i0 = fr.astype(int)
        w = (fr - i0)[:, None]
        Ac = A[i0] * (1 - w) + A[i0 + 1] * w
        y[idx] = (Ac * np.sin(k[None, :] * phase[idx, None])).sum(1)
    env = np.interp(t_audio, t_ctrl, amp)
    if style.grit:
        g = style.grit
        # rasp: intermittent weak subharmonics, rough amplitude, saturation and pulse noise
        y += 0.12 * g * np.sin(0.5 * phase) * np.clip(_lowpass_noise(rng, n_audio, 3, SR), 0, None)
        y *= 1 + 0.4 * g * _lowpass_noise(rng, n_audio, 150, SR)
        y = np.tanh((1 + 3 * g) * y / 2) / np.tanh((1 + 3 * g) / 2) * 2
        rasp = sosfilt(butter(2, 1500, "highpass", fs=SR, output="sos"), rng.standard_normal(n_audio))
        y += 0.25 * g * rasp * (0.5 + 0.5 * np.sin(phase))
    breath = sosfilt(butter(2, [500, 6000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n_audio))
    return y * env + style.breath * 4 * breath * env


def _noises(noises, n_audio, rng):
    y = np.zeros(n_audio)
    bands = {"s": (4000, 7500, -10), "t": (1800, 6000, -12), "breath": (800, 5000, -28), "": (3000, 7000, -16)}
    for start, dur, kind, level in noises:
        lo, hi, rel = bands.get(kind, bands[""])
        a, b = int(start * SR), int((start + dur) * SR)
        if b <= a or a < 0:
            continue
        b = min(b, n_audio)
        burst = sosfilt(butter(2, [lo, hi], "bandpass", fs=SR, output="sos"), rng.standard_normal(b - a))
        env = np.ones(b - a)
        ramp = min(int(0.01 * SR), (b - a) // 3)
        if ramp:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        if kind == "t":  # closure, then a burst
            env[: int(0.7 * len(env))] = 0.02
        y[a:b] += burst / (burst.std() + 1e-9) * env * 10 ** ((level + rel) / 20)
    return y


def _reverb(y, wet_db, rt60, rng):
    n = int(SR * min(rt60 * 1.2, 4.0))
    tt = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * tt / rt60)
    ir = sosfilt(butter(1, 5000, fs=SR, output="sos"), ir)
    ir[: int(0.012 * SR)] = 0  # pre-delay
    wet = fftconvolve(y, ir)[: len(y)]
    wet *= np.sqrt(np.mean(y ** 2) / (np.mean(wet ** 2) + 1e-12)) * 10 ** (wet_db / 20)
    return y + wet


def _bleed(n_audio, info, level_db, rng):
    """Instrument residue in a separated vocal: soft chord tones every couple of beats."""
    y = np.zeros(n_audio)
    beat = 60 / info["tempo"]
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    t = 0.0
    while t < n_audio / SR:
        tonic = names.index([name for k, name in info["keys"] if k <= t][-1].split()[0])
        root = tonic + int(rng.choice([0, 5, 7, 9]))
        for p in (root + 48, root + 51 + (rng.random() < 0.5), root + 55):
            f = float(_midi_hz(p))
            a = int(t * SR)
            m = min(n_audio - a, int(2 * beat * SR))
            tt = np.arange(m) / SR
            y[a:a + m] += sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.5 for h in (1, 2, 3, 4)) * np.exp(-2.5 * tt)
        t += 2 * beat
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 10 ** (level_db / 20)


PROGRESSIONS = {"major": ((0, "maj"), (7, "maj"), (9, "min"), (5, "maj")),  # I V vi IV
                "minor": ((0, "min"), (8, "maj"), (3, "maj"), (10, "maj"))}  # i VI III VII


def backing(info, seconds: float = 24.0, seed: int = 0):
    """An instrumental to go with a performance: chords, bass and drums in its key(s) and tuning,
    for the backing-track analysis (tuning and key) the app runs on the separated instrumental."""
    rng = np.random.default_rng(seed + 7)
    n = int(seconds * SR)
    y = np.zeros(n)
    beat = 60 / info["tempo"]
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    mode = MODE_OF[info["scale"]]

    def tone(f, dur, decay, partials=(1, 0.5, 0.33, 0.25, 0.2)):
        tt = np.arange(int(dur * SR)) / SR
        return sum(a * np.sin(2 * np.pi * f * h * tt) for h, a in enumerate(partials, 1) if f * h < 7600) * np.exp(-decay * tt)

    def put(sig, at):
        a = int(at * SR)
        if a < n:
            y[a:a + len(sig)] += sig[: n - a]

    bar = 0
    while bar * 4 * beat < seconds:
        t0 = bar * 4 * beat
        tonic = names.index([name for k, name in info["keys"] if k <= t0 + 1e-6][-1].split()[0])
        deg, quality = PROGRESSIONS[mode][bar % 4]
        root = tonic + deg
        chord = [root, root + (4 if quality == "maj" else 3), root + 7]
        for b in range(4):
            tb = t0 + b * beat
            for p in chord:
                put(0.18 * tone(_midi_hz(60 + (p - 60) % 12 + info["tuning"]), 1.2 * beat, 2.0), tb)
            put(0.35 * tone(_midi_hz(36 + root % 12 + info["tuning"]), beat, 3.0, (1, 0.4, 0.15)), tb)
            if b in (0, 2):  # kick
                tt = np.arange(int(0.15 * SR)) / SR
                put(0.6 * np.sin(2 * np.pi * (50 + 80 * np.exp(-30 * tt)) * tt) * np.exp(-20 * tt), tb)
            else:  # snare
                put(0.25 * rng.standard_normal(int(0.12 * SR)) * np.exp(-25 * np.arange(int(0.12 * SR)) / SR), tb)
            for h in (0, 0.5):  # hi-hat
                put(0.05 * sosfilt(butter(2, 6000, "highpass", fs=SR, output="sos"), rng.standard_normal(int(0.03 * SR))), tb + h * beat)
        bar += 1
    return (y / (np.abs(y).max() + 1e-9) * 0.8).astype(np.float32)


def render(style_name: str, seed: int, seconds: float = 24.0):
    """Audio (16 kHz), ground-truth notes and pitch track for one synthetic performance."""
    style = STYLES[style_name]
    evs, info = compose(style, seed, seconds)
    rng = np.random.default_rng(seed + 1_000_003)
    pitch, amp, formants, noises, voiced = perform(evs, info, style, rng, seconds)
    if style.autotune:
        pitch = _autotune(pitch, voiced, evs, info, style, rng)
    n_audio = int(seconds * SR)
    lead = _voice(pitch, amp, formants, info["voice"], style, rng, n_audio)
    sung = np.repeat(amp > 0.3 * amp.max(), SR // CTRL)[:n_audio]
    vowel_rms = np.sqrt(np.mean(lead[sung] ** 2)) if sung.any() else 1.0
    y = lead + _noises(noises, n_audio, rng) * vowel_rms
    if style.harmony:
        phrases = sorted({ev.phrase for ev in evs if ev.kind != "rap"})
        chosen = {p for p in phrases if rng.random() < style.harmony}
        if chosen:
            hevs = [replace(ev, sung=ev.sung + (4 if MODE_OF[style.scale] == "major" else 3), start=ev.start + rng.normal(0, 0.015))
                    for ev in evs if ev.phrase in chosen]
            hp, ha, hf, _, _ = perform(hevs, info, replace(style, scoop=0.1, fall=0.0), rng, seconds)
            harm = _voice(hp, ha, hf * 1.05, info["voice"], style, rng, n_audio)
            y += harm * 10 ** (rng.uniform(-15, -10) / 20)
    if style.reverb:
        y = _reverb(y, style.reverb[0], style.reverb[1], rng)
    peak_rms = np.sqrt(np.convolve(y ** 2, np.ones(SR // 10) / (SR // 10), mode="same").max())
    if style.bleed is not None:
        y += _bleed(n_audio, info, style.bleed, rng) * peak_rms
    y += rng.standard_normal(n_audio) * peak_rms * 10 ** (-62 / 20)
    y *= rng.uniform(0.3, 0.9) / (np.abs(y).max() + 1e-9)

    frames = int(seconds * FPS)
    step = CTRL // FPS
    v = voiced[: frames * step].reshape(frames, step)
    p = np.where(voiced, pitch, 0.0)[: frames * step].reshape(frames, step)
    count = v.sum(1)
    f0_hz = np.where(count > step // 2, _midi_hz(p.sum(1) / np.maximum(count, 1)), 0.0)
    notes = [{"start": round(ev.start, 4), "end": round(ev.end, 4), "pitch": ev.pitch,
              "hz": round(float(_midi_hz(ev.sung)), 3), "kind": ev.kind, "spoken": ev.kind == "rap",
              "phrase": ev.phrase, "link": ev.link} for ev in evs if ev.end - ev.start >= 0.03]
    return y.astype(np.float32), {**info, "notes": notes}, f0_hz.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("style", nargs="?", choices=list(STYLES))
    ap.add_argument("out", nargs="?", default="synth.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.style:
        for s in STYLES.values():
            print(f"{s.name:<11} {s.genre}")
        return
    import soundfile as sf
    audio, ref, _ = render(args.style, args.seed, args.seconds)
    sf.write(args.out, audio, SR)
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(ref, f, indent=1)
    print(f"{args.out}: {len(ref['notes'])} notes, {ref['genre']}, {ref['tempo']} BPM, keys {ref['keys']}")


if __name__ == "__main__":
    main()

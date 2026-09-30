"""Turn a frame-level pitch track into clean, discrete MIDI notes."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=float) - 69) / 12)


def hz_to_midi(hz):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69 + 12 * np.log2(np.asarray(hz, dtype=float) / 440.0)


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    name: str
    hz: float  # measured (median) frequency of the sung note
    note_hz: float  # equal-tempered frequency of `pitch` (A4 = 440 Hz)
    cents: float  # how far the singer was from `note_hz`
    velocity: int
    confidence: float

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d["duration"] = round(self.duration, 4)
        return d


@dataclass
class Settings:
    method: str = "hybrid"  # "hybrid": learned onsets + HMM pitch | "ml": learned onsets only | "hmm": pitch-only HMM
    onset_threshold: float | None = None  # ml: onset probability that starts a note; higher = fewer, longer notes
    smooth: bool = True  # ml: fold split-second notes whose pitch never settles (slides) into a neighbour
    blip: float = 0.12  # seconds; notes shorter than this are candidates for folding / glitch removal
    blip_spread: float = 0.5  # semitones (std) of pitch wobble that marks a short note as a slide
    conf_threshold: float = 0.4  # pitch-model confidence where a frame counts as sung
    silence_db: float = 45.0  # frames this far below the loudest vocal are unvoiced
    sigma: float = 0.5  # semitones of pitch wobble (vibrato, scoops) a note tolerates
    change_cost: float = 6.0  # evidence needed to switch notes; higher = fewer, steadier notes
    leap_cost: float = 1.0  # extra cost per semitone for leaps wider than 5 semitones
    octave_error_cost: float = 0.35  # per frame, to read the tracked pitch as an octave off
    onset_cost: float = 3.0  # evidence needed to start/stop singing
    outlier_cap: float = 4.0  # limits how much one stray frame can count against a note
    min_note: float = 0.06  # seconds
    split_db: float = 10.0  # loudness dip that re-articulates a held pitch
    legato: float = 0.25  # seconds; stretch notes across gaps shorter than this (0 = off)
    min_pitch: int = 36  # C2
    max_pitch: int = 96  # C7
    auto_tuning: bool = True  # correct for a song not tuned to A440 (see reference_tuning)
    reference_tuning: float | None = None  # semitones off A440, measured from the backing track
    autotune: str = "auto"  # "auto" | "on" | "off": snap to the song's key like pitch correction does
    key: str | None = None  # e.g. "G major" / "E minor"; None = detect from the vocal
    scale_penalty: float | None = None  # cost per frame for out-of-key notes; None = by mode
    blip_conf: float = 0.65  # hmm + autotune: short notes below this confidence are glitches


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
SCALE_STEPS = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def parse_key(key: str):
    """'G major', 'E minor', 'Bb', 'F#m' -> (root pitch class, mode)."""
    k = key.strip()
    mode = "minor" if k.lower().endswith(("minor", "min")) or (k.endswith("m") and not k.lower().endswith("major")) else "major"
    tonic = k.split()[0] if " " in k else k.rstrip("m")
    root = NOTE_NAMES.index(tonic[0].upper())
    root += tonic[1:].count("#") - tonic[1:].count("b")
    return root % 12, mode


def detect_key(midi: np.ndarray, key: str | None = None):
    """(name, 12-bool scale mask) from a pitch-class histogram (Krumhansl-Schmuckler)."""
    if key:
        root, mode = parse_key(key)
    else:
        hist = np.bincount(np.round(midi).astype(int) % 12, minlength=12).astype(float)
        root, mode = max(((r, m) for r in range(12) for m in ("major", "minor")),
                         key=lambda rm: np.corrcoef(hist, np.roll(
                             MAJOR_PROFILE if rm[1] == "major" else MINOR_PROFILE, rm[0]))[0, 1])
    mask = np.zeros(12, bool)
    mask[[(root + k) % 12 for k in SCALE_STEPS[mode]]] = True
    return f"{NOTE_NAMES[root]} {mode}", mask


def _runs(mask: np.ndarray):
    """(start, end) index pairs of consecutive True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def estimate_tuning(midi: np.ndarray) -> float:
    """Global offset (semitones, -0.5..0.5) of the performance from A440 tuning."""
    if len(midi) == 0:
        return 0.0
    phase = np.exp(2j * np.pi * (midi - np.round(midi)))
    return float(np.angle(phase.mean()) / (2 * np.pi))


def _viterbi(pitch, voice_p, weight, lo, hi, cfg: Settings, sigma: float, prior: np.ndarray) -> np.ndarray:
    """Most likely note per frame (-1 = silence), HMM-style as in pYIN/Tony.

    Every frame scores how well each semitone explains the sung pitch: a Gaussian
    `sigma` wide, clipped so a stray frame can't dominate, scaled by how confident
    the tracker was (`weight`), plus a per-note `prior` such as an out-of-key penalty.
    Reading the pitch as an octave off is allowed at a per-frame cost, because
    trackers do slip by an octave. Starting/stopping costs `onset_cost`; changing
    notes costs `change_cost` plus `leap_cost` per semitone beyond a 5th-ish leap.

    An excursion has to outweigh those costs over its whole duration: vibrato swings
    back too fast to pay for two note changes; a brief jump far away (or an octave
    slip) can't pay for two big leaps; real notes that hold their pitch do.
    """
    T, K = len(pitch), hi - lo + 1
    semis = np.arange(lo, hi + 1)
    log_v = np.log(np.clip(voice_p, 1e-4, 1))
    log_u = np.log(np.clip(1 - voice_p, 1e-4, 1))
    obs = np.nan_to_num(pitch, nan=-1e3)[:, None]

    def fit(d):
        return np.maximum(-d**2 / (2 * sigma**2), -cfg.outlier_cap)

    as_sung = fit(obs - semis)
    octave_off = np.maximum(fit(obs - 12 - semis), fit(obs + 12 - semis)) - cfg.octave_error_cost
    emit = log_v[:, None] + weight[:, None] * np.maximum(as_sung, octave_off) + prior[None, :]

    # States: S (silence, no melodic memory), N_k (singing note k), and R_k (a short
    # rest that still remembers note k). A leap is charged whether it happens
    # directly or across a breath/consonant gap. Each rest frame costs `rest_decay`,
    # so after roughly half a second forgetting (`rest_forget`) becomes cheaper and a
    # new phrase can start anywhere.
    leap_pen = cfg.leap_cost * np.maximum(0, np.abs(semis[:, None] - semis[None, :]) - 5).astype(float)
    n_to_n = -(cfg.change_cost + leap_pen)
    np.fill_diagonal(n_to_n, 0.0)
    r_to_n = -(cfg.onset_cost + leap_pen)
    rest_decay, rest_forget = 0.1, 8.0

    back = np.zeros((T, 2 * K + 1), dtype=np.int16)  # 0 = S, 1..K = N, K+1..2K = R
    sil = log_u[0]
    note = emit[0] - cfg.onset_cost
    rest = np.full(K, -np.inf)
    cols = np.arange(K)
    for t in range(1, T):
        # -> N_k
        a = note[:, None] + n_to_n
        b = rest[:, None] + r_to_n
        ja, jb = a.argmax(0), b.argmax(0)
        va, vb, vs = a[ja, cols], b[jb, cols], sil - cfg.onset_cost
        best = np.maximum(np.maximum(va, vb), vs)
        back[t, 1:K + 1] = np.where(best == va, ja + 1, np.where(best == vb, jb + K + 1, 0))
        new_note = best + emit[t]
        # -> R_k (only from N_k or R_k)
        from_note = note - cfg.onset_cost
        back[t, K + 1:] = np.where(from_note > rest, cols + 1, cols + K + 1)
        new_rest = np.maximum(from_note, rest) + log_u[t] - rest_decay
        # -> S
        k = int(np.argmax(rest))
        back[t, 0] = 0 if sil >= rest[k] - rest_forget else k + K + 1
        sil = max(sil, rest[k] - rest_forget) + log_u[t]
        note, rest = new_note, new_rest

    final = np.concatenate(([sil], note, rest))
    state = int(np.argmax(final))
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return np.where((path >= 1) & (path <= K), path - 1 + lo, -1)


def _split_on_dips(s: int, e: int, rms_db: np.ndarray, split_db: float, min_len: int):
    """Split a held note where loudness dips (a new syllable on the same pitch)."""
    level = median_filter(rms_db[s:e], size=5, mode="nearest")
    dip = level < np.median(level) - split_db
    pieces, cursor = [], s
    for ds, de in _runs(dip):
        if ds == 0 or de == len(dip):
            continue  # fades at the edges aren't re-articulations
        cut = s + ds + int(np.argmin(level[ds:de]))
        if cut - cursor >= min_len and e - cut >= min_len:
            pieces.append((cursor, cut))
            cursor = cut + 1
    pieces.append((cursor, e))
    return pieces


def _note(s, e, pitch, sung, fps, conf, rms_db, ref_db):
    loudness = float(np.percentile(rms_db[s:e], 90))
    return Note(
        start=round(s / fps, 4),
        end=round(e / fps, 4),
        pitch=pitch,
        name=note_name(pitch),
        hz=round(float(midi_to_hz(sung)), 2),
        note_hz=round(float(midi_to_hz(pitch)), 2),
        cents=round((sung - pitch) * 100, 1),
        velocity=int(np.clip(np.interp(loudness, [ref_db - 30, ref_db], [40, 115]), 1, 127)),
        confidence=round(float(np.mean(conf[s:e])), 3),
    )


def extract_notes(track: dict, fps: float, cfg: Settings = Settings(), onset_prob=None):
    """Returns (notes, info) where info holds the tuning offset and the pitch curve.

    `onset_prob` (per-frame note-onset probability) is normally computed by the onset
    model; passing it in lets training/benchmark code evaluate a specific model.
    """
    f0, conf, rms_db = track["f0"], track["conf"], track["rms_db"]
    midi = hz_to_midi(np.where(f0 > 0, f0, np.nan))

    # Soft voicing probability: confident, loud-enough, in-range frames.
    loud = rms_db > np.percentile(rms_db, 99) - cfg.silence_db
    in_range = (midi >= cfg.min_pitch - 0.5) & (midi <= cfg.max_pitch + 0.5)
    voice_p = 1 / (1 + np.exp(-(conf - cfg.conf_threshold) / 0.08))
    voice_p = np.where(loud & in_range, voice_p, 0.0)
    voiced = voice_p > 0.5

    info = {"tuning_cents": 0.0, "voiced": voiced, "midi_curve": midi, "key": None,
            "autotune": False, "on_note_ratio": 0.0}
    if not voiced.any():
        return [], info

    # Tuning: trust the backing track's reference pitch when we have it. Without one,
    # assume A440: estimating tuning from natural singing is unreliable (scoops and
    # slides skew it by up to ±40 cents, which pushes whole notes onto the wrong
    # semitone). Autotuned vocals are the exception: they sit exactly on their
    # reference, so the vocal itself reveals it precisely.
    tuning = cfg.reference_tuning if (cfg.auto_tuning and cfg.reference_tuning is not None) else 0.0
    clean = midi - tuning  # single-frame glitches removed
    for s, e in _runs(np.isfinite(clean)):
        clean[s:e] = median_filter(clean[s:e], size=min(5, (e - s) | 1), mode="nearest")

    # Autotune pins the voice to exact semitones, so far more frames sit within
    # 10 cents of a note than in natural singing (roughly 25-50%).
    on_note = float(np.mean(np.abs(clean[voiced] - np.round(clean[voiced])) < 0.1))
    autotuned = cfg.autotune == "on" or (cfg.autotune == "auto" and on_note >= 0.55)
    if autotuned and cfg.auto_tuning and cfg.reference_tuning is None:
        extra = estimate_tuning(clean[voiced])
        tuning += extra
        clean -= extra
    key_name, scale = detect_key(clean[voiced], cfg.key)
    info.update(key=key_name, autotune=autotuned, on_note_ratio=round(on_note, 3),
                tuning_cents=round(tuning * 100, 1))

    min_len = max(1, int(round(cfg.min_note * fps)))
    ref_db = np.percentile(rms_db[voiced], 95)
    if cfg.method == "hmm":
        notes = _notes_from_hmm(clean, voice_p, voiced, conf, rms_db, tuning, cfg, scale, autotuned,
                                min_len, fps, ref_db)
    else:
        # hybrid: the learned onsets decide where notes start and end; the HMM (with its
        # octave-slip, leap and key protections) decides which note each one is.
        path = _hmm_path(clean, voice_p, voiced, conf, cfg, scale, autotuned) if cfg.method == "hybrid" else None
        notes = _notes_from_onsets(track, clean, voiced, tuning, cfg, min_len, fps, ref_db, onset_prob, path)

    # Legato: hold each note until the next one starts when the gap is just a
    # consonant or breath, instead of leaving choppy silences.
    for n, nxt in zip(notes, notes[1:]):
        if 0 < nxt.start - n.end <= cfg.legato:
            n.end = nxt.start

    return notes, info


def _notes_from_onsets(track, clean, voiced, tuning, cfg, min_len, fps, ref_db, onset_prob, path=None):
    """Notes are the spans between learned onsets; each gets the pitch held in its middle
    (with an HMM `path`, the note the HMM holds there)."""
    from .onsets import get_model, model_info, pitch_curve, segment

    if onset_prob is None:
        onset_prob = get_model().onset_probability(track)
    threshold = cfg.onset_threshold if cfg.onset_threshold is not None else model_info()["threshold"]
    _, tracked = pitch_curve(track)
    notes = []
    for s, e in segment(onset_prob, tracked & voiced, threshold, min_len):
        trim = (e - s) // 5
        core = clean[s + trim:e - trim]
        core = core[np.isfinite(core)]
        if len(core) < 3:
            continue
        est = float(np.median(core))
        pitch = int(np.round(est))
        if path is not None:
            held = path[s + trim:e - trim]
            held = held[held >= 0]
            if len(held):
                pitch = int(np.bincount(held).argmax())
                est -= 12 * np.round((est - pitch) / 12)  # the HMM corrected an octave slip
        notes.append(_note(s, e, pitch, est + tuning, fps, track["conf"], track["rms_db"], ref_db))
    if cfg.smooth:
        notes = _fold_slides(notes, clean, fps, cfg.blip, cfg.blip_spread)
    return notes


def _fold_slides(notes, clean, fps, max_len, max_spread, max_gap=0.05):
    """A split-second note whose pitch never settles is a slide or a run's passing blur,
    not a note: fold it into the touching neighbour closest in pitch. Measured on expert
    transcriptions, these are right only about half the time, and on a piano they are
    the blips that sound off key."""
    notes = list(notes)
    k = 0
    while k < len(notes):
        n = notes[k]
        seg = clean[int(round(n.start * fps)):int(round(n.end * fps))]
        seg = seg[np.isfinite(seg)]
        if n.duration < max_len and (len(seg) < 3 or np.std(seg) > max_spread):
            touching = [j for j in (k - 1, k + 1) if 0 <= j < len(notes)
                        and (n.start - notes[j].end if j < k else notes[j].start - n.end) < max_gap]
            if touching:
                center = float(np.median(seg)) if len(seg) else notes[touching[0]].pitch
                j = min(touching, key=lambda j: abs(notes[j].pitch - center))
                if j < k:
                    notes[j].end = n.end
                else:
                    notes[j].start = n.start
                del notes[k]
                continue
        k += 1
    return notes


def _hmm_path(clean, voice_p, voiced, conf, cfg, scale, autotuned):
    lo = max(cfg.min_pitch, int(np.floor(np.nanpercentile(clean[voiced], 0.5))) - 2)
    hi = min(cfg.max_pitch, int(np.ceil(np.nanpercentile(clean[voiced], 99.5))) + 2)
    sigma = min(cfg.sigma, 0.35) if autotuned else cfg.sigma
    penalty = cfg.scale_penalty if cfg.scale_penalty is not None else (0.8 if autotuned else 0.25)
    prior = np.where(scale[np.arange(lo, hi + 1) % 12], 0.0, -penalty)
    weight = np.clip((conf - 0.2) / 0.6, 0.25, 1.0)  # trust confident frames more
    return _viterbi(clean, voice_p, weight, lo, hi, cfg, sigma, prior)


def _notes_from_hmm(clean, voice_p, voiced, conf, rms_db, tuning, cfg, scale, autotuned, min_len, fps, ref_db):
    path = _hmm_path(clean, voice_p, voiced, conf, cfg, scale, autotuned)

    bounds = np.flatnonzero(np.diff(path)) + 1
    notes = []
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(path)]):
        if path[s] < 0 or e - s < min_len:
            continue
        for ns, ne in _split_on_dips(s, e, rms_db, cfg.split_db, max(min_len, int(0.15 * fps))):
            if ne - ns < min_len:
                continue
            # Measure the sung pitch across the middle, skipping scoops in and falls off.
            trim = (ne - ns) // 5
            core = clean[ns + trim:ne - trim]
            core = core[np.isfinite(core)]
            est = float(np.mean(core)) if len(core) else float(path[s])
            est -= 12 * np.round((est - path[s]) / 12)  # undo octave slips the HMM corrected
            # Autotuned vocals sit on the snapped note; slides in and out would only
            # drag an average off it.
            if autotuned or abs(est - path[s]) < 0.75:
                pitch = int(path[s])
            else:
                pitch = int(np.round(est))
            notes.append(_note(ns, ne, pitch, est + tuning, fps, conf, rms_db, ref_db))

    # With autotune the pitch is rock-steady, so split-second, low-confidence notes
    # are the tracker catching a slide or a backing vocal; the previous note carries
    # through them instead (see legato). In natural fast singing they're often real.
    if autotuned:
        notes = [n for n in notes if not (n.duration < cfg.blip and n.confidence < cfg.blip_conf)]

    # Adjacent same-pitch fragments that touch become one note. Dip splits leave a
    # one-frame gap, so they stay separate.
    merged = []
    for n in notes:
        prev = merged[-1] if merged else None
        if prev and prev.pitch == n.pitch and n.start - prev.end < 0.5 / fps:
            prev.end = n.end
            prev.velocity = max(prev.velocity, n.velocity)
        else:
            merged.append(n)
    return merged

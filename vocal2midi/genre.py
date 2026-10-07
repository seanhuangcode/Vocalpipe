"""What kind of song this is: its genre, and how it is sung.

Genre: YAMNet (yamnet.py) rates the original mix every 0.48 s against AudioSet's genre
classes. These are averaged over the song and pooled into the families people mean
(Hip-hop, Rock, Electronic, ...): a family's score is the chance that any of its classes
applies. How the song is sung is measured from the separated vocal: rapped or sung (YAMNet's
speech vs. singing classes on the vocal, plus the pitch analysis in notes.py), vibrato,
autotune, vocal runs and range. A song that is mostly rapped is likely hip-hop; wide, steady
vibrato points to classical; autotune to pop, electronic or hip-hop. Those nudge the genre.

The genre in turn tunes the transcription (PROFILES): hip-hop is readier to hear a phrase
as rapped; R&B, soul and gospel keep the quick notes of vocal runs.
"""

from dataclasses import fields, replace

import numpy as np

from .notes import note_name, run_share

# AudioSet class indices in YAMNet's output, for the classes used here
CLASSES = {
    "Speech": 0, "Conversation": 2, "Narration, monologue": 3, "Singing": 24, "Choir": 25, "Yodeling": 26,
    "Chant": 27, "Mantra": 28, "Child singing": 29, "Synthetic singing": 30, "Rapping": 31, "Humming": 32,
    "Music": 132, "Pop music": 211, "Hip hop music": 212, "Rock music": 214, "Heavy metal": 215,
    "Punk rock": 216, "Grunge": 217, "Progressive rock": 218, "Rock and roll": 219, "Psychedelic rock": 220,
    "Rhythm and blues": 221, "Soul music": 222, "Reggae": 223, "Country": 224, "Swing music": 225,
    "Bluegrass": 226, "Funk": 227, "Folk music": 228, "Middle Eastern music": 229, "Jazz": 230, "Disco": 231,
    "Classical music": 232, "Opera": 233, "Electronic music": 234, "House music": 235, "Techno": 236,
    "Dubstep": 237, "Drum and bass": 238, "Electronica": 239, "Electronic dance music": 240,
    "Ambient music": 241, "Trance music": 242, "Music of Latin America": 243, "Salsa music": 244,
    "Flamenco": 245, "Blues": 246, "Music for children": 247, "New-age music": 248, "Vocal music": 249,
    "A capella": 250, "Music of Africa": 251, "Afrobeat": 252, "Christian music": 253, "Gospel music": 254,
    "Music of Asia": 255, "Carnatic music": 256, "Music of Bollywood": 257, "Ska": 258,
    "Traditional music": 259, "Silence": 494,
}
GENRES = {
    "Pop": ("Pop music",),
    "Rock": ("Rock music", "Punk rock", "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock"),
    "Metal": ("Heavy metal",),
    "Hip-hop": ("Hip hop music", "Rapping"),
    "R&B / Soul": ("Rhythm and blues", "Soul music", "Funk"),
    "Electronic": ("Electronic music", "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
                   "Electronic dance music", "Trance music", "Disco"),
    "Jazz": ("Jazz", "Swing music"),
    "Blues": ("Blues",),
    "Country": ("Country", "Bluegrass"),
    "Folk": ("Folk music", "Traditional music"),
    "Classical": ("Classical music", "Opera"),
    "Latin": ("Music of Latin America", "Salsa music", "Flamenco"),
    "Reggae": ("Reggae", "Ska"),
    "Gospel": ("Gospel music", "Christian music"),
    "World": ("Middle Eastern music", "Music of Africa", "Afrobeat", "Music of Asia", "Carnatic music",
              "Music of Bollywood"),
    "Ambient": ("Ambient music", "New-age music"),
    "Children's": ("Music for children",),
    "A cappella": ("A capella", "Vocal music"),
}
SPEECH = ("Speech", "Narration, monologue", "Conversation", "Rapping")
SINGING = ("Singing", "Choir", "Chant", "Humming", "Child singing", "Synthetic singing", "Yodeling", "Mantra")

# How each genre tunes note extraction (only settings still at their defaults are changed).
# Checked on tools/style_benchmark.py: a readier rap detector clears the last rapped syllables
# out of hip-hop melodies without touching sung phrases; R&B and gospel keep the quick notes
# of vocal runs more readily (any song does once a fifth of its notes are in runs).
PROFILES = {
    "Hip-hop": {"spoken": 0.5},
    "R&B / Soul": {"run_share": 0.1},
    "Gospel": {"run_share": 0.1},
}


def _idx(names):
    return [CLASSES[n] for n in names]


def genre_scores(mix_scores: np.ndarray) -> tuple[dict, float]:
    """YAMNet scores on the mix (patches, 521) -> ({family: 0..1}, how much of it is music)."""
    if not len(mix_scores):
        return {g: 0.0 for g in GENRES}, 0.0
    music = mix_scores[:, CLASSES["Music"]]
    keep = music >= min(0.2, np.percentile(music, 50))  # skip spoken intros, silence, applause
    mean = mix_scores[keep].mean(0)
    return {g: float(1 - np.prod(1 - mean[_idx(cls)])) for g, cls in GENRES.items()}, float(np.mean(music))


def speech_curve(vocal_scores: np.ndarray, n_frames: int, fps: float = 100.0):
    """Per pitch frame, from YAMNet on the separated vocal: how speech-like the voice is
    (0 = sung, 1 = spoken/rapped), and how sure YAMNet is that there's a voice at all (0..1)."""
    if not len(vocal_scores):
        return np.full(n_frames, 0.5), np.zeros(n_frames)
    speech = vocal_scores[:, _idx(SPEECH)].sum(1)
    sung = vocal_scores[:, _idx(SINGING)].sum(1)
    share = speech / np.maximum(speech + sung, 1e-6)
    weight = np.clip((speech + sung) / 0.2, 0, 1)
    from .yamnet import PATCH_SECONDS
    centres = (np.arange(len(share)) * PATCH_SECONDS + PATCH_SECONDS) * fps  # patch k: k*0.48 .. k*0.48 + 0.96 s
    t = np.arange(n_frames)
    return np.interp(t, centres, share), np.interp(t, centres, weight)


def vocal_style(notes, info) -> dict:
    """How the melody is sung, from the extracted notes and the extraction's measurements."""
    sung = [n for n in notes if not n.spoken]
    pitches = [n.pitch for n in sung]
    return {
        "rapped": info.get("spoken_share", 0.0),
        "autotune": bool(info.get("autotune")),
        "vibrato": info.get("vibrato") or {"share": 0.0, "rate": None, "extent_cents": None},
        "runs": info["runs"] if "runs" in info else round(run_share(sung), 3),
        "range": [note_name(int(np.percentile(pitches, 2))), note_name(int(np.percentile(pitches, 98)))] if pitches else None,
    }


def describe(families: dict, music: float, style: dict | None = None, tempo: float | None = None) -> dict:
    """The genre (YAMNet's family scores, nudged by how the song is sung) and human-readable tags."""
    scores = dict(families)
    tags = []
    if style:
        rapped, vib = style["rapped"], style["vibrato"]
        if rapped >= 0.6:
            tags.append("mostly rapped")
        elif rapped >= 0.12:
            tags.append("rapped verses")
        scores["Hip-hop"] *= 1 + 3 * rapped
        if style["autotune"]:
            tags.append("autotuned")
            for g in ("Pop", "Electronic", "Hip-hop"):
                scores[g] *= 1.3
        if vib["share"] >= 0.3 and (vib["extent_cents"] or 0) >= 25:
            tags.append("lots of vibrato")
            if vib["share"] >= 0.45 and vib["extent_cents"] >= 55:
                scores["Classical"] *= 1.8
        if style["runs"] >= 0.2:
            tags.append("vocal runs")
            for g in ("R&B / Soul", "Gospel"):
                scores[g] *= 1.3
    total = sum(scores.values())
    ranked = sorted(scores, key=scores.get, reverse=True)
    share = {g: scores[g] / total for g in ranked} if total > 0 else {g: 0.0 for g in ranked}
    top = ranked[0] if total > 0 and families[ranked[0]] >= 0.01 and music >= 0.05 else None
    return {
        "genre": top,
        "confidence": round(share[top], 3) if top else 0.0,
        "genres": [[g, round(share[g], 3)] for g in ranked[:3] if share[g] >= 0.05],
        "tempo": round(tempo, 1) if tempo else None,
        "tags": tags,
        "style": style,
    }


def text(g: dict | None) -> str:
    """One line for people: 'Hip-hop (62%), also Pop · ~92 BPM · rapped verses'."""
    if not g or not g.get("genre"):
        return "not recognised"
    others = [name for name, _ in g.get("genres", [])[1:]]
    parts = [f"{g['genre']} ({g['confidence']:.0%})" + (f", also {', '.join(others)}" if others else "")]
    if g.get("tempo"):
        parts.append(f"~{g['tempo']:.0f} BPM")
    return " · ".join(parts + g.get("tags", []))


def profile(cfg, genre: str | None):
    """`cfg` tuned for `genre` (see PROFILES), leaving settings someone already changed alone."""
    if genre not in PROFILES:
        return cfg
    defaults = {f.name: f.default for f in fields(cfg)}
    return replace(cfg, **{k: v for k, v in PROFILES[genre].items() if getattr(cfg, k) == defaults[k]})

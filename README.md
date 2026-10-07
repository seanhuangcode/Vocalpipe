# vocalpipe

Sing along to any song and get scored live. vocalpipe pulls the vocals out of a song, works
out the melody note by note, and turns it into a karaoke lane that tracks your pitch from
the microphone.

## What it does

- **Separates the vocals** from the backing track (Demucs, on your Mac's GPU).
- **Tracks the singer's pitch** with RMVPE and turns it into notes using a small
  note-onset model trained on expert-transcribed singing.
- **Recognises the genre and how the song is sung**: pop, rock, hip-hop, R&B, electronic,
  jazz, country, classical and more (Google's YAMNet sound classifier listening to the
  song), plus the vocal style measured from the singing itself: rapped verses, vibrato,
  autotune, vocal runs, tempo and key changes. Note detection adapts to it.
- **Clean notes for every style of singing**:
  - vibrato is heard as the note it centres on, not as a string of note changes;
  - rapped phrases become rhythm-only notes instead of a made-up melody;
  - reverb tails, backing vocals and instrument bleed left in the vocal don't turn into notes;
  - notes end when the singer stops, not when the reverb does;
  - legato note changes without a consonant are found;
  - vocal runs are kept for singers who do them;
  - key changes are followed.
- **Karaoke app in the browser**: songs from your Mac (found automatically), drag-and-drop,
  a quick start so a new song is singable in about 5 seconds, live pitch scoring, phrase
  looping, key and speed controls. Rapped parts show as hatched "rap" notes, scored on
  rhythm.
- **Command line** for batch processing: MIDI, note tables, a piano render and a piano-roll
  viewer for any song.

## Setup

Needs Python 3.12+ and ffmpeg (`brew install ffmpeg` on macOS).

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
mkdir -p models
curl -L -o models/rmvpe.pt https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
```

The pitch model (`rmvpe.pt`, 181 MB) isn't in the repo; the command above downloads it.
The Demucs model and the genre classifier (`models/yamnet.h5`, 15 MB, from Google) download
themselves the first time a song is processed. Without them, add `--pitch fcpe` (a pitch
model that ships inside its pip package) or `--genre off`.

## Run the karaoke app

```bash
venv/bin/python serve.py
```

Open http://localhost:8765, pick a song (or drop an mp3 onto the page), put on headphones
and press **🎤 Sing**. Press **?** in the app for keyboard shortcuts.

## Command line

```bash
venv/bin/python main.py song.mp3 --open
```

Each song gets a folder in `output/` with the separated stems, `melody.mid`,
`notes.csv` / `notes.json`, `genre.json`, `piano.wav` and an interactive `viewer.html`.
Rapped syllables go on their own MIDI track and are marked `spoken` in the tables.
Run `venv/bin/python main.py --help` for options (`--sensitivity`, `--key`, `--autotune`,
`--genre "Hip-hop"` to tell it the genre, `--no-rap`, …).

## Project layout

| Path | What it is |
|---|---|
| `serve.py` | App server: song library, processing queue, search |
| `main.py` | Command-line pipeline |
| `vocal2midi/` | Separation, pitch tracking, note extraction, the karaoke page |
| `vocal2midi/notes.py` | Pitch track → notes (onsets, HMM, rap, ghost notes, note ends) |
| `vocal2midi/vibrato.py` | Finds vibrato and the note it centres on |
| `vocal2midi/genre.py`, `yamnet.py` | Genre and vocal style; YAMNet in PyTorch |
| `vocal2midi/harmony.py` | Tuning, key, key changes and tempo from the backing track |
| `tools/` | Training and benchmarking the note-onset model; the multi-genre benchmark |
| `models/note_onsets.*` | Trained note-onset model |
| `tests/` | Unit tests (`venv/bin/pip install pytest && venv/bin/python -m pytest tests`) |

## Accuracy

Measured on the [Vocadito](https://zenodo.org/records/5578807) dataset (CC BY 4.0) with
5-fold cross-validation: note F1 0.69 (a second human transcriber scores 0.72), with
15% of note time on the wrong pitch. Retrain or re-check with:

```bash
venv/bin/python tools/train_onsets.py --data path/to/vocadito
venv/bin/python tools/benchmark_notes.py --data path/to/vocadito   # also prints the previous version
```

Vocadito is amateur a cappella singing, so it can't show how other genres are sung.
`tools/style_benchmark.py` renders synthetic singing in 14 styles with exact ground truth
(`tools/singsynth.py`: opera vibrato, R&B runs, rapped verses, hard autotune, rock grit,
jazz blue notes and swing, key changes, heavy reverb, backing harmonies, instrument bleed),
gives each a backing track, and scores the app's note extraction against it.
Results with FCPE pitch, 6 clips per style:

| Style | Note F1 before → now | F1 incl. note ends | Notes made up where nobody sang (per min) |
|---|---|---|---|
| Pop | 0.885 → **0.908** | 0.476 → **0.556** | 2.9 → **0.0** |
| Pop ballad (runs, vibrato) | 0.824 → **0.899** | 0.629 → **0.804** | 0 → 0 |
| Classical / opera | 0.506 → **0.840** | 0.203 → **0.713** | 7.1 → **3.3** |
| R&B / soul | 0.767 → **0.818** | 0.523 → **0.648** | 0.8 → **0.0** |
| Hip-hop (rapped verse, sung hook) | 0.450 → **0.951** | 0.281 → **0.681** | 0 → 0 |
| Pop / EDM, hard autotune | 0.817 → **0.847** | 0.383 → **0.438** | 4.6 → **0.0** |
| Rock | 0.800 → **0.844** | 0.403 → **0.470** | 6.2 → **0.0** |
| Jazz / blues | 0.818 → **0.828** | 0.257 → **0.320** | 1.7 → **0.0** |
| Country | 0.900 → **0.912** | 0.488 → **0.629** | 0 → 0 |
| Folk | 0.956 → **0.964** | 0.566 → **0.674** | 0.4 → **0.0** |
| Autotuned pop with a key change | 0.901 → **0.928** | 0.430 → **0.533** | 4.2 → **0.0** |
| Ballad drenched in reverb | 0.592 → **0.718** | 0.298 → **0.425** | 2.5 → **1.2** |
| Pop with backing harmonies | 0.886 → **0.947** | 0.426 → **0.509** | 8.3 → **0.0** |
| **Mean** | 0.777 → **0.877** | 0.413 → **0.569** | 3.0 → **0.4** |

Rapped syllables transcribed as melody: 100% → 3%; sung notes mistaken for rap: 0%.
Synthetic singing isn't real singing, so treat these as a check that each kind of singing
is handled, not as a promise for real recordings.

```bash
venv/bin/python tools/style_benchmark.py --variants default previous
venv/bin/python tools/singsynth.py rnb rnb.wav   # listen to a style
```

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) for vocal separation
- [RMVPE](https://github.com/Dream-High/RMVPE), with weights from
  [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [FCPE](https://github.com/CNChTu/FCPE) (MIT) for the optional `fcpe` pitch model
- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
  (Google, Apache 2.0), trained on [AudioSet](https://research.google.com/audioset/), for genre recognition
- Vocadito (Bittner et al., 2021) for training data

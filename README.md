# vocalpipe

Sing along to any song and get scored live. vocalpipe pulls the vocals out of a song, works
out the melody note by note, and turns it into a karaoke lane that tracks your pitch from
the microphone.

## What it does

- **Separates the vocals** from the backing track (Demucs, on your Mac's GPU).
- **Tracks the singer's pitch** with RMVPE and turns it into notes using a small
  note-onset model trained on expert-transcribed singing.
- **Karaoke app in the browser**: songs from your Mac (found automatically), drag-and-drop,
  a quick start so a new song is singable in about 5 seconds, live pitch scoring, phrase
  looping, key and speed controls.
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
The Demucs model downloads itself the first time a song is processed.

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
`notes.csv` / `notes.json`, `piano.wav` and an interactive `viewer.html`.
Run `venv/bin/python main.py --help` for options (`--sensitivity`, `--key`, `--autotune`, …).

## Project layout

| Path | What it is |
|---|---|
| `serve.py` | App server: song library, processing queue, search |
| `main.py` | Command-line pipeline |
| `vocal2midi/` | Separation, pitch tracking, note extraction, the karaoke page |
| `tools/` | Training and benchmarking the note-onset model |
| `models/note_onsets.*` | Trained note-onset model |

## Accuracy

Measured on the [Vocadito](https://zenodo.org/records/5578807) dataset (CC BY 4.0) with
5-fold cross-validation: note F1 0.69 (a second human transcriber scores 0.72), with
15% of note time on the wrong pitch. Retrain or re-check with:

```bash
venv/bin/python tools/train_onsets.py --data path/to/vocadito
venv/bin/python tools/benchmark_notes.py --data path/to/vocadito
```

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) for vocal separation
- [RMVPE](https://github.com/Dream-High/RMVPE), with weights from
  [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- Vocadito (Bittner et al., 2021) for training data

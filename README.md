# vocalpipe

Sing along to any song and get scored live. vocalpipe pulls the vocals out of a song, works
out the melody note by note, and turns it into a karaoke lane that tracks your pitch from
the microphone.

## What it does

- **Separates the vocals** from the backing track (Demucs, on your Mac's or NVIDIA graphics card, or the CPU).
- **Tracks the singer's pitch** with RMVPE and turns it into notes using a small
  note-onset model trained on expert-transcribed singing.
- **Karaoke app in the browser**: songs on your computer (found automatically), drag-and-drop,
  a quick start so a new song is singable in about 5 seconds, live pitch scoring, phrase
  looping, key and speed controls.
- **Command line** for batch processing: MIDI, note tables, a piano render and a piano-roll
  viewer for any song.

## Setup

### Windows

1. Install [Python 3.12 or newer](https://www.python.org/downloads/windows/), ticking
   **Add python.exe to PATH** in the installer.
2. Download this repository (green **Code** button → **Download ZIP**, then unzip it) or clone it.
3. Double-click **`setup.bat`**. It installs ffmpeg (with winget), the Python packages (the
   GPU build of PyTorch if you have an NVIDIA card) and the pitch model. It takes a while the
   first time; run it again any time, it skips what's already done.
4. Double-click **`start.bat`**. vocalpipe opens in your browser; close the window to stop it.

Songs are found in your Music, Downloads and Desktop folders (OneDrive ones too). Key changes
need an ffmpeg with the rubberband filter, which the build setup.bat installs includes.

### macOS / Linux

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

On Windows, double-click `start.bat` instead (or run `venv\Scripts\python serve.py`).

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
5-fold cross-validation, scored against annotator 1 (a second human transcriber is the
ceiling to compare with):

| | note F1 | F1 incl. note ends | wrong pitch |
|---|---|---|---|
| second human transcriber | 0.72 | 0.63 | 9.5% |
| vocalpipe | 0.70 | 0.54 | 13.9% |
| earlier model | 0.69 | 0.39 | 15.3% |

What changed: the onset model also sees the cues of the frames 30 ms either side, a note's
pitch is measured from more of the note, and notes are only stretched across consonant-length
gaps (100 ms), not breaths. On synthetic tests built from the same recordings, a backing
harmony a third away at -6 dB or a doubled take costs almost nothing (F1 0.69-0.71), a harmony
as loud as the lead costs a lot (0.63), and short ad-libs between phrases mostly get
transcribed (0.68): the app leaves out notes under 150 ms, which removes the briefest ones.
A loudness-based ad-lib filter was tried and rejected because it removed more real singing
than ad-libs. Retrain or re-check with:

```bash
venv/bin/python tools/train_onsets.py --data path/to/vocadito
venv/bin/python tools/benchmark_notes.py --data path/to/vocadito
```

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) for vocal separation
- [RMVPE](https://github.com/Dream-High/RMVPE), with weights from
  [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- Vocadito (Bittner et al., 2021) for training data

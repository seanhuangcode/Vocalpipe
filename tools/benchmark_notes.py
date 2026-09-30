"""Score note extraction against expert annotations (regression check).

    venv/bin/python tools/benchmark_notes.py --data path/to/vocadito

The installed onset model was trained on these recordings, so its score here is
in-sample; tools/train_onsets.py reports the honest cross-validated numbers.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocadito import TRACKS, Vocadito, evaluate, fmt  # noqa: E402

from vocal2midi.notes import Note, Settings, extract_notes  # noqa: E402
from vocal2midi.pitch import FPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="unzipped vocadito folder")
    args = ap.parse_args()
    data = Vocadito(args.data)
    tracks = {i: data.track(i) for i in TRACKS}
    human = lambda i: [Note(s, e, int(round(69 + 12 * np.log2(h / 440))), "", h, h, 0, 100, 1)
                       for (s, e), h in zip(*data.reference(i, "A2"))]
    print(fmt("human (annotator 2 vs 1)", evaluate(data, human)))
    print(fmt("pitch-only HMM", evaluate(data, lambda i: extract_notes(tracks[i], FPS, Settings(method="hmm"))[0])))
    print(fmt("onset model (in-sample)", evaluate(data, lambda i: extract_notes(tracks[i], FPS, Settings())[0])))


if __name__ == "__main__":
    main()

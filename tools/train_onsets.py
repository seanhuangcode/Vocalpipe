"""Train the note-onset model and report honest (cross-validated) accuracy.

    venv/bin/python tools/train_onsets.py --data path/to/vocadito

Every score comes from recordings the model wasn't trained on: 5-fold cross-validation
by recording, with the onset threshold chosen inside each training split. Notes are
produced by the production code path (vocal2midi.notes.extract_notes). The final model
is then trained on all recordings and saved to models/note_onsets.joblib (+ .json).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocadito import TRACKS, Vocadito, evaluate, fmt, objective  # noqa: E402

from vocal2midi.notes import Note, Settings, extract_notes  # noqa: E402
from vocal2midi.onsets import FEATURE_VERSION, MODEL_PATH, frame_features  # noqa: E402
from vocal2midi.pitch import FPS  # noqa: E402

THRESHOLDS = (0.2, 0.25, 0.3, 0.4, 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="unzipped vocadito folder")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    data = Vocadito(args.data)
    tracks = {i: data.track(i) for i in TRACKS}
    X = {i: frame_features(tracks[i]) for i in TRACKS}
    y = {i: data.onset_labels(i, len(X[i])) for i in TRACKS}
    print(f"{sum(len(v) for v in X.values())} frames, {X[1].shape[1]} features ({time.time() - t0:.0f}s)")

    def fit(ids):
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0).fit(
            np.concatenate([X[i] for i in ids]), np.concatenate([y[i] for i in ids]))

    def notes_for(i, prob, threshold):
        return extract_notes(tracks[i], FPS, Settings(onset_threshold=threshold), onset_prob=prob)[0]

    folds = [TRACKS[k::args.folds] for k in range(args.folds)]
    held_out, chosen = {}, []
    for k, test in enumerate(folds):
        train = [i for i in TRACKS if i not in test]
        inner = [train[j::4] for j in range(4)]
        oof = {}
        for part in inner:
            model = fit([i for i in train if i not in part])
            oof.update({i: model.predict_proba(X[i])[:, 1] for i in part})
        # most accurate threshold; among (near-)ties prefer the higher one = fewer, smoother notes
        best = max(THRESHOLDS, key=lambda t: (round(objective(evaluate(data, lambda i: notes_for(i, oof[i], t), train)), 2), t))
        chosen.append(best)
        model = fit(train)
        held_out.update({i: notes_for(i, model.predict_proba(X[i])[:, 1], best) for i in test})
        print(f"  fold {k + 1}/{args.folds}: threshold {best}")

    human = lambda i: [Note(s, e, int(round(69 + 12 * np.log2(h / 440))), "", h, h, 0, 100, 1)
                       for (s, e), h in zip(*data.reference(i, "A2"))]
    results = {
        "human (annotator 2 vs 1)": evaluate(data, human),
        "pitch-only HMM": evaluate(data, lambda i: extract_notes(tracks[i], FPS, Settings(method="hmm"))[0]),
        "onset model + HMM pitch (cross-validated)": evaluate(data, lambda i: held_out[i]),
    }
    for label, m in results.items():
        print(fmt(label, m))

    threshold = float(np.median(chosen))
    model = fit(TRACKS)
    args.out.parent.mkdir(exist_ok=True)
    joblib.dump(model, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({
        "feature_version": FEATURE_VERSION,
        "threshold": threshold,
        "trained_on": "Vocadito (Bittner et al., 2021), CC BY 4.0 — 40 solo-singing recordings, annotator A1",
        "cross_validated": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
    }, indent=1))
    print(f"saved {args.out} (threshold {threshold}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

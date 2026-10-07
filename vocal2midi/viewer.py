import json
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def write_viewer(path, title, notes, info, fps, duration, backend, audio, genre=None):
    """Self-contained piano-roll page. `audio` maps track -> path relative to the page."""
    curve = np.where(info["voiced"], info["midi_curve"], np.nan)
    data = {
        "title": title,
        "duration": round(float(duration), 3),
        "fps": fps,
        "backend": backend,
        "tuning_cents": info["tuning_cents"],
        "key": info["key"],
        "keys": info.get("keys"),
        "autotune": info["autotune"],
        "genre": genre,
        "audio": audio,
        "notes": [n.to_dict() for n in notes],
        "curve": [None if not np.isfinite(v) else round(float(v), 2) for v in curve],
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    Path(path).write_text(html)

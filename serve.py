"""vocalpipe app server.

    venv/bin/python serve.py            # http://localhost:8765
    venv/bin/python serve.py --port 9000

Finding songs: every song already on this Mac is listed (via Spotlight: instant, whole
disk, with title/artist/cover art), plus search results from ccMixter (full songs under
Creative Commons licences) and Apple's official 30-second previews (iTunes Search API).
Files can also be dropped onto the page.

Processing runs inside this server so the models stay loaded. Each song first gets a quick
preview of its opening seconds, so singing can start within seconds, then the full song is
processed in the background. Then sing along: the page tracks your pitch from the
microphone and scores it against the notes.

Songs live in songs/ (uploads and picks; library songs are linked, not copied) and any
.mp3/.m4a/.flac/.ogg in the project folder. Results are served from output/<song>/, with
HTTP Range support so the browser can seek within the audio.
"""

import argparse
import hashlib
import html
import itertools
import json
import os
import queue
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import vocal2midi  # noqa: F401  (sets MPS fallback before torch loads)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SONGS = ROOT / "songs"
ART = OUTPUT / ".art"
APP = ROOT / "vocal2midi" / "karaoke.html"
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".aif", ".aiff", ".opus"}
ROOT_EXT = AUDIO_EXT - {".wav"}  # loose .wav files in the project root are old outputs
MAX_UPLOAD = 300 * 1024 * 1024
META = SONGS / "meta.json"  # display info (title, artist, artwork) for added songs
LIBRARY_DIRS = [Path.home() / d for d in ("Music", "Downloads", "Desktop")]
SKIP_PARTS = ("/Library/", ".app/", "/node_modules/", "site-packages", "/.Trash/", "/venv/", "/.venv/",
              "/tests/", "/test/", "/fixtures/")
# leftovers from separation/transcription tools aren't songs you'd want to sing
STEM_NAMES = re.compile(r"^(vocals?|instrumental|no[_ ]vocals|accompaniment|acapella|drums|bass|other|stems?|"
                        r"melody[_ ]notes|detected[_ ]melody|piano[_ ]preview\d*|piano|rmvpe[_ ]\w+)( ?\(\d+\))?$"
                        r"|[_ -](vocals?|instrumental|no[_ ]vocals|accompaniment|acapella)( ?\(\d+\))?$"
                        r"|\((instrumental|karaoke[^)]*)\)$", re.I)  # no vocals left to sing along to
PREVIEW_HOSTS = (".apple.com", ".mzstatic.com")
CC_HOSTS = ("ccmixter.org",)

try:  # python.org builds of Python on macOS ship without root certificates
    import certifi
    TLS = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    TLS = ssl.create_default_context()


def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.,'()& ]", "_", name).strip(" .")[:120] or "song"


def load_meta() -> dict:
    try:
        return json.loads(META.read_text())
    except (OSError, ValueError):
        return {}


META_LOCK = threading.Lock()


def save_meta(stem: str, info: dict):
    with META_LOCK:
        meta = load_meta()
        meta[stem] = {**meta.get(stem, {}), **{k: v for k, v in info.items() if v not in (None, "")}}
        SONGS.mkdir(exist_ok=True)
        META.write_text(json.dumps(meta, indent=1))


def split_filename(stem: str):
    """'Artist - Title' filenames are common; use them when a file has no tags."""
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()
    return stem, ""


def read_tags(path) -> dict:
    """Title / artist / album / duration from the file's own tags (ffprobe)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags", "-of", "json", str(path)],
                           capture_output=True, timeout=15)
        fmt = json.loads(r.stdout or b"{}").get("format", {})
    except (subprocess.SubprocessError, ValueError, OSError):
        return {}
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    return {"title": tags.get("title"), "artist": tags.get("artist") or tags.get("album_artist"),
            "album": tags.get("album"), "duration": float(fmt.get("duration") or 0)}


def extract_cover(path, dest: Path) -> bool:
    """Embedded album art (if any) as a small JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-an", "-frames:v", "1",
                        "-vf", "scale=300:300:force_original_aspect_ratio=decrease", str(dest)],
                       capture_output=True, timeout=15)
    except subprocess.SubprocessError:
        return False
    return dest.exists() and dest.stat().st_size > 0


def audio_duration(path) -> float:
    return read_tags(path).get("duration") or 0.0


class Library:
    """Songs already on this Mac.

    Spotlight (mdfind) answers instantly for the whole disk and knows each file's title,
    artist and duration. Where Spotlight isn't available, walk ~/Music, ~/Downloads and
    ~/Desktop instead. Refreshed every couple of minutes so new downloads show up.
    """

    QUERY = ("kMDItemContentTypeTree == 'public.audio' && "
             "kMDItemDurationSeconds >= 60 && kMDItemDurationSeconds <= 1200")
    ATTRS = ["kMDItemAlbum", "kMDItemAuthors", "kMDItemDateAdded", "kMDItemDurationSeconds", "kMDItemTitle"]  # mdls sorts them

    def __init__(self):
        self.songs, self.by_path = [], {}
        self.ready, self.source = False, "spotlight" if shutil.which("mdfind") else "folders"
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                songs = self._dedupe(self._spotlight() if self.source == "spotlight" else self._walk())
                songs.sort(key=lambda s: s.get("added") or "", reverse=True)
                with self.lock:
                    self.songs, self.by_path, self.ready = songs, {s["path"]: s for s in songs}, True
            except Exception:
                traceback.print_exc()
            time.sleep(120)

    @staticmethod
    def _wanted(path: str) -> bool:
        p = Path(path)
        return (p.suffix.lower() in AUDIO_EXT and not any(part in path for part in SKIP_PARTS)
                and not any(part.startswith(".") for part in p.parts) and ROOT not in p.parents
                and not STEM_NAMES.search(p.stem))

    @staticmethod
    def _dedupe(songs):
        """The same file often sits in Downloads and in the Music library: keep one."""
        seen, out = set(), []
        for s in sorted(songs, key=lambda s: (s["where"].startswith("~/Music"), s.get("added") or ""), reverse=True):
            key = (s["title"].lower(), s["artist"].lower(), s["duration"])
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def _spotlight(self):
        found = subprocess.run(["mdfind", "-0", self.QUERY], capture_output=True, timeout=30).stdout
        paths = [p for p in found.decode("utf-8", "replace").split("\0") if p and self._wanted(p)]
        songs = []
        for i in range(0, len(paths), 150):
            chunk = paths[i:i + 150]
            args = ["mdls", "-raw"] + [x for a in self.ATTRS for x in ("-name", a)] + chunk
            raw = subprocess.run(args, capture_output=True, timeout=30).stdout.decode("utf-8", "replace").split("\0")
            for k, path in enumerate(chunk):
                vals = raw[k * len(self.ATTRS):(k + 1) * len(self.ATTRS)]
                if len(vals) < len(self.ATTRS):
                    break
                album, authors, added, dur, title = [None if v in ("(null)", "") else self._text(v) for v in vals]
                artist = ", ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', authors or "")) or (authors if authors and "(" not in authors else None)
                songs.append(self._entry(path, title, artist, album, float(dur) if dur else None, added))
        return songs

    @staticmethod
    def _text(v: str) -> str:
        """mdls -raw writes non-ASCII inside arrays as \\Uxxxx escapes (and decomposed Hangul)."""
        v = re.sub(r"\\U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), v)
        return unicodedata.normalize("NFC", v)

    def _walk(self):
        songs = []
        for base in LIBRARY_DIRS:
            for dirpath, dirnames, filenames in os.walk(base, onerror=lambda e: None):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "venv", "Library")]
                for f in filenames:
                    path = os.path.join(dirpath, f)
                    if self._wanted(path):
                        songs.append(self._entry(path, None, None, None, None, None))
        return songs

    @staticmethod
    def _entry(path, title, artist, album, duration, added):
        p = Path(path)
        guess_title, guess_artist = split_filename(p.stem)
        return {"path": path, "title": title or guess_title, "artist": artist or guess_artist, "album": album or "",
                "duration": round(duration) if duration else None, "added": added or "",
                "where": str(p.parent).replace(str(Path.home()), "~")}

    def browse(self, query: str = "", limit: int = 400):
        words = query.lower().split()
        with self.lock:
            songs = list(self.songs)
        if not words:
            return songs[:limit], len(songs)
        hits = []
        for s in songs:
            hay = f"{s['title']} {s['artist']} {s['album']} {Path(s['path']).stem}".lower()
            if all(w in hay for w in words):
                hits.append((0 if all(w in f"{s['title']} {s['artist']}".lower() for w in words) else 1, s))
        hits.sort(key=lambda h: h[0])
        return [s for _, s in hits[:limit]], len(hits)

    def get(self, path: str):
        with self.lock:
            return self.by_path.get(path)


def itunes_search(query: str, limit: int = 8):
    url = f"https://itunes.apple.com/search?term={quote(query)}&media=music&entity=song&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "vocalpipe/1.0"})
    with urllib.request.urlopen(req, timeout=8, context=TLS) as r:
        results = json.load(r).get("results", [])
    return [{
        "title": x.get("trackName", ""), "artist": x.get("artistName", ""), "album": x.get("collectionName", ""),
        "artwork": (x.get("artworkUrl100") or "").replace("100x100bb", "300x300bb"),
        "preview": x.get("previewUrl"), "buy": x.get("trackViewUrl", ""),
    } for x in results if x.get("previewUrl")]


def ccmixter_search(query: str, limit: int = 6):
    """Full songs under Creative Commons licences; prefer tracks tagged with vocals."""
    url = f"https://ccmixter.org/api/query?f=json&search={quote(query)}&tags=vocals&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "vocalpipe/1.0"})
    with urllib.request.urlopen(req, timeout=8, context=TLS) as r:
        results = json.load(r)
    out = []
    for x in results:
        mp3 = next((f for f in x.get("files", []) if f.get("download_url", "").lower().endswith(".mp3")), None)
        if not mp3:
            continue
        out.append({
            "title": x.get("upload_name", ""), "artist": x.get("user_real_name") or x.get("user_name", ""),
            "license": x.get("license_name", "").strip(), "license_url": x.get("license_url", ""),
            "page": x.get("file_page_url", ""), "url": mp3["download_url"],
            "length": (mp3.get("file_format_info") or {}).get("ps", ""),
        })
    return out


def download(url: str, dest: Path, hosts, max_mb: int):
    host = urlparse(url).hostname or ""
    if urlparse(url).scheme != "https" or not (host in hosts or host.endswith(hosts)):
        raise ValueError(f"downloads are only allowed from {', '.join(h.lstrip('.') for h in hosts)}")
    req = urllib.request.Request(url, headers={"User-Agent": "vocalpipe/1.0"})
    with urllib.request.urlopen(req, timeout=60, context=TLS) as r:
        data = r.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError("file too large")
    dest.write_bytes(data)


class KeyShifter:
    """Pitch-shifted copies of a song's stems (rubberband, formants preserved), cached on disk."""

    def __init__(self):
        self.locks, self.guard = {}, threading.Lock()

    def get(self, stem: str, semitones: int):
        src_dir = OUTPUT / stem
        if semitones == 0:
            return {"backing": "instrumental.wav", "guide": "vocals.wav"}
        with self.guard:
            lock = self.locks.setdefault((stem, semitones), threading.Lock())
        out_dir = src_dir / "keys"
        names = {"backing": f"instrumental_{semitones:+d}.m4a", "guide": f"vocals_{semitones:+d}.m4a"}
        with lock:
            out_dir.mkdir(exist_ok=True)
            jobs = []
            for track, src in (("backing", "instrumental.wav"), ("guide", "vocals.wav")):
                dest = out_dir / names[track]
                if dest.exists():
                    continue
                tmp = dest.with_suffix(".tmp.m4a")
                jobs.append((subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(src_dir / src), "-af",
                     f"rubberband=pitch={2 ** (semitones / 12):.6f}:formant=preserved:pitchq=quality",
                     "-c:a", "aac", "-b:a", "192k", str(tmp)]), tmp, dest))
            for proc, tmp, dest in jobs:  # both stems shift in parallel
                if proc.wait() != 0:
                    raise RuntimeError("ffmpeg pitch shift failed")
                tmp.rename(dest)
        return {k: f"keys/{v}" for k, v in names.items()}


SHIFTER = KeyShifter()


def list_sources():
    """Song files by stem; songs/ wins over the project root."""
    found = {}
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix.lower() in ROOT_EXT:
            found[p.stem] = p
    if SONGS.is_dir():
        for p in sorted(SONGS.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXT:
                found[p.stem] = p
    return found


class Jobs:
    """Processes songs one at a time, inside this server so the models stay loaded.

    Each song first gets a quick preview of its opening PREVIEW_SECONDS (state "preview"
    once it's singable), then the full song follows. Previews always go ahead of full runs,
    so a newly picked song is singable in seconds even while another one is finishing.
    """

    PREVIEW_SECONDS = 45
    # progress bands per pipeline stage: preview is mostly separation; full adds rendering
    BANDS = {"preview": {"Separating vocals": (0.0, 0.75), "Tracking pitch": (0.75, 0.9), "Finding notes": (0.9, 1.0)},
             "full": {"Separating vocals": (0.0, 0.8), "Tracking pitch": (0.8, 0.92), "Finding notes": (0.92, 0.96),
                      "Rendering": (0.96, 1.0)}}

    def __init__(self):
        self.status, self.lock = {}, threading.Lock()
        self.queue, self.seq = queue.PriorityQueue(), itertools.count()
        self.device, self.warm = "cpu", threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def get(self, stem):
        with self.lock:
            return dict(self.status.get(stem, {}))

    def _set(self, stem, **kw):
        with self.lock:
            self.status.setdefault(stem, {}).update(kw)

    def submit(self, stem, path):
        with self.lock:
            if self.status.get(stem, {}).get("state") in ("queued", "running", "preview"):
                return
            self.status[stem] = {"state": "queued", "stage": "Waiting for another song", "progress": 0.0}
        long_song = audio_duration(path) > self.PREVIEW_SECONDS + 15
        preview_ready = (OUTPUT / stem / "preview" / "notes.json").exists()
        if long_song and not preview_ready:
            self.queue.put((0, next(self.seq), "preview", stem, path))
        else:
            if preview_ready:
                self._set(stem, state="preview", stage="Loading the rest of the song")
            self.queue.put((1, next(self.seq), "full", stem, path))

    def _worker(self):
        self._warm_up()
        while True:
            _, _, kind, stem, path = self.queue.get()
            try:
                self._run(kind, stem, path)
            except Exception as e:  # keep the worker alive for the next song
                traceback.print_exc()
                self._set(stem, state="error", stage=f"Processing failed: {e}", log=[str(e)])

    def _warm_up(self):
        """Load the models and run a tiny clip through them, so the first song starts fast."""
        try:
            from vocal2midi.audio import pick_device
            from vocal2midi.pipeline import run
            import numpy as np
            import soundfile as sf
            self.device = pick_device("auto")
            tmp = OUTPUT / ".warmup"
            tmp.mkdir(parents=True, exist_ok=True)
            t = np.arange(44100 * 3) / 44100
            sf.write(tmp / "tone.wav", 0.2 * np.sin(2 * np.pi * 220 * t), 44100)
            t0 = time.time()
            run(tmp / "tone.wav", tmp, device=self.device, preview_seconds=3, log=lambda m: None)
            print(f"models ready on {self.device} ({time.time() - t0:.1f}s)")
        except Exception:
            traceback.print_exc()
        self.warm.set()

    def _run(self, kind, stem, path):
        from vocal2midi.pipeline import run

        bands = self.BANDS[kind]

        def progress(stage, frac):
            lo, hi = bands.get(stage, (0, 1))
            self._set(stem, stage=stage, progress=round(lo + (hi - lo) * frac, 3))

        t0 = time.time()
        if kind == "preview":
            self._set(stem, state="running", stage="Quick start", progress=0.0)
            _, _, meta = run(path, OUTPUT / stem, device=self.device, preview_seconds=self.PREVIEW_SECONDS,
                             progress=progress, log=lambda m: None)
            self._set(stem, state="preview", stage="Loading the rest of the song", progress=0.0,
                      preview_until=meta["duration"], preview_seconds=round(time.time() - t0, 1))
            self.queue.put((1, next(self.seq), "full", stem, path))
        else:
            if self.get(stem).get("state") != "preview":
                self._set(stem, state="running", stage="Starting", progress=0.0)
            run(path, OUTPUT / stem, device=self.device, progress=progress, log=lambda m: None)
            self._set(stem, state="done", stage="Ready", progress=1.0, full_seconds=round(time.time() - t0, 1))


JOBS = Jobs()
LIBRARY = Library()


def describe_song(stem: str, path: Path, info: dict):
    """Fill in title/artist/cover from the file's tags when the caller didn't know them."""
    tags = read_tags(path)
    guess_title, guess_artist = split_filename(path.stem)
    info.setdefault("title", tags.get("title") or guess_title)
    info["title"] = info["title"] or tags.get("title") or guess_title
    info["artist"] = info.get("artist") or tags.get("artist") or guess_artist
    info["album"] = info.get("album") or tags.get("album") or ""
    if not info.get("artwork") and extract_cover(path, OUTPUT / stem / "cover.jpg"):
        info["artwork"] = f"/{quote(stem)}/cover.jpg"
    return info


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        if not getattr(self, "cacheable", False):
            self.send_header("Cache-Control", "no-store")  # always see the latest run
        super().end_headers()

    # ---------- routing ----------
    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self.send_file(APP, "text/html; charset=utf-8")
        if url.path == "/library":
            return self.send_library()
        if url.path == "/api/songs":
            return self.send_json(self.songs())
        if url.path == "/api/search":
            return self.search(qs.get("q", [""])[0].strip())
        if url.path == "/api/library":
            songs, total = LIBRARY.browse(qs.get("q", [""])[0].strip(), int(qs.get("limit", ["400"])[0]))
            return self.send_json({"songs": songs, "total": total, "ready": LIBRARY.ready, "source": LIBRARY.source})
        if url.path == "/api/art":
            return self.send_art(qs.get("path", [""])[0])
        if url.path == "/api/shift":
            stem, key = qs.get("song", [""])[0], qs.get("key", ["0"])[0]
            if not (OUTPUT / stem / "notes.json").exists() or not re.fullmatch(r"-?\d+", key) or abs(int(key)) > 6:
                return self.send_json({"error": "unknown song or key out of range (-6..6)"}, 400)
            try:
                return self.send_json(SHIFTER.get(stem, int(key)))
            except Exception as e:
                return self.send_json({"error": f"Couldn't change key: {e}"}, 500)
        if url.path == "/api/status":
            stem = qs.get("song", [""])[0]
            return self.send_json(JOBS.get(stem) or {"state": "idle"})
        return self.send_static()

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if url.path == "/api/process":
            body = json.loads(self.rfile.read(length) or b"{}")
            sources = list_sources()
            stem = body.get("song", "")
            if stem not in sources:
                return self.send_json({"error": "unknown song"}, 404)
            JOBS.submit(stem, sources[stem])
            return self.send_json(JOBS.get(stem))
        if url.path == "/api/upload":
            return self.receive_upload(length)
        if url.path == "/api/add":
            return self.add_song(json.loads(self.rfile.read(length) or b"{}"))
        self.send_error(404)

    # ---------- api ----------
    def songs(self):
        meta = load_meta()
        out = []
        for stem, path in list_sources().items():
            out.append({
                "song": stem,
                "file": path.relative_to(ROOT).as_posix(),
                "processed": (OUTPUT / stem / "notes.json").exists(),
                "preview": (OUTPUT / stem / "preview" / "notes.json").exists(),
                "job": JOBS.get(stem) or None,
                "mtime": path.stat().st_mtime if path.exists() else 0,
                **meta.get(stem, {}),
            })
        return out

    def search(self, q):
        if len(q) < 2:
            return self.send_json({"library": [], "previews": [], "free": [], "library_ready": LIBRARY.ready})
        found, errors = {}, []

        def run(name, fn):
            try:
                found[name] = fn(q)
            except Exception as e:  # offline, rate-limited, ...
                found[name] = []
                errors.append(f"{name} search unavailable ({type(e).__name__})")

        threads = [threading.Thread(target=run, args=a) for a in (("previews", itunes_search), ("free", ccmixter_search))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.send_json({"library": LIBRARY.browse(q, 8)[0], "previews": found["previews"], "free": found["free"],
                               "library_ready": LIBRARY.ready, "error": "; ".join(errors) or None})

    def add_song(self, body):
        """Add a search/library result to songs/ and start processing it."""
        SONGS.mkdir(exist_ok=True)
        existing = list_sources()
        if body.get("source") == "library":
            entry = LIBRARY.get(body.get("path", ""))
            if not entry:
                return self.send_json({"error": "that file isn't in the music library index"}, 400)
            src = Path(entry["path"])
            for stem, p in existing.items():  # already added? just use it
                if p.is_symlink() and p.resolve() == src.resolve():
                    break
            else:
                stem = safe_name(f"{entry['artist']} - {entry['title']}" if entry["artist"] else entry["title"])
                if stem in existing:
                    stem = f"{stem} ({int(time.time()) % 10000})"
                (SONGS / f"{stem}{src.suffix.lower()}").symlink_to(src)  # full track, no copy
            info = describe_song(stem, src, {"title": entry["title"], "artist": entry["artist"],
                                             "album": entry["album"], "source": "library", "path": str(src)})
        elif body.get("source") == "preview":
            title, artist = body.get("title", "").strip(), body.get("artist", "").strip()
            stem = safe_name(f"{artist} - {title} (preview)" if artist else f"{title} (preview)")
            dest = SONGS / f"{stem}.m4a"
            if not dest.exists():
                try:
                    download(body.get("preview", ""), dest, PREVIEW_HOSTS, 25)
                except Exception as e:
                    return self.send_json({"error": f"Couldn't fetch the preview: {e}"}, 400)
            info = {"title": title, "artist": artist, "artwork": body.get("artwork", ""), "source": "preview",
                    "buy": body.get("buy", "")}
        elif body.get("source") == "free":
            title, artist = body.get("title", "").strip(), body.get("artist", "").strip()
            stem = safe_name(f"{artist} - {title}" if artist else title)
            dest = SONGS / f"{stem}.mp3"
            if not dest.exists():
                try:
                    download(body.get("url", ""), dest, CC_HOSTS, 60)
                except Exception as e:
                    return self.send_json({"error": f"Couldn't download the song: {e}"}, 400)
            info = {"title": title, "artist": artist, "source": "free", "license": body.get("license", ""),
                    "license_url": body.get("license_url", ""), "page": body.get("page", "")}
        else:
            return self.send_json({"error": "unknown source"}, 400)
        save_meta(stem, info)
        if not (OUTPUT / stem / "notes.json").exists():
            JOBS.submit(stem, list_sources()[stem])
        return self.send_json({"song": stem})

    def receive_upload(self, length):
        name = Path(unquote(self.headers.get("X-Filename", ""))).name
        name = re.sub(r"[^\w\-.,'()& ]", "_", name).strip(" .")[:120]
        if Path(name).suffix.lower() not in AUDIO_EXT or not name:
            self.rfile.read(length)
            return self.send_json({"error": "please add an audio file (mp3, m4a, wav, flac, ogg, aac, aiff)"}, 400)
        if length <= 0 or length > MAX_UPLOAD:
            return self.send_json({"error": "file is empty or larger than 300 MB"}, 400)
        SONGS.mkdir(exist_ok=True)
        dest = SONGS / name
        same = dest.exists() and dest.stat().st_size == length  # re-adding the same file: keep the cached work
        remaining = length
        with open(os.devnull if same else dest, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        stem = dest.stem
        save_meta(stem, describe_song(stem, dest, {"source": "upload"}))
        if not (OUTPUT / stem / "notes.json").exists():
            JOBS.submit(stem, dest)
        return self.send_json({"song": stem, "file": dest.relative_to(ROOT).as_posix()})

    # ---------- responses ----------
    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype):
        data = Path(path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_art(self, path):
        """Album art for a library song, extracted once and cached."""
        if not LIBRARY.get(path):
            return self.send_error(404)
        cache = ART / (hashlib.sha1(path.encode()).hexdigest() + ".jpg")
        none = cache.with_suffix(".none")
        if not cache.exists() and not none.exists() and not extract_cover(path, cache):
            none.parent.mkdir(parents=True, exist_ok=True)
            none.touch()
        if not cache.exists():
            return self.send_error(404)
        self.cacheable = True
        data = cache.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "max-age=86400")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_static(self):
        """Files from output/, honouring Range requests so audio can seek."""
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = Path(self.translate_path(self.path))
        if not match or not path.is_file():
            return super().do_GET()

        size = path.stat().st_size
        start_s, end_s = match.groups()
        if start_s:
            start, end = int(start_s), min(int(end_s) if end_s else size - 1, size - 1)
        else:  # suffix range: last N bytes
            start, end = max(0, size - int(end_s or 0)), size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled the request after a seek

    def send_library(self):
        songs = sorted(p.parent.name for p in OUTPUT.glob("*/viewer.html"))
        items = "".join(
            f'<li><a href="{html.escape(s)}/viewer.html">{html.escape(s)}</a></li>' for s in songs
        ) or "<li>No songs processed yet.</li>"
        body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Piano Roll Library</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#f6f5f2; --ink:#1d1d1f; --link:#3a6fd8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --ink:#ececef; --link:#6f9cff; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 -apple-system,system-ui,sans-serif; }}
main {{ max-width:640px; margin:0 auto; padding:32px 16px; }} a {{ color:var(--link); }}
</style></head><body><main><h1>Piano rolls</h1><p><a href="/">← vocalpipe</a></p><ul>{items}</ul></main></body></html>"""
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):
            super().log_message(fmt, *args)  # only log errors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    args = p.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(OUTPUT)))
    except OSError as e:
        if e.errno != 48 and "in use" not in str(e):
            raise
        sys.exit(f"Port {args.port} is already in use — the app may already be running at "
                 f"http://localhost:{args.port}\nStop it, or pick another port: venv/bin/python serve.py --port {args.port + 1}")
    print(f"vocalpipe at http://localhost:{args.port} (loading models in the background)")
    server.serve_forever()


if __name__ == "__main__":
    main()

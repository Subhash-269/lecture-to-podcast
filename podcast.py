"""
Turn a speaker-tagged script into a multi-host podcast episode with a
local text-to-speech model. No cloud API, no per-character cost.

Script format (one turn per line; unlabelled lines join the turn above):

    MAYA: [excited] Welcome back! Today we're unpacking recommender systems.
    JORDAN: [laughs] Which, honestly, run half my life.
    MAYA: [pause] Fair. So where do we start?

Labels map to speakers through "aliases" in the config. Tags in square
brackets are hints: backends that support expression use them, the
others ignore them. See tts_backends.py for the tags each backend reads.

Run:
    python podcast.py --validate                 # parse only, nothing synthesised
    python podcast.py                            # build the episode
    python podcast.py --limit 6                  # audition the first 6 turns
    python podcast.py --config my.json           # use another config
    python podcast.py --backend kokoro           # override the backend
    python podcast.py --force                    # rebuild even if the output exists

Every synthesised turn is cached in .cache/ under a hash of its text,
voice and settings, so an interrupted or edited run only re-renders the
turns that changed.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_backends import get_backend, split_tags

HERE = Path(__file__).parent
DEFAULT_CONFIG = "podcast.config.json"

# Matches "Name: text", "**Name:** text" and "**Name**: text".
TURN_RE = re.compile(r"^\*{0,2}\s*([^:*\[\]]{1,40}?)\s*:\s*\*{0,2}\s*(.*)$")


# ---- CONFIG ---------------------------------------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            "Config not found: %s\nCopy podcast.config.example.json to %s and edit it."
            % (path, DEFAULT_CONFIG)
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("script_path", "output_path", "speakers"):
        if not cfg.get(key):
            raise SystemExit('"%s" is missing or empty in %s.' % (key, path.name))
    cfg.setdefault("backend", "kokoro")
    cfg.setdefault("aliases", {})
    cfg.setdefault("pauses_ms", {})
    cfg.setdefault("backend_options", {})
    return cfg


def resolve(base: Path, p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else base / q


# ---- PARSE ----------------------------------------------------------------

def clean_text(text: str) -> str:
    """Drop markdown emphasis so the model doesn't read the markers aloud."""
    text = text.replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def is_noise(line: str) -> bool:
    """Headings, comments and rules are never spoken."""
    return (line.startswith(("#", "//", "<!--"))
            or set(line) <= set("-=*_ ~"))


def parse_script(path: Path, speakers: dict, aliases: dict):
    """Return (turns, skipped).

    turns   -- [(speaker, text)]; text may still contain [tags]
    skipped -- [(line_no, line)] non-dialogue lines, reported so nothing
               is dropped silently.
    """
    lookup = {k.strip().lower(): v for k, v in aliases.items()}
    for name in speakers:
        lookup.setdefault(name.strip().lower(), name)

    turns, skipped = [], []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue                      # comment
        if is_noise(line):
            skipped.append((lineno, line))
            continue
        m = TURN_RE.match(line)
        speaker = lookup.get(m.group(1).strip().lower()) if m else None
        if speaker:
            text = clean_text(m.group(2))
            if text:
                turns.append([speaker, text])
            continue
        if turns:
            # An unlabelled line continues the previous turn.
            turns[-1][1] = clean_text(turns[-1][1] + " " + line)
            continue
        skipped.append((lineno, line))
    return [tuple(t) for t in turns], skipped


def report(script: Path, turns, skipped, speakers) -> None:
    print("Script: %s" % script)
    print("Parsed %d turns." % len(turns))
    for name in speakers:
        n = sum(1 for s, _ in turns if s == name)
        print("  %-14s %4d turns" % (name, n))
    words = sum(len(split_tags(t)[1].split()) for _, t in turns)
    print("  %-14s %4d   (~%.0f min at 150 wpm)" % ("total words", words, words / 150))
    tags = sorted({tag for _, t in turns for tag in split_tags(t)[0]})
    if tags:
        print("  tags used: %s" % ", ".join(tags))
    if skipped:
        print("\n%d line(s) skipped as non-dialogue -- check none should be spoken:"
              % len(skipped))
        for lineno, line in skipped:
            print("  L%-4d %s" % (lineno, line if len(line) <= 88 else line[:85] + "..."))


# ---- RENDER ---------------------------------------------------------------

def turn_key(backend_name: str, speaker_cfg: dict, text: str) -> str:
    blob = json.dumps([backend_name, speaker_cfg, text], sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def silence(ms: int, sr: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


def speaker_settings(cfg: dict, base: Path, speaker: str, backend_name: str) -> dict:
    spk = dict(cfg["speakers"][speaker].get(backend_name, {}))
    for k, v in spk.items():
        if k.endswith("reference") and isinstance(v, str):
            spk[k] = str(resolve(base, v))
    return spk


def render(cfg: dict, base: Path, turns, backend_name: str):
    backend = get_backend(backend_name, cfg["backend_options"].get(backend_name, {}))
    cache = base / ".cache" / backend_name
    cache.mkdir(parents=True, exist_ok=True)

    pauses = {"turn": 300, "same_speaker": 120, "after_question": 170,
              "interrupted": 60, "backchannel": 140, "pause_tag": 650,
              "jitter": 0.4, "speed_jitter": 0.04, "overlap": False}
    pauses.update(cfg["pauses_ms"])

    clips = []            # (start_sample, audio, gain)
    cursor = 0            # where the next normal turn may start
    prev = None           # (speaker, text, start, end)
    sr = backend.sample_rate
    for i, (speaker, text) in enumerate(turns, 1):
        tags, _ = split_tags(text)
        spk = speaker_settings(cfg, base, speaker, backend_name)
        if "speed" in spk or backend_name == "kokoro":
            spk["speed"] = round(float(spk.get("speed", 1.0)) *
                                 (1 + pauses["speed_jitter"] * jitter(text, "speed")), 3)
        speak = speakable(text)
        wav_path = cache / ("%s.wav" % turn_key(backend_name, spk, speak))

        if wav_path.exists() and wav_path.stat().st_size > 0:
            audio, sr = sf.read(wav_path, dtype="float32")
            state = "cached"
        else:
            audio = np.asarray(backend.synthesize(speak, spk), dtype=np.float32).reshape(-1)
            sr = backend.sample_rate
            sf.write(wav_path, audio, sr)
            state = "rendered"
        audio = trim(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
        print("  [%3d/%d] %-8s %-10s %s" % (i, len(turns), state, speaker, text[:60]),
              flush=True)

        ms = lambda x: int(sr * x / 1000)
        if pauses["overlap"] and "over" in tags and prev is not None and prev[0] != speaker:
            # A listener's "mm-hm" under the end of the other speaker's line.
            p_start, p_end = prev[2], prev[3]
            start = max(p_start, p_end - len(audio) - ms(250 + 150 * jitter(text, "over")))
            clips.append((start, audio, 0.55))
            cursor = max(cursor, start + len(audio) - ms(80))
            continue

        if prev is None:
            start = 0
        else:
            p_text = prev[1].rstrip()
            if "over" in tags:
                gap = pauses["backchannel"]
            elif p_text.endswith("--"):
                gap = pauses["interrupted"]
            elif p_text.endswith("?"):
                gap = pauses["after_question"]
            elif prev[0] == speaker:
                gap = pauses["same_speaker"]
            else:
                gap = pauses["turn"]
            if gap > 0:
                gap *= 1 + pauses["jitter"] * jitter(text, "gap")
            if "pause" in tags:
                gap += pauses["pause_tag"]
            start = max(0, cursor + ms(gap))
        clips.append((start, audio, 1.0))
        cursor = start + len(audio)
        prev = (speaker, text, start, cursor)

    total = max(s + len(a) for s, a, _ in clips)
    out = np.zeros(total, dtype=np.float32)
    for s, a, g in clips:
        out[s:s + len(a)] += a * g
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.89).astype(np.float32), sr   # normalise to about -1 dBFS


def jitter(text: str, salt: str) -> float:
    """Deterministic value in [-1, 1] per line, so re-renders are identical."""
    h = hashlib.sha1((salt + "|" + text).encode("utf-8")).digest()
    return h[0] / 127.5 - 1.0


def speakable(text: str) -> str:
    """Text as the model should say it: cut-offs and dashes become pauses."""
    t = re.sub(r"\s*--\s*$", "", text.rstrip())       # trailing cut-off: just stop
    t = re.sub(r"\s*--\s*", ", ", t)                   # mid-line dash: a comma pause
    return t


def trim(audio: np.ndarray, sr: int, thresh: float = 0.01, keep_ms: int = 30) -> np.ndarray:
    """Strip the model's own leading/trailing silence so our gaps control timing."""
    loud = np.flatnonzero(np.abs(audio) > thresh)
    if not len(loud):
        return audio
    pad = int(sr * keep_ms / 1000)
    return audio[max(0, loud[0] - pad): loud[-1] + pad]


def save(audio: np.ndarray, sr: int, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".wav":
        sf.write(out, audio, sr)
        return out
    wav = out.with_suffix(".wav")
    sf.write(wav, audio, sr)
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH; kept %s instead of %s." % (wav.name, out.name))
        return wav
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-b:a", "160k", str(out)], check=True)
    wav.unlink()
    return out


# ---- MAIN -----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Script -> podcast audio with local TTS.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--backend", help="override the backend named in the config")
    ap.add_argument("--validate", action="store_true", help="parse only, no synthesis")
    ap.add_argument("--force", action="store_true", help="overwrite an existing episode")
    ap.add_argument("--limit", type=int, help="render only the first N turns (audition)")
    args = ap.parse_args()

    cfg_path = resolve(Path.cwd(), args.config)
    cfg = load_config(cfg_path)
    base = cfg_path.parent
    backend_name = args.backend or cfg["backend"]

    script = resolve(base, cfg["script_path"])
    if not script.exists():
        raise SystemExit("Script not found: %s" % script)
    turns, skipped = parse_script(script, cfg["speakers"], cfg["aliases"])
    if not turns:
        raise SystemExit("No dialogue found. Expected lines like 'NAME: text'. Known labels: %s"
                         % sorted(set(cfg["speakers"]) | set(cfg["aliases"])))
    report(script, turns, skipped, cfg["speakers"])

    missing = [s for s in cfg["speakers"] if backend_name not in cfg["speakers"][s]]
    if missing:
        print("\nNote: no '%s' settings for %s; backend defaults apply."
              % (backend_name, ", ".join(missing)))
    if args.validate:
        print("\n--validate: parsed only, nothing synthesised.")
        return

    out = resolve(base, cfg["output_path"])
    if args.limit:
        turns = turns[:args.limit]
        out = out.with_name(out.stem + "_preview" + out.suffix)
    elif out.exists() and not args.force:
        raise SystemExit("%s already exists. Delete it or use --force." % out)

    print("\nBackend: %s" % backend_name)
    audio, sr = render(cfg, base, turns, backend_name)
    final = save(audio, sr, out)
    print("Done: %s (%.1f min)" % (final, len(audio) / sr / 60))


if __name__ == "__main__":
    sys.exit(main())

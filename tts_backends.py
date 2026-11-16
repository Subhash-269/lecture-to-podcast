"""
Local text-to-speech backends.

Each backend turns one dialogue turn into a mono float32 numpy array at
`sample_rate`. Speaker settings come from the config, under the backend's
name, e.g.

    "speakers": {
      "Maya": {
        "kokoro":     {"voice": "af_heart", "speed": 1.0},
        "chatterbox": {"reference": "voices/maya.wav", "exaggeration": 0.55}
      }
    }

Expression tags in the script ([excited], [calm], [pause] ...) are read by
split_tags(). Each backend maps the tags it can honour onto its own
controls and silently drops the rest, so one script works everywhere.

Backends:
  kokoro      Kokoro-82M. Small, fast, runs on CPU. Clean narration
              voices; expression is limited to pace. Apache-2.0.
  chatterbox  Resemble AI Chatterbox. Zero-shot voice cloning from a
              ~10 s reference clip, with an "exaggeration" control for
              emotional intensity. Best on a CUDA GPU. MIT.

To add a backend: subclass Backend, implement synthesize(), register it
in BACKENDS.
"""

import re

import numpy as np

TAG_RE = re.compile(r"\[([a-zA-Z][a-zA-Z \-]{0,30})\]")

# Tag -> (pace multiplier, emotional intensity 0..1). Unknown tags are ignored.
EXPRESSION = {
    "excited":     (1.08, 0.80),
    "enthusiastic": (1.06, 0.75),
    "happy":       (1.04, 0.65),
    "amused":      (1.03, 0.65),
    "laughs":      (1.03, 0.70),
    "surprised":   (1.05, 0.75),
    "curious":     (1.00, 0.55),
    "thoughtful":  (0.95, 0.40),
    "calm":        (0.95, 0.35),
    "serious":     (0.96, 0.35),
    "sad":         (0.92, 0.45),
    "whispers":    (0.92, 0.30),
}


def split_tags(text: str):
    """Return ({tags}, text without tags)."""
    tags = {t.strip().lower() for t in TAG_RE.findall(text)}
    plain = re.sub(r"\s+", " ", TAG_RE.sub(" ", text)).strip()
    return tags, plain


def expression_of(tags):
    pace, intensity = 1.0, None
    for t in tags:
        if t in EXPRESSION:
            p, i = EXPRESSION[t]
            pace *= p
            intensity = i if intensity is None else max(intensity, i)
    return pace, intensity


def sentences(text: str, max_chars: int):
    """Split text into sentence groups no longer than max_chars."""
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        out.append(cur)
    return out


class Backend:
    name = "base"
    sample_rate = 24000

    def __init__(self, options: dict):
        self.options = options

    def synthesize(self, text: str, speaker: dict) -> np.ndarray:
        raise NotImplementedError


class KokoroBackend(Backend):
    """Kokoro-82M via the `kokoro` package (pip install kokoro)."""
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, options: dict):
        super().__init__(options)
        try:
            from kokoro import KPipeline
        except ImportError:
            raise SystemExit("Kokoro is not installed: pip install -r requirements-kokoro.txt")
        self.lang = options.get("lang_code", "a")   # a = American, b = British English
        self.pipeline = KPipeline(lang_code=self.lang, repo_id="hexgrad/Kokoro-82M")

    def synthesize(self, text, speaker):
        tags, plain = split_tags(text)
        pace, _ = expression_of(tags)
        voice = speaker.get("voice", "af_heart")
        speed = float(speaker.get("speed", 1.0)) * pace
        chunks = [np.asarray(audio, dtype=np.float32)
                  for _, _, audio in self.pipeline(plain, voice=voice, speed=speed)
                  if audio is not None]
        return np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)


class ChatterboxBackend(Backend):
    """Resemble AI Chatterbox (pip install chatterbox-tts)."""
    name = "chatterbox"

    def __init__(self, options: dict):
        super().__init__(options)
        try:
            import torch
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise SystemExit("Chatterbox is not installed: pip install -r requirements-chatterbox.txt")
        device = options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("Chatterbox is running on CPU: expect several seconds per sentence.")
        self.torch = torch
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = self.model.sr
        self.max_chars = int(options.get("max_chars", 280))
        self.seed = options.get("seed", 1234)

    def synthesize(self, text, speaker):
        tags, plain = split_tags(text)
        _, intensity = expression_of(tags)
        base = float(speaker.get("exaggeration", 0.5))
        exaggeration = base if intensity is None else min(1.0, (base + intensity) / 2 + 0.1)
        # Lower cfg_weight paces more naturally when exaggeration is high.
        cfg_weight = float(speaker.get("cfg_weight", 0.5 if exaggeration < 0.65 else 0.35))
        ref = speaker.get("reference")

        pieces = []
        for part in sentences(plain, self.max_chars):
            if self.seed is not None:
                self.torch.manual_seed(int(self.seed))
            wav = self.model.generate(part, audio_prompt_path=ref,
                                      exaggeration=exaggeration, cfg_weight=cfg_weight)
            pieces.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
            pieces.append(np.zeros(int(self.sample_rate * 0.12), dtype=np.float32))
        return np.concatenate(pieces[:-1]) if pieces else np.zeros(1, dtype=np.float32)


BACKENDS = {b.name: b for b in (KokoroBackend, ChatterboxBackend)}


def get_backend(name: str, options: dict) -> Backend:
    if name not in BACKENDS:
        raise SystemExit("Unknown backend '%s'. Available: %s" % (name, ", ".join(BACKENDS)))
    return BACKENDS[name](options)

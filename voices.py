"""
Audition Kokoro voices: renders one short line per voice, each introduced
by its name, into out/voice_sampler.wav.

    python voices.py                         # curated set
    python voices.py af_bella am_fenrir      # just these
    python voices.py --text "Your line here"

Voice names: first letter is the accent (a = American, b = British),
second is f/m. Full list:
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

CURATED = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_nova", "af_sky",
    "am_michael", "am_fenrir", "am_puck", "am_adam", "am_echo", "am_liam",
    "bf_emma", "bf_isabella", "bm_george", "bm_fable", "bm_lewis",
]
LINE = "Welcome back to the show! Today we're unpacking something you use every day."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voices", nargs="*")
    ap.add_argument("--text", default=LINE)
    ap.add_argument("--out", default="out/voice_sampler.wav")
    args = ap.parse_args()

    from kokoro import KPipeline
    pipes, sr, pieces = {}, 24000, []
    for v in args.voices or CURATED:
        lang = v[0]
        if lang not in pipes:
            pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
        intro = v.replace("_", " ").split(" ", 1)[1].capitalize()
        for text in (intro + ".", args.text):
            for _, _, a in pipes[lang](text, voice=v):
                pieces.append(np.asarray(a, dtype=np.float32))
            pieces.append(np.zeros(int(sr * 0.35), dtype=np.float32))
        pieces.append(np.zeros(int(sr * 0.6), dtype=np.float32))
        print("  %s" % v, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.concatenate(pieces), sr)
    print("Saved %s" % out)


if __name__ == "__main__":
    main()

# lecture-to-podcast

Turn a dialogue script (for example, two hosts unpacking a lecture) into a
podcast episode using **local** text-to-speech models. Nothing leaves your
machine, and there are no per-character costs.

```
script.txt  ──►  podcast.py (local TTS)  ──►  episode.mp3  ──►  publish.py (optional)
```

## Backends

| Backend | Quality | Expression | Hardware | License |
|---|---|---|---|---|
| `kokoro` | clean, natural narration | pace only | CPU is fine | Apache-2.0 |
| `chatterbox` | podcast-like, clones a voice from a ~10 s clip | emotion intensity (`exaggeration`) | NVIDIA GPU recommended (runs on CPU, slowly) | MIT |

Start with `kokoro` to check the script, then switch to `chatterbox` for
the final episode. Each backend is set up once per speaker in the config.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

pip install -r requirements-kokoro.txt        # CPU, small
# and/or
pip install -r requirements-chatterbox.txt    # for an NVIDIA GPU, install a CUDA torch first:
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

cp podcast.config.example.json podcast.config.json
```

ffmpeg on PATH is optional: with it you get `.mp3`, without it `.wav`.

## Write a script

```
MAYA: [excited] Welcome back! Today we're unpacking recommender systems.
JORDAN: [curious] The thing that decides what I watch next?
MAYA: [laughs] Exactly.
This line has no label, so it joins Maya's turn above.
JORDAN: [pause] Okay, where do we start?
```

- Labels map to speakers in `speakers`, or through `aliases` (`"HOST A": "Maya"`).
- Lines starting with `#` are ignored. Anything else that isn't dialogue is listed by `--validate`.
- Tags: `excited`, `enthusiastic`, `happy`, `amused`, `laughs`, `surprised`, `curious`,
  `thoughtful`, `calm`, `serious`, `sad`, `whispers` adjust pace and intensity; `pause` adds
  a beat before the turn. Backends drop tags they can't honour, so one script works everywhere.

See [examples/sample_script.txt](examples/sample_script.txt).

## Run

```bash
python podcast.py --validate          # parse and report, no audio
python podcast.py --limit 6           # audition the first 6 turns -> out/episode_preview.mp3
python podcast.py                     # full episode
python podcast.py --backend chatterbox --force
```

Each turn is cached in `.cache/<backend>/` under a hash of its text and voice
settings. Edit a line and re-run: only that turn is re-rendered.

## Voices

- **Kokoro**: run `python voices.py` to hear a sampler (`out/voice_sampler.wav`), then pick a built-in voice per speaker (`af_heart`, `af_bella`, `am_michael`,
  `am_fenrir`, `bf_emma`, `bm_george`, ...). See the
  [Kokoro voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
- **Chatterbox**: put an 8 to 15 s clean clip per speaker in `voices/` and point
  `reference` at it. `exaggeration` 0.3 is calm, 0.5 neutral, 0.7+ animated.
  Only clone voices you have permission to use.

## Publishing (optional)

`publish.py` uploads the episode to Spotify through the
`save-to-spotify` CLI using the `show` and `episode`
fields of the config. It is a dry run unless you pass `--confirm`, because
Spotify episodes have no draft state and their metadata can't be edited.

### Installing save-to-spotify

The CLI is Spotify's own tool and isn't bundled here. Get it from
[spotify/save-to-spotify releases](https://github.com/spotify/save-to-spotify/releases).

- **macOS / Linux**: `curl -fsSL https://saveto.spotify.com/install.sh | bash`
- **Windows**: download the `windows` `.exe` from the releases page, rename it
  to `save-to-spotify.exe`, and either put it on your `PATH` or save it as
  `%USERPROFILE%\tools\save-to-spotify\save-to-spotify.exe`.

`publish.py` looks for the CLI in this order: `cli_path` in your config, then
`PATH`, then `~/tools/save-to-spotify/save-to-spotify.exe`.

Sign in once before publishing:

```
save-to-spotify auth login
```

The login token is saved to `~/.config/save-to-spotify/token.json`, outside the repo.

**Only publish content you have the rights to.** Lecture recordings, slides
and course materials usually belong to the instructor or institution; get
permission before sharing anything derived from them.

## Adding a backend

Subclass `Backend` in [tts_backends.py](tts_backends.py), implement
`synthesize(text, speaker) -> np.ndarray`, and register it in `BACKENDS`.

## License

MIT. Model weights keep their own licenses (Kokoro: Apache-2.0, Chatterbox: MIT).

"""
Publish the generated episode to Spotify via the save-to-spotify CLI.

Reads podcast.config.json for the audio path and the show/episode metadata,
so the same file drives both generation (podcast.py) and publishing.

Prerequisites:
    - save-to-spotify installed (see cli_path below)
    - `save-to-spotify auth login` already completed
    - the audio file built by podcast.py

Run:
    python publish.py                      # dry run: print exactly what would happen
    python publish.py --confirm            # create/reuse the show and upload
    python publish.py --config my.json     # use another config

Only publish content you have the rights to. Lecture recordings and
course materials usually belong to the instructor or the institution.

Nothing is sent to Spotify without --confirm. This matters because the CLI
has no draft state (episodes go PROCESSING -> READY, i.e. live) and all
metadata is immutable once created -- fixing a title means delete + recreate.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

_cfg_arg = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--config"), "podcast.config.json")
CONFIG_PATH = (Path.cwd() / _cfg_arg) if not Path(_cfg_arg).is_absolute() else Path(_cfg_arg)
HERE = CONFIG_PATH.parent

DEFAULT_CLI = Path.home() / "tools" / "save-to-spotify" / "save-to-spotify.exe"

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg"}
MAX_AUDIO_BYTES = 1024 ** 3        # CLI limit: 1 GB
MAX_IMAGE_BYTES = 1024 * 1024      # CLI limit: 1 MB
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---- CLI PLUMBING ---------------------------------------------------------

def find_cli(cfg: dict) -> Path:
    """Locate save-to-spotify: config override, then PATH, then default."""
    configured = cfg.get("cli_path")
    if configured:
        p = Path(configured).expanduser()
        if not p.exists():
            raise SystemExit("cli_path in %s does not exist: %s"
                             % (CONFIG_PATH.name, p))
        return p

    on_path = shutil.which("save-to-spotify")
    if on_path:
        return Path(on_path)

    if DEFAULT_CLI.exists():
        return DEFAULT_CLI

    raise SystemExit(
        "save-to-spotify not found. Install it, or set \"cli_path\" in %s."
        % CONFIG_PATH.name
    )


def run_cli(cli: Path, *args: str) -> dict:
    """Invoke the CLI in JSON mode and return the parsed object.

    --json must precede the subcommand. Errors come back as
    {"error": "..."} with exit code 1, so both are checked.
    """
    cmd = [str(cli), "--json", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise SystemExit("CLI produced no output (exit %d).\n  cmd: %s\n  stderr: %s"
                         % (proc.returncode, " ".join(args), (proc.stderr or "").strip()))

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        raise SystemExit("CLI returned non-JSON output:\n%s" % stdout[:800])

    if isinstance(data, dict) and data.get("error"):
        raise SystemExit("CLI error on `%s`: %s" % (" ".join(args), data["error"]))
    if proc.returncode != 0:
        raise SystemExit("CLI exited %d on `%s`:\n%s"
                         % (proc.returncode, " ".join(args), stdout[:800]))
    return data


# ---- PREFLIGHT ------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit("Missing %s" % CONFIG_PATH)
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def check_audio(cfg: dict) -> Path:
    audio = HERE / cfg.get("output_path", "")
    if not cfg.get("output_path") or not audio.exists():
        raise SystemExit(
            "Audio file not found: %s\nRun `python podcast.py` first." % audio
        )
    if audio.suffix.lower() not in AUDIO_EXTS:
        raise SystemExit("Unsupported audio format %s (allowed: %s)"
                         % (audio.suffix, ", ".join(sorted(AUDIO_EXTS))))
    size = audio.stat().st_size
    if size == 0:
        raise SystemExit("Audio file is empty: %s" % audio)
    if size > MAX_AUDIO_BYTES:
        raise SystemExit("Audio file exceeds the 1 GB limit: %.2f GB"
                         % (size / MAX_AUDIO_BYTES))
    return audio


def check_image(section: dict, label: str) -> Path | None:
    """Cover art is optional; validate only if one is configured."""
    name = (section.get("image") or "").strip()
    if not name:
        return None
    img = HERE / name
    if not img.exists():
        raise SystemExit("%s image not found: %s" % (label, img))
    if img.suffix.lower() not in IMAGE_EXTS:
        raise SystemExit("%s image must be .jpg/.jpeg/.png, got %s"
                         % (label, img.suffix))
    if img.stat().st_size > MAX_IMAGE_BYTES:
        raise SystemExit("%s image exceeds the 1 MB limit: %.2f MB"
                         % (label, img.stat().st_size / MAX_IMAGE_BYTES))
    return img


def require_field(section: dict, key: str, label: str) -> str:
    value = (section.get(key) or "").strip()
    if not value:
        raise SystemExit('Set "%s" -> "%s" in %s first.'
                         % (label, key, CONFIG_PATH.name))
    return value


# ---- SHOW / EPISODE -------------------------------------------------------

def find_show(cli: Path, title: str) -> dict | None:
    """Reuse an existing show with this exact title, if there is one.

    Guards against a second run silently creating a duplicate show.
    """
    shows = run_cli(cli, "shows").get("shows") or []
    for show in shows:
        if (show.get("title") or "").strip() == title:
            return show
    return None


def show_id_of(show: dict) -> str:
    return show.get("show_uri") or show.get("uri") or show.get("id") or ""


def main() -> None:
    confirm = "--confirm" in sys.argv[1:]

    cfg = load_config()
    cli = find_cli(cfg)
    audio = check_audio(cfg)

    show_cfg = cfg.get("show") or {}
    ep_cfg = cfg.get("episode") or {}

    show_title = require_field(show_cfg, "title", "show")
    ep_title = require_field(ep_cfg, "title", "episode")
    show_summary = (show_cfg.get("summary") or "").strip()
    ep_summary = (ep_cfg.get("summary") or "").strip()
    language = (show_cfg.get("language") or "en").strip()

    show_image = check_image(show_cfg, "Show")
    ep_image = check_image(ep_cfg, "Episode")

    auth = run_cli(cli, "auth", "status")
    if not auth.get("authenticated"):
        raise SystemExit("Not authenticated. Run: %s auth login" % cli)

    existing = find_show(cli, show_title)

    print("CLI          : %s" % cli)
    print("Audio        : %s (%.1f MB)" % (audio.name, audio.stat().st_size / 1024 ** 2))
    print("Show         : %s%s" % (show_title, "  [reusing existing]" if existing else "  [will be created]"))
    print("Episode      : %s" % ep_title)
    print("Language     : %s" % language)
    print("Show cover   : %s" % (show_image.name if show_image else "(none)"))
    print("Episode cover: %s" % (ep_image.name if ep_image else "(none)"))

    if not confirm:
        print("\nDry run. Nothing was sent to Spotify.")
        print("Episodes have no draft state and metadata is immutable once created.")
        print("Re-run with --confirm to publish.")
        return

    # --- create or reuse the show
    if existing:
        show_id = show_id_of(existing)
        print("\nReusing show %s" % show_id)
    else:
        args = ["shows", "create", "--title", show_title, "--language", language]
        if show_summary:
            args += ["--summary", show_summary]
        if show_image:
            args += ["--image", str(show_image)]
        print("\nCreating show...")
        show_id = show_id_of(run_cli(cli, *args))
        if not show_id:
            raise SystemExit("Show created but no ID came back; check `%s --json shows`." % cli)
        print("  -> %s" % show_id)

    # --- upload the episode
    args = ["upload", str(audio), "--title", ep_title,
            "--show-id", show_id, "--language", language]
    if ep_summary:
        args += ["--summary", ep_summary]
    if ep_image:
        args += ["--image", str(ep_image)]

    print("Uploading episode...")
    result = run_cli(cli, *args)
    ep_uri = result.get("episode_uri") or ""
    print("  -> %s (%s)" % (ep_uri or "?", result.get("status", "?")))

    if not ep_uri:
        print("No episode URI returned; check `%s --json episodes --show-id %s`."
              % (cli, show_id))
        return

    # --- wait for processing
    ep_id = ep_uri.rsplit(":", 1)[-1]
    print("Waiting for processing (up to 5m)...")
    status = run_cli(cli, "episodes", "status", ep_id, "--wait")
    readiness = status.get("readiness", "?")
    print("  -> %s" % readiness)

    if readiness == "READY":
        print("\nPublished: %s" % ep_uri)
    elif readiness == "FAILED":
        raise SystemExit("Processing FAILED. Inspect, then delete with: "
                         "%s episodes delete %s" % (cli, ep_id))
    else:
        print("\nStill %s. Re-check with: %s --json episodes status %s --wait"
              % (readiness, cli, ep_id))


if __name__ == "__main__":
    main()

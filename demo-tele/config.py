"""Config loading for the alert demo.

alerts.yaml is the source of truth and the web UI is an editor over it, so
both directions have to work: the UI writes the file, and a hand-edit of the
file shows up without a restart. Every read stats the file and reloads when
the mtime moves, which is cheap enough at this request rate to just always do.
"""
import threading
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
ALERTS_PATH = BASE_DIR / "alerts.yaml"
SAMPLES_PATH = BASE_DIR / "sample_events.yaml"

# Re-emitted on every UI save. yaml.safe_dump does not preserve comments, so
# without this the schema note in alerts.yaml would be lost the first time
# someone pressed Save.
_BANNER = ALERTS_PATH.read_text(encoding="utf-8").split("\nmessages:", 1)[0].rstrip() + "\n\n"

_lock = threading.Lock()
_cache: dict[Path, tuple[float, dict]] = {}


class _BlockDumper(yaml.SafeDumper):
    """Dumps multi-line strings as `|-` blocks instead of folded scalars.

    Without this a saved template comes back as a single-quoted scalar with
    the blank lines re-flowed — valid YAML that round-trips fine, but no
    longer something anyone wants to hand-edit, which is half the point of
    keeping the file as the source of truth.
    """


def _represent_str(dumper: yaml.SafeDumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _represent_str)


def normalise_text(text: str) -> str:
    """Browser textareas submit CRLF. Telegram and YAML both want LF.

    A stray \\r also forces PyYAML out of block style, so this is what keeps a
    saved template readable in the file afterwards.
    """
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))


def _load(path: Path) -> dict:
    with _lock:
        mtime = path.stat().st_mtime
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _cache[path] = (mtime, data)
        return data


def alerts() -> dict:
    """messages / targets / rooms / dispatch, hot-reloaded from alerts.yaml."""
    return _load(ALERTS_PATH)


def sample_events() -> list[dict]:
    return _load(SAMPLES_PATH).get("events", [])


def messages() -> dict:
    return alerts().get("messages", {})


def targets() -> dict:
    return alerts().get("targets", {})


def dispatch_settings() -> dict:
    settings = {"default_target": None, "escalation_seconds": 20, "max_escalations": 1}
    settings.update(alerts().get("dispatch", {}) or {})
    return settings


def target_for_room(room_id: str) -> str | None:
    """Route an event to a chat target the way M8's room registry would."""
    rooms = alerts().get("rooms", {}) or {}
    return rooms.get(room_id) or dispatch_settings().get("default_target")


def known_chat_ids() -> set[str]:
    """Chat ids the bot will accept callbacks from (design doc §9.2)."""
    return {str(t.get("chat_id")) for t in targets().values() if t.get("chat_id")}


def save_alerts(data: dict) -> None:
    """Write alerts.yaml back, banner first, then flush the cache."""
    body = yaml.dump(
        data, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=100
    )
    tmp = ALERTS_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(_BANNER + body, encoding="utf-8")
    tmp.replace(ALERTS_PATH)
    with _lock:
        _cache.pop(ALERTS_PATH, None)

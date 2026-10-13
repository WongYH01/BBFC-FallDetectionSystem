"""Turn a message template + a fall event into a Telegram sendMessage payload.

Templates are user-edited from a web page, so they render in Jinja2's
sandboxed environment: a bad template should produce an error in the preview
pane, never reach into the process.

Only HTML and plain text are offered as parse modes. MarkdownV2 requires
escaping eighteen characters in every interpolated value, and a room name like
"Block A · Room 12 (west)" silently breaks the message when that is missed —
not a failure mode worth shipping in an alert channel.
"""
import html
import re
from datetime import datetime

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

ACTIONS = {"ack", "false_alarm"}

# Telegram's own HTML subset, minus links (no alert here carries one).
_ALLOWED_TAGS = ("b", "strong", "i", "em", "u", "s", "code", "pre")
_ANY_TAG = re.compile(r"<[^>]*>")
_ALLOWED_TAG = re.compile(r"^</?(" + "|".join(_ALLOWED_TAGS) + r")>$", re.IGNORECASE)

_env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
_env_html = SandboxedEnvironment(undefined=StrictUndefined, autoescape=True)


class RenderError(Exception):
    """A template did not render — surfaced to the editor, not to Telegram."""


def confidence_band(confidence: float) -> str:
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


def build_context(event: dict, escalation_level: int = 0) -> dict:
    """Add the derived fields templates are allowed to reference.

    The raw event carries what M5 knows; a message needs it in the shape a
    nurse reads at 3am — a local clock time, a confidence band, not an ISO
    timestamp and a float.
    """
    ctx = dict(event)
    stamp = event.get("ts_confirmed") or event.get("ts_trigger")
    when = None
    if stamp:
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            when = None
    when = when or datetime.now()

    confidence = float(event.get("confidence", 0.0))
    ctx.update(
        time=when.strftime("%H:%M"),
        # Built by hand rather than with %-d / %#d, which differ by platform.
        date=f"{when.day} {when.strftime('%b %Y')}",
        confidence=confidence,
        confidence_pct=round(confidence * 100),
        confidence_band=confidence_band(confidence),
        escalation_level=escalation_level,
    )
    return ctx


def render(message: dict, event: dict, escalation_level: int = 0) -> dict:
    """Render one message template into the payload sendMessage expects.

    Returns text, parse_mode and reply_markup separately so the caller can log
    the exact payload in dry-run mode without a transport in the way.
    """
    parse_mode = (message.get("parse_mode") or "").strip() or None
    if parse_mode and parse_mode.upper() != "HTML":
        raise RenderError(f"unsupported parse_mode {parse_mode!r} — use HTML or leave it empty")

    env = _env_html if parse_mode else _env
    ctx = build_context(event, escalation_level)
    try:
        text = env.from_string(message.get("text") or "").render(event=ctx).strip()
    except TemplateError as exc:
        raise RenderError(str(exc)) from exc
    if not text:
        raise RenderError("template rendered to an empty message")

    payload = {"text": text}
    if parse_mode:
        payload["parse_mode"] = "HTML"
    markup = _reply_markup(message.get("buttons") or [], str(ctx.get("event_id") or "preview"))
    if markup:
        payload["reply_markup"] = markup
    return payload


def _reply_markup(buttons: list[dict], event_id: str) -> dict | None:
    """Inline keyboard. Ack is a callback, never a free-text reply (§4.5).

    callback_data carries the event id so a tap identifies its alert even after
    a restart, and Telegram caps that field at 64 bytes.
    """
    row: list[dict] = []
    rows: list[list[dict]] = []
    for button in buttons:
        action = (button.get("action") or "").strip()
        if action not in ACTIONS:
            raise RenderError(f"unknown button action {action!r} — use one of {sorted(ACTIONS)}")
        data = f"{action}:{event_id}"
        if len(data.encode()) > 64:
            raise RenderError(f"callback data too long for event id {event_id!r}")
        row.append({"text": button.get("text") or action, "callback_data": data})
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return {"inline_keyboard": rows} if rows else None


def preview_html(text: str, parse_mode: str | None) -> str:
    """Text as the browser should show it, without trusting it as page HTML.

    In HTML mode the rendered text is already valid HTML — interpolated values
    were escaped as they were substituted — so this sanitises rather than
    escapes: tags outside Telegram's own subset are neutered, everything else
    including existing entities is left alone. Escaping again here would turn
    a room name's `&lt;` into a visible `&amp;lt;`.

    In plain mode nothing is markup, so everything is escaped.
    """
    if not parse_mode:
        body = html.escape(text)
    else:
        body = _ANY_TAG.sub(
            lambda m: m.group(0) if _ALLOWED_TAG.match(m.group(0)) else html.escape(m.group(0)),
            text,
        )
    return body.replace("\n", "<br>")

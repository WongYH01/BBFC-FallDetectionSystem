"""The M6 slice this demo covers: dispatch, acknowledgement, escalation.

One fall event becomes one alert. The alert stays open until somebody taps a
button; if nobody does inside the escalation window it is re-issued, to the
shift lead when the target names one. Acknowledging edits every message the
alert produced so a second responder can see it is already handled — which is
the whole point of putting the time in the message (§6.3).

State is in memory only. The real system writes the event to M7 before
dispatching, so the record survives a delivery failure; here the activity log
stands in for that store.
"""
import itertools
import json
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import config
import render
from telegram import DryRunTransport, LiveTransport, TransportError, bot_token

_lock = threading.RLock()
_seq = itertools.count(1)
_alert_ids = itertools.count(1)

_CHATS_PATH = Path(__file__).resolve().parent / "seen_chats.json"


def _load_seen_chats() -> dict[str, dict]:
    """Discovered chats survive a restart; the update that revealed them does not.

    "Bot added to a group" arrives exactly once, as a my_chat_member update.
    The poller confirms it by advancing the getUpdates offset, and Telegram
    never sends it again — so holding the result only in memory means one
    restart loses the group id for good, with no way to ask for it back.
    """
    try:
        return {c["id"]: c for c in json.loads(_CHATS_PATH.read_text(encoding="utf-8"))}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


ACTIVITY: deque = deque(maxlen=300)
ALERTS: dict[str, dict] = {}          # event_id -> alert record
SEEN_CHATS: dict[str, dict] = _load_seen_chats()   # chat_id -> {id, title, type}
_live_transport: LiveTransport | None = None
_dry_transport = DryRunTransport()


def note_chat(chat: dict | None) -> dict | None:
    """Remember a chat the bot has been seen in, so its id can be copied.

    The alert goes to a staff group, and a group's chat id is not something
    anyone can read off the Telegram UI. Recording every chat the bot hears
    from turns "add the bot to the group, then look here" into the whole
    setup step.
    """
    if not chat or chat.get("id") is None:
        return None
    entry = {
        "id": str(chat["id"]),
        "type": chat.get("type", "?"),
        "title": chat.get("title") or chat.get("username") or chat.get("first_name") or "(no title)",
        "is_forum": bool(chat.get("is_forum")),
    }
    with _lock:
        known = SEEN_CHATS.get(entry["id"])
        SEEN_CHATS[entry["id"]] = entry
        if known != entry:
            _save_seen_chats()
    return entry


def _save_seen_chats() -> None:
    """Best effort. Chat discovery is a convenience — never fail a tap over it."""
    try:
        _CHATS_PATH.write_text(
            json.dumps(list(SEEN_CHATS.values()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        log("error", f"could not save discovered chats: {exc}")


def seen_chats() -> list[dict]:
    with _lock:
        return list(SEEN_CHATS.values())


# ---------------------------------------------------------------------------
# Activity log — what the UI's right-hand pane shows
# ---------------------------------------------------------------------------
def log(kind: str, text: str, detail: dict | None = None) -> dict:
    entry = {
        "seq": next(_seq),
        "ts": datetime.now().strftime("%H:%M:%S"),
        "kind": kind,
        "text": text,
        "detail": detail or {},
    }
    with _lock:
        ACTIVITY.append(entry)
    return entry


def activity(since: int = 0) -> list[dict]:
    with _lock:
        return [e for e in ACTIVITY if e["seq"] > since]


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------
def transport(dry_run: bool):
    """One live transport per process so the poller and sender share a session."""
    global _live_transport
    if dry_run:
        return _dry_transport
    with _lock:
        if _live_transport is None:
            _live_transport = LiveTransport(bot_token())
        return _live_transport


def reset_live_transport() -> None:
    global _live_transport
    with _lock:
        _live_transport = None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def dispatch_event(
    event: dict,
    target_key: str | None = None,
    dry_run: bool = True,
    message: dict | None = None,
) -> dict:
    """Send one alert and arm its escalation timer.

    `message` is the template to render. The bench passes the one open in the
    editor, so what the preview shows is what gets sent. Pass None — as a real
    caller with no UI would — and the target's own configured template is used.

    Raises RenderError or TransportError; the caller decides how loudly to
    fail. A failed send still leaves a log entry, because a silent alerting
    path is the failure mode this system can least afford.
    """
    target_key = target_key or config.target_for_room(event.get("room_id", ""))
    targets = config.targets()
    if target_key not in targets:
        raise KeyError(f"unknown target {target_key!r}")

    event_id = str(event.get("event_id"))
    with _lock:
        existing = ALERTS.get(event_id)
        if existing and not existing["closed"]:
            # Debounce (§5.6): one fall, one alert.
            log("info", f"{event_id} already open on {existing['target']} — not re-sent")
            return existing

    record = {
        "alert_id": f"AL-{next(_alert_ids):04d}",
        "event": event,
        "event_id": event_id,
        "target": target_key,
        "dry_run": dry_run,
        "escalation_level": 0,
        "closed": False,
        "outcome": None,
        "acked_by": None,
        "acked_ts": None,
        "messages": [],
        "message": message,   # the template this alert was raised with, if given
        "timer": None,
    }
    with _lock:
        ALERTS[event_id] = record

    _send_for_target(record, target_key, escalation_level=0, message=message)
    _arm_escalation(record)
    return record


def _send_for_target(
    record: dict, target_key: str, escalation_level: int, message: dict | None = None
) -> None:
    target = config.targets()[target_key]
    if message is None:
        message_key = target.get("message")
        message = config.messages().get(message_key)
        if message is None:
            raise KeyError(f"target {target_key!r} names unknown message {message_key!r}")

    payload = render.render(message, record["event"], escalation_level)
    payload["chat_id"] = str(target["chat_id"])
    # Forum groups route by topic; a plain group ignores this and it is omitted.
    if target.get("message_thread_id"):
        payload["message_thread_id"] = int(target["message_thread_id"])

    try:
        result = transport(record["dry_run"]).call("sendMessage", payload)
    except TransportError as exc:
        log("error", f"send to {target_key} failed: {exc}", {"event_id": record["event_id"]})
        raise

    sent = {
        "chat_id": str(target["chat_id"]),
        "message_id": result.get("message_id"),
        "text": payload["text"],
        "parse_mode": payload.get("parse_mode"),
    }
    with _lock:
        record["messages"].append(sent)
        record["escalation_level"] = escalation_level

    kind = "escalation" if escalation_level else "dispatch"
    verb = f"escalation {escalation_level} to" if escalation_level else "alert sent to"
    log(
        kind,
        f"{record['event_id']} · {verb} {target.get('label', target_key)}",
        {
            "event_id": record["event_id"],
            "target": target_key,
            "chat_id": str(target["chat_id"]),
            "mode": "dry-run" if record["dry_run"] else "live",
            "payload": payload,
        },
    )


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------
def _arm_escalation(record: dict) -> None:
    settings = config.dispatch_settings()
    if record["escalation_level"] >= int(settings["max_escalations"]):
        return
    delay = float(settings["escalation_seconds"])
    timer = threading.Timer(delay, _escalate, args=(record["event_id"],))
    timer.daemon = True
    with _lock:
        record["timer"] = timer
    timer.start()
    log(
        "info",
        f"{record['event_id']} · escalating in {delay:.0f}s unless acknowledged",
        {"event_id": record["event_id"]},
    )


def _escalate(event_id: str) -> None:
    with _lock:
        record = ALERTS.get(event_id)
        if record is None or record["closed"]:
            return
        current_target = record["target"]
        next_level = record["escalation_level"] + 1

    target = config.targets().get(current_target, {})
    next_target = target.get("escalate_to") or current_target
    if next_target not in config.targets():
        log("error", f"{event_id} · escalate_to {next_target!r} is not a known target")
        return

    # Re-issuing into the same chat repeats the wording the alert was raised
    # with. A different target is a different audience — the shift lead gets
    # their own escalation template, not a copy of the ward message.
    repeat = record.get("message") if next_target == current_target else None

    try:
        _send_for_target(record, next_target, escalation_level=next_level, message=repeat)
    except (TransportError, render.RenderError, KeyError) as exc:
        log("error", f"{event_id} · escalation failed: {exc}")
        return

    with _lock:
        record["target"] = next_target
    _arm_escalation(record)


# ---------------------------------------------------------------------------
# Acknowledgement
# ---------------------------------------------------------------------------
def handle_callback(callback: dict, dry_run: bool = False) -> None:
    """Handle one callback_query from a button tap.

    Callbacks are only honoured from a chat listed in alerts.yaml, which is the
    "bot only accepts callbacks from the registered staff group" control in
    §9.2 of the design document.

    dry_run is False for taps arriving from the poller and True for the UI's
    simulated tap, which is what makes the ack and escalation paths demoable
    with no bot at all.
    """
    data = callback.get("data") or ""
    action, _, event_id = data.partition(":")
    chat_id = str((callback.get("message") or {}).get("chat", {}).get("id", ""))
    who = _describe(callback.get("from") or {})

    if chat_id not in config.known_chat_ids():
        log("error", f"callback from unregistered chat {chat_id} ignored")
        _answer(callback, "This bot is not configured for this chat.", dry_run)
        return

    with _lock:
        record = ALERTS.get(event_id)

    if record is None:
        _answer(callback, "That alert is no longer tracked.", dry_run)
        log("info", f"callback for unknown alert {event_id!r} from {who}")
        return
    if record["closed"]:
        _answer(callback, f"Already handled by {record['acked_by']}.", record["dry_run"])
        return

    outcome = "false_alarm" if action == "false_alarm" else "acknowledged"
    close(record, outcome=outcome, by=who)
    _answer(
        callback,
        "Acknowledged." if outcome == "acknowledged" else "Logged as a false alarm.",
        record["dry_run"],
    )


def close(record: dict, outcome: str, by: str) -> None:
    """Stop escalation and mark every message this alert produced as handled."""
    with _lock:
        if record["closed"]:
            return
        record["closed"] = True
        record["outcome"] = outcome
        record["acked_by"] = by
        record["acked_ts"] = datetime.now().strftime("%H:%M:%S")
        timer = record["timer"]
        record["timer"] = None
    if timer is not None:
        timer.cancel()

    banner = (
        f"✅ Acknowledged by {by} at {record['acked_ts']}"
        if outcome == "acknowledged"
        else f"🚫 Marked false alarm by {by} at {record['acked_ts']}"
    )
    for sent in record["messages"]:
        payload = {
            "chat_id": sent["chat_id"],
            "message_id": sent["message_id"],
            "text": f"{sent['text']}\n\n{banner}",
        }
        if sent["parse_mode"]:
            payload["parse_mode"] = sent["parse_mode"]
        try:
            # No reply_markup: sending none clears the keyboard, so the alert
            # cannot be acknowledged twice.
            transport(record["dry_run"]).call("editMessageText", payload)
        except TransportError as exc:
            log("error", f"could not update message {sent['message_id']}: {exc}")

    log(
        "ack" if outcome == "acknowledged" else "false_alarm",
        f"{record['event_id']} · {banner}",
        {"event_id": record["event_id"], "outcome": outcome, "by": by},
    )


def _answer(callback: dict, text: str, dry_run: bool) -> None:
    """answerCallbackQuery, or Telegram leaves a spinner on the button."""
    try:
        transport(dry_run).call(
            "answerCallbackQuery", {"callback_query_id": callback.get("id"), "text": text}
        )
    except TransportError as exc:
        log("error", f"answerCallbackQuery failed: {exc}")


def _describe(user: dict) -> str:
    username = user.get("username")
    if username:
        return f"@{username}"
    name = " ".join(filter(None, [user.get("first_name"), user.get("last_name")]))
    return name or f"user {user.get('id', '?')}"


# ---------------------------------------------------------------------------
# Introspection for the UI
# ---------------------------------------------------------------------------
def open_alerts() -> list[dict]:
    with _lock:
        return [
            {
                "alert_id": r["alert_id"],
                "event_id": r["event_id"],
                "target": r["target"],
                "escalation_level": r["escalation_level"],
                "closed": r["closed"],
                "outcome": r["outcome"],
                "acked_by": r["acked_by"],
                "mode": "dry-run" if r["dry_run"] else "live",
            }
            for r in ALERTS.values()
        ]


def reset() -> None:
    with _lock:
        for r in ALERTS.values():
            if r["timer"] is not None:
                r["timer"].cancel()
        ALERTS.clear()
        ACTIVITY.clear()
    _dry_transport.calls.clear()
    log("info", "state cleared")

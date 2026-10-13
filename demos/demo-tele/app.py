"""demo-tele — a bench for the Telegram alert path (design doc §6.3, I-5).

Run it, open the page, edit the message on the left, watch the rendered alert
on the right, pick a chat target and send. Dry-run shows the exact payload
without touching the network; live mode sends it and listens for the button
tap that closes the alert.

    python app.py        →  http://localhost:5001
"""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

import config  # noqa: E402  — imported after load_dotenv so the token is present
import orchestrator  # noqa: E402
import poller  # noqa: E402
import render  # noqa: E402
import telegram  # noqa: E402

app = Flask(__name__)
# Flask sorts JSON keys by default, which would reorder the templates and
# targets alphabetically and drop whatever order alerts.yaml was written in.
app.json.sort_keys = False


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    """Everything the page needs to draw itself, hot-read from the YAML."""
    return jsonify(
        messages=config.messages(),
        targets=config.targets(),
        rooms=config.alerts().get("rooms", {}),
        dispatch=config.dispatch_settings(),
        samples=config.sample_events(),
        bot=telegram.identity(),
        listener=poller.status(),
        alerts=orchestrator.open_alerts(),
        seen_chats=orchestrator.seen_chats(),
    )


# ---------------------------------------------------------------------------
# Preview — renders a draft that has not been saved yet
# ---------------------------------------------------------------------------
@app.route("/api/preview", methods=["POST"])
def preview():
    body = request.get_json(silent=True) or {}
    message = body.get("message") or {}
    event = body.get("event") or {}
    level = int(body.get("escalation_level") or 0)
    try:
        payload = render.render(message, event, level)
    except (render.RenderError, KeyError, ValueError, TypeError) as exc:
        return jsonify(ok=False, error=str(exc))
    return jsonify(
        ok=True,
        payload=payload,
        html=render.preview_html(payload["text"], payload.get("parse_mode")),
        buttons=[
            button
            for row in payload.get("reply_markup", {}).get("inline_keyboard", [])
            for button in row
        ],
    )


# ---------------------------------------------------------------------------
# Saving back to alerts.yaml
# ---------------------------------------------------------------------------
@app.route("/api/messages/<key>", methods=["PUT"])
def save_message(key: str):
    body = request.get_json(silent=True) or {}
    data = config.alerts()
    data.setdefault("messages", {})[key] = {
        "label": body.get("label") or key,
        "parse_mode": body.get("parse_mode") or None,
        "text": config.normalise_text(body.get("text") or ""),
        "buttons": body.get("buttons") or [],
    }
    config.save_alerts(data)
    orchestrator.log("info", f"message template '{key}' saved to alerts.yaml")
    return jsonify(ok=True)


@app.route("/api/targets/<key>", methods=["PUT"])
def save_target(key: str):
    body = request.get_json(silent=True) or {}
    chat_id = str(body.get("chat_id") or "").strip()
    if not chat_id:
        return jsonify(ok=False, error="chat_id is required"), 400
    data = config.alerts()
    target = data.setdefault("targets", {}).setdefault(key, {})
    target["label"] = body.get("label") or key
    target["chat_id"] = chat_id
    target["message"] = body.get("message") or "standard"
    for optional in ("escalate_to", "message_thread_id"):
        if body.get(optional):
            target[optional] = body[optional]
        else:
            target.pop(optional, None)
    config.save_alerts(data)
    orchestrator.log("info", f"chat target '{key}' saved to alerts.yaml")
    return jsonify(ok=True)


@app.route("/api/dispatch", methods=["PUT"])
def save_dispatch():
    body = request.get_json(silent=True) or {}
    data = config.alerts()
    settings = data.setdefault("dispatch", {})
    settings["default_target"] = body.get("default_target") or settings.get("default_target")
    settings["escalation_seconds"] = int(body.get("escalation_seconds") or 20)
    settings["max_escalations"] = int(body.get("max_escalations") or 0)
    config.save_alerts(data)
    orchestrator.log("info", "dispatch settings saved to alerts.yaml")
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
@app.route("/api/send", methods=["POST"])
def send():
    body = request.get_json(silent=True) or {}
    event = body.get("event") or {}
    if not event.get("event_id"):
        return jsonify(ok=False, error="event needs an event_id"), 400
    dry_run = bool(body.get("dry_run", True))
    # The bench sends the template open in the editor, so the preview is a
    # promise. Omit it and the target's own configured template is used.
    message = body.get("message") or None
    try:
        record = orchestrator.dispatch_event(
            event, body.get("target"), dry_run=dry_run, message=message
        )
    except (render.RenderError, telegram.TransportError, KeyError) as exc:
        return jsonify(ok=False, error=str(exc)), 400
    if not dry_run:
        poller.start()
    return jsonify(ok=True, alert_id=record["alert_id"], event_id=record["event_id"])


@app.route("/api/simulate-tap", methods=["POST"])
def simulate_tap():
    """Press a button on behalf of a nurse, without a real Telegram client.

    Lets the ack and escalation-cancel paths be demoed in dry-run, which is the
    only way to show them when the bot has no staff group to post into yet.
    """
    body = request.get_json(silent=True) or {}
    event_id = str(body.get("event_id") or "")
    action = body.get("action") or "ack"
    record = orchestrator.ALERTS.get(event_id)
    if record is None:
        return jsonify(ok=False, error=f"no open alert for {event_id}"), 404
    chat_id = record["messages"][-1]["chat_id"] if record["messages"] else ""
    orchestrator.handle_callback(
        {
            "id": "simulated",
            "data": f"{action}:{event_id}",
            "from": {"username": body.get("who") or "demo-nurse"},
            "message": {"chat": {"id": chat_id}},
        },
        dry_run=record["dry_run"],
    )
    return jsonify(ok=True)


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------
@app.route("/api/log")
def activity_log():
    since = request.args.get("since", type=int, default=0)
    return jsonify(
        entries=orchestrator.activity(since),
        alerts=orchestrator.open_alerts(),
        seen_chats=orchestrator.seen_chats(),
        listener=poller.status(),
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    orchestrator.reset()
    return jsonify(ok=True)


@app.route("/api/listen", methods=["POST"])
def listen():
    return jsonify(ok=True, status=poller.start())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    # 0.0.0.0 so the page is reachable from the Windows browser when this runs
    # under WSL. use_reloader=False because the reloader would fork a second
    # process, and a second poller means a 409 from Telegram.
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False, threaded=True)

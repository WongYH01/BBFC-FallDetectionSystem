# demo-tele — Telegram alert bench

A standalone bench for the alert path in the v2 design document: interface
**I-5**, the message content in **§6.3**, and the debounce / dispatch /
escalation / closure duties of **M6 — Alert Orchestrator** (§5.6).

It exists so the wording of the alert and the choice of chat can be changed
without touching code, and so the acknowledgement and escalation behaviour can
be demonstrated before the detector, the node or the gateway exist.

Not covered here: the skeleton clip (§6.4), the event store (M7), and any
detection logic. Fall events are fixtures in `sample_events.yaml`, shaped like
the `FallEvent` object in §7.1 so the real orchestrator can take the same
payload later.

---

## Run it

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Put your bot token in `.env`, then:

```bash
python app.py
```

Open <http://localhost:5001>. It binds `0.0.0.0`, so if you run it under WSL
the Windows browser can still reach it.

`demo-cam`'s `winvenv` already has every dependency, so you can just use that
interpreter instead of building a new environment.

---

## What the three columns do

**Message** — pick a template, edit its text, buttons and parse mode. Every
keystroke re-renders the preview. *Save template* writes it back to
`alerts.yaml`.

**Preview** — the alert as the staff group will see it, with the inline
keyboard drawn. Expand *sendMessage payload* to see the exact JSON that would
go to the Bot API. Choose a chat target underneath and send.

**Sending uses the template open in the editor**, not the one the target names.
The editor decides *what* is sent, the target decides *where* — so the preview
is a promise. A target's own `message:` is the fallback for callers with no
editor (a real orchestrator calling `dispatch_event` without one) and for
escalations routed *to* that target: escalating to `shift-lead` renders the
`escalation` template, not a copy of the ward message. Escalating into the same
chat repeats the wording the alert was raised with.

**Open alerts / Chats / Activity** — every dispatch, escalation and
acknowledgement, live. In dry-run you can tap the buttons from here, which is
how the ack and escalation paths are demoed with no Telegram at all.

The **Dry run** switch in the header is the important one. Dry-run builds the
identical payload and records it without a network call. Uncheck it to send for
real and start the listener that picks up button taps.

---

## Setting up the staff group

The alert goes to a group, which makes the group's chat id the first thing you
need — and Telegram does not show it anywhere in the client.

1. Create the group and add your bot to it.
2. Click **Start listener** in the header. The listener subscribes to
   `my_chat_member`, which fires when the bot is added to a group — and
   Telegram holds undelivered updates for 24 hours, so this still works if you
   added the bot before starting the listener.
3. The group appears in the **Chats the bot can see** panel with its id. If it
   does not — the bot was added more than a day ago — type `/id` in the group.
   That works because BotFather turns **group privacy mode** on by default: the
   bot never receives ordinary group chatter, but it always receives commands.
   It always receives button callbacks too, which is why acknowledgement works
   regardless of that setting.
4. Copy the id, click *Edit target…*, and paste it in.

Group ids are **negative**. A supergroup looks like `-1001234567890`. Only a
1:1 chat with a person is positive.

Two things that bite:

- **A bot cannot message a person who has never started it.** The
  `shift-lead` target in `alerts.yaml` escalates to a direct chat; that person
  must open the bot and press Start once, or escalation to them will fail. If
  that is awkward, point `escalate_to` at a second group instead.
- **A group can become a supergroup**, which changes its chat id. If alerts
  suddenly stop landing, re-read the id from the Chats panel.

If the ward group uses **Topics**, set `message_thread_id` on the target to
post into one topic. Plain groups ignore it.

---

## Configuration

`alerts.yaml` is the source of truth; the UI is an editor over it. Editing the
file by hand is picked up on the next preview or send — no restart. Saving
from the UI rewrites the file, and the YAML dumper does not preserve comments,
which is why the schema notes live in the banner at the top (re-emitted on
every save) rather than inline.

```yaml
messages:      # what the alert says — text, parse mode, buttons
targets:       # where it goes — chat_id, which message, escalation route
rooms:         # room_id → target, the routing M8's registry would own
dispatch:      # default target, escalation window, escalation count
```

One bot token, many chat targets. If you later want *different bots* per ward,
`LiveTransport` takes the token as a constructor argument — give the target an
optional `bot_token_ref` and pick the transport per target in
`orchestrator.transport()`.

### Template variables

Templates are Jinja2, rendered in a sandbox, against an `event` object:

| Variable | Example |
| --- | --- |
| `event.event_id` | `FE-20260905-0031` |
| `event.room` | `Block A · Room 12` |
| `event.room_id` | `room-12` |
| `event.time` | `03:42` |
| `event.date` | `5 Sep 2026` |
| `event.confidence` | `0.87` |
| `event.confidence_pct` | `87` |
| `event.confidence_band` | `High` / `Medium` / `Low` |
| `event.detection_type` | `Fall — confirmed after 8 s immobility` |
| `event.immobility_s` | `8` |
| `event.detector_version` | `rules-v1.2` |
| `event.escalation_level` | `0` first time, `1+` on re-issue |

Anything else you put in the event JSON is available too — the *Event* box in
the UI is editable, so you can add a field and reference it immediately.

**HTML or plain text only, no MarkdownV2.** MarkdownV2 needs eighteen
characters escaped in every interpolated value, and a room name like
`Block A · Room 12 (west)` breaks the message when that is missed. In an
alerting channel that is not a trade worth making. HTML mode auto-escapes
every value, so a stray `<` in a room name cannot break the message.

---

## Behaviour worth knowing

- **Debounce (§5.6).** A second send for an already-open `event_id` is
  logged and dropped. One fall, one alert.
- **Escalation (§6.5).** If nobody taps a button within
  `dispatch.escalation_seconds`, the alert is re-issued — to `escalate_to` if
  the target names one, otherwise to the same chat. `max_escalations` caps it.
  20 seconds is a demo value; the real window is a clinical decision.
- **Acknowledgement** edits *every* message the alert produced, appending who
  acknowledged and when, and removes the keyboard so it cannot be tapped
  twice. A second responder arriving at the group sees it is handled.
- **Callbacks are rejected from chats not listed in `alerts.yaml`** — the
  "bot only accepts callbacks from the registered staff group" control in §9.2.
- **State is in memory.** A restart forgets open alerts and their escalation
  timers. The real system writes the event to M7 before dispatching so the
  record survives a delivery failure; here the activity log stands in for that.
- **One listener per token.** Telegram answers a second poller with a 409, so
  the Flask reloader is off and exactly one thread polls.

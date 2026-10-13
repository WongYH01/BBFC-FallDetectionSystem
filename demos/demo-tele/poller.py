"""Long-polls getUpdates so button taps come back to us.

Acknowledgement is a callback, not a reply (§4.5), and callbacks only arrive
if something is asking Telegram for them. A webhook would be the production
choice; long polling is used here because it needs no public URL, which
matters on a laptop behind NAT.

Only one poller may exist per bot token. Telegram answers a second one with a
409 Conflict, so the demo starts exactly one thread and says so in the UI.

The alert lands in a staff group, which makes finding the group's chat id the
first practical problem: BotFather turns group privacy mode on by default, so
the bot never sees ordinary group chatter and getUpdates stays empty. Two
kinds of update do get through regardless, and both are subscribed to here:
my_chat_member (fires the moment the bot is added to a group) and messages
that are bot commands (so /id typed in the group works). Either one puts the
group in the seen-chats list, ready to paste into a target.
"""
import threading
import time

import orchestrator
from telegram import TransportError, bot_token

_POLL_TIMEOUT = 25


class Poller(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="telegram-poller", daemon=True)
        self._stop = threading.Event()
        self.offset = 0
        self.status = "starting"

    def run(self) -> None:
        orchestrator.log("info", "listening for button taps")
        while not self._stop.is_set():
            try:
                updates = orchestrator.transport(dry_run=False).call(
                    "getUpdates",
                    {
                        "offset": self.offset,
                        "timeout": _POLL_TIMEOUT,
                        "allowed_updates": ["callback_query", "message", "my_chat_member"],
                    },
                )
                self.status = "listening"
            except TransportError as exc:
                self.status = f"error: {exc}"
                orchestrator.log("error", f"poller: {exc}")
                # Back off rather than hammering a bad token or a dead link.
                self._stop.wait(5)
                continue

            for update in updates or []:
                self.offset = max(self.offset, update.get("update_id", 0) + 1)
                self._handle(update)
        self.status = "stopped"

    def _handle(self, update: dict) -> None:
        callback = update.get("callback_query")
        if callback:
            orchestrator.note_chat((callback.get("message") or {}).get("chat"))
            orchestrator.handle_callback(callback, dry_run=False)
            return

        # Not an alert action — but every update tells us about a chat the bot
        # is in, which is how a group's id gets discovered.
        if "my_chat_member" in update:
            member = update["my_chat_member"]
            chat = orchestrator.note_chat(member.get("chat"))
            state = (member.get("new_chat_member") or {}).get("status")
            if chat:
                orchestrator.log(
                    "info", f"bot is now '{state}' in {chat['title']} (chat_id {chat['id']})"
                )
            return

        message = update.get("message")
        if message:
            chat = orchestrator.note_chat(message.get("chat"))
            text = (message.get("text") or "").split("@")[0].strip()
            if chat and text in ("/id", "/start", "/chatid"):
                orchestrator.log(
                    "info", f"{chat['title']} → chat_id {chat['id']} ({chat['type']})"
                )

    def stop(self) -> None:
        self._stop.set()


_poller: Poller | None = None
_lock = threading.Lock()


def start() -> str:
    """Start the listener if a token is configured. Returns a status string."""
    global _poller
    if not bot_token():
        return "no token — dry-run only"
    with _lock:
        if _poller is not None and _poller.is_alive():
            return _poller.status
        _poller = Poller()
        _poller.start()
    time.sleep(0.1)
    return _poller.status


def status() -> str:
    if not bot_token():
        return "no token — dry-run only"
    with _lock:
        if _poller is None or not _poller.is_alive():
            return "not running"
        return _poller.status

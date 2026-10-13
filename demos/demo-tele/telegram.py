"""Telegram Bot API transport, in two interchangeable flavours.

Everything above this module builds payloads; only this module knows whether
they leave the machine. Dry-run exists so the message wording can be iterated
on without spamming a real staff group, and so the demo still runs with no
internet — the payload it records is byte-for-byte what live mode would post.
"""
import itertools
import os
import threading

import requests

API_ROOT = "https://api.telegram.org"


class TransportError(Exception):
    """The Bot API refused the call, or could not be reached."""


def mask(token: str) -> str:
    """Bot tokens are a gateway secret (§9.2) — never render one in full."""
    if not token:
        return "(not set)"
    head, _, tail = token.partition(":")
    return f"{head}:{'•' * 6}{tail[-4:]}" if tail else "•" * 8


class DryRunTransport:
    """Records calls instead of making them. Returns plausible ids."""

    name = "dry-run"
    live = False

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._ids = itertools.count(9001)
        self._lock = threading.Lock()

    def call(self, method: str, payload: dict) -> dict:
        with self._lock:
            self.calls.append({"method": method, "payload": payload})
            if method == "sendMessage":
                return {"message_id": next(self._ids), "chat": {"id": payload.get("chat_id")}}
            if method == "getUpdates":
                return []
            return {"ok": True}


class LiveTransport:
    """Posts to api.telegram.org with the bot token from the environment."""

    name = "live"
    live = True

    def __init__(self, token: str) -> None:
        if not token:
            raise TransportError("TELEGRAM_BOT_TOKEN is not set — live send is unavailable")
        self._token = token
        self._session = requests.Session()

    def call(self, method: str, payload: dict) -> dict:
        # getUpdates long-polls, so the HTTP timeout has to outlive its own.
        timeout = float(payload.get("timeout", 0)) + 15
        try:
            response = self._session.post(
                f"{API_ROOT}/bot{self._token}/{method}", json=payload, timeout=timeout
            )
        except requests.RequestException as exc:
            raise TransportError(f"{method}: {exc}") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise TransportError(f"{method}: HTTP {response.status_code}, non-JSON body") from exc
        if not body.get("ok"):
            raise TransportError(f"{method}: {body.get('description') or body}")
        return body.get("result")


def bot_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()


def make_transport(dry_run: bool):
    return DryRunTransport() if dry_run else LiveTransport(bot_token())


def identity() -> dict:
    """getMe, for showing which bot is configured. Never raises."""
    token = bot_token()
    if not token:
        return {"configured": False, "token": mask(""), "username": None, "error": None}
    try:
        me = LiveTransport(token).call("getMe", {})
        return {
            "configured": True,
            "token": mask(token),
            "username": me.get("username"),
            "error": None,
        }
    except TransportError as exc:
        return {"configured": True, "token": mask(token), "username": None, "error": str(exc)}

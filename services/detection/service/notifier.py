"""HTTP client for the notification service's ingest API.

The contract (services/notification, Spring Boot, context path
/api/v1/notification):

    POST /events                  JSON {eventId, roomId, roomName, confidence}
                                  202 accepted | 200 duplicate | 400 invalid
    PUT  /events/{id}/clip        multipart, field "file", video/mp4
                                  202 stored | 404 unknown event
                                  | 409 clip already attached | 413 too large

Errors are split into the two kinds the outbox worker acts on: a permanent
error (bad request, clip too large) is logged and dropped, a retryable one
(timeout, connection failure, 5xx, or a 404 because the event POST has not
landed yet) backs off and tries again.
"""
from __future__ import annotations

from pathlib import Path

import requests


class NotifyError(Exception):
    """Base for the two failure kinds."""


class PermanentNotifyError(NotifyError):
    """The request will never succeed as-is; do not retry."""


class RetryableNotifyError(NotifyError):
    """A transient failure; retry with backoff."""


class NotificationClient:
    def __init__(self, base_url: str, timeout_sec: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def _url(self, suffix: str) -> str:
        return f"{self.base_url}/{suffix.lstrip('/')}"

    def post_event(self, event_id: str, room_id: str, room_name: str,
                   confidence: float) -> str:
        """Raise the alert. Returns "accepted" (new) or "duplicate".

        Idempotent by contract: the same eventId twice returns 200 and does
        not dispatch a second alert, which is what makes replaying the outbox
        after a restart safe.
        """
        payload = {
            "eventId": event_id,
            "roomId": room_id,
            "roomName": room_name,
            "confidence": confidence,
        }
        try:
            response = requests.post(self._url("events"), json=payload,
                                     timeout=self.timeout_sec)
        except requests.RequestException as exc:
            raise RetryableNotifyError(f"POST /events: {exc}") from exc

        if response.status_code == 202:
            return "accepted"
        if response.status_code == 200:
            return "duplicate"
        raise self._error("POST /events", response)

    def put_clip(self, event_id: str, path: Path) -> str:
        """Upload the skeleton clip. Returns "stored" or "already attached"."""
        filename = Path(path).name
        try:
            with open(path, "rb") as fh:
                response = requests.put(
                    self._url(f"events/{event_id}/clip"),
                    files={"file": (filename, fh, "video/mp4")},
                    timeout=self.timeout_sec,
                )
        except requests.RequestException as exc:
            raise RetryableNotifyError(f"PUT clip {filename}: {exc}") from exc

        if response.status_code in (200, 202):
            return "stored"
        if response.status_code == 409:
            return "already attached"
        raise self._error(f"PUT clip {filename}", response)

    @staticmethod
    def _error(what: str, response: requests.Response) -> NotifyError:
        detail = response.text[:300].replace("\n", " ")
        message = f"{what} -> HTTP {response.status_code}: {detail}"
        if response.status_code == 404:
            # The alert POST has not landed (yet); the worker retries the
            # event first, so this is transient.
            return RetryableNotifyError(message)
        if 400 <= response.status_code < 500:
            return PermanentNotifyError(message)
        return RetryableNotifyError(message)

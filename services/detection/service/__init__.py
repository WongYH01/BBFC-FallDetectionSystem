"""Detection service: demo-cam v2 wired to the notification service's ingest API.

The Flask app and the fall decision stay in `demo-cam/v2`; this package adds the
deployment concerns -- always-on capture, a durable outbox, and the HTTP client
that turns a latched fall into an alert plus a skeleton clip.
"""
from __future__ import annotations

__version__ = "0.1.0"

"""NotificationClient against a stub HTTP server: the full status matrix."""
from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from service.notifier import (
    NotificationClient,
    PermanentNotifyError,
    RetryableNotifyError,
)


class _StubHandler(BaseHTTPRequestHandler):
    def _handle(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.records.append({
            "method": self.command,
            "path": self.path,
            "content_type": self.headers.get("Content-Type", ""),
            "body": body,
        })
        status = self.server.statuses.pop(0) if self.server.statuses else 202
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b"{}")

    do_POST = _handle
    do_PUT = _handle

    def log_message(self, fmt, *args):  # keep the test output clean
        pass


class NotifierTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.server.records = []
        self.server.statuses = []
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1/notification"
        self.client = NotificationClient(self.base, timeout_sec=5)

        self._tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self._tmp.name) / "fall_1.mp4"
        self.clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def _post(self):
        return self.client.post_event("e1", "r1", "Room 1", 0.75)

    # -- POST /events --------------------------------------------------------

    def test_post_event_accepted(self):
        self.server.statuses = [202]
        self.assertEqual(self._post(), "accepted")
        record = self.server.records[0]
        self.assertEqual(record["method"], "POST")
        self.assertEqual(record["path"], "/api/v1/notification/events")
        payload = json.loads(record["body"])
        self.assertEqual(payload, {"eventId": "e1", "roomId": "r1",
                                   "roomName": "Room 1", "confidence": 0.75})

    def test_post_event_duplicate_is_success(self):
        self.server.statuses = [200]
        self.assertEqual(self._post(), "duplicate")

    def test_post_event_bad_request_is_permanent(self):
        self.server.statuses = [400]
        with self.assertRaises(PermanentNotifyError):
            self._post()

    def test_post_event_server_error_is_retryable(self):
        self.server.statuses = [500]
        with self.assertRaises(RetryableNotifyError):
            self._post()

    # -- PUT /events/:id/clip ------------------------------------------------

    def test_put_clip_stored(self):
        self.server.statuses = [202]
        self.assertEqual(self.client.put_clip("e1", self.clip), "stored")
        record = self.server.records[0]
        self.assertEqual(record["method"], "PUT")
        self.assertEqual(record["path"], "/api/v1/notification/events/e1/clip")
        self.assertIn("multipart/form-data", record["content_type"])
        self.assertIn(b'name="file"', record["body"])
        self.assertIn(b'filename="fall_1.mp4"', record["body"])
        self.assertIn(b"video/mp4", record["body"])
        self.assertIn(b"ftypmp42", record["body"])

    def test_put_clip_already_attached_is_success(self):
        self.server.statuses = [409]
        self.assertEqual(self.client.put_clip("e1", self.clip), "already attached")

    def test_put_clip_missing_event_is_retryable(self):
        self.server.statuses = [404]
        with self.assertRaises(RetryableNotifyError):
            self.client.put_clip("e1", self.clip)

    def test_put_clip_too_large_is_permanent(self):
        self.server.statuses = [413]
        with self.assertRaises(PermanentNotifyError):
            self.client.put_clip("e1", self.clip)

    def test_put_clip_server_error_is_retryable(self):
        self.server.statuses = [503]
        with self.assertRaises(RetryableNotifyError):
            self.client.put_clip("e1", self.clip)

    # -- transport -----------------------------------------------------------

    def test_connection_refused_is_retryable(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        client = NotificationClient(f"http://127.0.0.1:{port}", timeout_sec=1)
        with self.assertRaises(RetryableNotifyError):
            client.post_event("e1", "r1", "Room 1", 0.5)


if __name__ == "__main__":
    unittest.main()

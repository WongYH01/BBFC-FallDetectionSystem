"""v2/fall_recorder: the keypoint file written beside each skeleton clip.

These drive `push_frame` through a whole clip -- lead-in, latch, the fixed
post-trigger run -- on a fake clock, with a stand-in for cv2.VideoWriter so no
codec or ffmpeg is needed, and read back what `clip_ready` reports. The
property that matters is that the keypoint file holds the same frames the clip
does, with the latch where the alarm went up.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))

import fall_recorder as fr  # noqa: E402

from fallcore import config as cfg  # noqa: E402
from fallcore.keypoint_file import Person  # noqa: E402

HZ = 15.0           # frames per second pushed, the live rate
PRE = 1.0           # FALL_CLIP_PRE_SECONDS for these tests
POST = 0.5          # FALL_CLIP_MAX_SECONDS


class FakeClock:
    def __init__(self, now: float = 1_000.0):
        self.now = now

    def time(self) -> float:
        return self.now


class FakeWriter:
    """cv2.VideoWriter stand-in: one byte per frame into the raw file."""

    instances: list[FakeWriter] = []

    def __init__(self, path, fourcc, fps, size):
        self.frames = 0
        self._fh = open(path, "wb")
        FakeWriter.instances.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames += 1
        self._fh.write(b"f")

    def release(self):
        self._fh.close()


class NeverOpensWriter:
    """A writer that failed to open: no file, nothing written."""

    def __init__(self, *args):
        pass

    def isOpened(self):
        return False

    def write(self, frame):
        pass

    def release(self):
        pass


def walker(track_id: int = 1, fallen: bool = False) -> Person:
    kp = np.full((cfg.N_KEYPOINTS, 3), 0.5, dtype=np.float32)
    return Person(track_id=track_id, keypoints=kp, centre=(0.5, 0.5),
                  area=0.1, fallen=fallen)


class FallRecorderKeypointsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.clips = Path(self._tmp.name)
        self.clock = FakeClock()
        self.canvas = np.zeros((48, 64, 3), dtype=np.uint8)
        FakeWriter.instances = []
        self._reset_recorder()
        for patch in (
            mock.patch.object(fr, "time", self.clock),
            mock.patch.object(fr, "CLIPS_DIR", str(self.clips)),
            mock.patch.object(fr, "PRE_EVENT_SECONDS", PRE),
            mock.patch.object(fr, "MAX_CLIP_SECONDS", POST),
            mock.patch.object(fr, "COOLDOWN_SECONDS", 0.0),
            mock.patch.object(fr, "KEYPOINTS_ENABLED", True),
            # No ffmpeg: the finished raw file is moved into place as-is.
            mock.patch.object(fr, "FFMPEG_BIN", "no-such-ffmpeg-binary"),
            mock.patch.object(fr.cv2, "VideoWriter", FakeWriter),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self._reset_recorder()
        self._tmp.cleanup()

    @staticmethod
    def _reset_recorder():
        fr._buffer.clear()
        fr._events.clear()
        fr._push_times.clear()
        fr._state.update(recording=False, converting=False, filename=None,
                         started_at=None, cooldown_until=0.0,
                         last_clip_filename=None, last_clip_at=None)
        fr._armed = True
        fr._clip_meta = None
        fr._write_queue = None

    def _push(self, n: int, fallen: bool, score: float = 0.0,
              lead: int | None = None) -> None:
        for _ in range(n):
            self.clock.now += 1 / HZ
            fr.push_frame(self.canvas, fallen, confidence=score,
                          people=(walker(fallen=fallen),), aspect=16 / 9,
                          lead=lead)

    def _fall(self) -> float:
        """Two seconds of walking, then a latch held past the clip's end;
        subject 2 leads until subject 1 falls and raises the alarm.

        Returns the trigger frame's time.
        """
        self._push(int(2 * HZ), fallen=False, score=0.1, lead=2)
        self._push(1, fallen=True, score=0.9, lead=1)
        trigger_t = self.clock.now
        self._push(int(2 * POST * HZ), fallen=True, score=0.9, lead=1)
        return trigger_t

    def _ready(self) -> dict:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            for event in fr.pop_events():
                if event["kind"] == "clip_ready":
                    return event
            time.sleep(0.02)
        self.fail("no clip_ready within 5 s")

    def test_the_keypoint_file_holds_the_clips_frames(self):
        trigger_t = self._fall()
        event = self._ready()

        clip = Path(event["filepath"])
        keypoints = Path(event["keypoints_filepath"])
        self.assertTrue(clip.exists())
        self.assertEqual(keypoints, clip.with_suffix(".npz"))
        self.assertTrue(keypoints.exists())

        with np.load(keypoints) as f:
            t, trigger = f["t"], int(f["trigger"])
            # Frame for frame what went into the video.
            self.assertEqual(len(t), FakeWriter.instances[0].frames)
            # The lead-in is the PRE seconds before the latch, then the latch.
            self.assertGreater(trigger, 0)
            self.assertAlmostEqual(t[trigger], trigger_t)
            self.assertGreaterEqual(t[0], trigger_t - PRE - 1e-9)
            self.assertTrue(np.all(np.diff(t) > 0))
            np.testing.assert_array_equal(f["alarm"][:trigger], False)
            np.testing.assert_array_equal(f["alarm"][trigger:], True)
            np.testing.assert_allclose(f["score"][trigger], 0.9, rtol=1e-6)
            # The subject that led when the alarm latched raised the Alert.
            self.assertEqual(int(f["raised_by"]), 1)
            np.testing.assert_array_equal(f["lead"][:trigger], 2)
            np.testing.assert_allclose(f["aspect"], 16 / 9, rtol=1e-6)
            # The fixed post-trigger run, then the clip closes.
            self.assertAlmostEqual(t[-1] - trigger_t, POST, delta=1 / HZ)
            # One walker in every frame, down from the latch on.
            np.testing.assert_array_equal(f["frame"], np.arange(len(t)))
            np.testing.assert_array_equal(f["track_id"], 1)
            np.testing.assert_array_equal(f["fallen"], f["alarm"])
            self.assertEqual(f["keypoints"].shape, (len(t), cfg.N_KEYPOINTS, 3))

    def test_no_keypoint_file_when_turned_off(self):
        with mock.patch.object(fr, "KEYPOINTS_ENABLED", False):
            self._fall()
            event = self._ready()

        self.assertIsNone(event["keypoints_filepath"])
        self.assertTrue(Path(event["filepath"]).exists())
        self.assertEqual(list(self.clips.glob("*.npz")), [])

    def test_keypoints_survive_a_video_writer_that_never_opened(self):
        with mock.patch.object(fr.cv2, "VideoWriter", NeverOpensWriter):
            self._fall()
            event = self._ready()

        self.assertFalse(Path(event["filepath"]).exists())
        keypoints = Path(event["keypoints_filepath"])
        with np.load(keypoints) as f:
            self.assertGreater(len(f["t"]), int(f["trigger"]))

    def test_a_failed_keypoint_write_still_publishes_the_clip(self):
        with mock.patch.object(fr.keypoint_file, "save",
                               side_effect=OSError("disk full")):
            self._fall()
            event = self._ready()

        self.assertIsNone(event["keypoints_filepath"])
        self.assertTrue(Path(event["filepath"]).exists())

    def test_a_held_latch_records_one_keypoint_file(self):
        # Someone still on the floor keeps the alarm up; the recorder needs the
        # alarm to clear before a second clip, and so a second file.
        self._fall()
        self._ready()
        self._push(int(3 * HZ), fallen=True, score=0.98)
        time.sleep(0.1)
        self.assertEqual(len(list(self.clips.glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()

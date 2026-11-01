"""scripts/run_refit.py: the refit workflow's entry point.

The seam is `main(argv, env)`: what it writes to the run summary and what it
exits with. `refit.run` is replaced by a stub returning each kind of verdict,
or raising, because the Refit itself is tested through its own seam
(test_refit.py) and a real run needs the deployed weights.
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_refit import main

from fallcore import refit

REPORT = refit.Report(untriaged=1, other_pose=0, unreadable=0, training=3,
                      new_training=3, new_genuine=1, eval=1, eval_genuine=0,
                      training_hash="ab" * 32)


def verdict(kind: refit.Kind, reason: str) -> refit.Verdict:
    return refit.Verdict(kind, reason, REPORT)


class RunRefitTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.summary = Path(tmp.name) / "summary.md"
        self.env = {"GITHUB_STEP_SUMMARY": str(self.summary)}

    def main(self, result=None, error=None):
        """Exit code, and everything printed, of one run on a stub Refit."""
        out = io.StringIO()
        with mock.patch.object(refit, "run", return_value=result,
                               side_effect=error) as run, \
                redirect_stdout(out), redirect_stderr(out):
            code = main(["mirror"], env=self.env)
        self.run_args = run.call_args
        return code, out.getvalue()

    def written(self) -> str:
        return self.summary.read_text(encoding="utf-8")

    def test_a_skipped_week_finishes_green_with_its_reason_in_the_summary(self):
        code, _ = self.main(verdict(refit.Kind.SKIPPED, "2 new, fewer than 5"))
        self.assertEqual(code, 0)
        self.assertIn("Refit skipped: 2 new, fewer than 5", self.written())
        self.assertIn("training Alerts: 3 (3 new, 1 of them Genuine)",
                      self.written())

    def test_the_refit_runs_the_live_rule_from_the_environment(self):
        self.env.update(BUFFER_LEN="2", FALL_CLEAR_WINDOWS="1")
        self.main(verdict(refit.Kind.SKIPPED, "2 new, fewer than 5"))
        folder, decision = self.run_args.args
        self.assertEqual(folder, "mirror")
        self.assertEqual((decision.buffer, decision.clear_windows), (2, 1))

    def test_a_failed_gate_finishes_green(self):
        code, _ = self.main(verdict(refit.Kind.GATE_FAILED, "1 fall lost"))
        self.assertEqual(code, 0)
        self.assertIn("Refit gate failed: 1 fall lost", self.written())

    def test_a_manifest_mismatch_finishes_red_and_says_why(self):
        code, _ = self.main(error=refit.PinnedDataError(
            "keypoints/heldout/fall/f1.npz does not match its SHA-256"))
        self.assertNotEqual(code, 0)
        self.assertIn("keypoints/heldout/fall/f1.npz does not match its "
                      "SHA-256", self.written())

    def test_an_unexpected_error_finishes_red_without_naming_an_alert(self):
        # Pinned clips are named in the manifest in git; an Alert's file is
        # not, and its message could name one (ADR 0002).
        code, printed = self.main(error=OSError(
            "keypoints/labelled/genuine/picam/event-0042.npz: bad block"))
        self.assertNotEqual(code, 0)
        self.assertIn("OSError", self.written())
        self.assertNotIn("event-0042", self.written() + printed)


if __name__ == "__main__":
    unittest.main()

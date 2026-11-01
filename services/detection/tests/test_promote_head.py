"""scripts/promote_head.py: the Promotion's rewrite of the weight fetcher.

The seam is `repin(source, revision, sha256)`, run on the real
fetch_weights.py: a Promotion PR may change only the pinned revision and the
head's SHA-256 (ADR 0003). The upload itself needs the Hub and is proved only
by a real Promotion.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fetch_weights
from promote_head import HEAD, repin

SOURCE = (SCRIPTS / "fetch_weights.py").read_bytes().decode("utf-8")
REVISION = "0123456789abcdef0123456789abcdef01234567"
SHA = "cd" * 32


class RepinTest(unittest.TestCase):
    def test_only_the_revision_and_the_heads_hash_change(self):
        out = repin(SOURCE, REVISION, SHA)
        self.assertIn(f'DEFAULT_REVISION = "{REVISION}"', out)
        self.assertIn(SHA, out)
        # Put the old values back: every other byte must be the original's.
        back = (out.replace(REVISION, fetch_weights.DEFAULT_REVISION)
                .replace(SHA, fetch_weights.FILES[HEAD][1]))
        self.assertEqual(back, SOURCE)

    def test_the_rewritten_fetcher_pins_the_new_head(self):
        pins = {}
        exec(compile(repin(SOURCE, REVISION, SHA), "fetch_weights.py",
                     "exec"), pins)
        self.assertEqual(pins["DEFAULT_REVISION"], REVISION)
        self.assertEqual(pins["FILES"][HEAD][1], SHA)
        others = {k: v for k, v in pins["FILES"].items() if k != HEAD}
        self.assertEqual(others, {k: v for k, v in fetch_weights.FILES.items()
                                  if k != HEAD})

    def test_a_fetcher_it_does_not_recognise_is_refused(self):
        with self.assertRaises(ValueError):
            repin(SOURCE.replace(HEAD, "checkpoints/renamed.npz"),
                  REVISION, SHA)


if __name__ == "__main__":
    unittest.main()

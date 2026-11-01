"""Run this week's Refit on a mirrored bucket: the refit workflow's one step
that decides anything.

    python scripts/run_refit.py <folder> [--promote-dir DIR]

`folder` is the bucket mirrored as `fallcore.refit` lays it out, with the
manifest copied in. The Fall Decision is the deployed one, the live rule read
from the environment as the detector reads it (`FallDecision.from_env`), on
the weights `fetch_weights.py` put in their default paths.

The verdict is printed and appended to the run summary
(`$GITHUB_STEP_SUMMARY`, when set). A skipped week and a failed gate exit 0,
so a red run always means the loop is broken: anything `refit.run` raises
exits 1, its message shown only when it is about the pinned data.

On a promotion, the Candidate Head is saved as `DIR/head.npz` and the
Promotion PR's body as `DIR/report.md`, and `promoted=true` is written to
`$GITHUB_OUTPUT` for the workflow's upload and PR steps.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from fallcore import refit
from fallcore.fall_decision import FallDecision


def main(argv: list[str] | None = None,
         env: Mapping[str, str] = os.environ) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="the mirrored bucket")
    ap.add_argument("--promote-dir", help="where a promotion's head and PR "
                    "body are saved")
    args = ap.parse_args(argv)

    decision = FallDecision.from_env(env)
    try:
        verdict = refit.run(args.folder, decision)
        text, code = str(verdict), 0
        if verdict.kind is refit.Kind.PROMOTED and args.promote_dir:
            _hand_off(verdict, Path(args.promote_dir), env)
    except refit.PinnedDataError as exc:
        text, code = f"Refit error: {exc}", 1
    except Exception as exc:
        # Its message or traceback could name an Alert's file, and the log
        # is public: the type alone (ADR 0002).
        text, code = (f"Refit error: {type(exc).__name__}, message withheld "
                      f"because it may name an Alert"), 1
    print(text, file=sys.stderr if code else sys.stdout)
    if env.get("GITHUB_STEP_SUMMARY"):
        with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"```text\n{text}\n```\n")
    return code


def _hand_off(verdict: refit.Verdict, folder: Path,
              env: Mapping[str, str]) -> None:
    """Save what the workflow promotes, then tell it to."""
    verdict.head.save(folder / "head.npz")
    # The report is the run summary's: published, so it names no Alert.
    (folder / "report.md").write_text(
        "The weekly Refit's Promotion (ADR 0003). It merges itself once "
        f"`gate` is green.\n\n```text\n{verdict}\n```\n", encoding="utf-8")
    if env.get("GITHUB_OUTPUT"):
        with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write("promoted=true\n")


if __name__ == "__main__":
    sys.exit(main())

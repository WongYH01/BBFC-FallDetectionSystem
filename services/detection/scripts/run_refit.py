"""Run this week's Refit on a mirrored bucket: the refit workflow's one step
that decides anything.

    python scripts/run_refit.py <folder>

`folder` is the bucket mirrored as `fallcore.refit` lays it out, with the
manifest copied in. The Fall Decision is the deployed one, the live rule read
from the environment as the detector reads it (`FallDecision.from_env`), on
the weights `fetch_weights.py` put in their default paths.

The verdict is printed and appended to the run summary
(`$GITHUB_STEP_SUMMARY`, when set). A skipped week and a failed gate exit 0,
so a red run always means the loop is broken: anything `refit.run` raises
exits 1, its message shown only when it is about the pinned data.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping

from fallcore import refit
from fallcore.fall_decision import FallDecision


def main(argv: list[str] | None = None,
         env: Mapping[str, str] = os.environ) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="the mirrored bucket")
    args = ap.parse_args(argv)

    decision = FallDecision.from_env(env)
    try:
        text, code = str(refit.run(args.folder, decision)), 0
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


if __name__ == "__main__":
    sys.exit(main())

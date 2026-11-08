"""Promote a Candidate Head: the refit workflow's upload step (ADR 0003).

    HF_WRITE_TOKEN=hf_... python scripts/promote_head.py <head.npz> <branch>

Uploads the head to the weights repository under the deployed head's name,
as a commit on a new `branch` made from the revision `fetch_weights.py`
pins, so the new revision differs from the deployed one by the head alone.
The repository's main never moves, so a Promotion whose PR is never
opened or merged leaves nothing for the next one to trip on. Then rewrites
`fetch_weights.py` to pin the new revision and the head's SHA-256, and
nothing else: that file is the Promotion PR.

The commit message is the training set's hash, the one thing that says what
the head was fitted on without naming an Alert (ADR 0002).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

import fetch_weights
from fallcore.fall_decision import DEPLOYED_HEAD

#: The deployed Camera Head's name in the weights repository.
REMOTE_HEAD = f"checkpoints/{DEPLOYED_HEAD}"
FETCHER = Path(fetch_weights.__file__)


def repin(source: str, revision: str, sha256: str) -> str:
    """`source`, the fetcher's text, pinning `revision` and the head's
    `sha256`; every other byte unchanged."""
    out, n = re.subn(r'^(DEFAULT_REVISION = ")[0-9a-f]{40}(")',
                     rf"\g<1>{revision}\g<2>", source, flags=re.M)
    out, m = re.subn(rf'("{re.escape(REMOTE_HEAD)}": \(\s*"[^"]*",\s*")[0-9a-f]{{64}}(")',
                     rf"\g<1>{sha256}\g<2>", out)
    if (n, m) != (1, 1):
        raise ValueError("fetch_weights.py does not pin DEFAULT_REVISION and "
                         f"{REMOTE_HEAD} as this script expects")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("head", help="the Candidate Head, as run_refit.py saved it")
    ap.add_argument("branch", help="the weights repository branch to create")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    head = Path(args.head)
    sha = fetch_weights.sha256(head)
    with np.load(head, allow_pickle=False) as blob:
        training_hash = str(blob["training_hash"])
    api, token = HfApi(), os.environ["HF_WRITE_TOKEN"]
    api.create_branch(fetch_weights.DEFAULT_REPO, branch=args.branch,
                      revision=fetch_weights.DEFAULT_REVISION, token=token)
    info = api.upload_file(
        path_or_fileobj=str(head), path_in_repo=REMOTE_HEAD,
        repo_id=fetch_weights.DEFAULT_REPO, token=token,
        revision=args.branch, parent_commit=fetch_weights.DEFAULT_REVISION,
        commit_message=f"Promote a Refit head, training set SHA-256 "
                       f"{training_hash}")
    # Bytes in, bytes out, so the line endings are the checkout's.
    source = FETCHER.read_bytes().decode("utf-8")
    FETCHER.write_bytes(repin(source, info.oid, sha).encode("utf-8"))
    print(f"{fetch_weights.DEFAULT_REPO} @ {info.oid[:12]}: {REMOTE_HEAD} "
          f"{sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

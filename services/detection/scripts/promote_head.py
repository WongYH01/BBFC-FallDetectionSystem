"""Promote a Candidate Head: the refit workflow's upload step (ADR 0003).

    HF_WRITE_TOKEN=hf_... python scripts/promote_head.py <head.npz>

Uploads the head to the weights repository under the deployed head's name,
as a commit on top of the revision `fetch_weights.py` pins, so the new
revision differs from the deployed one by the head alone, and the upload
fails if the repository has moved on since. Then rewrites `fetch_weights.py`
to pin the new revision and the head's SHA-256, and nothing else: that file
is the Promotion PR.

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

#: The deployed head's name in the weights repository.
HEAD = "checkpoints/probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz"
FETCHER = Path(fetch_weights.__file__)


def repin(source: str, revision: str, sha256: str) -> str:
    """`source`, the fetcher's text, pinning `revision` and the head's
    `sha256`; every other byte unchanged."""
    out, n = re.subn(r'^(DEFAULT_REVISION = ")[0-9a-f]{40}(")',
                     rf"\g<1>{revision}\g<2>", source, flags=re.M)
    out, m = re.subn(rf'("{re.escape(HEAD)}": \(\s*"[^"]*",\s*")[0-9a-f]{{64}}(")',
                     rf"\g<1>{sha256}\g<2>", out)
    if (n, m) != (1, 1):
        raise ValueError("fetch_weights.py does not pin DEFAULT_REVISION and "
                         f"{HEAD} as this script expects")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("head", help="the Candidate Head, as run_refit.py saved it")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    head = Path(args.head)
    sha = fetch_weights.sha256(head)
    with np.load(head, allow_pickle=False) as blob:
        training_hash = str(blob["training_hash"])
    info = HfApi().upload_file(
        path_or_fileobj=str(head), path_in_repo=HEAD,
        repo_id=fetch_weights.DEFAULT_REPO,
        token=os.environ["HF_WRITE_TOKEN"],
        parent_commit=fetch_weights.DEFAULT_REVISION,
        commit_message=f"Promote a Refit head, training set SHA-256 "
                       f"{training_hash}")
    # Bytes in, bytes out, so the line endings are the checkout's.
    source = FETCHER.read_bytes().decode("utf-8")
    FETCHER.write_bytes(repin(source, info.oid, sha).encode("utf-8"))
    print(f"{fetch_weights.DEFAULT_REPO} @ {info.oid[:12]}: {HEAD} "
          f"{sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

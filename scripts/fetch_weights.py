"""Fetch the deployment weights from the project's Hugging Face model repo.

The five files the stream deployment needs -- the ONNX pose backbone and the
four fall classifier checkpoints -- are artifacts, not source, so they are not
in git. This script downloads them from

    junyuu/fall-detection-bbfc

into exactly the paths `demo-cam/v2` resolves by default:

    demo-cam/models/yolo26n-pose.onnx
    runs/checkpoints/final_yolo26n.pt
    runs/checkpoints/hn10_full_hn_s99.pt
    runs/checkpoints/hn10_coords_hn_s99.pt
    runs/checkpoints/omnifall_cs_full.pt

Every file is checked against the SHA-256 recorded in the model card; a mismatch
is an error, not a warning. The revision is pinned to the commit those hashes
were taken from, so a deployment cannot silently change under you.

The repo is private: run `hf auth login` once on this machine, or set HF_TOKEN
(a read token is enough).

    python scripts/fetch_weights.py                 # into this repo
    python scripts/fetch_weights.py --check         # verify present files only
    python scripts/fetch_weights.py --force         # re-download everything
    python scripts/fetch_weights.py --root D:\\edge  # a different tree
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "junyuu/fall-detection-bbfc"
#: The commit the SHA-256s below were taken from. Move it forward only when the
#: model card in the repo is updated with new hashes.
DEFAULT_REVISION = "06984e6ecf35a46a92d220f803dd1ce4850f4e14"

#: remote path in the repo -> (path under the target root, sha256)
FILES = {
    "pose/yolo26n-pose-imgsz640.onnx": (
        "demo-cam/models/yolo26n-pose.onnx",
        "ea6b37044a5784d279dd93837b2fd89aaa2e0db720ad0928b4de2d78ad1f5c57"),
    "checkpoints/final_yolo26n.pt": (
        "runs/checkpoints/final_yolo26n.pt",
        "c046520aa4dba9d2472a0ae8c821a816f4df802948755f91a00144e7960d9ee7"),
    "checkpoints/hn10_full_hn_s99.pt": (
        "runs/checkpoints/hn10_full_hn_s99.pt",
        "19408d2d8845fc5a0382d3822363a2e18ced4f03f9a1022fba457ee0beabdb35"),
    "checkpoints/hn10_coords_hn_s99.pt": (
        "runs/checkpoints/hn10_coords_hn_s99.pt",
        "c25300c0723efa5a541ffca4f69db64da05145d42aed4a3eabff48870aeba63c"),
    "checkpoints/omnifall_cs_full.pt": (
        "runs/checkpoints/omnifall_cs_full.pt",
        "588320a91722500df43c74dc3a949cdd90708f5989ebf2ff7eae2acdea945ae5"),
}


def sha256(path: Path) -> str:
    """Streaming SHA-256, so a 12 MB blob does not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_one(repo: str, revision: str, remote: str, root: Path,
              force: bool, check: bool) -> str:
    """Return 'ok' or 'downloaded'; raise on a hash mismatch or missing file."""
    rel, want = FILES[remote]
    target = root / rel

    if target.exists() and not force:
        got = sha256(target)
        if got == want:
            print(f"  ok       {rel}")
            return "ok"
        if check:
            raise RuntimeError(f"{rel}: SHA-256 mismatch (have {got[:12]}…, "
                               f"want {want[:12]}…)")
        print(f"  stale    {rel} (hash differs, re-downloading)")

    if check:
        raise FileNotFoundError(f"{rel}: missing (run without --check to fetch)")

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        cached = Path(hf_hub_download(repo_id=repo, filename=remote,
                                      revision=revision))
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"cannot download {remote} from {repo}: {exc}. If the repo is "
            f"private, run `hf auth login` or set HF_TOKEN."
        ) from exc

    got = sha256(cached)
    if got != want:
        raise RuntimeError(
            f"{remote}: downloaded file has SHA-256 {got[:12]}… but the model "
            f"card records {want[:12]}… -- refusing to install it. If the repo "
            f"was updated on purpose, refresh the hashes in this script."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, target)
    print(f"  fetched  {rel}")
    return "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
                    help="tree the weights are written into (default: repo root)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--force", action="store_true",
                    help="re-download even when a valid local file exists")
    ap.add_argument("--check", action="store_true",
                    help="verify the local files only; never download")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"{args.repo} @ {args.revision[:12]}  ->  {root}")
    results = {}
    for remote in FILES:
        results[remote] = fetch_one(args.repo, args.revision, remote, root,
                                    args.force, args.check)

    fetched = sum(v == "downloaded" for v in results.values())
    ok = sum(v == "ok" for v in results.values())
    print(f"\n{ok} already present, {fetched} fetched "
          f"({len(FILES)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

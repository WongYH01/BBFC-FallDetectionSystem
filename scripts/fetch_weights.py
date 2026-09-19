"""Fetch the deployment weights from the project's Hugging Face model repo.

The three files the stream deployment needs -- the ONNX pose backbone and the
calibrated fall model (one transformer checkpoint plus the logistic head fitted
on the deployment camera) -- are artifacts, not source, so they are not in git.
This script downloads them from

    junyuu/fall-detection-bbfc

into exactly the paths `demo-cam/v2` resolves by default:

    demo-cam/models/yolo26n-pose.onnx
    runs/checkpoints/augnone_ms_coords_hn_s99.pt
    runs/checkpoints/probe_augnone_ms_coords_hn_s99.npz

The checkpoint without its head still runs (`CALIBRATION_HEAD=` empty) and
scores lower; the head without its checkpoint is refused. The four
single-corpus checkpoints the ensemble used to average
(`final_yolo26n`, `hn10_full_hn_s99`, `hn10_coords_hn_s99`,
`omnifall_cs_full`) were removed from the release when the calibrated model
replaced them -- they remain downloadable from the previous revision
`06984e6ecf35a46a92d220f803dd1ce4850f4e14` if the old ensemble is ever wanted
back.

Every file is checked against the SHA-256 recorded in the model card; a mismatch
is an error, not a warning. The revision is pinned to the commit those hashes
were taken from, so a deployment cannot silently change under you.

The repo is private, so a Hugging Face **read** token is required, and it must
come from demo-cam/.env:

    FALL_WEIGHTS_TOKEN=hf_xxx

That file is already the deployment secrets file for the Flask app and is
gitignored. Create the token at https://huggingface.co/settings/tokens (read
scope), copy demo-cam/.env.example to demo-cam/.env if it does not exist yet,
and add the line.

There is no other auth path: no CLI flag, no process environment variable, no
stored `hf auth login` session. The token is passed to the Hub call and never
printed. `--env-file` changes which file is read; `--check` verifies the local
files and needs no token at all.

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

#: The one variable the script reads; written into demo-cam/.env by the operator.
TOKEN_VAR = "FALL_WEIGHTS_TOKEN"

DEFAULT_REPO = "junyuu/fall-detection-bbfc"
#: The commit the SHA-256s below were taken from -- the one that replaced the
#: four single-corpus checkpoints with the calibrated pair. Move it forward only
#: when the model card in the repo is updated with new hashes.
DEFAULT_REVISION = "09c33ba5d37d0bf38ab3f42d4be631732daaef00"

#: remote path in the repo -> (path under the target root, sha256)
FILES = {
    "pose/yolo26n-pose-imgsz640.onnx": (
        "demo-cam/models/yolo26n-pose.onnx",
        "ea6b37044a5784d279dd93837b2fd89aaa2e0db720ad0928b4de2d78ad1f5c57"),
    "checkpoints/augnone_ms_coords_hn_s99.pt": (
        "runs/checkpoints/augnone_ms_coords_hn_s99.pt",
        "b65d8a35668a57a711660eeb2d584db34bd0cbc1f1aedbc78134b3a290f7d36a"),
    "checkpoints/probe_augnone_ms_coords_hn_s99.npz": (
        "runs/checkpoints/probe_augnone_ms_coords_hn_s99.npz",
        "6a97474e133b019ccbdf82c048251a33c7cb4dbcb8ae36d3520e0ac55cd0dc92"),
}


def sha256(path: Path) -> str:
    """Streaming SHA-256, so a 12 MB blob does not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_token(env_file: Path) -> str:
    """The read token from `env_file`, or a clear error explaining how to set it.

    A deliberately small parser rather than python-dotenv: this script is the
    bootstrap that runs before a demo venv is guaranteed to exist, so it should
    not depend on anything outside the standard library. Handles blank lines,
    `#` comments, an optional leading `export`, and surrounding quotes around
    the value.
    """
    if not env_file.exists():
        raise FileNotFoundError(
            f"{env_file} not found -- copy demo-cam/.env.example to "
            f"demo-cam/.env and add {TOKEN_VAR}=hf_... (read token from "
            f"https://huggingface.co/settings/tokens)")
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == TOKEN_VAR:
            token = value.strip().strip("'\"")
            if token:
                return token
            break
    raise ValueError(
        f"{TOKEN_VAR} is not set in {env_file} -- add {TOKEN_VAR}=hf_... "
        f"(read token from https://huggingface.co/settings/tokens)")


def fetch_one(repo: str, revision: str, remote: str, root: Path,
              force: bool, check: bool, token: str | None = None) -> str:
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
                                      revision=revision, token=token))
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"cannot download {remote} from {repo}: {exc}. The repo is "
            f"private: set {TOKEN_VAR} in demo-cam/.env to a read token."
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
    ap.add_argument("--env-file", default=str(
        Path(__file__).resolve().parents[1] / "demo-cam" / ".env"),
        help="secrets file holding FALL_WEIGHTS_TOKEN (default: demo-cam/.env)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"{args.repo} @ {args.revision[:12]}  ->  {root}")
    results = {}
    try:
        token = None
        if not args.check:
            token = read_token(Path(args.env_file))
            print(f"auth: {TOKEN_VAR} from {args.env_file}")
        for remote in FILES:
            results[remote] = fetch_one(args.repo, args.revision, remote, root,
                                        args.force, args.check, token)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    fetched = sum(v == "downloaded" for v in results.values())
    ok = sum(v == "ok" for v in results.values())
    print(f"\n{ok} already present, {fetched} fetched "
          f"({len(FILES)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

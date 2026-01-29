#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = {
    "train": "scripts/train_seg.py",
    "predict": "scripts/predict_seg.py",
    "eval": "scripts/eval_seg.py",
    "inference": "scripts/inference_seg.py",
}


def run_script(script_rel: str, forwarded: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / script_rel
    if not script_path.exists():
        print(f"[ERROR] Cannot find {script_path}", file=sys.stderr)
        return 2

    # If user used: cli.py inference -- --bids_dir ...
    # strip the leading "--" so argparse in inference_seg.py works
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    cmd = [sys.executable, str(script_path)] + forwarded
    print("[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(repo_root)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="NipoSimCortex",
        description="CLI wrapper that forwards args to underlying scripts.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Subcommands (no extra args defined here; we forward unknown flags)
    for name in SCRIPTS:
        sub.add_parser(name, help=f"Run {name}")

    # Key trick: parse_known_args keeps unknown flags instead of erroring
    args, unknown = parser.parse_known_args()

    script = SCRIPTS.get(args.cmd)
    if script is None:
        print(f"[ERROR] Unknown command: {args.cmd}", file=sys.stderr)
        return 2

    return run_script(script, unknown)


if __name__ == "__main__":
    raise SystemExit(main())

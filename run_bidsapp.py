#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent
INFER_SCRIPT = REPO_ROOT / "scripts" / "inference_seg.py"
DEFAULT_CKPT = REPO_ROOT / "assets" / "seg_best_dice.pt"
DEFAULT_MNI  = REPO_ROOT / "assets" / "MNI152_T1_1mm.nii.gz"


def _norm_sub(s: str) -> str:
    return s if s.startswith("sub-") else f"sub-{s}"


def _norm_ses(s: str) -> str:
    return s if s.startswith("ses-") else f"ses-{s}"


def _list_subjects(bids_dir: Path):
    subs = sorted([p.name for p in bids_dir.glob("sub-*") if p.is_dir()])
    return subs

def write_deriv_desc(path: Path, name: str, version: str, tag: str):
    path.mkdir(parents=True, exist_ok=True)
    dd = path / "dataset_description.json"
    if dd.exists():
        return
    payload = {
        "Name": name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {"Name": "NipoSimCortex", "Version": version,
             "Container": {"Type": "docker", "Tag": tag}}
        ],
    }
    dd.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description="NipoSimCortex BIDS-App entrypoint (participant level)."
    )
    p.add_argument("bids_dir", type=str, help="Input BIDS dataset root (raw).")
    p.add_argument("output_dir", type=str, help="Output directory (BIDS-App style).")
    p.add_argument("analysis_level", choices=["participant"], help="Only participant supported.")

    # BIDS-App common flags
    p.add_argument("--participant_label", nargs="+", default=None,
                   help="Participant(s) without 'sub-' (e.g., 0001) OR with it (sub-0001). "
                        "If omitted: run all subjects in bids_dir.")
    p.add_argument("--session_label", default=None,
                   help="Optional session without 'ses-' (e.g., 01) or with it (ses-01).")
    p.add_argument("--skip_bids_validation", action="store_true",
                   help="Accepted for compatibility; not used.")

    # Model/runtime options
    p.add_argument("--ckpt_path", default=str(DEFAULT_CKPT), help="Model checkpoint path.")
    p.add_argument("--mni_template", default=str(DEFAULT_MNI), help="MNI template path.")
    p.add_argument("--force_reg", action="store_true", help="Force rerun registration.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    p.add_argument("--cuda", type=int, default=0, help="CUDA device index (if --gpu).")

    args = p.parse_args()

    version = "0.4"
    tag = f"niposimcortex:{version}"
    write_deriv_desc(Path(args.output_dir) / "derivatives" / "niposimcortex_preproc",
                    "NipoSimCortex Preprocessing", version, tag)
    write_deriv_desc(Path(args.output_dir) / "derivatives" / "niposimcortex_seg",
                    "NipoSimCortex Segmentation", version, tag)


    bids_dir = Path(args.bids_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not INFER_SCRIPT.exists():
        print(f"[ERROR] Missing inference script: {INFER_SCRIPT}", file=sys.stderr)
        return 2

    # Resolve subjects
    if args.participant_label is None:
        subjects = _list_subjects(bids_dir)
        if not subjects:
            print(f"[ERROR] No subjects found under {bids_dir} (expected sub-*)", file=sys.stderr)
            return 2
    else:
        subjects = [_norm_sub(x) for x in args.participant_label]

    rc = 0
    for sub in subjects:
        t0 = time.time()

        cmd = [
            sys.executable, str(INFER_SCRIPT),
            "--bids_dir", str(bids_dir),
            "--subject_id", sub,
            "--output_dir", str(out_dir),
            "--ckpt_path", str(Path(args.ckpt_path).resolve()),
            "--mni_template", str(Path(args.mni_template).resolve()),
        ]

        if args.session_label:
            cmd += ["--session_id", _norm_ses(args.session_label)]
        if args.force_reg:
            cmd += ["--force_reg"]
        if args.gpu:
            cmd += ["-g", "--cuda", str(args.cuda)]

        print("[RUN]", " ".join(cmd))
        r = subprocess.run(cmd).returncode
        dt = time.time() - t0
        print(f"[BIDSAPP] {sub} finished in {dt:.2f}s ({dt/60:.2f} min) rc={r}")

        if r != 0:
            rc = r  # keep last non-zero, continue

    return rc


if __name__ == "__main__":
    raise SystemExit(main())


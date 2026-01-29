#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import time

import numpy as np
import nibabel as nib
import torch

# ---- Make repo imports work regardless of cwd ----
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from model.UNet import Unet  # noqa: E402


# -------------------------
# Utilities
# -------------------------f
def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize(vol: np.ndarray, mode: str) -> np.ndarray:
    vol = vol.astype(np.float32)
    if mode == "none":
        return vol
    if mode == "zscore":
        m = float(vol.mean())
        s = float(vol.std()) + 1e-8
        return (vol - m) / s
    if mode == "minmax":
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax - vmin < 1e-8:
            return vol * 0.0
        return (vol - vmin) / (vmax - vmin)
    raise ValueError(f"Unknown norm mode: {mode}")


def pad_to_multiple(vol: np.ndarray, mult: int) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """Pad (D,H,W) symmetrically to multiples of `mult`, return padded vol and unpad slices."""
    pads = []
    for dim in vol.shape:
        r = dim % mult
        pad_total = 0 if r == 0 else (mult - r)
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pads.append((pad_before, pad_after))

    vol_p = np.pad(vol, pads, mode="constant", constant_values=0)

    sl = tuple(slice(p[0], p[0] + s) for p, s in zip(pads, vol.shape))
    return vol_p, sl  # type: ignore


# -------------------------
# BIDS input resolution
# -------------------------
def t1_candidates(anat_dir: Path) -> List[Path]:
    if not anat_dir.exists():
        return []
    c = []
    c += sorted(anat_dir.glob("*_T1w.nii.gz"))
    c += sorted(anat_dir.glob("*_T1w.nii"))
    return c


def find_raw_t1w_bids(bids_dir: Path, subject_id: str, session_id: Optional[str]) -> Path:
    """
    BIDS-first resolver with session optional.
    - If session_id provided: bids/sub/ses/anat/*_T1w.nii(.gz)
    - If no session_id:
        * try bids/sub/anat/*_T1w.nii(.gz)
        * else, if session folders exist:
            - if exactly ONE session has T1w -> use it
            - if multiple sessions have T1w -> error (must pass --session_id)
    """
    subj_dir = bids_dir / subject_id
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject folder not found: {subj_dir}")

    if session_id:
        anat = subj_dir / session_id / "anat"
        c = t1_candidates(anat)
        if not c:
            raise FileNotFoundError(f"No T1w found under: {anat}")
        if len(c) > 1:
            raise RuntimeError(
                "Multiple T1w files found (please pass --t1_path to disambiguate):\n"
                + "\n".join(str(x) for x in c)
            )
        return c[0]

    # no-session BIDS
    c0 = t1_candidates(subj_dir / "anat")
    if c0:
        if len(c0) > 1:
            raise RuntimeError(
                "Multiple T1w files found (please pass --t1_path to disambiguate):\n"
                + "\n".join(str(x) for x in c0)
            )
        return c0[0]

    # sessioned BIDS but session not provided
    ses_dirs = sorted([p for p in subj_dir.glob("ses-*") if p.is_dir()])
    if not ses_dirs:
        raise FileNotFoundError(f"No anat/ and no ses-*/ found under: {subj_dir}")

    all_hits = []
    for sdir in ses_dirs:
        all_hits += [(sdir.name, p) for p in t1_candidates(sdir / "anat")]

    if not all_hits:
        raise FileNotFoundError(f"No T1w found under any ses-*/anat for: {subj_dir}")

    sessions_with_t1 = sorted(set(s for s, _ in all_hits))
    if len(sessions_with_t1) > 1:
        raise RuntimeError(
            "Multiple sessions contain T1w. Please pass --session_id. Available:\n"
            + "\n".join(sessions_with_t1)
        )

    # exactly one session has T1w
    only = [p for s, p in all_hits if s == sessions_with_t1[0]]
    if len(only) > 1:
        raise RuntimeError(
            f"Multiple T1w found in {sessions_with_t1[0]} (please pass --t1_path):\n"
            + "\n".join(str(x) for x in only)
        )
    return only[0]


# -------------------------
# Preprocessing: raw T1w -> MNI (NiftyReg)
# -------------------------
def register_t1_to_mni(
    t1_path: Path,
    mni_template: Path,
    out_dir: Path,
    force: bool = False,
) -> Tuple[Path, Path]:
    """
    Uses NiftyReg:
      reg_aladin  -> affine txt
      reg_resample -> t1w_mni.nii.gz in template grid

    Caches outputs if exist unless force=True.
    """
    ensure_dir(out_dir)

    affine_path = out_dir / "transform_affine.txt"
    t1_mni_path = out_dir / "t1w_mni.nii.gz"
    

    if (not force) and t1_mni_path.exists() and affine_path.exists():
        print("[INFO] Reusing cached MNI registration:", t1_mni_path)
        return t1_mni_path, affine_path

    # Basic tool availability checks
    for tool in ("reg_aladin", "reg_resample"):
        if _which(tool) is None:
            raise RuntimeError(
                f"Required tool not found in PATH: {tool}\n"
                "Install NiftyReg or make sure modules are loaded (reg_aladin/reg_resample)."
            )

   
    tmp_res = out_dir / "t1w_aladin_tmp.nii.gz"

    run_cmd([
        "reg_aladin",
        "-ref", str(mni_template),
        "-flo", str(t1_path),
        "-aff", str(affine_path),
        "-res", str(tmp_res),
        "-voff",
    ])

    # cleanup
    if tmp_res.exists():
        tmp_res.unlink()


    run_cmd([
        "reg_resample",
        "-ref", str(mni_template),
        "-flo", str(t1_path),
        "-trans", str(affine_path),
        "-res", str(t1_mni_path),
        "-inter", "3",   # cubic for intensity
        "-voff",
    ])

    return t1_mni_path, affine_path


# -------------------------
# Model loading
# -------------------------
def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(str(ckpt_path), map_location=device)
    # allow both plain state_dict and {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="NipoSimCortex segmentation inference (BIDS raw T1w -> MNI -> UNet -> derivatives)."
    )

    # Nipoppy-style core args
    ap.add_argument("--bids_dir", required=True, type=str, help="Root of BIDS dataset (raw input).")
    ap.add_argument("--subject_id", required=True, type=str, help="BIDS subject id (sub-XXXX).")
    ap.add_argument("--session_id", default=None, type=str, help="Optional BIDS session id (ses-YYYY).")
    ap.add_argument("--output_dir", required=True, type=str, help="Output root (typically same as bids_dir).")

    # Allow explicit input override (useful when multiple T1w exist)
    ap.add_argument("--t1_path", default=None, type=str, help="Optional explicit raw T1w path override.")

    # Preproc requirement
    ap.add_argument("--force_reg", action="store_true", help="Force re-run registration even if cached outputs exist.")

    default_ckpt = REPO_ROOT / "assets" / "seg_best_dice.pt"
    default_mni  = REPO_ROOT / "assets" / "MNI152_T1_1mm.nii.gz"

    ap.add_argument("--mni_template", default=str(default_mni), type=str,
                    help="Path to MNI template NIfTI (default: assets/MNI152_T1_1mm.nii.gz).")
    ap.add_argument("--ckpt_path", default=str(default_ckpt), type=str,
                    help="Path to model checkpoint (default: assets/seg_best_dice.pt).")


    # Model args
    ap.add_argument("--in_channels", default=1, type=int)
    ap.add_argument("--out_channels", default=9, type=int)

    # Inference options
    ap.add_argument("--pad_mult", default=16, type=int, help="Pad D/H/W to multiples of this before UNet.")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "minmax", "none"], help="Intensity normalization.")

    # GPU control (Agitation-like)
    ap.add_argument("-g", "--gpu", action="store_true", help="Enable GPU inference if available.")
    ap.add_argument("--cuda", default=0, type=int, help="CUDA device index (used if --gpu).")

    args = ap.parse_args()

    bids_dir = Path(args.bids_dir)
    out_root = Path(args.output_dir)
    subject_id = args.subject_id
    session_id = args.session_id
    ses_part = session_id # if session_id else "ses-NA"

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mni_template = Path(args.mni_template)
    if not mni_template.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_template}")

    # ---- Resolve raw input T1w ----
    if args.t1_path:
        t1_path = Path(args.t1_path)
    else:
        t1_path = find_raw_t1w_bids(bids_dir, subject_id, session_id)

    if not t1_path.exists():
        raise FileNotFoundError(f"Raw T1w not found: {t1_path}")
    t0 = time.time()

    # ---- Preprocess: raw -> MNI ----
    preproc_dir = out_root / "derivatives" / "niposimcortex_preproc" / subject_id
    if ses_part:
        preproc_dir = preproc_dir / ses_part
    preproc_dir = preproc_dir / "anat"
    
    t1_mni_path, affine_path = register_t1_to_mni(
        t1_path=t1_path,
        mni_template=mni_template,
        out_dir=preproc_dir,
        force=args.force_reg,
    )

    # ---- Device ----
    if args.gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.cuda)}")
    else:
        device = torch.device("cpu")

    # ---- Load MNI image ----
    img = nib.load(str(t1_mni_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={data.shape} from {t1_mni_path}")

    data = normalize(data, args.norm)
    data_p, unpad_sl = pad_to_multiple(data, args.pad_mult)

    x = torch.from_numpy(data_p[None, None]).to(device)  # [1,1,D,H,W]

    # ---- Model ----
    model = Unet(c_in=args.in_channels, c_out=args.out_channels).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    with torch.inference_mode():
        logits = model(x)                      # [1,C,D,H,W]
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)  # [D,H,W]

    # Unpad
    pred = pred[unpad_sl]

    # ---- Save segmentation ----
    seg_dir = out_root / "derivatives" / "niposimcortex_seg" / subject_id
    if ses_part:
        seg_dir = seg_dir / ses_part
    seg_dir = seg_dir / "anat"

    ensure_dir(seg_dir)
    if ses_part:
        seg_name = f"{subject_id}_{ses_part}_desc-segmentation_mni.nii.gz"
    else:
        seg_name = f"{subject_id}_desc-segmentation_mni.nii.gz"

    seg_path = seg_dir / seg_name


    out_img = nib.Nifti1Image(pred, img.affine)
    nib.save(out_img, str(seg_path))

    # ---- Print summary ----
    print("[OK] Subject:", subject_id, "Session:", ses_part)
    print("[OK] Raw T1w:", t1_path)
    print("[OK] MNI T1w:", t1_mni_path)
    print("[OK] Affine :", affine_path)
    print("[OK] Segm  :", seg_path)
    print("[OK] Device:", device)

    dt = time.time() - t0
    print(f"[TIME] Total inference time: {dt:.2f} s ({dt/60:.2f} min)")



if __name__ == "__main__":
    main()

import os
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import hydra
from omegaconf import OmegaConf

from monai.metrics import compute_surface_dice

from util.dataloader_seg import EvalSegDataset


def setup_logger(log_dir: str, filename: str = "eval_seg.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def dice_np(gt, pred, num_classes: int, exclude_background: bool = True, eps: float = 1e-6) -> float:
    dices = []
    start_cls = 1 if exclude_background else 0
    for c in range(start_cls, num_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        inter = np.logical_and(gt_c, pred_c).sum()
        union = gt_c.sum() + pred_c.sum()
        if union == 0:
            continue
        dices.append((2.0 * inter + eps) / (union + eps))
    if len(dices) == 0:
        return 0.0
    return float(np.mean(dices))


def accuracy_np(gt, pred) -> float:
    return float((gt == pred).sum() / gt.size)


def nsd_monai(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    tolerance_vox: float = 1.0,
    include_background: bool = False,
    spacing=(1.0, 1.0, 1.0),
) -> float:

    gt_t = torch.from_numpy(gt).long().unsqueeze(0)    
    pred_t = torch.from_numpy(pred).long().unsqueeze(0) 

   
    gt_1h = F.one_hot(gt_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    pred_1h = F.one_hot(pred_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    if include_background:
        class_thresholds = [float(tolerance_vox)] * num_classes    
    else:
        class_thresholds = [float(tolerance_vox)] * (num_classes - 1) 

    nsd_per_class = compute_surface_dice(
        y_pred=pred_1h,
        y=gt_1h,
        class_thresholds=class_thresholds,
        include_background=include_background,
        distance_metric="euclidean",
        spacing=spacing,
        use_subvoxels=False,
    ) 

    nsd_per_class = nsd_per_class[0]  

    vals = nsd_per_class[~torch.isnan(nsd_per_class)]
    if vals.numel() == 0:
        return 0.0

    return float(vals.mean().item())


@hydra.main(version_base="1.1", config_path="configs", config_name="eval_seg")
def main(cfg):
    setup_logger(cfg.outputs.log_dir, "eval_seg.log")
    logging.info("=== Eval config ===")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    ds = EvalSegDataset(
        data_root=cfg.dataset.path,
        split_csv=cfg.dataset.split_file,
        pred_dir=cfg.outputs.pred_save_dir,
        split_name=cfg.dataset.split_name,
    )
    logging.info(f"Evaluating {len(ds)} subjects on split={cfg.dataset.split_name}")

    num_classes   = cfg.evaluation.num_classes
    exclude_bg    = cfg.evaluation.exclude_background
    nsd_tol       = getattr(cfg.evaluation, "nsd_tolerance_vox", 1.0)

    # MNI 1 mm
    spacing = (1.0, 1.0, 1.0)

    records = []

    for i in range(len(ds)):
        gt9, pred_arr, sub = ds[i]  

        d   = dice_np(gt9, pred_arr, num_classes=num_classes, exclude_background=exclude_bg)
        acc = accuracy_np(gt9, pred_arr)
        nsd = nsd_monai(
            gt9,
            pred_arr,
            num_classes=num_classes,
            tolerance_vox=nsd_tol,
            include_background=False,
            spacing=spacing,
        )

        records.append({"subject": sub, "dice": d, "accuracy": acc, "nsd": nsd})
        logging.info(f"{sub}: Dice={d:.4f}, Acc={acc:.4f}, NSD={nsd:.4f}")

    if not records:
        logging.warning("No subjects evaluated (no records).")
        return

    df = pd.DataFrame(records)
    mean_dice = df["dice"].mean()
    mean_acc  = df["accuracy"].mean()
    mean_nsd  = df["nsd"].mean()

    std_dice = df["dice"].std()
    std_acc  = df["accuracy"].std()
    std_nsd  = df["nsd"].std()

    logging.info(
        f"MEAN Dice={mean_dice:.4f} (SD={std_dice:.4f}), "
        f"MEAN Acc={mean_acc:.4f} (SD={std_acc:.4f}), "
        f"MEAN NSD={mean_nsd:.4f} (SD={std_nsd:.4f})"
    )


    logging.info(
            f"MEAN Dice={mean_dice:.4f} (SD={std_dice:.4f}), "
            f"MEAN Acc={mean_acc:.4f} (SD={std_acc:.4f}), "
            f"MEAN NSD={mean_nsd:.4f} (SD={std_nsd:.4f})"
        )

    os.makedirs(os.path.dirname(cfg.outputs.eval_csv), exist_ok=True)
    df.to_csv(cfg.outputs.eval_csv, index=False)
    logging.info(f"Saved per-subject metrics to {cfg.outputs.eval_csv}")


if __name__ == "__main__":
    main()

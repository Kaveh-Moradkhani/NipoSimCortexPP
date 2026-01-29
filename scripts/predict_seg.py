import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
CONFIG_PATH = str(REPO_ROOT / "configs")

import logging

import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import numpy as np

from model.UNet import Unet
from utils.dataloader_seg import PredictSegDataset


def setup_logger(log_dir: str, filename: str = "predict_seg.log"):
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


@hydra.main(version_base="1.1", config_path=CONFIG_PATH, config_name="predict_seg")
def main(cfg):
    setup_logger(cfg.outputs.log_dir, "predict_seg.log")
    logging.info("=== Predict config ===")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")

    # Dataset / DataLoader
    ds = PredictSegDataset(
        data_root=cfg.dataset.path,
        split_csv=cfg.dataset.split_file,
        split_name=cfg.dataset.split_name,
        pad_mult=cfg.dataset.pad_mult,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
    )

    os.makedirs(cfg.outputs.pred_save_dir, exist_ok=True)

    # Model
    model = Unet(c_in=cfg.model.in_channels, c_out=cfg.model.out_channels).to(device)
    logging.info(f"Loading checkpoint from {cfg.model.ckpt_path}")
    state = torch.load(cfg.model.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    writer = SummaryWriter(cfg.outputs.log_dir)

    with torch.no_grad():
        pbar = tqdm(dl, desc="Predicting", total=len(dl))
        for step, (vol, sub, affine, orig_shape) in enumerate(pbar):
            vol = vol.to(device, non_blocking=True)  # [B,1,D',H',W']

            if isinstance(orig_shape, torch.Tensor):
                shapes = orig_shape.cpu().numpy()   # [B,3]
            else:
                shapes = np.array(orig_shape)

            logits = model(vol)                      # [B,C,D',H',W']
            pred = logits.argmax(dim=1).cpu().numpy()  # [B,D',H',W']

            for b in range(pred.shape[0]):
                sid = sub[b]
                aff = affine[b].numpy()
                D, H, W = shapes[b].tolist()  

                pred_b = pred[b, :D, :H, :W]

                out_img = nib.Nifti1Image(pred_b.astype("int16"), aff)
                out_path = os.path.join(
                    cfg.outputs.pred_save_dir,
                    f"{sid}_prediction_mni.nii.gz",
                )
                nib.save(out_img, out_path)
                logging.info(f"Saved prediction for {sid} to {out_path}")


            writer.add_scalar("predict/processed_subjects",
                              (step + 1) * cfg.trainer.batch_size, step)

    writer.close()
    logging.info("Prediction finished.")


if __name__ == "__main__":
    main()

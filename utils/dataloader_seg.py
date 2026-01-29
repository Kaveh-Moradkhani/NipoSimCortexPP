import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset
from typing import Dict, Set

from monai.transforms import (
    Compose,
    RandAffined,           
    RandGaussianNoised,    
    RandAdjustContrastd,   
    RandBiasFieldd,        
)

# ---------------- Label mapping ----------------
LABEL_GROUPS: Dict[int, Set[int]] = {
    1: {2, 5, 10, 11, 12, 13, 26, 28, 30, 31},                      # lh white matter
    2: {41, 44, 49, 50, 51, 52, 58, 60, 62, 63},                    # rh white matter
    3: set(range(1000, 1004)) | set(range(1005, 1036)),             # lh cortex (pial)
    4: set(range(2000, 2004)) | set(range(2005, 2036)),             # rh cortex (pial)
    5: {17, 18},                                                    # lh amyg/hip
    6: {53, 54},                                                    # rh amyg/hip
    7: {4},                                                         # lh ventricle
    8: {43},                                                        # rh ventricle
}

def map_labels(seg_arr: np.ndarray, filled_arr: np.ndarray) -> np.ndarray:
    seg_mapped = np.zeros_like(seg_arr, dtype=np.int32)
    for cls, labels in LABEL_GROUPS.items():
        mask = np.isin(seg_arr, list(labels))
        seg_mapped[mask] = cls
    ambiguous = np.isin(seg_arr, [77, 80])
    seg_mapped[ambiguous & (filled_arr == 255)] = 1   
    seg_mapped[ambiguous & (filled_arr == 127)] = 2   
    return seg_mapped

# ---------------- ROBUST Normalization ----------------
def robust_normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    positive = vol[vol > 0]
    if positive.size == 0:
        return vol  

    p99 = np.percentile(positive, 99)
    if p99 <= 0:
        return vol

    vol = np.clip(vol, 0, p99)
    vol = vol / p99
    return vol

# ---------------- Augmentation Pipeline ----------------
def get_augmentations():
    return Compose([

        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"), 
            padding_mode="zeros",
        ),
        
        RandBiasFieldd(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.1, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    ])

# ---------------- Padding Helpers ----------------
def pad_vol_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    if x.ndim == 3: x = x.unsqueeze(0)
    _, D, H, W = x.shape
    pads = (0, (mult - W % mult) % mult, 0, (mult - H % mult) % mult, 0, (mult - D % mult) % mult)
    return F.pad(x, pads, mode="replicate")

def pad_seg_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    if x.ndim == 3: x = x.unsqueeze(0)
    _, D, H, W = x.shape
    pads = (0, (mult - W % mult) % mult, 0, (mult - H % mult) % mult, 0, (mult - D % mult) % mult)
    return F.pad(x, pads, mode="constant", value=0)

# ---------------- TRAIN Dataset ----------------
class SegDataset(Dataset):
    def __init__(self, data_root: str, split_csv: str, split: str = "train", pad_mult: int = 16, augment: bool = False):
        super().__init__()
        self.data_root = data_root
        self.split_csv = split_csv
        self.split = split
        self.pad_mult = pad_mult
        
        df = pd.read_csv(split_csv)
        self.subjects = df[df["split"] == split]["subject"].astype(str).tolist()
        
        if split == "train" and augment:
            self.transforms = get_augmentations()
        else:
            self.transforms = None

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        subj_dir = os.path.join(self.data_root, sub)
        
        img = nib.load(os.path.join(subj_dir, f"{sub}_t1w_mni.nii.gz"))
        seg_arr = nib.load(os.path.join(subj_dir, f"{sub}_aparc_aseg_mni.nii.gz")).get_fdata().astype(np.int32)
        fill_arr = nib.load(os.path.join(subj_dir, f"{sub}_filled_mni.nii.gz")).get_fdata().astype(np.int32)

        vol = img.get_fdata().astype(np.float32)
        
        vol = robust_normalize(vol)
        
        seg9 = map_labels(seg_arr, fill_arr)

        data = {
            "image": vol[None],  
            "label": seg9[None]  
        }
        
        if self.transforms:
            data = self.transforms(data)
        
        vol_out = torch.as_tensor(data["image"])
        seg_out = torch.as_tensor(data["label"]).long()

        vol_out = pad_vol_to_multiple(vol_out, self.pad_mult)
        seg_out = pad_seg_to_multiple(seg_out, self.pad_mult)
        
        return vol_out, seg_out.squeeze(0)


# ---------------- PREDICT Dataset ----------------
class PredictSegDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        split_csv: str,
        split_name: str = "test",
        pad_mult: int = 16,
    ):
        super().__init__()
        self.data_root = data_root
        self.pad_mult = pad_mult
        df = pd.read_csv(split_csv)
        self.subjects = df[df["split"] == split_name]["subject"].astype(str).tolist()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        subj_dir = os.path.join(self.data_root, sub)
        t1_path = os.path.join(subj_dir, f"{sub}_t1w_mni.nii.gz")

        img = nib.load(t1_path)
        vol = img.get_fdata().astype(np.float32)
        affine = img.affine
        orig_shape = np.array(vol.shape[:3], dtype=np.int16) 

        vol = robust_normalize(vol)
        vol_t = torch.from_numpy(vol[None])  # [1,D,H,W]
        vol_t = pad_vol_to_multiple(vol_t, mult=self.pad_mult)

        return vol_t, sub, affine, orig_shape


# ---------------- EVAL Dataset ----------------
class EvalSegDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        split_csv: str,
        pred_dir: str,
        split_name: str = "test",
    ):
        super().__init__()
        self.data_root = data_root
        self.pred_dir = pred_dir

        df = pd.read_csv(split_csv)
        self.subjects = df[df["split"] == split_name]["subject"].astype(str).tolist()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        subj_dir = os.path.join(self.data_root, sub)

        gt_path = os.path.join(subj_dir, f"{sub}_aparc_aseg_mni.nii.gz")
        fill_path = os.path.join(subj_dir, f"{sub}_filled_mni.nii.gz")
        pred_path = os.path.join(self.pred_dir, f"{sub}_prediction_mni.nii.gz")

        gt_arr = nib.load(gt_path).get_fdata().astype(np.int32)
        fill_arr = nib.load(fill_path).get_fdata().astype(np.int32)
        pred_arr = nib.load(pred_path).get_fdata().astype(np.int32)

        gt9 = map_labels(gt_arr, fill_arr)

        D, H, W = gt9.shape
        pred_arr = pred_arr[:D, :H, :W]

        return gt9, pred_arr, sub
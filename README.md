# BIDS T1w → MNI → UNet Segmentation Pipeline

This project provides a command-line pipeline for running brain image segmentation on BIDS-formatted T1-weighted MRI data.  
The workflow performs spatial normalization to MNI space, applies a UNet-based segmentation model, and saves outputs following the BIDS **derivatives** convention.

---

## 📌 Overview

**Pipeline steps:**

1. Load T1w images from a BIDS dataset
2. Register T1w images to MNI space using **NiftyReg**
3. Run UNet-based segmentation
4. Save results under `derivatives/`

**Workflow summary:**

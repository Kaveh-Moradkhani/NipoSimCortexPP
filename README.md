
# NipoSimCortex

**BIDS T1w → MNI Registration → U-Net Segmentation Pipeline**

This project provides a command-line pipeline for running brain image segmentation on **BIDS-formatted T1-weighted MRI** data. The workflow normalizes T1w images to **MNI space**, runs a **U-Net segmentation model**, and writes outputs following the BIDS **derivatives** convention.

---

## 📌 Overview

**Pipeline steps:**

1. Load T1w images from a BIDS dataset.
2. Register T1w images to MNI space using **NiftyReg** (`reg_aladin`, `reg_resample`).
3. Run **U-Net** based segmentation inference.
4. Save results under `derivatives/` following BIDS standards.

**Workflow Highlights:**

* ✅ **Nipoppy Ready:** Fully packaged for Nipoppy as a processing pipeline (`v0.5`).
* ✅ **Containerized:** Support for both **Docker** and **Apptainer**.

---

## 📦 Containers

### Docker

**Image:** `kavehmo/niposimcortex:0.5`

```bash
docker pull kavehmo/niposimcortex:0.5

```

### Apptainer 

To build or pull the image from DockerHub:

```bash
apptainer pull niposimcortex_0.5.sif docker://kavehmo/niposimcortex:0.5

```

---

## 🚀 Quickstart (BIDS-App CLI)

The entrypoint script is located at `/opt/niposimcortex/run_bidsapp.py`.

### Display Help

```bash
docker run --rm kavehmo/niposimcortex:0.5 python /opt/niposimcortex/run_bidsapp.py --help

```

### Run Single Subject (Apptainer Example)

```bash
apptainer exec --bind /path/to/BIDS:/bids,/path/to/output:/out \
  niposimcortex_0.5.sif \
  python /opt/niposimcortex/run_bidsapp.py /bids /out participant \
    --participant_label sub-ED01 --session_label BL

```

---

## 🧩 Integration with Nipoppy

### 1. Install Pipeline Configs

Unzip the `niposimcortex-0.5_nipoppy_bundle.zip` and copy the contents to your Nipoppy dataset root:

```text
NipoDataset/
└── pipelines/
    └── processing/
        └── niposimcortex-0.5/
            ├── config.json
            ├── descriptor.json
            ├── invocation.json
            └── tracker.json

```

### 2. Run via Nipoppy CLI

```bash
nipoppy process \
  --dataset /path/to/NipoppyDataset \
  --pipeline niposimcortex \
  --pipeline-version 0.5 \
  --participant-id ED01 \
  --session-id BL \
  --verbose

```

*Tip: Use `--dry-run` first to preview the generated command.*

---

## 🧾 Outputs

The pipeline generates BIDS-compliant derivatives:

* `niposimcortex_preproc/sub-<ID>/anat/t1w_mni.nii.gz`
* `niposimcortex_seg/sub-<ID>/anat/sub-<ID>_desc-segmentation_mni.nii.gz`
* `dataset_description.json` (Derivatives metadata)

---

## 📌 Versioning & Links

* **Pipeline Name:** niposimcortex
* **Current Version:** 0.5
* **Docker Tag:** `kavehmo/niposimcortex:0.5`

---

## 📄 License

*Add your license information here.*

---

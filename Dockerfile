FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/niposimcortex

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# NiftyReg provides reg_aladin / reg_resample
# The pytorch/pytorch image may pin conda=23.5.2; unpin first or installs can fail.
RUN rm -f /opt/conda/conda-meta/pinned || true \
 && conda config --set channel_priority flexible \
 && conda config --add channels defaults \
 && conda config --add channels conda-forge \
 && conda install -y -c defaults conda \
 && conda install -y -c conda-forge niftyreg \
 && conda clean -afy

# Copy requirements first for better layer caching
COPY requirements_infer.txt /opt/niposimcortex/requirements_infer.txt
RUN pip install --no-cache-dir -r requirements_infer.txt
COPY bids_dataset_description.json /bids_dataset_description.json

# Copy the rest of the repo
COPY . /opt/niposimcortex
RUN chmod -R a+rX /opt/niposimcortex
ENTRYPOINT ["python", "/opt/niposimcortex/run_bidsapp.py"]


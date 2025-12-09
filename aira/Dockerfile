FROM nvcr.io/nvidia/base/ubuntu:jammy-20250415.1

# Install uv package manager, Docker client, build tools, and dependencies for GROMACS + MM-GBSA
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    pip \
    docker.io \
    wget \
    build-essential \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip uv

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Miniconda for gmx_MMPBSA
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda init bash

# Accept conda TOS and create gmx_mmpbsa environment
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true \
    && /opt/conda/bin/conda create -y -n mmpbsa python=3.11 \
    && /opt/conda/bin/conda install -y -n mmpbsa -c conda-forge gmx_mmpbsa ambertools gromacs \
    && /opt/conda/bin/conda clean -afy

# Add conda and mmpbsa environment to PATH
ENV PATH="/opt/conda/envs/mmpbsa/bin:/opt/conda/bin:$PATH"
ENV AMBERHOME="/opt/conda/envs/mmpbsa"

RUN uv python install 3.12

# Copy the local build context (your project files) into the container's /app directory
COPY ./aira/entrypoint.sh /entrypoint.sh
COPY ./aira/. /app

RUN uv venv --python-preference managed
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIQ_AIRA="0.0.0"

# Use uv to install Python dependencies 
RUN uv pip install /app 

# Install the GROMACS MD refinement pipeline from GitHub
RUN uv pip install git+https://github.com/briandasantini/gromacs-md-refinement-pipeline.git

RUN chmod +x /entrypoint.sh

# syntax=docker/dockerfile:1
#
# SIAM prediction image, one Dockerfile for both GPU and CPU.
#
#   GPU (default, CUDA 12.6 wheels: Pascal .. Hopper)
#     docker build -t siam:gpu .
#   GPU for recent cards (Blackwell, RTX 50xx) needs CUDA >= 12.8 wheels
#     docker build --build-arg TORCH_VARIANT=cu128 -t siam:gpu-cu128 .
#   CPU only (much smaller image)
#     docker build --build-arg TORCH_VARIANT=cpu -t siam:cpu .
#
# Model weights are NOT in the image: they are downloaded at the first run
# into /models. Mount a host folder there so they are kept between runs:
#
#   docker run --rm --gpus all -u $(id -u):$(id -g) \
#       -v ~/siam_params:/models -v $PWD:/data \
#       siam:gpu -i /data/my_image.nii.gz
#
#   docker run --rm -u $(id -u):$(id -g) \
#       -v ~/siam_params:/models -v $PWD:/data \
#       siam:cpu -i /data/my_image.nii.gz -device cpu

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-slim AS builder

# cpu | cu126 | cu128 ... (any tag of https://download.pytorch.org/whl/)
ARG TORCH_VARIANT=cu126

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch first, from the PyTorch index, so that nnunetv2 does not pull the
# default (CUDA) wheel from PyPI. torchvision is not needed.
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# the rest of the dependencies (nnunetv2, torchio, ...) then the package itself
COPY pyproject.toml README.md LICENSE labels.json /src/
COPY SIAMpred /src/SIAMpred
RUN pip install /src \
 # trim what is not needed at run time
 && find /opt/venv -depth \( -name __pycache__ -o -name tests -o -name test \) \
        -type d -path '*/site-packages/*' -exec rm -rf {} + \
 && find /opt/venv -name '*.pyc' -delete

# ---------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.title="siam-pred" \
      org.opencontainers.image.source="https://github.com/romainVala/SIAM" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIAM_MODEL_DIR=/models \
    # writable caches when running as an arbitrary user (-u uid:gid)
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.matplotlib \
    # silence nnU-Net warnings about unset paths (not used for inference)
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/models

# world writable so the weights can be downloaded whatever the user id
RUN mkdir -p /models /data && chmod 1777 /models /data
VOLUME ["/models"]
WORKDIR /data

ENTRYPOINT ["siam-pred"]
CMD ["--help"]

# copy from totalsegmentor (modify november 2025)
# Use python:3.11-slim for a much smaller base image
#FROM python:3.11-slim
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

#a tester FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04
# ou ne pas faire  l'install torch

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    SIAM_MODEL_DIR=/model_weights

WORKDIR /app

# Install system dependencies, PyTorch, and application packages in one layer
# This minimizes intermediate layers and reduces final image size
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        xvfb \
        ca-certificates \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy<2.0.0" \
#    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch \
    && pip install --no-cache-dir fury \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && conda clean -a -y

RUN echo "force rebuild from here (2)"

# Copy application and install
COPY . /app
RUN pip install --no-cache-dir /app \
    && find /opt/conda/lib/python3.11 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/conda/lib/python3.11 -type f -name "*.pyc" -delete \
    && find /opt/conda/lib/python3.11 -type f -name "*.pyo" -delete

RUN echo "force rebuild from here (3)"

#RUN mkdir -p /model_weights \
#    && python /app/SIAMpred/download_model_weights.py
RUN mkdir -p /model_weights
#plus simple si les poid sont deja dasn apps avec le copy precedent
RUN ln -s /app/v0.3  /model_weights/

ENTRYPOINT ["siam-pred"]
CMD ["--help"]
# CMD ["siam-pred"]
# expose not needed if using -p
# If using only expose and not -p then will not work
# EXPOSE 80

[← back to the README](../README.md)

# Installation

## Local installation

Note that you need to have a python3 installation for SIAM to work. 
pip version > 22 and setuptool > 61

1. Clone this repository:
   ```bash
   git clone https://github.com/romainVala/SIAM
   ```
2. Go into the repository (the folder with the pyproject.toml file) and install:
   optionally create a conda env before the following commande 
   ```
   pip install -e .
   ```
   when testing on windows, I had to remove the -e flag ... (no idea why)

3. Model parameters will be downloaded to ~/siam_params/v0.x the first time you
   run an inference. If the installation is for multiple user, setup the environement variable `SIAM_MODEL_DIR` 
   and run `python /instal_dir/SIAMpred/download_model_weights.py`. Then you only need to setup ` export SIAM_MODEL_DIR` before running the main command `siam-pred`

## Docker or Singularity

### Build the image
The same `Dockerfile` builds a GPU or a CPU image (the CPU one is much smaller):
   ```bash
   docker build -t siam:gpu .                                  # GPU, CUDA 12.6 (Pascal to Hopper cards), ~7.5 GB
   docker build --build-arg TORCH_VARIANT=cu128 -t siam:gpu .  # GPU, for recent cards (Blackwell, RTX 50xx)
   docker build --build-arg TORCH_VARIANT=cpu -t siam:cpu .    # CPU only, ~1.6 GB
   ```

### Run
The model weights are not in the image: they are downloaded at the first run into `/models`
(~5 GB for the default model, up to ~20 GB for `-m 4`, less with `-r`, `-n` or `--lowres`). Mount a host folder there so they are kept between runs:
   ```bash
   # GPU (needs the NVIDIA container toolkit)
   docker run --rm --gpus all --shm-size=8g -u $(id -u):$(id -g) \
       -v ~/siam_params:/models -v $PWD:/data siam:gpu -i /data/my_image.nii.gz
   # CPU
   docker run --rm --shm-size=8g -u $(id -u):$(id -g) \
       -v ~/siam_params:/models -v $PWD:/data siam:cpu -i /data/my_image.nii.gz -device cpu
   ```
   `-u $(id -u):$(id -g)` makes the output files belong to you instead of root.

### Singularity / Apptainer
   ```bash
   singularity build siam.sif docker-daemon://siam:gpu
   singularity exec --nv -B ~/siam_params:/models -B $PWD:/data siam.sif siam-pred -i /data/my_image.nii.gz
   ```
   With limited disk space on $HOME or /tmp, set `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` to a location with enough space.

## Apple Silicon (Metal / MPS)

To be tested:
```bash
siam-pred -i INPUT_FILENAME -device mps -nbworker 1
```

Alternatively, you can try [MedMLX](https://medmlx.org/models/siam/) who reimplement the model's inference inside their tool.

# SIAM
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

SIAM : **Segment It All Model** a head tissue segmentation model designed to be robust to contrast, resolution, and **pathology**.
It can process any 3D human head volume (T1, T2, FLAIR, etc., and even CT).

The current version performs **tissue segmentation**, at 0.75 mm

- **Model 1** (-m 1) is performing a 39 regions segmentation task, as illustrated in the figure. It is an old version trained on 3 subjects, but may be interesting for extra-cerebral [labels](labels.json) 

- **Model 2** (-m 2) is described [here](https://arxiv.org/abs/2605.02737), trained from 6 high-quality templates it shows great performance in healthy brain for tissue segmentation (12 brain tissue + head\/skull\/dura matter\/vessel)

- **Model 3** (defautl or -m 0), extends siam's v2 robustness toward anatomical **anomalies**, which are segmented as an extra [Label](labels.json).  


> Valabregue, R., Khemir, I., Bardinet, E., Rousseau, F., Auzias, G. & Dorent R. (2026).
> _SIAM : Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training._
> ArXiv. [https://arxiv.org/abs/2605.02737)

![SIAM](https://github.com/user-attachments/assets/ef94239e-60fe-463c-94f3-88b85fced7d4)


## Local Installation 

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



## How to use it


```bash
siam-pred -i INPUT_FILENAME 
```

INPUT_FILENAME must be a nifti file containing 3D volume data. 4D image sequences are not supported 


```bash
siam-pred -i INPUT_FOLDER -o OUTPUT_PREFIX -m 3
```

The above command will look for all nifti files in the INPUT_FOLDER
and save the predictions in a sub-folder containing the OUTPUT_PREFIX name.
if `-o` is not specify, result are store in the same folder, with a prefix
-m model number (default 3, int within [1 2 3] )

### More options:

**NO Interpolation**: Use `-voxel_size 0.75`  for inputs with isotropic resolution in the range [0.6 1] mm. Adding this option
will avoid the standard process which consist of : 1) reslice input data to 0.75 mm (as it was the choosen training resolution)
2) interpolate back the predicted labels to the original input resolution. (Since we have to use nearest neighbor this will 
induced a loss of smalll structures)
Removing interpolation is important for small vessel, but also for GM ... it worth the try ...

For very small baby brain, you need to scale the volume up, for the model to work. 
You can achieve it, without resampling,  by changing nifti header voxel size.
This will be done on the fly when using the `-voxelsize x.x` where `x.x` is a float for the
new (fake) voxel size. Typically for newborn brain we multiply the voxel size by 1.4, in order to get a 
Total Intracranial Volume similar to an adult brain. The prediction result is then converted back to the
original resolution
Only use if you have near isotropic resolution

Use `-nbthread 1` if you run with memory issues. This will reduce the number of thread for processing your input (default is 4) 
 
To summarize all inputs parameters, refer to the help functionality:

```bash
siam-pred -h
```

## Docker or Singularity
   ```bash
   docker pull romainvalabregue/siam 
   ```
   
   Or with Singularity
   ```bash
  singularity build siam_v0.3.simg docker://romainvalabregue/siam
   ```
   Example to run with gpu ()
   ```bash
  singularity run --nv -B `pwd`:/data  siam_v0.3.simg siam-pred  -i /data/my_image.nii.gz 
   ```
   For linux user, with limited disk space on $HOME or on /tmp, you can set following environement variable to location with enough space
   ```bash
   export SINGULARITY_TMPDIR=/data/singularity/tpm
   export SINGULARITY_CACHEDIR=/data/singularity/cache
   ```
   (may be also DOCKER_CONFIG=  is you do not want to use the .docker in your home)

   image will be ~10G and cache dir ~ 10G (can be removed after) tmp dir ~32G (but automaticaly deleted )


## Apple Silicon (Metal / MPS)

On Apple Silicon (M1–M4) the model can run on the integrated GPU via PyTorch's
Metal backend. Either request it explicitly:

```bash
siam-pred -i INPUT_FILENAME -device mps -nbthread 1
```

or just leave `-device cuda` as-is: when CUDA isn't available, SIAM now falls
back to MPS automatically (and only to CPU if MPS is also unavailable). A few
3D ops aren't implemented on Metal yet; `PYTORCH_ENABLE_MPS_FALLBACK=1` is
set automatically so those transparently run on CPU.

Note that nnU-Net's sliding-window inference moves tiles between device and
host on non-CUDA backends, so MPS is faster than CPU but not as fast as CUDA.
On a 48 GB unified-memory machine, use `-nbthread 1` or `2` to avoid OOM
kills from parallel worker processes.

## Memory issues

Running with GPU requires more than 12 G on the GPU card

Execution time, and Memory usage will depend on the input image Field Of View (FOV). 
The input resolution do not matter too much since it is resliced to 0.75 mm resolution. 
So only the FOV will change the total datasize that will be feed into the network. 

With a large FOV, covering the nec : 166x240x256 mm^3, 

running with  `-device cpu -nbthread 8` took ~ 25 mn (seems to take ~ 20G of RAM)

running with  `-device cpu -nbthread 1` took ~ 2h20  (but sill ~17 G of RAM ... not sure why)


## Thanks
We thank the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) team for providing the training framework,
and we built this inference tool based on [HD-BET](https://github.com/MIC-DKFZ/HD-BET).

We thank B. Billot and E. Iglesias for their original [SynthSeg](https://github.com/BBillot/SynthSeg) method:
training on synthetic data enables robust, contrast-agnostic segmentation models.
It worth the try, and we are still working to improve it, adding more tissue, being robust to anomalies

Thanks to Fernando and its [torchio](https://github.com/TorchIO-project/torchio) lib ! my favorite 3D augmentation tools which we used to build the generative model.


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/benoit-beranger/"><img src="https://avatars.githubusercontent.com/u/16976839?v=4?s=100" width="100px;" alt="Benoît Béranger"/><br /><sub><b>Benoît Béranger</b></sub></a><br /><a href="https://github.com/romainVala/SIAM/commits?author=benoitberanger" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://scholar.google.com/citations?user=00jLGq8AAAAJ&hl=en"><img src="https://avatars.githubusercontent.com/u/8930807?v=4?s=100" width="100px;" alt="Chris Rorden"/><br /><sub><b>Chris Rorden</b></sub></a><br /><a href="https://github.com/romainVala/SIAM/commits?author=neurolabusc" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
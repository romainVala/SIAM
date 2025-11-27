# SIAM 

This repository provides easy-to-use access to our SIAM model for inference.
SIAM is an attempt to segment all tissue (of the human head) from any 3D volume.
Publication in preparation, meanwhile the method is similar to

    Valabregue, R., Girka, F., Pron, A., Rousseau, F., & Auzias, G. (2024).  
    "Comprehensive analysis of synthetic learning applied to neonatal brain MRI segmentation".
     Human Brain Mapping https://doi.org/10.1002/hbm.26674

We thank  [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) repository which provides the code for training, and we built this inference tool
starting from [HD-BET](https://github.com/MIC-DKFZ/HD-BET). 
We thanks  B. Billot and E. Iglesias for their initial proposition of [SynthSeg](https://github.com/BBillot/SynthSeg):
Training on synthetic data opens avenues for robust, contrast-agnostic segmentation models.
We built this tool upon their idea, but re-implement the synthetic generative model with
[torchio](https://github.com/TorchIO-project/torchio) augmentations (Thanks !)

Our contribution is two fold:
1) improving the label space: adding more labels (skull/vessel/dura matter ...) towards a denser labeling of the head.
2) adding specific label augmentation, to allow less bias predictions while training with a few subject.

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
3. Per default, model parameters will be downloaded to ~/siam_params/v0.1. If you
   wish to use a different folder, set the environement variable `SIAM_MODEL_DIR` to the path you want



## How to use it

Using siam-pred is straightforward. You can use it in any terminal on your linux
system. The siam-pred command was installed automatically. We provide GPU as well
as MPS and CPU support. Running on GPU is a lot faster but it requires around 15G GPU memory :

```bash
siam-pred -i INPUT_FILENAME 
```

INPUT_FILENAME must be a nifti file containing 3D volume data. 4D
image sequences are not supported (however can be split upfront into the
individual temporal volumes using fslsplit<sup>1</sup>). INPUT_FILENAME can be
any MRI sequence. 


For batch processing it is faster to process an entire folder at once as this
will mitigate the overhead of loading and initializing the model for each case (at least if you have GPU)

```bash
siam-pred -i INPUT_FOLDER -o OUTPUT_PREFIX
```

The above command will look for all nifti files in the INPUT_FOLDER
and save the predictions in a sub-folder containing the OUTPUT_PREFIX name.
if `-o` is not specify, result are store in the same folder, with a prefix


### More options:
For very small baby brain, you need to scale the volume up, for the model to work. 
You can achieve it, without resampling,  by changing nifti header voxel size.
This will be done on the fly when using the `-voxelsize x.x` where `x.x` is a float for the
new (fake) voxel size. Typically for newborn brain we multiply the voxel size by 1.4, in order to get a 
Total Intracranial Volume similar to an adult brain. The prediction result is then converted back to the
original resolution
Only use if you have near isotropic resolution

To summarize all inputs parameters, refer to the help functionality:

```bash
siam-pred -h
```

## Docker or Singularity
   ```bash
   docker pull nipreps/fmriprep 
   ```
   
   Or with Singularity
   ```bash
  singularity build siam_v0.3.simg romainvalabregue/siam
   ```
   Example to run with gpu ()
   ```bash
  singularity run --nv -B `pwd`:/data   -i /data/my_image.nii.gz 
   ```

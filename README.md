# SIAM

SIAM : **Segment It All Model** a head tissue segmentation tool designed to be robust to contrast, resolution, and **pathology**.
It can process any 3D human head volume (T1, T2, FLAIR, etc., and even CT).

The current version performs **tissue segmentation**, including:
17 tissues : WM anomalies, skull, vessels, dura mater, head,
and brain tissues: WM, GM, CSF, cerebellum, ventricles, 5 deep nuclei, hippocampus, and amygdala.

Stay tuned, we plan to add the region / sub-region segmentation task soon.

A publication is currently in preparation. In the meantime, the method is similar to:

> Valabregue, R., Girka, F., Pron, A., Rousseau, F., & Auzias, G. (2024).
> *Comprehensive analysis of synthetic learning applied to neonatal brain MRI segmentation*.
> Human Brain Mapping. [https://doi.org/10.1002/hbm.26674](https://doi.org/10.1002/hbm.26674)

We thank the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) team for providing the training framework,
and we built this inference tool based on [HD-BET](https://github.com/MIC-DKFZ/HD-BET).

We also thank B. Billot and E. Iglesias for their original [SynthSeg](https://github.com/BBillot/SynthSeg) method:
training on synthetic data enables robust, contrast-agnostic segmentation models.
Our tool is inspired by their approach, but we reimplemented the synthetic data generator using
[torchio](https://github.com/TorchIO-project/torchio) augmentations.


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
  singularity run --nv -B `pwd`:/data   -i /data/my_image.nii.gz 
   ```
## Memory issues
the docker image is quite big, 15 G (but it include the model weight)
when using a local version, the model weights (~5G) will be download at first usage, 

Running with GPU requires more than 12 G on the GPU card

Execution time, and Memory usage will depend on the input image Field Of View (FOV). 
The input resolution do not matter too much since it is resliced to 0.75 mm resolution. 
So only the FOV will change the total datasize that will be feed into the network. 

With a large FOV, covering the nec : 166x240x256 mm^3, 
running with  `-device cpu -nbthread 8` took ~ 25 mn (seems to take ~ 20G of RAM)
running with  `-device cpu -nbthread 1` took ~ 2h30  (seems to take ~ 20G of RAM)


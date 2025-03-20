# SIAM 

This repository provides easy-to-use access to our SIAM model for inference.
SIAM is an attempt to segment all tissue (of the human head) from any 3D volume.
While waiting for a publication related to this work, the method is similar to

    Valabregue, R., Girka, F., Pron, A., Rousseau, F., & Auzias, G. (2024).  
    "Comprehensive analysis of synthetic learning applied to neonatal brain MRI segmentation".
     Human Brain Mapping https://doi.org/10.1002/hbm.26674

We are greatful to [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) repository which provides the code for training, and we built this inference tool
starting from [HD-BET](https://github.com/MIC-DKFZ/HD-BET). 
We would also like to thank B. Billot and E. Iglesias for their initial proposition of [SynthSeg](https://github.com/BBillot/SynthSeg):
Training on synthetic data opens avenues for robust, contrast-agnostic segmentation models.
We built this tool upon their work, our contribution is limited to the label space: adding more labels towards a denser
labeling of the head

![SIAM](https://github.com/user-attachments/assets/ef94239e-60fe-463c-94f3-88b85fced7d4)


## Installation Instructions

Note that you need to have a python3 installation for SIAM to work. 
We recommend installing in a virtual environment.

**Install either as a python package (TODO):**
```bash
#not yet available    pip install siam-predic
```

**(Recomended) install the most recent master from GitHub**
1. Clone this repository:
   ```bash
   git clone https://github.com/romainVala/SIAM
   ```
2. Go into the repository (the folder with the pyproject.toml file) and install:
   ```
   pip install -e .
   ```
3. Per default, model parameters will be downloaded to ~/siam_params/v0.1. If you
   wish to use a different folder, open HD_BET/paths.py in a text editor and
   modify `folder_with_parameter_files`

## How to use it

Using siam-pred is straightforward. You can use it in any terminal on your linux
system. The siam-pred command was installed automatically. We provide GPU as well
as MPS and CPU support. Running on GPU/MPS is a lot faster and should always be
preferred. :

```bash
siam-pred -i INPUT_FILENAME 
```

INPUT_FILENAME must be a nifti (.nii.gz) file containing 3D MRI image data. 4D
image sequences are not supported (however can be split upfront into the
individual temporal volumes using fslsplit<sup>1</sup>). INPUT_FILENAME can be
any MRI sequence. 
Input images must match the orientation of standard MNI152
template! Use fslreorient2std <sup>2</sup> upfront to ensure that this is the
case.

By default, siam-pred will run in GPU mode and use test time data
augmentation by mirroring along all axes.

For batch processing it is faster to process an entire folder at once as this
will mitigate the overhead of loading and initializing the model for each case:

```bash
siam-pred -i INPUT_FOLDER -o OUTPUT_PREFIX
```

The above command will look for all nifti files (\*.nii.gz) in the INPUT_FOLDER
and save the predictions in a sub-folder containing the OUTPUT_PREFIX name.
(the -o is optional)

### GPU is nice, but I don't have one of those... What now?

siam-pred has CPU support. Running on CPU takes a lot longer though and you will
need quite a bit of RAM. To run on CPU, we recommend you use the following
command:

```bash
siam-pred -i INPUT_FILE_OR_FOLDER -device cpu --disable_tta
```

The option --disable_tta will disable test time data augmentation
(speedup of 8x).

HD-BET should also run on mps, just specify `-device mps`

### More options:

For more information, please refer to the help functionality:

```bash
siam-pred -h
```

## FAQ
1. **Will you provide the training code?** It's basically 
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) the training data is what matter, we hope to publish them soon
2. **What run time can I expect on CPU/GPU?** This depends on your MRI image
   size. Typical run times (preprocessing and resampling
   included) are just a couple of seconds for GPU and about 2 Minutes on CPU
   (using `--disable_tta`)

<sup>1</sup>https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutils

<sup>2</sup>https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained

[← back to the README](../README.md)

# How to use it

```bash
siam-pred -i INPUT_FILENAME 
```

INPUT_FILENAME must be a nifti file containing 3D volume data. 4D image sequences are not supported 


```bash
siam-pred -i INPUT_FOLDER -o OUTPUT_PREFIX -m 4
```

The above command will look for all nifti files in the INPUT_FOLDER
and save the predictions in a sub-folder containing the OUTPUT_PREFIX name.
if `-o` is not specify, result are store in the same folder, with a prefix
if `-o` is an absolute path (ex: `-o /data/siam_out`), it is used as the output folder (created if needed)
and the output files keep the default model prefix
-m model number (default 0 = model 3, possible values 0, 1, 2, 4)

Already existing outputs are not recomputed: a model is skipped when all its outputs exist,
so a command can be rerun safely, for instance to add a region or to only regroup the labels.

## Model 4 (v0.4, tissue + regions) : choose what to download
The v0.4 weights are stored on [hugging face](https://huggingface.co/RomainValabregue/siam-model-v04) as separate
files (one per model and per fold): only the models needed by your command are downloaded (~20G for everything).

```bash
siam-pred -i INPUT -m 4                      # tissue (cascade 0.5mm) + all the region models
siam-pred -i INPUT -m 4 -r CSF,GM_allen,WM   # tissue + only these 3 region models
siam-pred -i INPUT -m 4 -r none              # tissue only
siam-pred -i INPUT -m 4 --lowres             # tissue only, at 1mm (only the low res model), same output names
siam-pred -i INPUT -m 4 --lowres -n 1        # very light version : first fold of the low res model only (ex: brain mask)
```
- `-r` : comma separated list of region names: CSF, GM_AssN (GM sub-regions from AssemblyNet), GM_allen, WM, Cereb, deep_nuclei, Thal, Head, Skull, Lesion
  (an unknown name prints the list of available regions)
- `-n N` : only the first N folds of each model are used (and downloaded)
- the list of the repo files is cached in `siam_model_v04/hf_repo_files.txt` : once the files are downloaded, no
  internet connection is needed. Delete this file to see new models added on hugging face.

- `--regroup_regions` : regroups the tissue and the sub-region predictions into one label map (`All_lab_*`)
  at the end of the prediction. The labels are fixed, whatever the `-r` selection: a tissue without sub-region prediction
  gets the first final label of its region (CSF 1, WM 3, Cereb 5, Head 11, GM 39, Thal 239, deep_nuclei 273, Skull 329, Lesion 353).
  Since existing predictions are not recomputed, you can rerun the same command with `--regroup_regions` to only do the regrouping.

To download in advance (shared install with `SIAM_MODEL_DIR`), the same options work with
`python SIAMpred/download_model_weights.py -m 4 -r CSF,GM_allen,WM -n 1`

## More options

See [Voxel size](voxel_size.md) for the `-voxelsize` option.

To summarize all inputs parameters, refer to the help functionality:

```bash
siam-pred -h
```

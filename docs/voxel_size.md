[← back to the README](../README.md)

# Voxel size (`-voxelsize`)

- `-voxelsize 0` (default) : the voxel size is read from the nifti header.
- `-voxelsize x.x` : the header voxel size is ignored and replaced by x.x mm (isotropic), without reslicing: this is a zoom
  of the head. Useful for small heads: for newborn brains, multiply the voxel size by ~1.4 to get a Total Intracranial Volume
  similar to an adult brain. Only use it with (nearly) isotropic inputs.
- `-voxelsize -1` (model 4 only) : the header voxel size is kept for the low resolution (1 mm) models (no zoom), and set to
  0.5 mm for the high resolution models, to avoid the resampling (and the nearest neighbour interpolation back of the labels,
  which can lose small structures). Best for inputs with an isotropic resolution close to 0.5 mm.

Note: the prediction always reorients the image to canonical (RAS) and resamples it to the model resolution
(0.75 mm for models 2 and 3, 1 mm and 0.5 mm for model 4); the labels are then resampled back to the original grid and orientation.

**Warning**: the models were trained with a scaling factor in the range [0.7 1.3]. Changing the voxel size beyond
(real voxel size / forced voxel size outside this range) will probably start failing.

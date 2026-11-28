# SIAM
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

> **SIAM v4 is there** (`-m 4`): tissue segmentation at 0.5 mm plus sub-region models (GM parcellations, WM, deep nuclei, thalamus, head, skull, lesion ...).
> It is available now and will soon become the default model. See [How to use it](docs/usage.md).

SIAM : **Segment It All Model** a head tissue segmentation model designed to be robust to contrast, resolution, and **pathology**.
It can process any 3D human head volume (T1, T2, FLAIR, etc., and even CT).

The current version performs **tissue segmentation**, at 0.75 mm

- **Model 1** (-m 1) is performing a 39 regions segmentation task, as illustrated in the figure. It is an old version trained on 3 subjects, but may be interesting for extra-cerebral [labels](labels.json) 

- **Model 2** (-m 2) is described [here](https://arxiv.org/abs/2605.02737), trained from 6 high-quality templates it shows great performance in healthy brain for tissue segmentation (12 brain tissue + head\/skull\/dura matter\/vessel)

- **Model 3** (default or -m 0), extends siam's v2 robustness toward anatomical **anomalies**, which are segmented as an extra [Label](labels.json).  

- **Model 4** (-m 4, soon the default) : a cascade tissue model (1 mm then 0.5 mm) followed by sub-region models, see [How to use it](docs/usage.md).

If usefull, please cite (and may be add a star here)
> Valabregue, R., Khemir, I., Bardinet, E., Rousseau, F., Auzias, G. & Dorent R. (2026).
> _SIAM : Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training._
> ArXiv. [https://arxiv.org/abs/2605.02737)

![SIAM](https://github.com/user-attachments/assets/ef94239e-60fe-463c-94f3-88b85fced7d4)

## Documentation

- [Installation](docs/installation.md) : local install with pip, Docker / Singularity, Apple Silicon
- [How to use it](docs/usage.md) : basic commands, model 4 (choosing the regions, `--lowres`, `-n`, `--regroup_regions`), other options
- [Voxel size](docs/voxel_size.md) : the `-voxelsize` option (no interpolation, zoom for baby brains) and its limits
- [Memory needs](docs/memory.md) : GPU / CPU memory and execution time

Quick start:
```bash
pip install -e .            # from the cloned repository
siam-pred -i INPUT_FILENAME -m 4
```

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
import os.path
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import json, re

import torch, numpy as np
import pandas as pd
import torchio as tio
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, load_pickle, \
    load_json

sys.stdout = open(os.devnull, 'w')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nnunetv2.inference.predict_from_raw_data as _nnunet_pfrd
from nnunetv2.inference.export_prediction import export_prediction_from_logits as _nnunet_export_prediction_from_logits
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
sys.stdout = sys.__stdout__

from SIAMpred.paths import get_model_path_and_fold, addprefixtofilenames, get_parent_path, gfile, gdir


SEG_EXT = '.nii.gz'  # extension of everything written by nnUNet / intermediate files
# nnUNet readers sharing the same axis convention: array (c, z, y, x), spacing (z, y, x)
COMPATIBLE_READERS = ('SimpleITKIO', 'NibabelIO', 'NibabelIOWithReorient')


@contextmanager
def _filter_stdout(drop_substrings):
    """Suppress stdout lines containing any of the given substrings."""
    real = sys.stdout

    class _Filter:
        def __init__(self):
            self._buf = ''

        def write(self, s):
            self._buf += s
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                if not any(sub in line for sub in drop_substrings):
                    real.write(line + '\n')

        def flush(self):
            if self._buf and not any(sub in self._buf for sub in drop_substrings):
                real.write(self._buf)
            self._buf = ''
            real.flush()

    sys.stdout = _Filter()
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = real


# ============================================================================
# Canonical orientation on the fly (no toCano_ file)
# ============================================================================
def _find_nifti(f):
    """Accept file names with or without extension."""
    if f.endswith(('.nii.gz', '.nii')):
        return f
    for ext in ('.nii.gz', '.nii'):
        if os.path.isfile(f + ext):
            return f + ext
    raise FileNotFoundError(f)


def _tio_to_nnunet(images):
    """
    List of torchio images (one per input channel, same grid, original orientation)
    -> nnUNet array (c, z, y, x) in canonical (RAS) orientation + siam_stuff (needed to go back).
    Shared by the file reader and the in-memory region inputs : both see exactly the same data.
    """
    arrays, st = [], None
    for ti in images:
        tc = tio.ToCanonical()(ti)
        if tc.num_channels != 1:
            raise ValueError(f'3D image expected : {ti.path} has {tc.num_channels} channels')
        if st is None:
            st = dict(orientation=tuple(ti.orientation),
                      original_shape=tuple(ti.spatial_shape),
                      original_affine=np.array(ti.affine),
                      canonical_shape=tuple(tc.spatial_shape),
                      canonical_affine=np.array(tc.affine),
                      native_spacing=[float(s) for s in tc.spacing[::-1]])  # (x, y, z) -> (z, y, x)
        else:
            if tuple(tc.spatial_shape) != st['canonical_shape']:
                raise ValueError(f'shape mismatch between input channels : {ti.path}')
            if not np.allclose(tc.affine, st['canonical_affine'], atol=1e-3):
                print(f'WARNING affine mismatch between input channels : {ti.path}')
        data = tc.numpy()[0].astype(np.float32)          # (x, y, z)
        arrays.append(data.transpose((2, 1, 0))[None])  # -> (1, z, y, x) as SimpleITKIO
    return np.vstack(arrays), st


def _canonical_to_original(image, st):
    """
    Bring an image defined in canonical (RAS) space back to the original grid.
    Same grid up to axis permutation / flip -> nearest neighbour resampling is exact.
    The reference grid is rebuilt from (shape, affine) : works for in-memory crops too.
    """
    if st['orientation'] == ('R', 'A', 'S'):
        return image  # ToCanonical did nothing, canonical grid == original grid
    reference = tio.LabelMap(tensor=torch.zeros((1, *st['original_shape']), dtype=torch.uint8),
                             affine=st['original_affine'])
    return tio.Resample(target=reference)(image)


class CanonicalNiftiIO(BaseReaderWriter):
    """
    In-memory equivalent of the former toCano_ file:
      read  : reorientation to RAS (tio.ToCanonical) ; spacing optionally forced (forced_spacing)
      write : optional tissue mask (sub-regions), then back to the original grid with the true affine
    forced_spacing is given in nnUNet array order (z, y, x), before transpose_forward.
    """
    supported_file_endings = ['.nii.gz', '.nii']

    def __init__(self, forced_spacing=None):
        super().__init__()
        self.forced_spacing = forced_spacing

    def read_images(self, image_fnames):
        data, st = _tio_to_nnunet([tio.ScalarImage(_find_nifti(f)) for f in image_fnames])
        spacing = self.forced_spacing if self.forced_spacing is not None else st['native_spacing']
        return data, {'spacing': [float(s) for s in spacing], 'siam_stuff': st}

    def read_seg(self, seg_fname):
        return self.read_images((seg_fname,))

    def write_seg(self, seg, output_fname, properties):
        st = properties['siam_stuff']
        mask = st.get('tissue_mask')  # (z, y, x) bool, sub-regions only
        if mask is not None:
            if mask.shape != seg.shape:
                raise ValueError(f'tissue mask shape {mask.shape} != prediction shape {seg.shape} for {output_fname}')
            # check there is no background left inside the mask (should never happen)
            nb_bg_in_mask = int(((seg == 0) & mask).sum())
            if nb_bg_in_mask > 0:
                raise ValueError(
                    f'{nb_bg_in_mask} background voxels ({100 * nb_bg_in_mask / int(mask.sum()):.2f} % of the mask) '
                    f'predicted inside the tissue mask for {output_fname} : force_no_background not applied ?')
            seg = seg * mask
        seg = seg.transpose((2, 1, 0))  # (z, y, x) -> (x, y, z)
        seg = seg.astype(np.uint8 if seg.max() < 256 else np.int16)
        il = tio.LabelMap(tensor=torch.from_numpy(np.ascontiguousarray(seg))[None],
                          affine=st['canonical_affine'])
        _canonical_to_original(il, st).save(output_fname)


class PlansManagerCanonical(PlansManager):
    """Plans manager returning CanonicalNiftiIO (must stay at module level : pickled for nnUNet workers)."""
    @property
    def image_reader_writer_class(self):
        return partial(CanonicalNiftiIO, forced_spacing=self.plans.get('siam_forced_spacing'))


# ============================================================================
# High resolution output (segmentation at the model resolution, no resampling to the input grid)
# ============================================================================
def _highres_prefix(spacing):
    """0.5 -> 'r05_' , 0.75 -> 'r075_'"""
    return 'r' + f'{min(spacing):g}'.replace('.', '') + '_'


def _highres_name(file_truncated, prefix):
    return os.path.join(os.path.dirname(file_truncated), prefix + os.path.basename(file_truncated))


def _label_mask(data, labels):
    mask = torch.zeros_like(data, dtype=torch.bool)
    for lab in labels:
        mask |= data == lab
    return mask


def _export_highres(predicted_logits, properties, plans_manager, dataset_json, output_file_truncated, prefix):
    """
    Segmentation at the model resolution, uncropped into a full high res grid.
    Saved in canonical (RAS) orientation, with the affine matching the nnUNet resampling
    (edge aligned, effective voxel size = cropped extent / resampled shape).
    Sub-regions : restricted to the region label of the high res tissue prediction.
    """
    st = properties['siam_stuff']
    output_file = _highres_name(output_file_truncated, prefix) + SEG_EXT
    hm = st.get('highres_mask')
    if hm is not None and not os.path.isfile(hm['file']):
        print(f'WARNING no high res tissue prediction {hm["file"]} : high res output {output_file} skipped')
        return
    if isinstance(dataset_json, str):
        dataset_json = load_json(dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)

    logits = torch.from_numpy(predicted_logits) if isinstance(predicted_logits, np.ndarray) else predicted_logits
    if logits.ndim != 4:
        raise ValueError('save_highres is only supported for 3d configurations')
    with torch.no_grad():
        if label_manager.has_regions:
            seg = label_manager.convert_logits_to_segmentation(logits.float())
            seg = seg.cpu().numpy() if torch.is_tensor(seg) else np.asarray(seg)
        else:  # argmax slice by slice : no float32 copy of the whole 4D logits (many classes)
            seg = torch.empty(logits.shape[1:], dtype=torch.int16)
            for z in range(logits.shape[1]):
                seg[z] = logits[:, z].float().argmax(0)
            seg = seg.numpy()

    # all shapes / bbox below are in nnUNet transposed order (transpose_forward)
    n_crop = np.array(properties['shape_after_cropping_and_before_resampling'], dtype=float)
    full = np.array(properties['shape_before_cropping'], dtype=float)
    bbox_start = np.array([b[0] for b in properties['bbox_used_for_cropping']], dtype=float)
    m = np.array(seg.shape)
    r = n_crop / m                                          # input voxels per high res voxel
    start_hr = np.round(bbox_start / r).astype(int)
    full_hr = np.maximum(np.ceil(full / r).astype(int), start_hr + m)
    t = 0.5 * r - 0.5 + bbox_start - start_hr * r           # input (continuous) index of high res voxel 0

    seg_full = np.zeros(full_hr, dtype=np.uint8 if seg.max() < 256 else np.int16)
    seg_full[tuple(slice(s, s + k) for s, k in zip(start_hr, m))] = seg

    # back to the reader array order (z, y, x), then torchio order (x, y, z)
    tb = list(plans_manager.transpose_backward)
    seg_full, r, t = seg_full.transpose(tb), r[tb], t[tb]
    seg_xyz = np.ascontiguousarray(seg_full.transpose((2, 1, 0)))
    M = np.eye(4)
    M[:3, :3] = np.diag(r[::-1])
    M[:3, 3] = t[::-1]
    il = tio.LabelMap(tensor=torch.from_numpy(seg_xyz)[None], affine=st['canonical_affine'] @ M)

    if hm is not None:
        itis = tio.LabelMap(hm['file'])
        imask = tio.LabelMap(tensor=_label_mask(itis.data, hm['labels']).to(torch.uint8), affine=itis.affine)
        mask = tio.Resample(target=il)(imask).data > 0  # nearest neighbour, grids almost aligned
        nb_out = int((mask & (il.data == 0)).sum())
        if nb_out > 0:
            print(f'WARNING {nb_out} high res tissue voxels outside the region crop for {output_file}')
        il.set_data(il.data * mask.to(il.data.dtype))

    il.save(output_file)
    print(f'saving high res {output_file}')


def _export_prediction_from_logits_siam(predicted_array_or_file, properties_dict, configuration_manager,
                                        plans_manager, dataset_json_dict_or_file, output_file_truncated,
                                        save_probabilities=False, *args, **kwargs):
    """nnUNet export + optional high res export (module level : pickled for the export workers)."""
    prefix = plans_manager.plans.get('siam_highres_prefix')
    if prefix is not None and 'siam_stuff' in properties_dict:
        _export_highres(predicted_array_or_file, properties_dict, plans_manager,
                        dataset_json_dict_or_file, output_file_truncated, prefix)
    return _nnunet_export_prediction_from_logits(predicted_array_or_file, properties_dict, configuration_manager,
                                                 plans_manager, dataset_json_dict_or_file, output_file_truncated,
                                                 save_probabilities, *args, **kwargs)


# predict_from_data_iterator looks up this global at call time : inject our export
# (the original is imported from export_prediction, so a module reload cannot wrap twice)
_nnunet_pfrd.export_prediction_from_logits = _export_prediction_from_logits_siam


# ============================================================================
# Predictor
# ============================================================================
def _model_spacing_before_transpose(plans_manager, configuration_manager):
    """Target spacing of the model, expressed in the reader array order (before transpose_forward)."""
    spacing = configuration_manager.spacing
    if len(spacing) != 3:
        raise ValueError('model spacing options (voxel_size < 0, save_highres) only supported for 3d configurations')
    return [float(spacing[i]) for i in plans_manager.transpose_backward]


def _get_model_spacing(model_folder):
    """Target spacing of a model folder (Trainer__Plans__configuration), without initializing the network."""
    configuration = os.path.basename(os.path.normpath(model_folder)).split('__')[-1]
    pm = PlansManager(join(model_folder, 'plans.json'))
    return _model_spacing_before_transpose(pm, pm.get_configuration(configuration))


def _flatten_models(folder_with_parameter_files):
    if isinstance(folder_with_parameter_files, (list, tuple)):
        return [m for sub in folder_with_parameter_files for m in _flatten_models(sub)]
    return [folder_with_parameter_files]


class nnUNetPredictorSIAM(nnUNetPredictor):
    """
    - force_no_background : label 0 cannot be predicted (sub-regions)
    - CanonicalNiftiIO reader : no toCano_ file written on disk
    - voxel_size : 0 header spacing / >0 forced isotropic spacing / <0 reference spacing (see spacing_mode)
    - predict_from_nnunet_arrays : in-memory inputs (sub-region crops)
    - save_highres : also save the segmentation at the model resolution (models at highres_spacing only)
    """
    force_no_background = False
    # -1e4 and not -inf : logits are float16 in nnUNet (max ~65504),
    # and -inf would give NaN when the logits are resampled (spline)
    BG_LOGIT = -1e4
    voxel_size = 0
    reference_spacing = None    # voxel_size < 0 : finest model spacing of the pipeline (set by nn_predict)
    spacing_mode = 'reference'  # 'reference' or 'matching' (see nn_predict)
    save_highres = False        # toggled by nn_predict (False for the cascade low res stage)
    highres_spacing = None      # spacing of the final tissue model (set by nn_predict)
    highres_prefix = None

    def _compute_forced_spacing(self):
        if self.voxel_size > 0:
            return [float(self.voxel_size)] * 3
        if self.voxel_size == 0:
            return None
        model_spacing = _model_spacing_before_transpose(self.plans_manager, self.configuration_manager)
        reference = self.reference_spacing if self.reference_spacing is not None else model_spacing
        if np.allclose(model_spacing, reference, atol=1e-3):
            return list(reference)  # model at reference resolution : no interpolation
        if self.spacing_mode == 'reference':
            return list(reference)  # declared at reference spacing, nnUNet resamples to the model spacing
        if self.spacing_mode == 'matching':
            return None             # model at another resolution : true header spacing
        raise ValueError(f'unknown spacing_mode {self.spacing_mode}')

    def _compute_highres_prefix(self):
        if not self.save_highres or self.highres_spacing is None:
            return None
        model_spacing = _model_spacing_before_transpose(self.plans_manager, self.configuration_manager)
        if np.allclose(model_spacing, self.highres_spacing, atol=1e-3):
            print(f'high res output enabled ({self.highres_prefix})')
            return self.highres_prefix
        print(f'model spacing {model_spacing} != high res spacing {self.highres_spacing} : no high res output')
        return None

    def initialize_from_trained_model_folder(self, *args, **kwargs):
        super().initialize_from_trained_model_folder(*args, **kwargs)
        rw_name = self.plans_manager.plans['image_reader_writer']
        if rw_name not in COMPATIBLE_READERS:
            raise ValueError(f'reader {rw_name} : axis convention not checked with CanonicalNiftiIO')
        if self.dataset_json['file_ending'] != SEG_EXT:
            raise ValueError(f'file_ending {self.dataset_json["file_ending"]} != {SEG_EXT}')

        forced_spacing = self._compute_forced_spacing()
        if self.voxel_size != 0:
            print(f'model spacing {self.configuration_manager.spacing} : '
                  f'input spacing {"forced to " + str(forced_spacing) if forced_spacing else "from header"}')
        self.plans_manager.plans['siam_forced_spacing'] = forced_spacing
        self.plans_manager.plans['siam_highres_prefix'] = self._compute_highres_prefix()
        self.plans_manager.__class__ = PlansManagerCanonical  # inject our reader/writer

    @torch.inference_mode()  # needed : in-place modification of an "inference tensor"
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        prediction = super().predict_logits_from_preprocessed_data(data)
        if self.force_no_background:
            if self.label_manager.has_regions:
                raise ValueError('force_no_background not compatible with a region-based model (sigmoid)')
            prediction[0] = self.BG_LOGIT
        return prediction

    def _save_output_jsons(self, output_folder):
        maybe_mkdir_p(output_folder)
        save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)

    def predict_from_files_and_prev_seg(self, list_of_lists, seg_from_prev_stage_files, output_files_truncated,
                                        save_probabilities, num_processes_preprocessing,
                                        num_processes_segmentation_export):
        """
        predict_from_files with an explicit list of previous stage segmentations
        (avoids nnUNet caseid deduction by truncation, and therefore the symlinks)
        """
        self._save_output_jsons(os.path.dirname(output_files_truncated[0]))
        todo = [i for i, o in enumerate(output_files_truncated) if not os.path.isfile(o + SEG_EXT)]  # overwrite=False
        if len(todo) == 0:
            print('second cascade stage : all outputs exist, skipping')
            return
        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            [list_of_lists[i] for i in todo], [seg_from_prev_stage_files[i] for i in todo],
            [output_files_truncated[i] for i in todo], num_processes_preprocessing)
        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_nnunet_arrays(self, arrays, siam_stuffs, output_files_truncated,
                                   num_processes_preprocessing, num_processes_segmentation_export):
        """In-memory inputs (output of _tio_to_nnunet) : nothing written on disk except the predictions."""
        self._save_output_jsons(os.path.dirname(output_files_truncated[0]))  # needed by regroup_regions
        forced = self.plans_manager.plans.get('siam_forced_spacing')
        properties = [{'spacing': [float(s) for s in (forced if forced is not None else st['native_spacing'])],
                       'siam_stuff': st} for st in siam_stuffs]
        return self.predict_from_list_of_npy_arrays(
            arrays, None, properties, output_files_truncated,
            num_processes=num_processes_preprocessing,
            save_probabilities=False,
            num_processes_segmentation_export=num_processes_segmentation_export)


def get_nn_predictor(
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        verbose: bool = False,
        voxel_size=0,
):
    os.environ['nnUNet_compile'] = 'F'
    # nnUNet only honours perform_everything_on_device on CUDA, and prints
    # the same warning unconditionally on non-CUDA devices regardless of the
    # value we pass. Silence just that one line here.
    with _filter_stdout(['perform_everything_on_device=True is only supported for cuda']):
        predictor = nnUNetPredictorSIAM(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_tta,
            perform_everything_on_device=(device.type == 'cuda'),
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose
        )
    predictor.voxel_size = voxel_size

    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
    return predictor


# ============================================================================
# General utilities
# ============================================================================
def _split_ext(fname):
    """Return (extension, name_without_extension) for .nii or .nii.gz"""
    fname_ext = '.nii.gz' if fname.endswith('.nii.gz') else '.nii'
    return fname_ext, fname[:-len(fname_ext)]


def _get_model_type(folder_with_parameter_files):
    """
    'simple'            : str
    'subregion'         : [tissue_model, [region_models...]]
    'cascade'           : [lowres_model, fullres_model]
    'cascade_subregion' : [[lowres_model, fullres_model], [region_models...]]
    """
    if not isinstance(folder_with_parameter_files, list):
        return 'simple'
    first, second = folder_with_parameter_files[0], folder_with_parameter_files[1]
    if isinstance(first, list):
        if not isinstance(second, list):
            raise ValueError('cascade_subregion : second element must be a list of region models')
        return 'cascade_subregion'
    if isinstance(second, list):
        return 'subregion'
    return 'cascade'


def _run_prediction(predictor, model_folder, folds, input_files_lists, output_files,
                    save_probabilities, num_processes_preprocessing, num_processes_segmentation_export,
                    folder_with_segs_from_prev_stage=None):
    predictor.initialize_from_trained_model_folder(model_folder, folds)
    predictor.predict_from_files(
        input_files_lists,
        output_files,
        save_probabilities=save_probabilities,
        overwrite=False,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
        num_parts=1,
        part_id=0
    )


def _convert_proba_to_nifti(output_files):
    """npz probabilities are in canonical array space : same back-transformation as write_seg."""
    print(f'converting proba npz, in nifti')
    for fo in output_files:
        fprob1, fprob2 = fo + '.npz', fo + '.pkl'
        if not os.path.isfile(fprob1):
            print(f'no proba file {fprob1}, skip')
            continue
        st = load_pickle(fprob2)['siam_stuff']
        proba = np.load(fprob1)['probabilities'].transpose((0, 3, 2, 1))  # (c,z,y,x) -> (c,x,y,z)
        il = tio.LabelMap(tensor=torch.from_numpy(np.ascontiguousarray(proba)), affine=st['canonical_affine'])
        fonew = addprefixtofilenames(fo, 'proba_')[0]
        _canonical_to_original(il, st).save(f'{fonew}{SEG_EXT}')
        os.remove(fprob1)
        os.remove(fprob2)


def _cleanup_json_files(output_files, out_prefix):
    # may fail if two single file predictions run in the same folder : not a big deal
    try:
        dout = os.path.dirname(output_files[0])
        os.replace(join(dout, 'dataset.json'), join(dout, f'label_{out_prefix}.json'))
        os.remove(join(dout, 'plans.json'))
        os.remove(join(dout, 'predict_from_raw_data_args.json'))
    except Exception:
        pass


def _get_num_processes(num_process, n_files, device):
    n = max(1, min(num_process, n_files))  # no need for more workers than cases
    if device.type == 'cpu':
        n = max(1, min(n, 2))  # the network already uses all threads
    return n, n  # (preprocessing, export)


# ============================================================================
# Inputs / outputs
# ============================================================================
@dataclass
class _IOFiles:
    input_is_file: bool
    input_files: list   # original input files (never modified, no more toCano_)
    output_files: list  # final tissue predictions, WITHOUT extension (nnUNet appends .nii.gz)
    out_prefix: str


def _resolve_output_dir(input_file_or_folder, output_option, out_prefix):
    """Output folder and output prefix from the -o option.

    - None             : <input folder>/<out_prefix>
    - relative string  : appended to the prefix -> <input folder>/<out_prefix><output_option>
    - absolute path    : used as the output folder, prefix unchanged
    (<input folder> is the parent folder when the input is a file)
    """
    if output_option:
        output_option = os.path.expanduser(output_option)
        if os.path.isabs(output_option):
            return os.path.normpath(output_option), out_prefix
        out_prefix += output_option
    if os.path.isdir(input_file_or_folder):
        input_dir = input_file_or_folder
    else:
        input_dir = get_parent_path(input_file_or_folder)[0]
    return join(input_dir, out_prefix), out_prefix


def _resolve_input_output(input_file_or_folder, output_file_or_folder, out_prefix):
    """Return an _IOFiles, or None if no input file is found."""
    if not input_file_or_folder.startswith('/'):
        input_file_or_folder = join(os.path.curdir, input_file_or_folder)

    if os.path.isdir(input_file_or_folder):
        # ignore files generated by this script (toCano_ from older versions, outputs, probas)
        excluded_prefix = ('toCano_', out_prefix, 'proba_')
        input_files = [f for f in gfile(input_file_or_folder, '.*nii')
                       if not os.path.basename(f).startswith(excluded_prefix)]
    else:
        input_files = [input_file_or_folder]
    output_file_or_folder, out_prefix = _resolve_output_dir(input_file_or_folder, output_file_or_folder, out_prefix)

    if len(input_files) == 0:
        return None
    maybe_mkdir_p(output_file_or_folder)

    output_files = [join(output_file_or_folder, f'{out_prefix}{_split_ext(os.path.basename(i))[1]}')
                    for i in input_files]
    if len(set(output_files)) != len(output_files):  # ex : a.nii and a.nii.gz in the same folder
        raise ValueError(f'duplicate output names : {output_files}')

    # outputs always go in a dedicated sub-folder, so it is handled as a folder input
    return _IOFiles(False, input_files, output_files, out_prefix)


def _get_first_stage_paths(output_files, input_is_file, out_prefix, fname_noext):
    """Intermediate folder + names of the first stage predictions (Pre_*), without extension."""
    diroutput = os.path.dirname(output_files[0])
    if not input_is_file:
        dirout1 = diroutput
    else:
        dirout1 = os.path.join(diroutput, f'region_{out_prefix}_{fname_noext}')
    maybe_mkdir_p(dirout1)

    output_files_first_stage = []
    for oo in output_files:
        file_basname = os.path.basename(oo)[len(out_prefix):]
        output_files_first_stage.append(os.path.join(dirout1, f'Pre_{out_prefix}{file_basname}'))
    return dirout1, output_files_first_stage


# ============================================================================
# nnUNet cascade : second stage (full res)
# ============================================================================
def _predict_cascade_fullres(predictor, model_folder, folds, input_files, output_files_first_stage,
                             output_files, save_proba,
                             num_processes_preprocessing, num_processes_segmentation_export):
    print(f"runing second cascade on {len(input_files)} first input {input_files[0]}, output {output_files[0]}")
    predictor.initialize_from_trained_model_folder(model_folder, folds)
    predictor.predict_from_files_and_prev_seg(
        [[f] for f in input_files],
        [f + SEG_EXT for f in output_files_first_stage],
        output_files,
        save_probabilities=save_proba,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export)


# ============================================================================
# Sub-regions
# ============================================================================
def _load_tissue_labels(tissue_seg_files):
    fjson = gfile(get_parent_path(tissue_seg_files)[0], 'dataset.json')[0]
    with open(fjson) as f:
        dsj = json.load(f)
    return dsj['labels']


def _get_region_names(region_model_folders, name_from_level=2):
    sub_model_name = get_parent_path(region_model_folders, name_from_level)[1]
    region, region_dir = [], []
    for DSname in sub_model_name:
        ind_underscore = [m.start() for m in re.finditer('_', DSname)]
        region_name = DSname[ind_underscore[0] + 1:ind_underscore[1]]
        if region_name == "reg":  # new naming convention
            region_name = DSname[ind_underscore[1] + 1:ind_underscore[2]]
        if region_name == 'deep':  # quick fix for region deep_nuclei
            region_name = 'deep_nuclei'

        region.append(region_name)
        region_dir.append('pred_' + DSname[ind_underscore[0] + 1:])
    return region, region_dir


def _region_pred_name(region_dir, input_file, out_prefix):
    """Final region prediction, WITHOUT extension (name expected by regroup_regions)."""
    return os.path.join(region_dir, f'{out_prefix}' + _split_ext(os.path.basename(input_file))[1])


def _region_labels(reg, dic_lab):
    """Tissue labels defining the region mask (None for Lesion : whole image)."""
    if reg == "Lesion":
        return None
    if reg == 'Thal' and "Striatum" in dic_lab:  # quick hack for model 4040 -> thal contains striatum and RU
        return [dic_lab['Thal'], dic_lab['Striatum'], dic_lab['RU']]
    return [dic_lab[reg]]


def _prepare_region_case(ti, il, reg, dic_lab, crop_to_mask=True, min_voxels=1000, highres_tissue_file=None):
    """
    In-memory input of one region model for one subject.
    ti : input image, il : tissue segmentation (same grid, original orientation)
    highres_tissue_file : high res tissue prediction, used to mask the high res region output
    Returns (array (c, z, y, x), siam_stuff) or None if the region mask is too small.
    Crops are done in the original orientation, then canonicalized as by the file reader.
    """
    labels = _region_labels(reg, dic_lab)
    data = il.data > 0 if labels is None else _label_mask(il.data, labels)
    if data.sum() < min_voxels:
        print(f"Skiping region {reg} because first level mask has less than {min_voxels} voxel")
        return None

    if labels is None:  # Lesion : whole image, second input is the tissue segmentation, no tissue mask
        arr, st = _tio_to_nnunet([ti, il])
        return arr, st

    iout = tio.LabelMap(tensor=data.to(torch.uint8), affine=il.affine)
    images = [ti, iout]
    if crop_to_mask:
        suj = tio.Subject({'t1': ti, 'lab': iout})
        tc = tio.Compose([tio.CropOrPad(mask_name='lab'), tio.Pad(4), tio.EnsureShapeMultiple(2)])
        sujt = tc(suj)
        images = [sujt.t1, sujt.lab]
    arr, st = _tio_to_nnunet(images)
    st['tissue_mask'] = arr[1] > 0  # applied in CanonicalNiftiIO.write_seg
    if highres_tissue_file is not None:
        st['highres_mask'] = dict(file=highres_tissue_file, labels=labels)  # applied in _export_highres
    return arr, st


def _predict_subregions(predictor, region_model_folders, region_folds, input_files, tissue_seg_files,
                        dirout1, num_processes_preprocessing, num_processes_segmentation_export,out_prefix,
                        crop_to_mask=True, min_voxels=1000, max_subjects_in_memory=20):
    """
    tissue_seg_files : tissue segmentations (without extension) used to define the masks
                       (Pre_* for case 2, cascade full res output for case 4)
    Region inputs are built in memory (nothing written except the predictions).
    Region predictions are written in the original orientation, on the cropped grid
    (regroup_regions resamples them to the tissue grid).
    """
    dic_lab = _load_tissue_labels(tissue_seg_files)
    region, region_dir = _get_region_names(region_model_folders)
    dirout_list = [os.path.join(dirout1, f'{reg}') for reg in region_dir]

    # subjects processed by chunks to bound memory (Lesion inputs are whole images)
    for start in range(0, len(input_files), max_subjects_in_memory):
        cases = [[] for _ in region]  # cases[ii] = [(subj_idx, array, siam_stuff), ...]
        for k in range(start, min(start + max_subjects_in_memory, len(input_files))):
            todo = [ii for ii in range(len(region))
                    if not os.path.isfile(_region_pred_name(dirout_list[ii], input_files[k], out_prefix) + SEG_EXT)]
            if len(todo) == 0:
                continue
            ti = tio.ScalarImage(input_files[k])
            il = tio.LabelMap(tissue_seg_files[k] + SEG_EXT)
            ti.load(), il.load()  # read once for all regions
            highres_tissue_file = None
            if predictor.save_highres and predictor.highres_prefix is not None:
                highres_tissue_file = _highres_name(tissue_seg_files[k], predictor.highres_prefix) + SEG_EXT
            for ii in todo:
                case = _prepare_region_case(ti, il, region[ii], dic_lab, crop_to_mask, min_voxels,
                                            highres_tissue_file)
                if case is not None:
                    cases[ii].append((k, *case))

        for ii, (mfolder, efold) in enumerate(zip(region_model_folders, region_folds)):
            if len(cases[ii]) == 0:
                print(f'region {region[ii]} : nothing to predict (already done or mask too small)')
                continue
            print(f"predicting subregion {region[ii]} on {len(cases[ii])} subjects")
            predictor.force_no_background = region[ii] != "Lesion"
            try:
                predictor.initialize_from_trained_model_folder(mfolder, efold)
                predictor.predict_from_nnunet_arrays(
                    [c[1] for c in cases[ii]],
                    [c[2] for c in cases[ii]],
                    [_region_pred_name(dirout_list[ii], input_files[c[0]], out_prefix) for c in cases[ii]],
                    num_processes_preprocessing, num_processes_segmentation_export)
            finally:
                predictor.force_no_background = False  # do not pollute the next predictions
        del cases

    return dict(dic_lab=dic_lab, region=region, dirout_list=dirout_list)


# ============================================================================
# Main functions
# ============================================================================
def nn_predict(
        input_file_or_folder: str,
        output_file_or_folder: str,
        device: torch.device = torch.device('cuda'),
        num_model: int = 1,
        nb_fold: int = 0,
        region_regex: str = '.*',
        voxel_size=0,
        model_spacing_mode='reference',
        verbose: bool = False,
        num_process=8,
        save_proba=False,
        save_highres=False,
        do_merge_regions=False,
):
    """
    voxel_size :
        0  : spacing from the nifti header
        >0 : forced isotropic spacing, for all models
        <0 : reference spacing = finest model spacing of the pipeline ; models at this
             resolution see the image with exactly their spacing (no interpolation)
    model_spacing_mode (only used if voxel_size < 0), for models at another resolution :
        'reference' : image declared at the reference spacing, then resampled by nnUNet
                      (consistent scaling for all models, as the cascade low res stage)
        'matching'  : true header spacing (only models at the reference resolution are forced)
    save_highres :
        also save the predictions at the resolution of the final tissue model (prefix r05_, r075_ ...)
        for the final tissue model and the region models at the same resolution
        (not the cascade low res stage)
    """
    folder_with_parameter_files, existing_fold, out_prefix = get_model_path_and_fold(num_model, region_regex,nb_fold)
    if not input_file_or_folder.startswith('/'):
        input_file_or_folder = join(os.path.curdir, input_file_or_folder)

    model_type = _get_model_type(folder_with_parameter_files)
    print(f"save PROBA is {save_proba}  (model type: {model_type})")

    # ---- models
    first_model = fullres_model = region_models = None
    first_fold = fullres_fold = region_folds = None
    if model_type == 'cascade_subregion':
        (first_model, fullres_model), region_models = folder_with_parameter_files
        (first_fold, fullres_fold), region_folds = existing_fold
    elif model_type == 'cascade':
        first_model, fullres_model = folder_with_parameter_files
        first_fold, fullres_fold = existing_fold
    elif model_type == 'subregion':
        first_model, region_models = folder_with_parameter_files
        first_fold, region_folds = existing_fold
    if model_type == 'simple':
        tissue_model = folder_with_parameter_files
    elif 'cascade' in model_type:
        tissue_model = fullres_model
    else:
        tissue_model = first_model

    # ---- inputs / outputs
    io_files = _resolve_input_output(input_file_or_folder, output_file_or_folder, out_prefix)
    if io_files is None:
        print(f"no file found in {input_file_or_folder} ... so nothing done ....")
        return
    out_prefix = io_files.out_prefix
    input_files = io_files.input_files
    output_files = io_files.output_files  # without extension

    predictor = get_nn_predictor(use_tta=False, device=device, verbose=verbose, voxel_size=voxel_size)
    if voxel_size < 0:
        spacings = [_get_model_spacing(m) for m in _flatten_models(folder_with_parameter_files)]
        predictor.reference_spacing = min(spacings, key=lambda s: float(np.prod(s)))
        predictor.spacing_mode = model_spacing_mode
        print(f'reference spacing (finest model) {predictor.reference_spacing}, mode {model_spacing_mode}')
    if save_highres:
        predictor.highres_spacing = _get_model_spacing(tissue_model)
        predictor.highres_prefix = _highres_prefix(predictor.highres_spacing)
        print(f'high res spacing (tissue model) {predictor.highres_spacing}, prefix {predictor.highres_prefix}')
    predictor.save_highres = save_highres
    fname_noext = _split_ext(os.path.basename(input_files[0]))[1]

    num_processes_preprocessing, num_processes_segmentation_export = \
        _get_num_processes(num_process, len(output_files), device)
    print(f'num process {num_processes_preprocessing} / {num_processes_segmentation_export}  (preproc/export)')

    # ---- prediction
    if model_type == 'simple':
        _run_prediction(predictor, folder_with_parameter_files, existing_fold,
                        [[i] for i in input_files], output_files, save_proba,
                        num_processes_preprocessing, num_processes_segmentation_export)
    else:
        dirout1, output_files_first_stage = _get_first_stage_paths(
            output_files, io_files.input_is_file, out_prefix, fname_noext)

        if 'cascade' in model_type:
            # first stage (low res) -> first_stage_pred/Pre_* (save_proba and save_highres discarded)
            output_files_first_stage = addprefixtofilenames(output_files_first_stage, 'first_stage_pred/')
            predictor.save_highres = False
            _run_prediction(predictor, first_model, first_fold,
                            [[i] for i in input_files], output_files_first_stage, False,
                            num_processes_preprocessing, num_processes_segmentation_export)
            predictor.save_highres = save_highres
            # second stage (full res) -> output_files
            _predict_cascade_fullres(predictor, fullres_model, fullres_fold, input_files,
                                     output_files_first_stage, output_files, save_proba,
                                     num_processes_preprocessing, num_processes_segmentation_export)
        else:
            # first stage (tissue) -> Pre_*
            _run_prediction(predictor, first_model, first_fold,
                            [[i] for i in input_files], output_files_first_stage, save_proba,
                            num_processes_preprocessing, num_processes_segmentation_export)

        # sub-regions
        if model_type in ('subregion', 'cascade_subregion'):
            # case 2 : masks from Pre_* ; case 4 : from the cascade full res output
            tissue_seg_files = output_files_first_stage if model_type == 'subregion' else output_files
            _predict_subregions(predictor, region_models, region_folds, input_files, tissue_seg_files, dirout1,
                                num_processes_preprocessing, num_processes_segmentation_export,out_prefix)
            do_merge_regions = True

    # ---- post-processing
    if save_proba:
        _convert_proba_to_nifti(output_files)

    if do_merge_regions:
        regroup_regions(input_file_or_folder, output_file_or_folder, num_model, region_regex=region_regex)

    if io_files.input_is_file:  # otherwise keep the json files in the output folder
        _cleanup_json_files(output_files, out_prefix)


def regroup_regions(
        input_file_or_folder: str,
        output_file_or_folder: str,
        num_model: int = 1,
        region_regex: str = '.*',
):
    folder_with_parameter_files, existing_fold, out_prefix = get_model_path_and_fold(num_model, region_regex)

    model_type = _get_model_type(folder_with_parameter_files)
    print(f" model {num_model} type: {model_type})")
    pred_dir, out_prefix = _resolve_output_dir(input_file_or_folder, output_file_or_folder, out_prefix)

    # regex anchored with ^ : high res outputs (r05_pred_...) must not be matched
    if os.path.isfile(input_file_or_folder):
        input_name = get_parent_path(input_file_or_folder)[1]
        pred_regex = f'^{out_prefix}.*{input_name}'
    else:
        pred_regex = f'^{out_prefix}.*nii'

    fpreds = gfile(pred_dir, pred_regex)

    dreg = gdir(pred_dir, '^pred_reg')
    for region_dir in dreg:
        if not (len(gfile(region_dir, pred_regex)) == len(fpreds)):
            raise FileNotFoundError(f"missing prediction in {region_dir}")

    print(f'found  {len(fpreds)} predictions and {len(dreg)} region (all predicted)')

    region, region_dir = _get_region_names(dreg, name_from_level=1)
    dir_code = get_parent_path(os.path.realpath(__file__))[0]
    fall_csv_vasc, fall_csv_allen = gfile(dir_code, 'vascular_All')[0], gfile(dir_code, 'allen_All')[0]
    dfv = pd.read_csv(fall_csv_vasc)
    dfa = pd.read_csv(fall_csv_allen)
    dicNv = {dfs.Name: dfs.label for _, dfs in dfv.iterrows()}
    dicNaRegLobe = {dfs.Name: dfs.Region for _, dfs in dfa.iterrows()}
    dicNvRegLobe = {dfs.Name: dfs.Lobes_Name for _, dfs in dfv.iterrows() if not pd.isna(dfs.Lobes)}
    dicVasLobe_name2val = {dfs.Lobes_Name: dfs.Lobes for _, dfs in dfv.iterrows() if not pd.isna(dfs.Lobes)}

    for ft in fpreds:
        fn = f'{out_prefix}' + get_parent_path(ft)[1][len(out_prefix):]
        # name of the merged outputs : 'pred_' is dropped (older prefixes), other prefixes (siamV03_ ...) are kept
        fn_out = fn.removeprefix('pred_')
        fo_all = os.path.join(pred_dir, 'All_lab_AllenAssn_' + fn_out)
        if os.path.isfile(fo_all):
            print(f'skiping existing out {fo_all}')
            continue

        dic_lab_tissue = _load_tissue_labels(pred_dir + '/toto')
        il = tio.LabelMap(ft)
        dataremap = torch.zeros_like(il.data.to(torch.int16))

        for regn, regd in zip(region, region_dir):
            fpred = gfile(os.path.join(pred_dir, regd), '^' + fn)
            if not len(fpred) == 1:
                raise FileNotFoundError(f'missing prediction {fn} in {os.path.join(pred_dir, regd)}')
            ilreg = tio.LabelMap(fpred[0])
            ilreg.data = ilreg.data.to(torch.int16)  # WM labels above 256 : needed for the remap
            dicreg = _load_tissue_labels(fpred)
            if 'allen' in regd:
                fo = os.path.join(pred_dir, 'Lobes_GMAllen_' + fn_out)
                if not os.path.isfile(fo):
                    tmap_reg = tio.RemapLabels({v: dicVasLobe_name2val[dicNaRegLobe[k]] for k, v in dicreg.items()})
                    if ilreg.shape == dataremap.shape:
                        ilreg = tmap_reg(ilreg)
                    else:
                        tr = tio.Resample(target=ft)
                        ilreg = tmap_reg(tr(ilreg))
                    ilreg.save(fo)
                    print(f"saving{fo}")

            else:
                tmap_reg = tio.RemapLabels({v: dicNv[k] for k, v in dicreg.items()})
                if ilreg.shape == dataremap.shape:
                    ilreg = tmap_reg(ilreg)
                else:
                    tr = tio.Resample(target=ft)
                    ilreg = tmap_reg(tr(ilreg))

                dataremap += ilreg.data.to(torch.int16)
                dic_lab_tissue.pop(regn)
                if "GM" in regd:  # GM AssN
                    tmap_reg = tio.RemapLabels({v: dicVasLobe_name2val[dicNvRegLobe[k]] for k, v in dicreg.items()})
                    fo = os.path.join(pred_dir, 'Lobes_GMAssN_' + fn_out)
                    if not os.path.isfile(fo):
                        ilreg = tio.LabelMap(fpred[0])
                        if ilreg.shape == dataremap.shape:
                            ilreg = tmap_reg(ilreg)
                        else:
                            tr = tio.Resample(target=ft)
                            ilreg = tmap_reg(tr(ilreg))
                        ilreg.save(fo)
                        print(f"saving{fo}")

        dic_lab_tissue.pop('background')
        if len(dic_lab_tissue) == 0:
            print('all region added')

        ilall = tio.LabelMap(tensor=dataremap, affine=il.affine)
        ilall.save(fo_all)
        print(f"saving{fo_all}")

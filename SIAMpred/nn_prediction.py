import os.path
import sys
import torch, numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir
sys.stdout = open(os.devnull, 'w')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
sys.stdout = sys.__stdout__
from SIAMpred.paths import get_model_path_and_fold, addprefixtofilenames, get_parent_path, gfile, addsuffixtofilenames, \
    remove_extension
import json, re
import torchio as tio

#from scipy.ndimage import binary_dilation

from contextlib import contextmanager
from dataclasses import dataclass

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

def get_nn_predictor(
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        verbose: bool = False
):
    os.environ['nnUNet_compile'] = 'F'
    # nnUNet only honours perform_everything_on_device on CUDA, and prints
    # the same warning unconditionally on non-CUDA devices regardless of the
    # value we pass. Silence just that one line here.
    with _filter_stdout(['perform_everything_on_device=True is only supported for cuda']):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_tta,
            perform_everything_on_device=(device.type == 'cuda'),
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose
        )

    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count())
    return predictor

def convert_to_canonical_if_needed(file_list, voxel_size, output_dir=None, fout_cano=None):
    if fout_cano is None:
        fout_cano = addprefixtofilenames(file_list, 'toCano_')
        if output_dir is not None:
            input_dir, fname = get_parent_path(fout_cano)
            fout_cano = [os.path.join(output_dir,ff) for ff in fname]

    fout = []
    tcano = tio.ToCanonical()
    reoriented_list = []
    for fi, fo in zip(file_list, fout_cano):
        # nnUNet suppose que les inputs ont l'extension du dataset (.nii.gz) :
        # on force la conversion des .nii en .nii.gz (fichier temporaire toCano_)
        if fo.endswith('.nii'):
            fo = fo + '.gz'
        ti = tio.ScalarImage(fi)
        if (ti.orientation == ('R', 'A', 'S')) and (voxel_size == 0) and fi.endswith('.nii.gz'):
            fout.append(fi)
            reoriented_list.append(False)
        else:
            if not os.path.isfile(fo):
                tc = tcano(ti)
                reoriented_list.append(True)
                if voxel_size > 0:
                    print(f'CHANING NIFTI voxel size to {voxel_size} saving {fo}')
                    new_aff = tc.affine
                    new_aff[0, 0] = voxel_size
                    new_aff[1, 1] = voxel_size
                    new_aff[2, 2] = voxel_size
                    tc.affine = new_aff
                else:
                    print(f'cononical (and/or .nii.gz) saving {fo}')
                tc.save(fo)
            fout.append(fo)

    return fout, reoriented_list


# ============================================================================
# Utilitaires généraux
# ============================================================================
def _split_ext(fname):
    """Retourne (extension, nom_sans_extension) pour .nii ou .nii.gz"""
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
            raise ValueError('cascade_subregion : le 2ieme element doit etre une liste de modeles region')
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

def _reorient_prediction_back(fi_orig, fo_cano_noext, fo_wanted, fname_ext, voxel_size):
    tr = tio.Resample(target=fi_orig)
    ffo = fo_cano_noext + fname_ext  # prédiction nnUNet (toujours .nii.gz maintenant)
    if fo_wanted.endswith('.nii'):  # input d'origine en .nii -> sortie forcée en .nii.gz
        fo_wanted = fo_wanted + '.gz'
    ipred = tio.LabelMap(ffo)
    ipred.load()  # IMPORTANT pour pouvoir modifier l'affine
    if voxel_size > 0:
        ilorig = tio.ToCanonical()(tio.LabelMap(fi_orig))
        ipred.affine = ilorig.affine
    io = tr(ipred)
    io.save(fo_wanted)
    print(f'removing {ffo}')
    os.remove(ffo)


# ============================================================================
# Gestion des entrées / sorties
# ============================================================================
@dataclass
class _IOFiles:
    input_is_file: bool
    input_file_orig: list
    input_files: list
    converted_to_canonical: list
    output_files: list          # avec extension (retirée plus tard dans nn_predict)
    output_files_wanted: list
    out_prefix: str


def _resolve_input_output(input_file_or_folder, output_file_or_folder, out_prefix, voxel_size):
    """Retourne un _IOFiles, ou None si la sortie existe déjà (cas fichier unique)."""
    if not input_file_or_folder.startswith('/'):
        input_file_or_folder = join(os.path.curdir, input_file_or_folder)

    if os.path.isdir(input_file_or_folder):
        input_is_file = False
        #input_file_orig = gfile(input_file_or_folder, '.*nii')
        excluded_prefix = ('toCano_', out_prefix, 'proba_')
        input_file_orig = [f for f in gfile(input_file_or_folder, '.*nii')
                   if not os.path.basename(f).startswith(excluded_prefix)]

        if output_file_or_folder is None:
            output_file_or_folder = ''
            #output_files = addprefixtofilenames(input_files, out_prefix)
            #output_files_wanted = addprefixtofilenames(input_file_orig, out_prefix)

        if not output_file_or_folder.startswith('/'):
            output_file_or_folder = join(input_file_or_folder, f'{out_prefix}{output_file_or_folder}')
            print(f'setting output dir to {output_file_or_folder}')
        maybe_mkdir_p(output_file_or_folder)

        input_files, converted_to_canonical = convert_to_canonical_if_needed(input_file_orig, voxel_size, output_dir=output_file_or_folder)

        output_files = [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in input_files]
        output_files_wanted = [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in input_file_orig]
    else:
        input_is_file = True
        input_file_orig = [input_file_or_folder]

        if output_file_or_folder is not None:
            out_prefix += output_file_or_folder
            output_file_or_folder = os.path.join(get_parent_path(input_file_orig)[0][0],out_prefix)
            maybe_mkdir_p(output_file_or_folder)
            input_files, converted_to_canonical = convert_to_canonical_if_needed([input_file_or_folder], voxel_size,output_dir=output_file_or_folder)

            output_files =  [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in input_files]

            output_files_wanted = [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in
                                   input_file_orig]

        else:
            input_files, converted_to_canonical = convert_to_canonical_if_needed([input_file_or_folder], voxel_size)
            output_files_wanted = addprefixtofilenames(input_file_orig, out_prefix)
            output_files = addprefixtofilenames(input_files, out_prefix)
        if os.path.isfile(output_files[0]):
            print(f'Existing output file {output_files} \n Skiping')
            return None

    return _IOFiles(input_is_file, input_file_orig, input_files, converted_to_canonical,
                    output_files, output_files_wanted, out_prefix)

def _get_first_stage_paths(output_files, input_is_file, out_prefix, fname_noext):
    """Dossier intermédiaire + noms des prédictions du 1er étage (Pre_*), sans extension."""
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
# Cascade nnUNet : 2ieme étage (full res)
# ============================================================================
def _predict_cascade_fullres(predictor, model_folder, folds, input_files, output_files_first_stage,
                             output_files, dirout1, save_proba,
                             num_processes_preprocessing, num_processes_segmentation_export):
    input_files_nnunet = []
    for ff1, ff2 in zip(input_files, output_files_first_stage):
        ddn, fn = get_parent_path(ff1)
        fnnew = os.path.join(dirout1, addsuffixtofilenames(fn, '_0000'))
        previous_stage_dir, fn2 = get_parent_path(ff2)
        _, fn2 = get_parent_path(gfile(previous_stage_dir, fn2)[0])  # pour récupérer l'extension
        dd, subdir_first_stage = get_parent_path(previous_stage_dir) #because first_stage is in a subdir
        fnnew2 = os.path.join(previous_stage_dir, fn)

        if not os.path.isfile(fnnew):
            if dd==ddn:
                os.symlink(f'./{fn}', fnnew)
            else:
                os.symlink(f'../{fn}', fnnew)
        if not os.path.isfile(fnnew2):
            os.symlink(f'./{fn2}', fnnew2)
        input_files_nnunet.append([fnnew])
    print(f"runing second cascade on {len(input_files)} first input {input_files_nnunet[0]}, output {output_files[0]}")

    _run_prediction(predictor, model_folder, folds, input_files_nnunet, output_files,
                    save_probabilities=save_proba,
                    num_processes_preprocessing=num_processes_preprocessing,
                    num_processes_segmentation_export=num_processes_segmentation_export,
                    folder_with_segs_from_prev_stage=previous_stage_dir)


# ============================================================================
# Sous-régions
# ============================================================================
def _load_tissue_labels(tissue_seg_files):
    fjson = gfile(get_parent_path(tissue_seg_files)[0], 'dataset.json')[0]
    with open(fjson) as f:
        dsj = json.load(f)
    return dsj['labels']


def _get_region_names(region_model_folders):
    sub_model_name = get_parent_path(region_model_folders, 2)[1]
    region, region_dir = [], []
    for DSname in sub_model_name:
        ind_underscore = [m.start() for m in re.finditer('_', DSname)]
        region_name = DSname[ind_underscore[0] + 1:ind_underscore[1]]
        if region_name == "reg" : #new naming convention
            region_name = DSname[ind_underscore[1] + 1:ind_underscore[2]]
        if region_name=='deep': #quick bad fix for region deep_nuclei
            region_name='deep_nuclei'

        region.append(region_name)
        region_dir.append(DSname[ind_underscore[0] + 1:])
    return region, region_dir


def _prepare_region_inputs(input_files, tissue_seg_files, region, dirout_list, dic_lab,
                           crop_to_mask=True, min_voxels=1000):
    """
    Pour chaque sujet et chaque région : crée le masque de la région (depuis la seg tissu)
    et l'image croppée. Retourne :
      next_input_file[ii]  = [[t1_noext, mask_noext], ...]
      next_input_index[ii] = [indice du sujet correspondant, ...]
    """
    next_input_file = [[] for _ in region]
    next_input_index = [[] for _ in region]

    for subj_idx, (fin, fo) in enumerate(zip(input_files, tissue_seg_files)):
        fname = get_parent_path(fin)[1]
        fin = os.path.abspath(fin)
        fname_ext, fname_noext = _split_ext(fname)
        fname_noext = fname_noext if fname_noext.endswith('_0000') else fname_noext + '_0000'
        il = tio.LabelMap(fo + fname_ext)

        for ii, (reg, dirout2) in enumerate(zip(region, dirout_list)):
            if reg == 'Thal' and "Striatum" in dic_lab:  # quick hack for model 4040  -> thal contains striatum et RU
                data = (il.data == dic_lab['Thal']) | (il.data == dic_lab['Striatum']) | (il.data == dic_lab['RU'])
            else:
                data = il.data == dic_lab[reg]
            if data.sum() < min_voxels:
                print(f"Skiping region {reg} because first level mash has less than {min_voxels} voxel")
                continue

            iout = tio.LabelMap(tensor=data, affine=il.affine)
            maybe_mkdir_p(dirout2)
            fname1 = os.path.join(dirout2, fname_noext + fname_ext)
            fname2 = os.path.join(dirout2, fname_noext[:-1] + '1' + fname_ext)
            if crop_to_mask:
                suj = tio.Subject({'t1': tio.ScalarImage(fin), 'lab': iout, 'mask': iout})
                tc = tio.Compose([tio.CropOrPad(mask_name='mask'), tio.Pad(4), tio.EnsureShapeMultiple(2)])
                sujt = tc(suj)
                sujt.lab.save(fname2)
                sujt.t1.save(fname1)
            else:
                if not os.path.isfile(fname1):
                    os.symlink(fin, fname1)
                iout.save(fname2)

            next_input_file[ii].append([fname1[:-len(fname_ext)], fname2[:-len(fname_ext)]])
            next_input_index[ii].append(subj_idx)

    return next_input_file, next_input_index


def _force_region_prediction_in_mask(fo, mask_file, fname_ext):
    """Force la prédiction à ne pas être BG à l'intérieur du masque tissu."""
    im = tio.LabelMap(mask_file)
    pred_logit = np.load(fo + ".npz")
    data = torch.from_numpy(pred_logit["probabilities"].transpose([0, 3, 2, 1]))
    data[0, :] = torch.zeros(data.shape[1:])  # interdit le label 0
    seg = data.argmax(0).unsqueeze(0) * im.data  # masque avec le masque d'entrée
    iout = tio.LabelMap(fo + fname_ext)
    vol_change = (seg > 0).sum() / (iout.data > 0).sum()
    print(f'extending predicted volume of {vol_change}')  # > 1.05 : le modèle préférait le BG
    iout['data'] = seg
    iout.save(fo + fname_ext)
    os.remove(fo + ".npz")
    os.remove(fo + ".pkl")


def _reorient_region_predictions_back(output_files_reg, next_input_file_reg, next_input_index_reg,
                                      input_file_orig, converted_to_canonical, fname_ext, voxel_size):
    for fo2, fi2, subj_idx in zip(output_files_reg, next_input_file_reg, next_input_index_reg):
        if not converted_to_canonical[subj_idx]:
            continue
        print('remove cano for region predictions')
        fo1 = os.path.join(os.path.dirname(fo2), os.path.basename(fo2).replace('_toCano_', '_'))
        if fo1.endswith('_0000'):
            fo1 = fo1[:-5]
        fo1 += fname_ext
        _reorient_prediction_back(input_file_orig[subj_idx], fo2, fo1, fname_ext, voxel_size)
        print(f'removing {fi2}')
        for ffi2 in fi2:  # plusieurs contrastes en entrée
            os.remove(ffi2 + fname_ext)


def _predict_subregions(predictor, region_model_folders, region_folds, input_files, tissue_seg_files,
                        dirout1, input_file_orig, converted_to_canonical, fname_ext, voxel_size,
                        num_processes_preprocessing, num_processes_segmentation_export,
                        crop_to_mask=True, min_voxels=1000):
    """
    tissue_seg_files : segmentations tissu (sans extension) servant à définir les masques
                       (Pre_* pour le cas 2, sortie cascade full res pour le cas 4)
    """
    dic_lab = _load_tissue_labels(tissue_seg_files)
    region, region_dir = _get_region_names(region_model_folders)
    dirout_list = [os.path.join(dirout1, f'pred_{reg}') for reg in region_dir]

    next_input_file, next_input_index = _prepare_region_inputs(
        input_files, tissue_seg_files, region, dirout_list, dic_lab,
        crop_to_mask=crop_to_mask, min_voxels=min_voxels)

    output_files_regions = []
    for ii, (mfolder, efold) in enumerate(zip(region_model_folders, region_folds)):
        output_files_reg = [addprefixtofilenames(ffi[0], 'pred_')[0] for ffi in next_input_file[ii]]
        output_files_regions.append(output_files_reg)
        if len(output_files_reg) == 0:
            continue
        if os.path.isfile(output_files_reg[-1] + fname_ext):
            print(f'skiping Prediction {output_files_reg[-1]} because exist')
            continue

        _run_prediction(predictor, mfolder, efold, next_input_file[ii], output_files_reg,
                        save_probabilities=True,
                        num_processes_preprocessing=num_processes_preprocessing,
                        num_processes_segmentation_export=num_processes_segmentation_export)

        for iiff, fo in enumerate(output_files_reg):
            if os.path.isfile(fo + fname_ext) and not os.path.isfile(fo + '.npz'):
                print(f'skiping {fo} because exist and npz has been delete ')
                continue
            print(f'{ii} Merging : {fo + fname_ext}')
            _force_region_prediction_in_mask(fo, next_input_file[ii][iiff][1] + fname_ext, fname_ext)

        _reorient_region_predictions_back(output_files_reg, next_input_file[ii], next_input_index[ii],
                                          input_file_orig, converted_to_canonical, fname_ext, voxel_size)

    return dict(dic_lab=dic_lab, region=region, dirout_list=dirout_list,
                output_files_regions=output_files_regions)


def _merge_regions_into_tissue(tissue_seg_files, output_files, region_info, fname_ext):
    """Ancien bloc DOMERGE (non utilisé pour l'instant, conservé tel quel)."""
    dic_lab = region_info['dic_lab']
    region = region_info['region']
    dirout_list = region_info['dirout_list']
    output_files_regions = region_info['output_files_regions']

    ind_label, ind_region = 0, 0
    dic_region_list = []
    for one_region in dic_lab.keys():
        if one_region in region:
            fjson = gfile(dirout_list[ind_region], 'dataset.json')[0]
            with open(fjson) as f:
                dic_subregion = json.load(f)['labels']
            del dic_subregion['background']
            for k in dic_subregion.keys():
                dic_subregion[k] = ind_label
                ind_label += 1
            dic_region_list.append(dic_subregion)
            ind_region += 1
        else:
            dic_lab[one_region] = ind_label
            ind_label += 1
    tmap_no_region = tio.RemapLabels({iii: vv for iii, vv in enumerate(dic_lab.values())})
    for one_region in region:
        del dic_lab[one_region]
    for dd in dic_region_list:
        dic_lab.update(dd)
    for ii, (fi, fo) in enumerate(zip(tissue_seg_files, output_files)):
        il = tmap_no_region(tio.LabelMap(fi + fname_ext))
        for fregion, dic_map in zip(output_files_regions, dic_region_list):
            tmap = tio.RemapLabels({ind_val + 1: v for ind_val, v in enumerate(dic_map.values())})
            tc = tio.Resample(target=il)
            iregion = tc(tmap(tio.LabelMap(fregion[ii] + fname_ext)))
            mask = iregion.data > 0
            il['data'], iregion['data'] = il['data'].type(torch.ByteTensor), iregion['data'].type(torch.ByteTensor)
            il['data'][mask] = iregion['data'][mask]
        il.save(fo + fname_ext)

    dic_lab = {k: v for k, v in sorted(dic_lab.items(), key=lambda item: item[1])}
    fo_dic = get_parent_path(fo)[0] + '/dataset_region_labels.json'
    with open(fo_dic, 'w') as file:
        json.dump(dic_lab, file, indent=4, sort_keys=False)


# ============================================================================
# Post-traitements finaux
# ============================================================================
def _convert_proba_to_nifti(output_files):
    print(f'converting proba npz, in nifti')
    for fo in output_files:
        fprob1 = remove_extension(fo) + '.npz'
        fprob2 = remove_extension(fo) + '.pkl'
        if not os.path.isfile(fprob1):
            print(f'no proba file {fprob1}, skip')
            continue
        fonew = addprefixtofilenames(fo, 'proba_')
        pred_logit = np.load(fprob1)
        ddo, ffo = get_parent_path(fo)
        ffo = gfile(ddo, f'{ffo}.nii')[0]
        iref = tio.LabelMap(ffo)
        data = torch.from_numpy(pred_logit["probabilities"].transpose([0, 3, 2, 1]))
        il = tio.LabelMap(tensor=data, affine=iref.affine)
        il.save(f'{fonew[0]}.nii.gz')
        os.remove(fprob1)
        os.remove(fprob2)


def _reorient_outputs_back(io_files, output_files, fname_ext, voxel_size):
    for fi1, fi2, fo1, fo2, reorient_back in zip(io_files.input_file_orig, io_files.input_files,
                                                 io_files.output_files_wanted, output_files,
                                                 io_files.converted_to_canonical):
        if not reorient_back:
            continue
        ffo = fo2 + fname_ext + ('.gz' if len(fname_ext) == 4 else '')
        if os.path.isfile(ffo):
            print('remove cana')
            _reorient_prediction_back(fi1, fo2, fo1, fname_ext, voxel_size)
        else:  # ex : cas sous-région sans merge, pas de sortie finale
            print(f'no final prediction {ffo}, skip reorientation')
        print(f'removing {fi2}')
        os.remove(fi2)


def _cleanup_json_files(output_files, out_prefix):
    # peut échouer si deux prédictions single-file tournent dans le même dossier : pas grave
    try:
        dout = os.path.dirname(output_files[0])
        os.replace(join(dout, 'dataset.json'), join(dout, f'label_{out_prefix}.json'))
        os.remove(join(dout, 'plans.json'))
        os.remove(join(dout, 'predict_from_raw_data_args.json'))
    except Exception:
        pass

def _get_num_processes(num_process, n_files, device):
    n = max(1, min(num_process, n_files))       # inutile d'avoir plus de workers que de cas
    if device.type == 'cpu':
        n = max(1, min(n, 2))                   # le réseau utilise déjà tous les threads
    return n, n                                 # (preprocessing, export)

# ============================================================================
# Fonction principale
# ============================================================================
def nn_predict(
        input_file_or_folder: str,
        output_file_or_folder: str,
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        num_model: int = 1,
        region_regex: str = '.*',
        voxel_size=0,
        verbose: bool = False,
        num_process=8,
        save_proba=False,
        do_merge_regions=False,
):
    folder_with_parameter_files, existing_fold, out_prefix = get_model_path_and_fold(num_model, region_regex)
    if not input_file_or_folder.startswith('/'):
        input_file_or_folder = join(os.path.curdir, input_file_or_folder)

    model_type = _get_model_type(folder_with_parameter_files)
    print(f"save PROBA is {save_proba}  (model type: {model_type})")

    # ---- entrées / sorties
    io_files = _resolve_input_output(input_file_or_folder, output_file_or_folder, out_prefix, voxel_size)
    if io_files is None:
        print(f"no file found in {input_file_or_folder} ... so nothing done ....")
        return
    out_prefix = io_files.out_prefix
    input_files = io_files.input_files

    predictor = get_nn_predictor(use_tta=use_tta, device=device, verbose=verbose)

    fname = get_parent_path(input_files[0])[1]
    fname_ext, fname_noext = _split_ext(fname)
    output_files = [oo[:-len(fname_ext)] for oo in io_files.output_files]  # sans extension

    n_files = len(output_files)
    num_processes_preprocessing, num_processes_segmentation_export = _get_num_processes(num_process, n_files, device)
    print(f'num process {num_processes_preprocessing} / {num_processes_segmentation_export}  (preproc/export)')

    # ---- prédiction
    if model_type == 'simple':
        _run_prediction(predictor, folder_with_parameter_files, existing_fold,
                        [[i] for i in input_files], output_files, save_proba,
                        num_processes_preprocessing, num_processes_segmentation_export)
    else:
        dirout1, output_files_first_stage = _get_first_stage_paths(
            output_files, io_files.input_is_file, out_prefix, fname_noext)

        if model_type == 'cascade_subregion':
            (first_model, fullres_model), region_models = folder_with_parameter_files
            (first_fold, fullres_fold), region_folds = existing_fold
        elif model_type == 'cascade':
            first_model, fullres_model = folder_with_parameter_files
            first_fold, fullres_fold = existing_fold
        else:  # subregion
            first_model, region_models = folder_with_parameter_files
            first_fold, region_folds = existing_fold

        # 1er étage (tissu ou low res) -> Pre_*
        if 'cascade' in model_type: #discard save_proba
            output_files_first_stage = addprefixtofilenames(output_files_first_stage,'first_stage_pred/')
            _run_prediction(predictor, first_model, first_fold,
                            [[i] for i in input_files], output_files_first_stage, False,
                            num_processes_preprocessing, num_processes_segmentation_export)

        else:
            _run_prediction(predictor, first_model, first_fold,
                            [[i] for i in input_files], output_files_first_stage, save_proba,
                            num_processes_preprocessing, num_processes_segmentation_export)

        # 2ieme étage cascade -> output_files
        if model_type in ('cascade', 'cascade_subregion'):
            _predict_cascade_fullres(predictor, fullres_model, fullres_fold, input_files,
                                     output_files_first_stage, output_files, dirout1, save_proba,
                                     num_processes_preprocessing, num_processes_segmentation_export)

        # sous-régions
        if model_type in ('subregion', 'cascade_subregion'):
            # cas 2 : masques depuis Pre_* ; cas 4 : depuis la sortie cascade full res
            tissue_seg_files = output_files_first_stage if model_type == 'subregion' else output_files
            region_info = _predict_subregions(
                predictor, region_models, region_folds, input_files, tissue_seg_files, dirout1,
                io_files.input_file_orig, io_files.converted_to_canonical, fname_ext, voxel_size,
                                     num_processes_preprocessing, num_processes_segmentation_export)
            if do_merge_regions:
                _merge_regions_into_tissue(tissue_seg_files, output_files, region_info, fname_ext)

    # ---- post-traitements
    if save_proba:
        _convert_proba_to_nifti(output_files)

    _reorient_outputs_back(io_files, output_files, fname_ext, voxel_size)

    if io_files.input_is_file:  # sinon on laisse les json dans le dossier de sortie
        _cleanup_json_files(output_files, out_prefix)

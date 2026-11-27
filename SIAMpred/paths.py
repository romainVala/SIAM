import os,re
import os
import zipfile, tarfile
from typing import Optional

import requests
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p
from tqdm import tqdm



def install_model_from_zip_file(zip_file: str, folder_with_parameter_files):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder_with_parameter_files)

def install_model_from_tar_file(filename_tar: str, folder_with_parameter_files):
    with tarfile.open(filename_tar, "r:gz") as tar:
        tar.extractall(folder_with_parameter_files)


def download_file(url: str, local_filename: str, chunk_size: Optional[int] = 8192 * 16) -> str:
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(local_filename, 'wb'), "write", total=int(r.headers.get("Content-Length"))) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return local_filename


def maybe_download_parameters(folder_with_parameter_files, ZENODO_DOWNLOAD_URL):
    if not isfile(join(folder_with_parameter_files, 'fold_0', 'checkpoint_final.pth')):
        maybe_mkdir_p(folder_with_parameter_files)
        print(f'Downlod and install model weight in your model dir {folder_with_parameter_files}')
        print(f'Sorry the model is quite big (~5G) do not forget to delete the model folder, if no more use ')
        fname = download_file(ZENODO_DOWNLOAD_URL, join(folder_with_parameter_files, os.pardir, 'tmp_download.zip'))
        install_model_from_zip_file(fname, folder_with_parameter_files)
        os.remove(fname)

def has_checkpoint(model_folder):
    """True if model_folder/fold_0 contains a final checkpoint (.pth or .safetensors)"""
    return any(isfile(join(model_folder, 'fold_0', f'checkpoint_final{ext}')) for ext in ('.pth', '.safetensors'))

def maybe_download_parameters_tarfile(folder_with_parameter_files, DOWNLOAD_URL, extract_dir=None):
    """Download and untar the model weights if folder_with_parameter_files does not contain a checkpoint yet.
    DOWNLOAD_URL : direct link to a .tar.gz file (zenodo or hugging face 'resolve' url)
    extract_dir : where to untar the archive (default: parent dir of folder_with_parameter_files)
    """
    if has_checkpoint(folder_with_parameter_files):
        return
    if extract_dir is None:
        extract_dir = os.path.dirname(folder_with_parameter_files)
    maybe_mkdir_p(extract_dir)
    print(f'Downlod and install model weight in your model dirs {extract_dir}')
    print(f'Sorry the model is quite big (~5G) do not forget to delete the model folder, if no more use ')
    fname = download_file(DOWNLOAD_URL, join(extract_dir, os.pardir, 'tmp_download.tar.gz'))
    install_model_from_tar_file(fname, extract_dir)
    os.remove(fname)
    if not has_checkpoint(folder_with_parameter_files):
        raise FileNotFoundError(f'After extracting {DOWNLOAD_URL} in {extract_dir}, no checkpoint_final found in '
                                f'{join(folder_with_parameter_files, "fold_0")}. Check the archive structure')

def get_siam_model_dir():
    if 'SIAM_MODEL_DIR' in os.environ:
        model_dir = os.environ['SIAM_MODEL_DIR']
        if os.path.isdir(model_dir):
            print(f'Selecting model dir from env variable SIAM_MODEL_DIR ')
        else:
            raise ValueError(f'Error the environement variable SIAM_MODEL_DIR point to a no existing dir: {model_dir}')
    elif 'nnUNet_results' in os.environ: #just an other (natural option to quickly test fresh trained model)
        print(f'Selecting model dir from env variable nnUNet_results ')
        model_dir = os.environ.get('nnUNet_results')

    else:
        model_dir = os.path.join(os.path.expanduser('~'), 'siam_params')
    return model_dir

# ============================================================================
# v0.4 model : one file per model / fold on hugging face, only the needed ones are downloaded
# ============================================================================
HF_REPO_V04 = 'RomainValabregue/siam-model-v04'
V04_DIR = 'siam_model_v04'  # local folder (in the siam model dir) = root of the hugging face repo
V04_LOWRES = 'Dataset782_Ano4DS_vAllen_nolesP2/nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_lowres'
V04_FULLRES = 'Dataset782_Ano4DS_vAllen_nolesP2/nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_cascade_fullres'
V04_REGION_TRAINER = 'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'
REPO_LISTING_FILE = 'hf_repo_files.txt'  # cached listing of the repo files (delete it to refresh)


def parse_region_list(region_arg):
    """-r option : 'CSF,GM_allen,WM' -> ['CSF', 'GM_allen', 'WM'] ; 'all' -> 'all' (all regions) ; 'none' -> []"""
    if region_arg is None or (isinstance(region_arg, str) and region_arg.strip().lower() == 'all'):
        return 'all'
    if isinstance(region_arg, (list, tuple)):
        names = [str(r).strip() for r in region_arg]
    else:
        names = [r.strip() for r in region_arg.split(',')]
    names = [r for r in names if r]
    if len(names) == 0 or [n.lower() for n in names] == ['all']:
        return 'all'
    if [n.lower() for n in names] == ['none']:
        return []
    return names


def region_list_to_regex(regions):
    """for the local (non hugging face) models, still selected with a regex on the dataset folder names"""
    if regions == 'all':
        return '.*'
    if len(regions) == 0:
        return '(?!)'  # matches nothing
    return '(' + '|'.join(re.escape(r) for r in regions) + ')'


def region_name_from_dataset(dataset_name):
    """'Dataset7824_reg_GM_allen_P4' -> 'GM_allen', 'Dataset786_Lesion_P2' -> 'Lesion'
    (the '_vascular' suffix is removed in region_names, unless ambiguous : CSF, WM ... but GM_vascular)"""
    name = dataset_name.split('_', 1)[1] if '_' in dataset_name else dataset_name
    name = name.removeprefix('reg_')
    return re.sub(r'_P\d+$', '', name)  # training version suffix


REGION_ALIASES = {'GM_AssN': 'GM_vascular'}  # GM sub-regions from AssemblyNet


def region_names(dataset_names):
    """{short name: dataset name} : CSF, GM_vascular, GM_allen, WM, Cereb, deep_nuclei, Thal, Head, Skull, Lesion"""
    names = [region_name_from_dataset(d) for d in dataset_names]
    out = {}
    for d, n in zip(dataset_names, names):
        sn = n.removesuffix('_vascular')
        # keep '_vascular' if an other model of the same structure exists (GM_vascular / GM_allen)
        ambiguous = any(o != n and (o == sn or o.startswith(sn + '_')) for o in names)
        out[n if ambiguous else sn] = d
    return out


def _list_local_files(local_root):
    files = []
    for dp, dns, fns in os.walk(local_root):
        dns[:] = [d for d in dns if not d.startswith('.')]  # skip .cache (hugging face metadata)
        files += [os.path.relpath(join(dp, f), local_root).replace(os.sep, '/') for f in fns]
    return [f for f in files if f != REPO_LISTING_FILE]


def list_hf_repo_files(repo_id, local_root, download=True):
    """Files of the hugging face repo. The listing is cached in local_root/hf_repo_files.txt so that
    nothing is asked to hugging face once the needed files are there (compute nodes without internet).
    Fallback (no cache, no internet or download=False) : the files already in local_root."""
    cache = join(local_root, REPO_LISTING_FILE)
    if isfile(cache):
        with open(cache) as f:
            return [l.strip() for l in f if l.strip()]
    if download and os.environ.get('HF_HUB_OFFLINE', '0') in ('0', '', 'false', 'False'):
        try:
            from huggingface_hub import HfApi
            files = HfApi().list_repo_files(repo_id)
            maybe_mkdir_p(local_root)
            with open(cache, 'w') as f:
                f.write('\n'.join(files) + '\n')
            return files
        except Exception as e:
            print(f'WARNING cannot list the files of {repo_id} on hugging face ({e}), using the local files only')
    return _list_local_files(local_root)


def model_files_to_download(repo_files, model_dir, nb_fold=0):
    """json files of a model folder + checkpoint of the first nb_fold folds (all folds if nb_fold=0)"""
    prefix = model_dir.rstrip('/') + '/'
    jsons = [f for f in repo_files if f.startswith(prefix) and '/' not in f[len(prefix):] and f.endswith('.json')]
    folds = sorted({int(m.group(1)) for f in repo_files
                    if (m := re.match(re.escape(prefix) + r'fold_(\d+)/checkpoint_(final|best)\.', f))})
    if nb_fold:
        folds = folds[:nb_fold]
    ckpts = []
    for k in folds:
        fk = [f for f in repo_files if f.startswith(f'{prefix}fold_{k}/checkpoint_final.')]
        if len(fk) == 0:
            fk = [f for f in repo_files if f.startswith(f'{prefix}fold_{k}/checkpoint_best.')]
        ckpts += fk
    return jsons + ckpts


def download_hf_files(repo_id, files, local_root):
    """download (from hugging face) the files missing in local_root"""
    missing = [f for f in files if not isfile(join(local_root, f))]
    if len(missing) == 0:
        return
    from huggingface_hub import hf_hub_download
    print(f'Downloading {len(missing)} model files from hugging face ({repo_id}) into {local_root}')
    print('(delete the model folder if no more used, the full v0.4 model is ~20G)')
    for f in missing:
        hf_hub_download(repo_id, f, local_dir=local_root)


def get_v04_model_folders(siam_model_dir, regions='all', nb_fold=0, lowres=False, download=True):
    """
    regions : list of region names ('all' : all regions, [] : no region)
    lowres  : only the low res (1mm) tissue model, no region
    nb_fold : only the first nb_fold folds of each model (0 : all)
    Only the needed files are downloaded.
    """
    local_root = os.path.join(siam_model_dir, V04_DIR)
    repo_files = list_hf_repo_files(HF_REPO_V04, local_root, download=download)

    if lowres:
        if regions != 'all' and len(regions) > 0:
            print(f'WARNING --lowres : tissue only, the regions {regions} are ignored')
        tissue_models, region_models = [V04_LOWRES], []
    else:
        tissue_models = [V04_LOWRES, V04_FULLRES]
        # region models : regions/<Dataset>/<trainer>/
        region_ds = sorted({f.split('/')[1] for f in repo_files
                            if f.startswith('regions/') and f.split('/')[2:3] == [V04_REGION_TRAINER]})
        available = region_names(region_ds)
        if regions == 'all':
            selected = region_ds
        else:
            lower = {k.lower(): v for k, v in available.items()}
            lower.update({a.lower(): available[t] for a, t in REGION_ALIASES.items() if t in available})
            unknown = [r for r in regions if r.lower() not in lower]
            if len(unknown) > 0:
                shown = {t: a for a, t in REGION_ALIASES.items()}  # GM_vascular shown as GM_AssN
                raise ValueError(f'Unknown region(s) {unknown}. Available regions are : '
                                 f'{",".join(shown.get(k, k) for k in available)}')
            selected = list(dict.fromkeys(lower[r.lower()] for r in regions))  # no duplicates
        region_models = [f'regions/{ds}/{V04_REGION_TRAINER}' for ds in selected]

    if download:
        to_download = []
        for m in tissue_models + region_models:
            fm = model_files_to_download(repo_files, m, nb_fold)
            if not any('/fold_' in f for f in fm):
                raise FileNotFoundError(f'no checkpoint found for model {m} in {HF_REPO_V04}')
            to_download += fm
        download_hf_files(HF_REPO_V04, to_download, local_root)

    tissue = [os.path.join(local_root, m) for m in tissue_models]
    tissue = tissue[0] if lowres else tissue
    if lowres or (regions != 'all' and len(region_models) == 0):
        return tissue  # 'simple' (low res) or 'cascade' (no region)
    return [tissue, [os.path.join(local_root, m) for m in region_models]]


def get_model_path_and_fold(num_model: int, regions='all', nb_fold=0, lowres=False, download=True):
    """
    regions : list of region names ('all' : all regions, [] : no region)
    lowres  : (model 4 only) only the low res tissue model
    download: if False, nothing is downloaded (used by regroup_regions)
    """
    siam_model_dir = get_siam_model_dir()
    region_regex = region_list_to_regex(regions)
    if lowres and num_model != 4:
        raise ValueError('--lowres is only available for model 4 (-m 4)')

    if num_model==4: #next release region predictions, from hugging face
        res_folder = get_v04_model_folders(siam_model_dir, regions=regions, nb_fold=nb_fold,
                                           lowres=lowres, download=download)
        out_prefix = 'siamV04'

    elif num_model==0: #get it from zenodo
        # the link where to get zip version of the models
        ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/17733431/files/pred_DS108_LcsfP_Ano.tar.gz?download=1'
        res_folder = os.path.join(siam_model_dir, 'v0.3','pred_DS108_LcsfP_Ano')
        out_prefix = 'siamV03_'
        maybe_download_parameters_tarfile(res_folder, ZENODO_DOWNLOAD_URL)

    elif num_model==2: #get it from zenodo
        # the link where to get zip version of the models
        ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/15780983/files/DS715_NODA.tar.gz?download=1' # https://doi.org/10.5281/zenodo.15780983
        res_folder = os.path.join(siam_model_dir, 'v0.2','DS715_NODA')

        out_prefix = 'siamV02_'
        maybe_download_parameters_tarfile(res_folder, ZENODO_DOWNLOAD_URL)
    elif num_model==1: #get it from zenodo
        # the link where to get zip version of the models
        ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/15055596/files/model_708.zip?download=1'
        res_folder = os.path.join(siam_model_dir, 'v0.1')

        out_prefix = 'siamV01_'
        maybe_download_parameters(res_folder, ZENODO_DOWNLOAD_URL)

    else:

        if num_model == 111:
            res_folder = os.path.join(siam_model_dir, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred_DS710_3ResEncL_'

        elif num_model == 1:
            res_folder = os.path.join(siam_model_dir, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS710_5ResEncXL_'

        elif num_model == 102:
            res_folder = os.path.join(siam_model_dir, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetPlans_50G__3d_fullres')
            out_prefix = 'pred_DS710_5nn2Pass_'

        elif num_model == 2:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(siam_model_dir, 'Dataset709_Vasc2suj_v3',
                                      'nnUNetTrainer__nnUNetPlans_50G__3d_fullres_first_pass')
            out_prefix = 'pred_DS709_3nnP50G_'

        elif num_model == 9:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = [os.path.join(siam_model_dir, 'Dataset709_Vasc2suj_v3',
                                       'nnUNetTrainer_onlyMirror01__nnUNetResEncUNetLPlans__3d_lowres'),
                          os.path.join(siam_model_dir, 'Dataset709_Vasc2suj_v3',
                                      'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_cascade_fullres') ]
            out_prefix = 'pred_DS709_CascadeNoDA_'

        elif num_model == 8:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(siam_model_dir, 'Dataset708_Ultra_SkulVasc40',
                                      'model_rXL708_v0.1')
            out_prefix = 'pred_DS708_5nnResXL_'

        elif num_model == 4:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(siam_model_dir, 'Dataset706_Vasc2suj_l22_v1',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS706_5nnResXL_'

        elif num_model == 5:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(siam_model_dir, 'Dataset704_Uhcp_skv52_mot_elaBig',
                                      'nnUNetTrainer__nnUNetPlannerResEncXL_80G__3d_fullres')
            out_prefix = 'pred_DS704_3nnResXL_'

        elif num_model ==88: #8
            res_folder = os.path.join(siam_model_dir, 'Dataset718_Ultra_SkulVasc40_TumorBrast', 'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred_DS718_5nnResL'

        elif num_model ==888:
            res_folder = os.path.join(siam_model_dir, 'Dataset718_Ultra_SkulVasc40_TumorBrast', 'nnUNetTrainer__nnUNetResEncUNetXXLPlans__3d_fullres')
            out_prefix = 'pred_DS718_5nnResXXL'

        elif num_model ==12:
            res_folder = os.path.join(siam_model_dir, 'Dataset712_Vasc2suj_v3_Few', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS712_5ResXL'

        elif num_model ==122:
            res_folder = os.path.join(siam_model_dir, 'Dataset712_Vasc2suj_v3_Few', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS712_NODA3ResXL'

        elif num_model ==1212:
            res_folder = os.path.join(siam_model_dir, 'Dataset712_Vasc2suj_v3_Few', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS712_SubR_'
            sub_region = gdir(siam_model_dir,['Dataset712[0123456789]','nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans'])
            res_folder = [res_folder, sub_region]

        elif num_model ==14:
            res_folder = os.path.join(siam_model_dir, 'Dataset714_MidaSuj3_RFew', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS714_5ResXL'

        elif num_model ==1414:
            res_folder = os.path.join(siam_model_dir, 'Dataset714_MidaSuj3_RFew', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS714_SubR_'
            sub_region = gdir(siam_model_dir,['Dataset714[0123456789]','nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans'])
            res_folder = [res_folder, sub_region]

        elif num_model ==13:
            res_folder = os.path.join(siam_model_dir, 'Dataset713_MidaSuj1', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS713_3ResXL'

        elif num_model ==15:
            res_folder = os.path.join(siam_model_dir, 'Dataset715_MixSuj6', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS715_NODA3ResXL'
        elif num_model ==152:#do not exist (bad Skull)
            res_folder = os.path.join(siam_model_dir, 'Dataset715_MixSuj6', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS715_3ResXL'
        elif num_model ==16:
            res_folder = os.path.join(siam_model_dir, 'Dataset716_MixLowDill', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS716_NODA'
        elif num_model ==17:
            res_folder = os.path.join(siam_model_dir, 'Dataset717_MixLowDill_Ano', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS717_Ano'
            
        elif num_model == 101:
            # XXX ines tumor + ms + 716 train
            res_folder = os.path.join(siam_model_dir, 'Dataset1001_ms_tumor_MixLowDill', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS101_ms_tumor_MixLowDill'
        elif num_model == 102:
            # XXX ines (tumor + ms) without contrast constraint + 716 train : probleme plan normale pas XL
            res_folder = os.path.join(siam_model_dir, 'Dataset1002_ms_tumor_WC_MixLowDill', 'nnUNetTrainer__nnUNetPlans__3d_fullres')
            out_prefix = "pred_DS102_ms_tumor_MixLowDill"
        elif num_model == 106:
            # XXX ines healthy with GM and cerb CSF push
            res_folder = os.path.join(siam_model_dir, 'Dataset1006_Partialization_MixLowDill',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = "pred_DS106_MixLow_csfPush"
        elif num_model == 107:
            # XXX ines 33% healthy with GM and cerb CSF push, 33% ms and 33% of tumor
            res_folder = os.path.join(siam_model_dir, 'Dataset1007_Partialization_MixLowDill',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = "pred_DS107_MixLow_csfPush_ms_tumor"
        elif num_model == 108:
            # same as 107 ep4000
            res_folder = os.path.join(siam_model_dir, 'Dataset1007_Partialization_MixLowDill', 'pred_DS108_LcsfP_Ano')
            out_prefix = 'pred_DS108_LcsfP_Ano'
        elif num_model == 109:
            # same as 107 ep4000
            res_folder = os.path.join(siam_model_dir, 'Dataset1007_Partialization_MixLowDill', 'nnUNetTrainerNoDA4000__nnUNetResEncUNetXLPlans__3d_fullres_average')
            out_prefix = 'pred_DS109_LcsfP_Ano'
        elif num_model == 999:
            res_folder = os.path.join(siam_model_dir, 'Dataset005_SN', 'nnUNetTrainer__nnUNetPlans__3d_fullres/')
            out_prefix = 'siamV03_'
        elif num_model ==20:
            res_folder = os.path.join(siam_model_dir, 'Dataset720_V4_nDS5', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS720_AnoV4_nDS5'
        elif num_model ==24:
            res_folder = os.path.join(siam_model_dir, 'Dataset724_allen_V4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS724_AnoV4_allen'
        elif num_model ==25:
            res_folder = os.path.join(siam_model_dir, 'Dataset725_bigbrain_V4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS725_AnoV4_bigbrain'

        elif num_model ==2020:
            res_folder = os.path.join(siam_model_dir, 'Dataset720_V4_nDS5', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS720_RegAllen'
            sub_region = gdir(gdir(siam_model_dir,'region_jzay'),['allen_V4','nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'])
            res_folder = [res_folder, sub_region]
        elif num_model ==2021:
            res_folder = os.path.join(siam_model_dir, 'Dataset720_V4_nDS5', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS720_RegVasc'
            sub_region = gdir(gdir(siam_model_dir,'region_jzay'),['vascular_V4','nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'])
            res_folder = [res_folder, sub_region]

        elif num_model ==2420:
            res_folder = os.path.join(siam_model_dir, 'Dataset724_allen_V4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS724_RegAllen'
            sub_region = gdir(gdir(siam_model_dir,'region_jzay'),['allen_V4','nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'])
            res_folder = [res_folder, sub_region]
        elif num_model ==30:
            res_folder = os.path.join(siam_model_dir, 'Dataset730_V42_nDS4', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS730_nDS4'
        elif num_model ==40:
            res_folder = os.path.join(siam_model_dir, 'Dataset740_V42_nDS4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS740_2KnDS4'
        elif num_model ==41:
            res_folder = os.path.join(siam_model_dir, 'Dataset740_V42_nDS4', 'nnUNetTrainerNoDA4000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS741_3KnDS4'
        elif num_model ==4040:
            res_folder = os.path.join(siam_model_dir, 'Dataset740_V42_nDS4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS740_Reg'
            sub_region = gdir(gdir(siam_model_dir,'region_jzayV42'),[region_regex,'nnUNetTrainerNoDA__nnUNetResEncUNet.*'])
            res_folder = [res_folder, sub_region]
        elif num_model ==4041:
            res_folder = os.path.join(siam_model_dir, 'Dataset740_V42_nDS4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS740_RegAllenSkull'
            sub_region = gdir(gdir(siam_model_dir,'region_jzayV42'),['(allen|skull)_V42','nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'])
            res_folder = [res_folder, sub_region]

        elif num_model ==50:
            #res_folder = os.path.join(siam_model_dir, 'Dataset750_AnoV42_nsD4', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            #out_prefix = 'pred_DS750_1KnDS4'
            res_folder = os.path.join(siam_model_dir, 'Dataset750_AnoV42_nsD4', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS750_Les_2KDS4'

        elif num_model ==60:
            res_folder = os.path.join(siam_model_dir, 'Dataset760_AnoV42_nsD4_and_720', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS760_Mix50_20_Ano'
        elif num_model ==70:
            res_folder = os.path.join(siam_model_dir, 'Dataset770_mergeNoLes', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS770_NoL_Mix50_202K'
        elif num_model ==71:
            res_folder = os.path.join(siam_model_dir, 'Dataset770_mergeNoLes', 'nnUNetTrainerNoDA2000__nnUNetResEncUNetXLPlans__3d_fullresinit720')
            out_prefix = 'pred_DS770_NoL_Mix20_502K'

        elif num_model == 610:
            res_folder = os.path.join(siam_model_dir, 'Dataset610_3DS_noAugment', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS610_nnAug3DS'

        elif num_model == 33:
            res_folder = os.path.join(siam_model_dir, '..','insightec_les','results', 'Dataset333_lesionHIFU_T1w', 'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred33_LesT1'

        elif num_model == 82:
            res_folder = [os.path.join(siam_model_dir, 'Dataset782_Ano4DS_vAllen_nolesP2',
                                       'nnUNetTrainerNoDA2000__nnUNetResEncUNetLPlans__3d_lowres'),
                          os.path.join(siam_model_dir, 'Dataset782_Ano4DS_vAllen_nolesP2',
                                       'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_cascade_fullres')
                          ]
            out_prefix = 'pred_DS782_lAtis_cas05_regions' #pred_DS782_TNoLes_cas05_regions
        elif num_model == 8282:
            res_folder = [os.path.join(siam_model_dir, 'Dataset782_Ano4DS_vAllen_nolesP2',
                                       'nnUNetTrainerNoDA2000__nnUNetResEncUNetLPlans__3d_lowres'),
                          os.path.join(siam_model_dir, 'Dataset782_Ano4DS_vAllen_nolesP2',
                                       'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_cascade_fullres')
                          ]
            sub_region = gdir(gdir(siam_model_dir,'region82_jzay'),[region_regex,'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres'])
            res_folder = [res_folder, sub_region]
            out_prefix = 'pred_DS782_Tis05_regions'

        elif num_model == 80:
            res_folder = os.path.join(siam_model_dir, 'Dataset782_Ano4DS_vAllen_nolesP2',
                                      'nnUNetTrainerNoDA2000__nnUNetResEncUNetLPlans__3d_lowres')
            out_prefix = 'pred_DS780_lAtis_lowres'

        elif num_model == 83:
            res_folder = [os.path.join(siam_model_dir, 'Dataset783_Ano4DS_vAllen_LesionP2',
                                      'nnUNetTrainerNoDA2000__nnUNetResEncUNetLPlans__3d_lowres'),
                          os.path.join(siam_model_dir, 'Dataset783_Ano4DS_vAllen_LesionP2',
                                       'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_cascade_fullres')
                          ]
            out_prefix = 'pred_DS783_Tles_cas05'
        elif num_model == 84:
            res_folder = os.path.join(siam_model_dir, 'Dataset784_Ano4DS_vAllen_nolesP3',
                                      'nnUNetTrainerNoDA2000__nnUNetResEncUNetLPlans__3d_fullres$')
            out_prefix = 'pred_DS784_Tis075'

        else:
            raise(f'Sorry no model file in {siam_model_dir}  ')



    if isinstance(res_folder,list): #subregion or cascade
        existing_fold = []
        for rrr in res_folder:
            if isinstance(rrr,list): #subregion case
                existing_fold.append([get_fold_list(rrrr,nb_fold) for rrrr in rrr ])
            else:
                existing_fold.append( get_fold_list(rrr,nb_fold) )
        if isinstance(res_folder[1],list):
            print(f"Selecting model {num_model} with {len(res_folder[1])} subreg")
        else :
            print(f"Slecting casscade, model {num_model}")

    else:
        existing_fold = get_fold_list(res_folder,nb_fold)

        print(f'Selecting model {num_model} out Prefix {out_prefix} and found {len(existing_fold)} FOLD ')
        print(f'model folder is {res_folder}')

    return res_folder, existing_fold, out_prefix


def get_fold_list(res, nb_fold=0):
    existing_fold = gdir(res,'fold')
    if len(existing_fold) == 0:  # not downloaded (regroup_regions)
        return []
    if nb_fold:
        existing_fold = existing_fold[:nb_fold]
    model_files = gfile(existing_fold, 'checkpoint_final') #.pth
    if len(model_files)==0:
        model_files = gfile(existing_fold, 'checkpoint_best') #extension can be now .pth or .safetensors
        print(f'get_mode{model_files}')
    existing_fold = get_parent_path(model_files)[0] #selec only dir with checkpoint_final
    return [int(ss[-1]) for ss in existing_fold]



#files and dir utils
def gdir(dirs, regex, verbose=False):
    """ get sudirs from dirs depending in the regular expression
    dirs is a list of input directories
    regex is a list of regular expresions. Each item of the list is a subdirectory
    being the last regex the one refering to the filename
    items is the maximum number of items for each folder

    Items are sorted in each file by alphabetical order
    """
    # check inputs
    if isinstance(dirs, str):
        dirs = [dirs]
    elif len(dirs) == 0:
        print(" ** NO directories found!!")
        return []
    if isinstance(regex, str):
        regex = [regex]

    # search subdir levels
    if len(regex) == 1:
        finaldirs = []

        # this is the final level
        comp = re.compile(regex[0])
        # print " - Compiled "+regex[0]
        for d in dirs:
            d = os.path.abspath(d)
            if not os.path.isdir(d):
                continue

            files = os.listdir(d)
            files.sort()

            for f in files:
                # print " file "+f
                ff = d + os.sep + f
                if not os.path.isdir(ff):
                    continue
                if comp.search(f) is None:
                    continue

                # print " -- Found: "+ff
                finaldirs.append(ff)

        return finaldirs
    else:
        # decomposing recursively
        finaldirs = dirs
        for r in regex:
            finaldirs = gdir(finaldirs, r)

        if verbose:
            for d in finaldirs:
                print(d)

        return finaldirs

def gfile(dirs, regex, opts={"items": -1}, list_flaten=True):
    """ get files from dirs depending in the regular expression
    dirs: is a list of input directories
    regex: is a list of regular expresions. Each item of the list is a subdirectory
        being the last regex the one refering to the filename
    items: is the maximum number of items for each folder

        Items are sorted in each file by alphabetical order
    """

    # check inputs
    if isinstance(dirs, str):
        dirs = [dirs]
    elif len(dirs) == 0:
        print(" ** Error: No dirs found!!")
        return []

    if isinstance(regex, str):
        regex = [regex]

    # extracting options
    verbose = False
    if "verbose" in opts and opts["verbose"] == True:
        verbose = True

    items = -1
    if "items" in opts:
        items = int(opts["items"])

    # search subdir levels
    if len(regex) == 1:
        finaldirs = []

        # this is the final level
        comp = re.compile(regex[0])
        for d in dirs:
            d = os.path.abspath(d)
            if not os.path.isdir(d):
                continue

            files = os.listdir(d)
            files.sort()
            files_in_one_dir = []
            i = 0
            for f in files:
                # print " file "+f
                ff = d + os.sep + f
                # if not os.path.isdir(ff):
                # continue
                if comp.search(f) is None:
                    continue
                if list_flaten:
                    finaldirs.append(ff)
                else:
                    files_in_one_dir.append(ff)
                i = i + 1

            if items > 0 and i != items:
                print("WARNING found %d item and not %s in %s" % (i, items, d))
            if list_flaten is False:
                finaldirs.append(files_in_one_dir)

        return finaldirs
    else:
        # decomposing recursively
        finaldirs = dirs
        for r in range(len(regex) - 1):
            finaldirs = gdir(finaldirs, regex[r], opts)
        finaldirs = gfile(finaldirs, regex[-1], opts)
        return finaldirs

def addprefixtofilenames(file_names,prefix):
    fout = []
    if isinstance(file_names, str):
        file_names = [file_names]
    for ff in file_names:
        basdir = os.path.dirname(ff)
        fn = os.path.basename(ff)
        fout.append(basdir + os.sep + prefix + fn)
    return fout

def get_parent_path(fin,level=1, remove_ext=False):

    return_string=False
    if isinstance(fin, str):
        fin = [fin]
        return_string=True

    path_name, file_name  = [], []
    concat = False
    if level <0:
        level = -level
        concat = True

    for ff in fin:
        if ff[-1]== os.sep : 
            ff=ff[:-1]
        dd = ff.split(os.sep)
        ll = len(dd)
        if remove_ext:
            dd[-1] = remove_extension(dd[-1])
        if concat:
            ss='_'.join(dd[ll-level:])

            file_name.append(ss)
        else:
            file_name.append(dd[ll-level])
        path_name.append((os.sep).join(dd[:ll-level]))


    if return_string:
        return path_name[0], file_name[0]
    else:
        return path_name, file_name

def remove_extension(str_in,retrun_ext=False):
    return_str = False
    second_extention_list = ['.nii', '.pt']
    first_extention_list = ['.gz']
    if isinstance(str_in, str):
        str_in = [str_in];
        return_str = True
    #remove up to 2 extension
    #res = [os.path.splitext(os.path.splitext(ss)[0])[0] for ss in str_in]
    res,ext = [], []
    for ss in str_in:
        rr1, ext1 = os.path.splitext(ss)
        rr2, ext2 = os.path.splitext(rr1)
        if (ext2 in second_extention_list) | (ext1 in first_extention_list):
            res.append(rr2)
            ext.append(ext2+ext1)
        else:
            res.append(rr1)
            ext.append(ext1)
    #res = [os.path.splitext(ss)[0] for ss in str_in]  #one shot
    if retrun_ext:
        if return_str:
            return res[0],ext[0]
        else:
            return res,ext

    return res[0] if return_str else res

def addsuffixtofilenames(file_names,suffix):
    fout = []
    return_str = False
    if isinstance(file_names, str):
        file_names = [file_names]
        return_str = True

    for ff in file_names:
        basdir = os.path.dirname(ff)
        fn = os.path.basename(ff)
        fn,ex = remove_extension(fn,retrun_ext=True)
        fout.append(os.path.join(basdir , fn + suffix + ex))
    if return_str:
        return fout[0]
    return fout

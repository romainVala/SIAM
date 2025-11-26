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
        print(f'Downlod and install model weight in your home {folder_with_parameter_files}')
        print(f'Sorry the model is quite big (~5G) do not forget to delete the model folder, if no more use ')
        fname = download_file(ZENODO_DOWNLOAD_URL, join(folder_with_parameter_files, os.pardir, 'tmp_download.zip'))
        install_model_from_zip_file(fname, folder_with_parameter_files)
        os.remove(fname)

def maybe_download_parameters_tarfile(folder_with_parameter_files, ZENODO_DOWNLOAD_URL):
    if not isfile(join(folder_with_parameter_files, 'fold_0', 'checkpoint_final.pth')):
        folder_with_parameter_files = os.path.dirname(folder_with_parameter_files)
        maybe_mkdir_p(folder_with_parameter_files)
        print(f'Downlod and install model weight in your home {folder_with_parameter_files}')
        print(f'Sorry the model is quite big (~4G) do not forget to delete the model folder, if no more use ')
        fname = download_file(ZENODO_DOWNLOAD_URL, join(folder_with_parameter_files, os.pardir, 'tmp_download.tar.gz'))
        install_model_from_tar_file(fname, folder_with_parameter_files)
        os.remove(fname)

def get_siam_model_dir():
    if 'SIAM_MODEL_DIR' in os.environ:
        model_dir = os.environ['SIAM_MODEL_DIR']
        if os.path.isdir(model_dir):
            print(f"model parameter from SIAM_MODEL_DIR env variable {model_dir}")
        else:
            raise ValueError(f'Error the environement variable SIAM_MODEL_DIR point to a no existing dir: {model_dir}')
    else:
        model_dir = os.path.join(os.path.expanduser('~'), 'siam_params')

    return model_dir

def get_model_path_and_fold(num_model:int ):
    if num_model==-1: #get it from zenodo
        # the link where to get zip version of the models
        ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/15055596/files/model_708.zip?download=1'
        siam_model_dir = get_siam_model_dir()
        res_folder = os.path.join(siam_model_dir, 'v0.1')

        out_prefix = 'siamV01_'
        maybe_download_parameters(res_folder, ZENODO_DOWNLOAD_URL)

    elif num_model==0: #get it from zenodo
        # the link where to get zip version of the models
        ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/15780983/files/DS715_NODA.tar.gz?download=1' # https://doi.org/10.5281/zenodo.15780983

        siam_model_dir = get_siam_model_dir()
        res_folder = os.path.join(siam_model_dir, 'v0.2','DS715_NODA')

        out_prefix = 'siamV02_'
        maybe_download_parameters_tarfile(res_folder, ZENODO_DOWNLOAD_URL)

    else:

        nnres_path = os.environ.get('nnUNet_results')
        if nnres_path is None:
            print(f'you choose num_model {num_model} For local modem (-m num  (with num>0) you need to specify the environement varaible  nnUNet_results')
            raise('rrr')

        if num_model == 111:
            res_folder = os.path.join(nnres_path, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred_DS710_3ResEncL_'

        elif num_model == 1:
            res_folder = os.path.join(nnres_path, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS710_5ResEncXL_'

        elif num_model == 102:
            res_folder = os.path.join(nnres_path, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetPlans_50G__3d_fullres')
            out_prefix = 'pred_DS710_5nn2Pass_'

        elif num_model == 2:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset709_Vasc2suj_v3',
                                      'nnUNetTrainer__nnUNetPlans_50G__3d_fullres_first_pass')
            out_prefix = 'pred_DS709_3nnP50G_'

        elif num_model == 9:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = [os.path.join(nnres_path, 'Dataset709_Vasc2suj_v3',
                                       'nnUNetTrainer_onlyMirror01__nnUNetResEncUNetLPlans__3d_lowres'),
                          os.path.join(nnres_path, 'Dataset709_Vasc2suj_v3',
                                      'nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_cascade_fullres') ]
            out_prefix = 'pred_DS709_CascadeNoDA_'

        elif num_model == 3:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset708_Ultra_SkulVasc40',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS708_5nnResXL_'

        elif num_model == 4:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset706_Vasc2suj_l22_v1',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS706_5nnResXL_'

        elif num_model == 5:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset704_Uhcp_skv52_mot_elaBig',
                                      'nnUNetTrainer__nnUNetPlannerResEncXL_80G__3d_fullres')
            out_prefix = 'pred_DS704_3nnResXL_'

        elif num_model ==8:
            res_folder = os.path.join(nnres_path, 'Dataset718_Ultra_SkulVasc40_TumorBrast', 'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred_DS718_5nnResL'

        elif num_model ==88:
            res_folder = os.path.join(nnres_path, 'Dataset718_Ultra_SkulVasc40_TumorBrast', 'nnUNetTrainer__nnUNetResEncUNetXXLPlans__3d_fullres')
            out_prefix = 'pred_DS718_5nnResXXL'

        elif num_model ==12:
            res_folder = os.path.join(nnres_path, 'Dataset712_Vasc2suj_v3_Few', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS712_5ResXL'

        elif num_model ==122:
            res_folder = os.path.join(nnres_path, 'Dataset712_Vasc2suj_v3_Few', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS712_NODA3ResXL'

        elif num_model ==1212:
            res_folder = os.path.join(nnres_path, 'Dataset712_Vasc2suj_v3_Few', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS712_SubR_'
            sub_region = gdir(nnres_path,['Dataset712[0123456789]','nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans'])
            res_folder = [res_folder, sub_region]

        elif num_model ==14:
            res_folder = os.path.join(nnres_path, 'Dataset714_MidaSuj3_RFew', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS714_5ResXL'

        elif num_model ==1414:
            res_folder = os.path.join(nnres_path, 'Dataset714_MidaSuj3_RFew', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS714_SubR_'
            sub_region = gdir(nnres_path,['Dataset714[0123456789]','nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans'])
            res_folder = [res_folder, sub_region]

        elif num_model ==13:
            res_folder = os.path.join(nnres_path, 'Dataset713_MidaSuj1', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS713_3ResXL'

        elif num_model ==15:
            res_folder = os.path.join(nnres_path, 'Dataset715_MixSuj6', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS715_NODA3ResXL'
        elif num_model ==152:#do not exist (bad Skull)
            res_folder = os.path.join(nnres_path, 'Dataset715_MixSuj6', 'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS715_3ResXL'
        elif num_model ==16:
            res_folder = os.path.join(nnres_path, 'Dataset716_MixLowDill', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS716_NODA'
        elif num_model ==17:
            res_folder = os.path.join(nnres_path, 'Dataset717_MixLowDill_Ano', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS717_Ano'
            
        elif num_model == 101:
            # XXX ines tumor + ms + 716 train
            ines_nnunet_res = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/Results/"
            res_folder = os.path.join(ines_nnunet_res, 'Dataset1001_ms_tumor_MixLowDill', 'nnUNetTrainerNoDA__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS101_ms_tumor_MixLowDill'
        elif num_model == 102:
            # XXX ines (tumor + ms) without contrast constraint + 716 train : probleme plan normale pas XL
            ines_nnunet_res = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/Results/"
            res_folder = os.path.join(ines_nnunet_res, 'Dataset1002_ms_tumor_WC_MixLowDill', 'nnUNetTrainer__nnUNetPlans__3d_fullres')
            out_prefix = "pred_DS102_ms_tumor_MixLowDill"
        elif num_model == 104:
            # XXX ines ms only without constraint : probleme plan normale pas XL
            ines_nnunet_res = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/Results/"
            res_folder = os.path.join(ines_nnunet_res, 'Dataset1004_ms_MixLowDill', 'nnUNetTrainer__nnUNetPlans__3d_fullres')
            out_prefix = "pred_DS104_ms_MixLowDill"
        elif num_model == 106:
            # XXX ines healthy with GM and cerb CSF push
            ines_nnunet_res = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/Results/"
            res_folder = os.path.join(ines_nnunet_res, 'Dataset1006_Partialization_MixLowDill',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = "pred_DS106_MixLow_csfPush"
        elif num_model == 107:
            # XXX ines 33% healthy with GM and cerb CSF push, 33% ms and 33% of tumor
            ines_nnunet_res = "/network/iss/cenir/analyse/irm/users/ines.khemir/nnunet/Results/"
            res_folder = os.path.join(ines_nnunet_res, 'Dataset1007_Partialization_MixLowDill',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = "pred_DS107_MixLow_csfPush_ms_tumor"
        elif num_model == 108:
            # same as 107 ep4000
            res_folder = os.path.join(nnres_path, 'Dataset1007_Partialization_MixLowDill', 'nnUNetTrainerNoDA4000__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS108_LcsfP_Ano'
        elif num_model == 109:
            # same as 107 ep4000
            res_folder = os.path.join(nnres_path, 'Dataset1007_Partialization_MixLowDill', 'nnUNetTrainerNoDA4000__nnUNetResEncUNetXLPlans__3d_fullres_average')
            out_prefix = 'pred_DS109_LcsfP_Ano'


    if isinstance(res_folder,list): #subregion or cascade
        existing_fold = []
        for rrr in res_folder:
            if isinstance(rrr,list): #subregion case
                existing_fold.append([get_fold_list(rrrr) for rrrr in rrr ])
            else:
                existing_fold.append( get_fold_list(rrr) )

    else:
        existing_fold = get_fold_list(res_folder)

        print(f'Selecting model {num_model} out {out_prefix} and found {len(existing_fold)} FOLD')

    return res_folder, existing_fold, out_prefix


def get_fold_list(res):
    existing_fold = gdir(res,'fold')
    model_files = gfile(existing_fold, 'checkpoint_final.pth')
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


import os,re
import os
import zipfile
from typing import Optional

import requests
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p
from tqdm import tqdm

# the link where to get zip version of the models
ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/15055596/files/model_708.zip?download=1'
folder_with_parameter_files = os.path.join(os.path.expanduser('~'), 'siam_params', 'v0.1')


def install_model_from_zip_file(zip_file: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder_with_parameter_files)


def download_file(url: str, local_filename: str, chunk_size: Optional[int] = 8192 * 16) -> str:
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=100) as r:
        r.raise_for_status()
        with tqdm.wrapattr(open(local_filename, 'wb'), "write", total=int(r.headers.get("Content-Length"))) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    return local_filename


def maybe_download_parameters():
    if not isfile(join(folder_with_parameter_files, 'fold_0', 'checkpoint_final.pth')):
        maybe_mkdir_p(folder_with_parameter_files)
        print(f'Downlod and install model weight in your home {folder_with_parameter_files}')
        print(f'Sorry the model is quite big (~4G) do not forget to delete the model folder, if no more use ')
        fname = download_file(ZENODO_DOWNLOAD_URL, join(folder_with_parameter_files, os.pardir, 'tmp_download.zip'))
        install_model_from_zip_file(fname)
        os.remove(fname)



def get_model_path_and_fold(num_model:int ):
    if num_model==0: #get it from zenodo
        out_prefix = 'siamV01_'
        maybe_download_parameters()
        res_folder = folder_with_parameter_files

    else:
        nnres_path = os.environ.get('nnUNet_results')
        if nnres_path is None:
            raise('For local modem (-m num  (with num>0) you need to specify the environement varaible  nnUNet_results')

        if num_model == 111:
            res_folder = os.path.join(nnres_path, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred_DS710_3ResEncL_'

        if num_model == 1:
            res_folder = os.path.join(nnres_path, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS710_5ResEncXL_'

        if num_model == 102:
            res_folder = os.path.join(nnres_path, 'Dataset710_Vasc2suj_v3_Region',
                                      'nnUNetTrainer__nnUNetPlans_50G__3d_fullres')
            out_prefix = 'pred_DS710_5nn2Pass_'

        if num_model == 2:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset709_Vasc2suj_v3',
                                      'nnUNetTrainer__nnUNetPlans_50G__3d_fullres_first_pass')
            out_prefix = 'pred_DS709_3nnP50G_'

        if num_model == 3:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset708_Ultra_SkulVasc40',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS708_5nnResXL_'

        if num_model == 4:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset706_Vasc2suj_l22_v1',
                                      'nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres')
            out_prefix = 'pred_DS706_5nnResXL_'

        if num_model == 5:
            #img 450^3 18 mn GPU A100 80G 24 cpu
            res_folder = os.path.join(nnres_path, 'Dataset704_Uhcp_skv52_mot_elaBig',
                                      'nnUNetTrainer__nnUNetPlannerResEncXL_80G__3d_fullres')
            out_prefix = 'pred_DS704_3nnResXL_'
        if num_model ==8:
            res_folder = os.path.join(nnres_path, 'Dataset718_Ultra_SkulVasc40_TumorBrast', 'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')
            out_prefix = 'pred_DS718_5nnResL'

    existing_fold = gdir(res_folder,'fold')
    model_files = gfile(existing_fold, 'checkpoint_final.pth')
    existing_fold = get_parent_path(model_files)[0] #selec only dir with checkpoint_final
    existing_fold = [int(ss[-1]) for ss in existing_fold]

    print(f'Selecting model {num_model} out {out_prefix} and found {len(existing_fold)} FOLD')

    return res_folder, existing_fold, out_prefix


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
        fout.append(basdir + '/' + prefix + fn)
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
        if ff[-1]=='/':
            ff=ff[:-1]
        dd = ff.split('/')
        ll = len(dd)
        if remove_ext:
            dd[-1] = remove_extension(dd[-1])
        if concat:
            ss='_'.join(dd[ll-level:])

            file_name.append(ss)
        else:
            file_name.append(dd[ll-level])
        path_name.append('/'.join(dd[:ll-level]))


    if return_string:
        return path_name[0], file_name[0]
    else:
        return path_name, file_name


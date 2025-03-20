import os.path
import sys
import torch
from batchgenerators.utilities.file_and_folder_operations import nifti_files, join, maybe_mkdir_p, isdir
sys.stdout = open(os.devnull, 'w')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
sys.stdout = sys.__stdout__
from predNN.paths import get_model_path_and_fold, addprefixtofilenames



def get_nn_predictor(
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        verbose: bool = False
):
    os.environ['nnUNet_compile'] = 'F'
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose
    )

    if device == torch.device('cpu'):
        torch.set_num_threads(os.cpu_count())
    return predictor


def nn_predict(
        input_file_or_folder: str,
        output_file_or_folder: str,
        use_tta: bool = False,
        device: torch.device = torch.device('cuda'),
        num_model: int =1,
        verbose: bool = False,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=8,
):

    folder_with_parameter_files, existing_fold, out_prefix = get_model_path_and_fold(num_model)

    # find input file or files
    if not input_file_or_folder.startswith('/'):
        input_file_or_folder = join(os.path.curdir, input_file_or_folder)
    input_is_file = True
    if os.path.isdir(input_file_or_folder):
        input_is_file = False
        input_files = nifti_files(input_file_or_folder)
        # output_file_or_folder must be folder in this case
        if output_file_or_folder is None:
            output_files = addprefixtofilenames(input_files, out_prefix)
        else:
            if not output_file_or_folder.startswith('/'):
                output_file_or_folder = join( input_file_or_folder, f'{out_prefix}{output_file_or_folder}')
                print(f'setting output dir to {output_file_or_folder}')

            maybe_mkdir_p(output_file_or_folder)
            output_files = [join(output_file_or_folder, f'{out_prefix}{os.path.basename(i)}') for i in input_files]

        # output_files = [join(output_file_or_folder, os.path.basename(i)) for i in input_files]
        # brain_mask_files = [i[:-7] + out_prefix + '.nii.gz' for i in output_files]
    else:
        #assert not isdir(output_file_or_folder), 'If input is a single file then output must be a filename, not a directory'
        #assert output_file_or_folder.endswith('.nii.gz'), 'Output file must end with .nii.gz'
        if output_file_or_folder is not None:
            out_prefix += output_file_or_folder

        input_files = [input_file_or_folder]
        output_files = addprefixtofilenames(input_files,out_prefix)
        if os.path.isfile(output_files[0]):
            print(f'Existing output file {output_files} \n Skiping')
            return
        #output_files = [join(os.path.curdir, output_file_or_folder)]
        #brain_mask_files = [join(os.path.curdir, output_file_or_folder[:-7] + out_prefix + '.nii.gz')]


    # Model prediction
    predictor = get_nn_predictor(
        use_tta=use_tta,
        device=device,
        verbose=verbose
    )

    predictor.initialize_from_trained_model_folder(
        folder_with_parameter_files,
        existing_fold
    )

    predictor.predict_from_files(
        [[i] for i in input_files],
        output_files,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    if input_is_file:  #otherwise let it in the output folder
        try :
            # remove unnecessary json files
            # may fail if two single file prediction are launch in the same folder (because first process already removed the dataset.json)
            # not a big deal so skip it if errors
            label_file = join(os.path.dirname(output_files[0]), f'label_{out_prefix}.json')
            os.replace(join(os.path.dirname(output_files[0]), 'dataset.json'), label_file)
            #os.remove(join(os.path.dirname(output_files[0]), 'dataset.json'))
            os.remove(join(os.path.dirname(output_files[0]), 'plans.json'))
            os.remove(join(os.path.dirname(output_files[0]), 'predict_from_raw_data_args.json'))
        except:
            we_do_not_care=1


import torch,os
from torch.optim.optimizer import required

from . import __version__

def test_cuda_memory_greate_than(min_memory_GB=12):
    cuda_ok=torch.cuda.is_available()
    if cuda_ok:
        mem_disp,mem_tot = torch.cuda.mem_get_info()
        mem_disp = mem_disp / 1024**3
        if mem_disp < min_memory_GB:
            cuda_ok = False
            print(f' NOT enought GPU mem disponible {mem_disp} over {mem_tot/1024**3} ')
        else :
            print(f' Cuda memory available is {mem_disp}')
    else :
        print(f'NO cuda is_available')
    return cuda_ok

def main():
    #print("\n########################")
    print(f"SIAM: Segment it all model, version {__version__}  \n")
    #print("########################\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input. Can be either a single file name or an input folder. If file: must be '
                                       'nifti (.nii.gz) and can only be 3D. No support for 4d images, use fslsplit to '
                                       'split 4d sequences into 3d images. If folder: all files ending with .nii.gz '
                                       'within that folder will be brain extracted.', required=True, type=str)
    parser.add_argument('-o', '--output', help='Optional. A relative string is added to the output prefix (outputs in '
                                          '<input folder>/<prefix><string>). An absolute path is used as the '
                                          'output folder (default prefix kept).', required=False, type=str)
    parser.add_argument('-device', default='cuda', type=str, required=False,
                        help='used to set on which device the prediction will run. Can be \'cuda\' (=GPU), \'cpu\' or '
                             '\'mps\'. Default: cuda')
    #parser.add_argument('--disable_tta', required=False, action='store_true',
    #                    help='Set this flag to disable test time augmentation. This will make prediction faster at a '
    #                         'slight decrease in prediction quality. Recommended for device cpu')

    parser.add_argument('--verbose', action='store_true', required=False,
                        help="Talk to me.")
    parser.add_argument('-m', '--model', help='Optional: For local use only : An integer to specify which model '
                                              'use -1 to have version 0.1 model '
                        ,default=0, type=int, required=False)
    parser.add_argument('-n', '--nb_fold', help='Optional: int between 1 and 4. use 1 to make quick prediction using '
                                                'only the first fold of each model (only this fold is downloaded). '
                                                'default is 0, it will take all 5 folds  '
                        ,default=0, type=int, required=False)

    parser.add_argument('-r', '--region', help='Optional (model 4): comma separated list of the region models to run '
                                               '(only those are downloaded), ex: -r CSF,GM_allen,WM '
                                               'default is all, use -r none for the tissue only. '
                                               'An unknown name prints the list of the available regions '
                                               '(note the Lesion region, to add Lesion detection)'
                        ,default='all', type=str, required=False)
    parser.add_argument('--lowres', action='store_true', required=False,
                        help="model 4 only : only the low resolution (1mm) tissue prediction, no region "
                             "(only the low res model is downloaded). Same output names as the complete prediction. "
                             "With -n 1 this is a very light version (ex: for a brain mask)")

    parser.add_argument('--save_proba', action='store_true', required=False,
                        help="add this flag, to also save 4D model output (probabilities per class) ")
    parser.add_argument('--regroup_regions', action='store_true', required=False,
                        help="add this flag, with -i -o -m options used at prediction to do posprocessing region grouping and remapping")

    parser.add_argument('-voxelsize', default=0, type=float, required=False,
                        help=" float default 0. Important for too small head size (compare to adult ) "
                             "if not zero, this will set the nifit header voxel resolution (!!without reslicing!!) to isotropic resolution with the given value."
                             " This is equivalent to apply a zoom factor, but changing only the nifti header avoid an extra interpolation. "
                             "The result is converted back to original resolution and orientation")
    parser.add_argument('-nbworker', default=4, type=int, required=False,
                        help=" gpu only : number of worker processes reading/preprocessing and writing the cases "
                             "(each one holds a whole case in RAM), default is 4, reduce if memory issues. "
                             "On cpu, at most 2 workers, and the network uses all the available cpus")
    parser.add_argument('--HR', action='store_true', required=False,
                        help="SAve HR (model 0.5)")

    args = parser.parse_args()

    device_arg = args.device
    if device_arg == 'cuda':
        check_mem = test_cuda_memory_greate_than()
        if check_mem is False:
            if torch.backends.mps.is_available():
                print('Falling back to Apple Metal (MPS)')
                device_arg = 'mps'
            else:
                print('Warning : forcing to cpu')
                device_arg = 'cpu'

    if device_arg == 'mps':
        if not torch.backends.mps.is_available():
            print('MPS not available on this system; forcing to cpu')
            device_arg = 'cpu'
        else:
            # A handful of 3D ops aren't implemented on Metal yet; let them
            # transparently fall back to CPU instead of crashing.
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    if device_arg == 'cpu':
        # all the cpus allocated to this process (taskset / SLURM), not all the cpus of the node
        try:
            nb_cpu = len(os.sched_getaffinity(0))
        except AttributeError:  # macOS / Windows
            nb_cpu = os.cpu_count()
        torch.set_num_threads(nb_cpu)
        # nnUNet caps the network threads with nnUNet_def_n_proc (default 8), read once when nnunetv2 is imported
        os.environ['nnUNet_def_n_proc'] = str(nb_cpu)
        print(f'cpu prediction with {nb_cpu} threads')
    # I put the import here so the default_num_processes set by nnunet does use the env set done here
    from SIAMpred.nn_prediction import nn_predict,regroup_regions
    from SIAMpred.paths import parse_region_list
    regions = parse_region_list(args.region)

    if args.regroup_regions:
        regroup_regions(
            args.input, args.output,
            num_model=args.model,
            regions=regions,
        )
    else:
        nn_predict(args.input,args.output,
                   device=torch.device(device_arg),
                   num_model = args.model,
                   nb_fold = args.nb_fold,
                   regions = regions,
                   lowres = args.lowres,
                   verbose=args.verbose,
                   voxel_size=args.voxelsize,
                   num_process=args.nbworker,
                   save_proba = args.save_proba,
                   save_highres = args.HR
                   )


if __name__ == '__main__':
    main()

    #create model zip
    # nnUNetv2_export_model_to_zip -d 710 -o nn710_nnP50G_f5.zip -c 3d_fullres -tr nnUNetTrainer -p nnUNetPlans_50G


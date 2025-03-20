import torch

from predNN.nn_prediction import nn_predict

def main():
    #print("\n########################")
    print("SIAM (Segment it all model), version 0.1  \n")
    #print("########################\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input. Can be either a single file name or an input folder. If file: must be '
                                       'nifti (.nii.gz) and can only be 3D. No support for 4d images, use fslsplit to '
                                       'split 4d sequences into 3d images. If folder: all files ending with .nii.gz '
                                       'within that folder will be brain extracted.', required=True, type=str)
    parser.add_argument('-o', '--output', help='Optional. A string to specify an output prefix. '
                                     , required=False, type=str)
    parser.add_argument('-device', default='cuda', type=str, required=False,
                        help='used to set on which device the prediction will run. Can be \'cuda\' (=GPU), \'cpu\' or '
                             '\'mps\'. Default: cuda')
    parser.add_argument('--disable_tta', required=False, action='store_true',
                        help='Set this flag to disable test time augmentation. This will make prediction faster at a '
                             'slight decrease in prediction quality. Recommended for device cpu')

    parser.add_argument('--verbose', action='store_true', required=False,
                        help="Talk to me.")
    parser.add_argument('-m', '--model', help='Optional: For local use only : An integer to specify which model '
                                              '(need the variable nnUNet_results to be define)'
                        ,default=0, type=int, required=False)


    args = parser.parse_args()

    nn_predict(args.input,args.output,
                  use_tta=not args.disable_tta,
                  device=torch.device(args.device),
                  num_model = args.model,
                  verbose=args.verbose
                  )


if __name__ == '__main__':
    main()

    #create model zip
    # nnUNetv2_export_model_to_zip -d 710 -o nn710_nnP50G_f5.zip -c 3d_fullres -tr nnUNetTrainer -p nnUNetPlans_50G

    # (torch23) icm-cenir-le70 nnunet/testing_set/ULTRA_all [800] L>
    # for i in {vol_ct,vol_ute,vol_flair,vol_inv2,vol_uni}; do predNN -i $i -o res; done

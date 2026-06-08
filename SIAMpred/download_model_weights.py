from SIAMpred.nn_prediction import get_model_path_and_fold
from SIAMpred.paths import  get_siam_model_dir
import os

if __name__ == "__main__":
    """
    Download all pretrained weights default num 0 model
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', help='Optional: For local use only : An integer to specify which model '
                                              'use -1 to have version 0.1 model '
                        ,default=0, type=int, required=False)
    args = parser.parse_args()

    get_model_path_and_fold(num_model=args.model )
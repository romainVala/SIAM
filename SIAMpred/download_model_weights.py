from SIAMpred.nn_prediction import get_model_path_and_fold
from SIAMpred.paths import  get_siam_model_dir
import os

if __name__ == "__main__":
    """
    Download all pretrained weights default num 0 model
    """
    get_model_path_and_fold(num_model=0 )
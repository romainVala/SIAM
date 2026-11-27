
from SIAMpred.paths import get_model_path_and_fold, parse_region_list

if __name__ == "__main__":
    """
    Download the pretrained weights (default model 0)
    For model 4, only the requested models / folds are downloaded (same options as siam-pred)
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', help='Optional: An integer to specify which model (default 0)'
                        ,default=0, type=int, required=False)
    parser.add_argument('-r', '--region', help='model 4: comma separated list of regions, ex: -r CSF,GM_allen,WM '
                                               '(default all, none for the tissue only)',
                        default=None, type=str, required=False)
    parser.add_argument('-n', '--nb_fold', help='model 4: only the first n folds of each model (default 0 : all)',
                        default=0, type=int, required=False)
    parser.add_argument('--lowres', action='store_true', required=False,
                        help='model 4: only the low res (1mm) tissue model')
    args = parser.parse_args()

    get_model_path_and_fold(num_model=args.model, regions=parse_region_list(args.region),
                            nb_fold=args.nb_fold, lowres=args.lowres)

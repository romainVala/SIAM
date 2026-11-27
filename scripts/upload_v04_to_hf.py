"""
Upload the v0.4 model folder to hugging face as separate files (instead of one big tar.gz),
so that siam-pred only downloads the models / folds it needs.

    pip install huggingface_hub
    hf auth login            # token with write access
    python scripts/upload_v04_to_hf.py ~/siam_params/siam_model_v04

The folder must have the layout expected by SIAMpred/paths.py :
    Dataset782_Ano4DS_vAllen_nolesP2/<trainer>__3d_lowres/{plans,dataset}.json + fold_k/checkpoint_final.*
    Dataset782_Ano4DS_vAllen_nolesP2/<trainer>__3d_cascade_fullres/...
    regions/<DatasetXXX_name>/nnUNetTrainerNoDA__nnUNetResEncUNetLPlans__3d_fullres/...
Only the json files and the checkpoint_final files are uploaded (no validation, logs, checkpoint_latest ...).
A small config.json is also written at the root of the repo : hugging face only counts the downloads
of root files like config.json (siam-pred asks for it at each run), not of the files in sub folders.
Use --delete-tar to also remove the old siam_model_v04.tar.gz from the repo
(siam-pred versions before this change download that archive).
"""
import argparse
import json
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument('folder', help='local siam_model_v04 folder')
parser.add_argument('--repo', default='RomainValabregue/siam-model-v04')
parser.add_argument('--delete-tar', action='store_true', help='delete siam_model_v04.tar.gz from the repo')
args = parser.parse_args()

api = HfApi()
api.upload_large_folder(
    repo_id=args.repo, repo_type='model', folder_path=args.folder,
    allow_patterns=['*.json', '*/fold_*/checkpoint_final.*'],
    ignore_patterns=['hf_repo_files.txt', '.cache/*', '*/validation/*'],
)
# root config.json : the file hugging face uses to count the downloads (siam-pred fetches it at each run)
config = {'model': 'SIAM', 'version': '0.4', 'framework': 'nnUNetv2',
          'usage': 'siam-pred -i INPUT -m 4 (only the needed model files are downloaded)',
          'code': 'https://github.com/romainVala/SIAM'}
api.upload_file(path_or_fileobj=(json.dumps(config, indent=2) + '\n').encode(), path_in_repo='config.json',
                repo_id=args.repo, repo_type='model', commit_message='Add root config.json (download counter)')
files = api.list_repo_files(args.repo)
print(f'{len(files)} files in {args.repo}')
if args.delete_tar and 'siam_model_v04.tar.gz' in files:
    api.delete_file('siam_model_v04.tar.gz', repo_id=args.repo, commit_message='Remove the tar.gz : separate files now')
    print('siam_model_v04.tar.gz deleted')

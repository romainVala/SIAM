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
Use --delete-tar to also remove the old siam_model_v04.tar.gz from the repo
(siam-pred versions before this change download that archive).
"""
import argparse
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
files = api.list_repo_files(args.repo)
print(f'{len(files)} files in {args.repo}')
if args.delete_tar and 'siam_model_v04.tar.gz' in files:
    api.delete_file('siam_model_v04.tar.gz', repo_id=args.repo, commit_message='Remove the tar.gz : separate files now')
    print('siam_model_v04.tar.gz deleted')
